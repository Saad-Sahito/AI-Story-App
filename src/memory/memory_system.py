# src/memory/memory_system.py

from typing import Dict, Any
from asyncio import Lock
import asyncio
from fastapi import HTTPException
from .qdrant_store import QdrantStore
from .shared_resources import SHARED_QDRANT, get_sqlite_store



class StoryMemorySystem:
    def __init__(self, user_id: str, story_id: str, db_path: str = None):
        self._lock = Lock()
        self.user_id = user_id
        self.story_id = story_id
        self.db_path = db_path  # Optional custom database path

        # Long-term storage using SQLite (lazy-initialized via getter methods)
        self._long_term_story = None
        self._long_term_characters = None
        self._long_term_worlds = None
        self._long_term_docs = None
        self._long_term_story_progress = None
        self._long_term_users = None

        # Episodic storage using Qdrant
        self.episodic_story = None
        self.episodic_characters = None
        self.episodic_worlds = None

    # ---------- Long-term SQLite getters (async) ----------
    async def long_term_story(self):
        async with self._lock:
            if self._long_term_story is None:
                # get_sqlite_store is async and returns a SQLiteStore instance
                self._long_term_story = await get_sqlite_store(
                    table="story_texts",
                    user_id=self.user_id,
                    story_id=self.story_id,
                    # db_path=self.db_path
                )
            return self._long_term_story

    async def long_term_characters(self):
        async with self._lock:
            if self._long_term_characters is None:
                self._long_term_characters = await get_sqlite_store(
                    table="characters",
                    user_id=self.user_id,
                    story_id=self.story_id,
                )
            return self._long_term_characters

    async def long_term_worlds(self):
        async with self._lock:
            if self._long_term_worlds is None:
                self._long_term_worlds = await get_sqlite_store(
                    table="world_elements",
                    user_id=self.user_id,
                    story_id=self.story_id,
                )
            return self._long_term_worlds

    async def long_term_docs(self):
        async with self._lock:
            if self._long_term_docs is None:
                self._long_term_docs = await get_sqlite_store(
                    table="director_notes",
                    user_id=self.user_id,
                    story_id=self.story_id,
                )
            return self._long_term_docs

    async def long_term_story_progress(self):
        async with self._lock:
            if self._long_term_story_progress is None:
                self._long_term_story_progress = await get_sqlite_store(
                    table="story_progress",
                    user_id=self.user_id,
                    story_id=self.story_id,
                )
            return self._long_term_story_progress

    async def long_term_users(self):
        async with self._lock:
            if self._long_term_users is None:
                self._long_term_users = await get_sqlite_store(
                    table="users",
                    user_id=self.user_id,
                    story_id=None,
                )
            return self._long_term_users

    # ---------- Qdrant initialization (episodic) ----------
    async def qdrant_initialize(self):
        """
        Initialize Qdrant with retry logic for handling timeouts.
        """
        async with self._lock:
            if self.episodic_story is None:
                max_retries = 3
                last_error = None
                
                for attempt in range(max_retries):
                    try:
                        print(f"🔄 Initializing Qdrant (attempt {attempt + 1}/{max_retries})")
                        
                        base = QdrantStore(
                            collection="episodic_story_memory",
                            user_id=self.user_id,
                            story_id=self.story_id,
                            client=SHARED_QDRANT,
                        )
                        
                        # This is where the timeout can occur
                        await base.async_init()
                        
                        # Create namespaced stores
                        self.episodic_story = base.with_namespace("episodic_story")
                        self.episodic_characters = base.with_namespace("episodic_characters")
                        self.episodic_worlds = base.with_namespace("episodic_worlds")
                        
                        print(f"✅ Qdrant initialized successfully")
                        return  # Success!
                        
                    except Exception as e:
                        last_error = e
                        error_msg = str(e).lower()
                        
                        # Check if it's a timeout or connection issue
                        if "timeout" in error_msg or "408" in error_msg:
                            if attempt < max_retries - 1:
                                wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                                print(f"⚠️ Qdrant timeout on attempt {attempt + 1}, retrying in {wait_time}s...")
                                await asyncio.sleep(wait_time)
                            else:
                                print(f"❌ Qdrant initialization failed after {max_retries} attempts")
                                raise HTTPException(
                                    status_code=503,
                                    detail=f"Qdrant service unavailable: {str(e)}"
                                )
                        else:
                            # Non-timeout error, don't retry
                            print(f"❌ Qdrant initialization failed with non-timeout error: {e}")
                            raise
                
                # If we get here, all retries failed
                if last_error:
                    raise last_error

    # ---------- Episodic methods ----------
    async def add_story_summary(self, summary: str, metadata: dict = None):
        await self.episodic_story.put(summary, metadata=metadata or {})

    async def add_character_summary(self, summary: Dict[str, str], metadata: dict[str, int] = None):
        await self.episodic_characters.put_dict_replace_character(
            data=summary, metadata=metadata or {}
        )

    async def add_world_summary(self, summary: Dict[str, str], metadata: dict[str, int] = None):
        await self.episodic_worlds.put_dict_replace_world(data=summary, metadata=metadata or {})

    async def search_episodic(self, query: str, metadata: dict = None, k: int = 5):
        return {
            "story": await self.episodic_story.search(query, metadata=metadata, k=k),
            "characters": await self.episodic_characters.search(query, metadata=metadata, k=k),
            "world": await self.episodic_worlds.search(query, metadata=metadata, k=k),
        }

    async def search_episodic_story_summary(self, chapter_number):
        hits = await self.episodic_story.get_chapter_content(chapter_number=chapter_number)
        return "\n".join(hits)

    async def get_context_for_scene(self, current_chapter_number, query: str, k: int = 10):
        episodic_raw = await self.search_episodic(
            query, metadata={"chapter_id": current_chapter_number}, k=k
        )
        return episodic_raw

    # ---------- Long-Term (SQLite) operations ----------
    async def add_story_scene_cluster(self, text: list, metadata: dict[str, Any] = None):
        store = await self.long_term_story()
        for entry in text:
            await store.put_text(entry, metadata=metadata or {})

    async def add_character_detail(self, scene_bundle, metadata):
        store = await self.long_term_characters()
        await store.put_characters_or_world(details_dict=scene_bundle, metadata=metadata or {})

    async def add_world_detail(self, scene_bundle, metadata):
        store = await self.long_term_worlds()
        await store.put_characters_or_world(details_dict=scene_bundle, metadata=metadata or {})

    async def get_story_cluster(self, chapter_id):
        store = await self.long_term_story()
        return await store.get_text(chapter_id)

    async def get_long_term_characters(self):
        store = await self.long_term_characters()
        return await store.get_all_characters_or_worlds()

    async def get_long_term_worlds(self):
        store = await self.long_term_worlds()
        return await store.get_all_characters_or_worlds()

    # ---------- Director Docs (Long-Term) ----------
    async def add_long_term_document(self, text: str, metadata: dict = None):
        store = await self.long_term_docs()
        # For director notes, we'll store text directly as an entry
        await store.put_text({"content": text}, metadata=metadata or {})

    async def get_long_term_document(self, name: str) -> str:
        store = await self.long_term_docs()
        docs = await store.get_text(name)  # Use name as chapter_id
        if docs:
            # Assuming it's stored as [{"content": "..."}], return the last one's content
            last_doc = docs[-1] if isinstance(docs, list) else docs
            if isinstance(last_doc, dict):
                return last_doc.get("content", "")
        return ""

    # ----------- Story Progress (Long-Term) ----------
    async def update_story_progress(self, metadata: dict = None):
        store = await self.long_term_story_progress()
        await store.put_progress(metadata=metadata or {})

    async def get_story_progress(self) -> dict:
        store = await self.long_term_story_progress()
        return await store.get_progress() or {}

    # ---------- Unified scene ingestion ----------
    async def add_post_scene_bundle(self, scene_bundle: Dict[str, Any], metadata: Dict[str, Any]):
        if scene_bundle.get("story_summary"):
            await self.add_story_summary(scene_bundle["story_summary"], metadata)
        if scene_bundle.get("character_details"):
            await self.add_character_detail(scene_bundle["character_details"], metadata)
        if scene_bundle.get("world_details"):
            await self.add_world_detail(scene_bundle["world_details"], metadata)

    async def add_post_chapter_bundle(self, parts: Dict[str, Dict], metadata: Dict[str, int]):
        if parts.get("character_summary"):
            await self.add_character_summary(parts["character_summary"], metadata)
        if parts.get("world_summary"):
            await self.add_world_summary(parts["world_summary"], metadata)

    async def get_director_context(self, current_chapter_number, query: str, k: int = 5):
        return await self.get_context_for_scene(current_chapter_number, query, k)

    async def close(self):
        """Close all storage connections."""
        for attr in [
            "_long_term_story",
            "_long_term_characters",
            "_long_term_worlds",
            "_long_term_docs",
            "_long_term_story_progress",
            "_long_term_users",
            "episodic_story",
            "episodic_characters",
            "episodic_worlds",
        ]:
            store = getattr(self, attr, None)
            if store is not None:
                # Some store.close may be async; try awaiting if coroutine
                close_func = getattr(store, "close", None)
                if close_func:
                    # call and await if coroutine
                    result = close_func()
                    if hasattr(result, "__await__"):
                        await result

        import gc
        gc.collect()

    async def cleanup(self):
        await self.close()
