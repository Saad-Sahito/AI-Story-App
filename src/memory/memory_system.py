# src/memory/memory_system.py

from typing import Dict, Any
from asyncio import Lock
from .qdrant_store import QdrantStore
#from .sqlite_store import SQLiteStore
from .shared_resources import SHARED_QDRANT, get_sqlite_store


class StoryMemorySystem:
    def __init__(self, user_id: str, story_id: str, db_path: str = None):
        self._lock = Lock()
        self.user_id = user_id
        self.story_id = story_id
        self.db_path = db_path  # Optional custom database path

        # Long-term storage using SQLite
        self._long_term_story = None
        self._long_term_characters = None
        self._long_term_worlds = None
        self._long_term_docs = None
        self._long_term_story_progress = None

        # Episodic storage using Qdrant (keeping this as you had it)
        self.episodic_story = None
        self.episodic_characters = None
        self.episodic_worlds = None

    @property
    async def long_term_story(self):
        async with self._lock:
            if self._long_term_story is None:
                self._long_term_story = await get_sqlite_store(
                    table="story_texts", 
                    user_id=self.user_id, 
                    story_id=self.story_id,
                    db_path=self.db_path
                )
            return self._long_term_story
    
    @property
    async def long_term_characters(self):
        async with self._lock:
            if self._long_term_characters is None:
                self._long_term_characters = await get_sqlite_store(
                    table="characters", 
                    user_id=self.user_id, 
                    story_id=self.story_id,
                    db_path=self.db_path
                )
            return self._long_term_characters

    @property
    async def long_term_worlds(self):
        async with self._lock:
            if self._long_term_worlds is None:
                self._long_term_worlds = await get_sqlite_store(
                    table="world_elements", 
                    user_id=self.user_id, 
                    story_id=self.story_id,
                    db_path=self.db_path
                )
            return self._long_term_worlds
    
    @property
    async def long_term_docs(self):
        async with self._lock:
            if self._long_term_docs is None:
                self._long_term_docs = await get_sqlite_store(
                    table="director_notes", 
                    user_id=self.user_id, 
                    story_id=self.story_id,
                    db_path=self.db_path
                )
            return self._long_term_docs
    
    @property
    async def long_term_story_progress(self):
        async with self._lock:
            if self._long_term_story_progress is None:
                self._long_term_story_progress = await get_sqlite_store(
                    table="story_progress", 
                    user_id=self.user_id, 
                    story_id=self.story_id,
                    db_path=self.db_path
                )
            return self._long_term_story_progress

    @property
    async def long_term_users(self):
        async with self._lock:
            if self._long_term_users is None:
                self._long_term_users = await get_sqlite_store(
                    table="users", 
                    user_id=self.user_id, 
                    story_id=None,
                    db_path=self.db_path
                )
            return self._long_term_users

    async def qdrant_initialize(self):
        if self.episodic_story is None:
            base = QdrantStore(
                collection="episodic_story_memory", 
                user_id=self.user_id, 
                story_id=self.story_id, 
                client=SHARED_QDRANT
            )
            await base.async_init()
            self.episodic_story = base.with_namespace("episodic_story")
            self.episodic_characters = base.with_namespace("episodic_characters")
            self.episodic_worlds = base.with_namespace("episodic_worlds")

    # ---------- Episodic (keeping your existing methods) ----------
    async def add_story_summary(self, summary: str, metadata: dict = None):
        await self.episodic_story.put(summary, metadata=metadata or {})

    async def add_character_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        await self.episodic_characters.put_dict_replace_character(
            data=summary, metadata=metadata or {}
        )

    async def add_world_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        await self.episodic_worlds.put_dict_replace_world(data=summary, metadata=metadata or {})

    async def search_episodic(self, query: str, metadata: dict = None, k=5):
        return {
            "story": await self.episodic_story.search(query, metadata=metadata, k=k),
            "characters": await self.episodic_characters.search(query, metadata=metadata, k=k),
            "world": await self.episodic_worlds.search(query, metadata=metadata, k=k),
        }
    
    async def search_episodic_story_summary(self, chapter_number):
        hits = await self.episodic_story.get_chapter_content(chapter_number=chapter_number)
        return "\n".join(hits)

    async def get_context_for_scene(self, current_chapter_number, query: str, k=10):
        episodic_raw = await self.search_episodic(
            query, metadata={"chapter_id": current_chapter_number}, k=k
        )
        return episodic_raw
    
    # ---------- Long-Term (SQLite) ----------
    async def add_story_scene_cluster(self, text: list, metadata: dict[str, Any] = None):
        for entry in text:
            await self.long_term_story.put_text(entry, metadata=metadata or {})

    async def add_character_detail(self, scene_bundle, metadata):
        await self.long_term_characters.put_characters_or_world(
            details_dict=scene_bundle, metadata=metadata
        )

    async def add_world_detail(self, scene_bundle, metadata):
        await self.long_term_worlds.put_characters_or_world(
            details_dict=scene_bundle, metadata=metadata
        )

    async def get_story_cluster(self, chapter_id):
        return await self.long_term_story.get_text(chapter_id=chapter_id)

    async def get_long_term_characters(self):
        return await self.long_term_characters.get_all_characters_or_worlds()
    
    async def get_long_term_worlds(self):
        return await self.long_term_worlds.get_all_characters_or_worlds()

    # ---------- Director Docs (Long-Term) ----------
    async def add_long_term_document(self, text: str, metadata: dict = None):
        # For director notes, we'll store text directly
        await self.long_term_docs.put_text({"content": text}, metadata=metadata)

    async def get_long_term_document(self, name: str) -> str:
        docs = await self.long_term_docs.get_text(name)  # Use name as chapter_id
        if docs:
            # Assuming it's stored as [{"content": "..."}], return the last one's content
            last_doc = docs[-1] if isinstance(docs, list) else docs
            if isinstance(last_doc, dict):
                return last_doc.get("content", "")
        return ""
    
    # ----------- Story Progress (Long-Term) ----------
    async def update_story_progress(self, metadata: dict = None):
        await self.long_term_story_progress.put_progress(metadata=metadata or {})
    
    async def get_story_progress(self) -> dict:
        return await self.long_term_story_progress.get_progress() or {}

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

    async def get_director_context(self, current_chapter_number, query: str, k=5):
        return await self.get_context_for_scene(current_chapter_number, query, k)
    
    async def close(self):
        """Close all storage connections."""
        for attr in ["_long_term_story",
            "_long_term_characters",
            "_long_term_worlds",
            "_long_term_docs",
            "_long_term_story_progress",
            "episodic_story",
            "episodic_characters",
            "episodic_worlds"]:
            store = getattr(self, attr, None)
            if store is not None and hasattr(store, "close"):
                await store.close()
        
        import gc
        gc.collect()

    async def cleanup(self):
        await self.close()