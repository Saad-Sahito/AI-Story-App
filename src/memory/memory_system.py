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

    async def long_term_characters_raw(self):
        async with self._lock:
            if self._long_term_characters is None:
                self._long_term_characters = await get_sqlite_store(
                    table="characters_raw",
                    user_id=self.user_id,
                    story_id=self.story_id,
                )
            return self._long_term_characters

    async def long_term_worlds_raw(self):
        async with self._lock:
            if self._long_term_worlds is None:
                self._long_term_worlds = await get_sqlite_store(
                    table="world_elements_raw",
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

    # ---------- Episodic Add methods ----------
    async def add_story_summary(self, summary: str, metadata: dict = None):
        await self.episodic_story.put(summary, metadata=metadata or {})

    async def add_character_summary(self, summary: Dict[str, dict], metadata: dict[str, Any] = None):
        await self.episodic_characters.put_dict_replace_character(
            data=summary, metadata=metadata or {}
        )

    async def add_world_summary(self, summary: Dict[str, dict], metadata: dict[str, Any] = None):
        await self.episodic_worlds.put_dict_replace_world(data=summary, metadata=metadata or {})

    # ---------- Episodic search methods ----------
    async def get_entire_act_chapters_for_act_ingestion_episodic_story(self, current_act: int, chapters: int = 0) -> list:
        act_chapter_text = []
        for i in range(0, chapters + 1):
            print(i)
            print(current_act)
            text = await self.search_single_episodic_story(act_number=current_act, chapter_number=i, summary_type = "chapter summary")
            print(text)
            #if len(text) > 0:
            text_extract = f"Chapter number: {i}\n{text}"
            act_chapter_text.append(text_extract)
        return act_chapter_text


    async def search_episodic_chars_worlds(self, query: str, metadata: dict = None, k: int = 5) -> dict:
        return {
            "characters": await self.episodic_characters.search(query, metadata=metadata, k=k),
            "worlds": await self.episodic_worlds.search(query, metadata=metadata, k=k),
        }
    
    async def search_multiple_episodic_story(self, query: str, metadata: dict = None, k: int = 5) -> str:
        hits = await self.episodic_story.search(query, metadata=metadata, k=k)
        return "\n".join(hits)

    async def search_single_episodic_story(self, act_number, chapter_number, summary_type = "scene summary"):
        hits = await self.episodic_story.get_chapter_content(metadata={"act_number": act_number, "chapter_number": chapter_number, "type": summary_type})
        return "\n".join(hits)

    async def get_char_world_context_for_scene(self, current_chapter_number, query: str, k: int = 10) -> dict:
        episodic_raw = await self.search_episodic_chars_worlds(
            query, metadata={"chapter_id": current_chapter_number}, k=k
        )
        return episodic_raw

    async def get_director_context(self, current_act_number: int, current_chapter_number: int, query: str, k: int = 5) -> dict:
        char_world_context = await self.get_char_world_context_for_scene(current_chapter_number, query, 50)
        chapters_context = await self.search_multiple_episodic_story(
            query, metadata={"chapter_id": current_chapter_number, "type": "chapter summary"}, k=k
        )

        prev_act_summaries = []
        last_few_chapters_summaries = []

        # --- Collect summaries of all previous acts (from 1 to current - 1) ---
        if current_act_number > 1:
            for act_num in range(1, current_act_number):
                act_text = await self.search_single_episodic_story(
                    act_number=act_num,
                    chapter_number=0,
                    summary_type="act summary"
                )
                if act_text:
                    text_extract = f"Act number: {act_num}\n{act_text}"
                    prev_act_summaries.append(text_extract)
                del act_text, text_extract

        # --- Collect summaries for the past 3 chapters only (relative to current act) ---
        for offset in range(3, 0, -1):  # 3, 2, 1
            if current_chapter_number > offset:
                chapt_num = current_chapter_number - offset
                chapt_text = await self.search_single_episodic_story(
                    act_number=current_act_number,
                    chapter_number=chapt_num,
                    summary_type="chapter summary"
                )
                if chapt_text:
                    text_extract = f"Chapter number: {chapt_num}\n{chapt_text}"
                    last_few_chapters_summaries.append(text_extract)
                

        chapters_combined_string = '\n'.join(last_few_chapters_summaries)
        acts_combined_string = '\n'.join(prev_act_summaries)
        return {
            "previous act summaries":acts_combined_string,
            "last few chapter summaries from current act": chapters_combined_string,
            "relevant characters": char_world_context.get("characters", []),
            "relevant worlds": char_world_context.get("worlds", []),
            "relevant chapter context": chapters_context}
    
    # ---------- Long-Term (SQLite) operations ----------
    async def add_story_scene_cluster(self, text: list, metadata: dict[str, Any] = None):
        store = await self.long_term_story()
        for entry in text:
            await store.put_text(entry, metadata=metadata or {})

    async def add_character_detail(self, scene_bundle, metadata):
        store = await self.long_term_characters_raw()
        await store.put_characters_or_world(details_dict=scene_bundle, metadata=metadata or {})

    async def add_world_detail(self, scene_bundle, metadata):
        store = await self.long_term_worlds_raw()
        await store.put_characters_or_world(details_dict=scene_bundle, metadata=metadata or {})

    async def get_story_cluster(self, chapter_id):
        store = await self.long_term_story()
        return await store.get_text(chapter_id)

    async def get_long_term_recent_characters(self, current_chapter_number):
        store = await self.long_term_characters_raw()
        return await store.get_all_characters_or_worlds_by_chapter(chapter_number=current_chapter_number)
    
    async def get_long_term_recent_worlds(self, current_chapter_number):
        store = await self.long_term_worlds_raw()
        return await store.get_all_characters_or_worlds_by_chapter(chapter_number=current_chapter_number)

    async def get_long_term_characters_names(self):
        store = await self.long_term_characters_raw()
        return await store.get_all_character_or_world_names()

    async def get_long_term_worlds_names(self):
        store = await self.long_term_worlds_raw()
        return await store.get_all_character_or_world_names()

    # ---------- Director Docs (Long-Term) ----------
    async def add_long_term_document(self, text: Any, metadata: dict = None):
        """
        Saves a long-term document (e.g. director note, outline, or chapter plan)
        into the long-term memory store.
        """
        store = await self.long_term_docs()
        
        # For director notes, the entry can be a string or dict
        await store.put_to_director_notes(entry=text, metadata=metadata or {})

    async def get_long_term_document(self, metadata: dict):
        """
        Retrieves a long-term document from the store, filtered by metadata fields
        (e.g., chapter_id, type, story_title).
        Returns either the stored text (string) or the last item if multiple.
        """
        store = await self.long_term_docs()
        docs = await store.get_from_director_notes(metadata)

        if not docs:
            return ""

        # If stored as a list of dicts
        if isinstance(docs, list):
            return docs

        # If it's a single dict
        if isinstance(docs, dict):
            return docs.get("content") or docs.get("text") or str(docs)

        # If it's just raw text
        return str(docs)

    # ----------- Story Progress (Long-Term) ----------
    async def update_story_progress(self, metadata: dict = None):
        store = await self.long_term_story_progress()
        await store.update_story_progress(metadata=metadata or {})

    async def increment_chapter(self, word_count_delta: int, scene_id: int) -> dict:
        store = await self.long_term_story_progress()
        return await store.increment_chapter(word_count_delta=word_count_delta, scene_id=scene_id) or {}
    
    async def increment_act(self, new_act_number: int) -> dict:
        store = await self.long_term_story_progress()
        return await store.increment_act(new_act_number=new_act_number)
    
    async def get_story_progress(self) -> dict:
        store = await self.long_term_story_progress()
        return await store.get_story_progress() or {}
    
    async def get_act_progress_summary(self) -> dict:
        store = await self.long_term_story_progress()
        return await store.get_act_progress_summary() or {}
    
    async def mark_story_complete(self):
        store = await self.long_term_story_progress()
        return await store.mark_story_complete()

    # ---------- Unified scene ingestion ----------
    async def add_post_scene_bundle(self, scene_bundle: Dict[str, Any], metadata: Dict[str, Any]):
        if scene_bundle.get("story_summary"):
            await self.add_story_summary(scene_bundle["story_summary"], metadata={"chapter_id": metadata.get("chapter_id", 0),"scene_id": metadata.get("scene_id", 0), "type": "scene summary", "act_id": metadata.get("act_id", 0)})
        if scene_bundle.get("character_details"):
            await self.add_character_detail(scene_bundle["character_details"], metadata)
        if scene_bundle.get("world_details"):
            await self.add_world_detail(scene_bundle["world_details"], metadata)

    async def add_post_chapter_bundle(self, parts: Dict[str, Dict], metadata: Dict[str, Any]):
        if parts.get("summary"):
            await self.add_story_summary(parts["summary"], metadata={"chapter_id": metadata.get("chapter_id", 0), "type": "chapter summary", "act_id": metadata.get("act_id", 0)})
        if parts.get("character_summary"):
            await self.add_character_summary(parts["character_summary"], metadata)
        if parts.get("world_summary"):
            await self.add_world_summary(parts["world_summary"], metadata)

    async def add_post_act_bundle(self, act_bundle: Dict[str, str], metadata: Dict[str, Any]):
        if act_bundle.get("act_summary"):
            await self.add_story_summary(act_bundle["act_summary"], metadata={"type": "act summary", "act_id": metadata.get("act_id", 0)})
        if act_bundle.get("character_progression"):
            await self.add_character_summary(act_bundle["character_progression"], metadata)
        if act_bundle.get("world_progression"):
            await self.add_world_summary(act_bundle["world_progression"], metadata)

    #---------------User Management------------------
    async def update_user_monthly_word_count(self, word_count):
        store = await self.long_term_users()
        return await store.increment_monthly_word_count(words_added=word_count)

    async def get_monthly_word_count(self):
        store = await self.long_term_users()
        return await store.get_monthly_word_count()
    
    # ---------- Cleanup ----------
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
