# src/memory/memory_system.py
import time
from typing import Dict, Any
from .qdrant_store import QdrantStore
#from .sqlite_store import SQLiteStore
from .shared_resources import SHARED_QDRANT, get_sqlite_store


class StoryMemorySystem:
    def __init__(self, user_id: str, story_id: str, db_path: str = None):
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
    def long_term_story(self):
        if self._long_term_story is None:
            self._long_term_story = get_sqlite_store(
                table="story_texts", 
                user_id=self.user_id, 
                story_id=self.story_id,
                db_path=self.db_path
            )
        return self._long_term_story
    
    @property
    def long_term_characters(self):
        if self._long_term_characters is None:
            self._long_term_characters = get_sqlite_store(
                table="characters", 
                user_id=self.user_id, 
                story_id=self.story_id,
                db_path=self.db_path
            )
        return self._long_term_characters

    @property
    def long_term_worlds(self):
        if self._long_term_worlds is None:
            self._long_term_worlds = get_sqlite_store(
                table="world_elements", 
                user_id=self.user_id, 
                story_id=self.story_id,
                db_path=self.db_path
            )
        return self._long_term_worlds
    
    @property
    def long_term_docs(self):
        if self._long_term_docs is None:
            self._long_term_docs = get_sqlite_store(
                table="director_notes", 
                user_id=self.user_id, 
                story_id=self.story_id,
                db_path=self.db_path
            )
        return self._long_term_docs
    
    @property
    def long_term_story_progress(self):
        if self._long_term_story_progress is None:
            self._long_term_story_progress = get_sqlite_store(
                table="story_progress", 
                user_id=self.user_id, 
                story_id=self.story_id,
                db_path=self.db_path
            )
        return self._long_term_story_progress

    @property
    def long_term_users(self):
        if self._long_term_users is None:
            self._long_term_users = get_sqlite_store(
                table="users", 
                user_id=self.user_id, 
                story_id=None,
                db_path=self.db_path
            )
        return self._long_term_users

    def qdrant_initialize(self):
        # Qdrant (episodic) - keeping your existing implementation
        if self.episodic_story is None:
            base = QdrantStore(
                collection="episodic_story_memory", 
                user_id=self.user_id, 
                story_id=self.story_id, 
                client=SHARED_QDRANT
            )

            self.episodic_story = base.with_namespace("episodic_story")
            self.episodic_characters = base.with_namespace("episodic_characters")
            self.episodic_worlds = base.with_namespace("episodic_worlds")

    # ---------- Episodic (keeping your existing methods) ----------
    def add_story_summary(self, summary: str, metadata: dict = None):
        self.episodic_story.put(summary, metadata=metadata or {})

    def add_character_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        self.episodic_characters.put_dict_replace_character(
            data=summary, metadata=metadata or {}
        )

    def add_world_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        self.episodic_worlds.put_dict_replace_world(data=summary, metadata=metadata or {})

    def search_episodic(self, query: str, metadata: dict = None, k=5):
        return {
            "story": self.episodic_story.search(query, metadata=metadata, k=k),
            "characters": self.episodic_characters.search(query, metadata=metadata, k=k),
            "world": self.episodic_worlds.search(query, metadata=metadata, k=k),
        }
    
    def search_episodic_story_summary(self, chapter_number):
        hits = self.episodic_story.get_chapter_content(chapter_number=chapter_number)
        return "\n".join(hits)

    def get_context_for_scene(self, current_chapter_number, query: str, k=10):
        episodic_raw = self.search_episodic(
            query, metadata={"chapter_id": current_chapter_number}, k=k
        )
        return episodic_raw
    
    # ---------- Long-Term (SQLite) ----------
    def add_story_scene_cluster(self, text: list, metadata: dict[str, Any] = None):
        for entry in text:
            self.long_term_story.put_text(entry, metadata=metadata or {})

    def add_character_detail(self, scene_bundle, metadata):
        self.long_term_characters.put_characters_or_world(
            details_dict=scene_bundle, metadata=metadata
        )

    def add_world_detail(self, scene_bundle, metadata):
        self.long_term_worlds.put_characters_or_world(
            details_dict=scene_bundle, metadata=metadata
        )

    def get_story_cluster(self, chapter_id):
        return self.long_term_story.get_text(chapter_id=chapter_id)

    def get_long_term_characters(self):
        return self.long_term_characters.get_all_characters_or_worlds()
    
    def get_long_term_worlds(self):
        return self.long_term_worlds.get_all_characters_or_worlds()

    # ---------- Director Docs (Long-Term) ----------
    def add_long_term_document(self, text: str, metadata: dict = None):
        # For director notes, we'll store text directly
        self.long_term_docs.put_text({"content": text}, metadata=metadata)

    def get_long_term_document(self, name: str) -> str:
        docs = self.long_term_docs.get_text(name)  # Use name as chapter_id
        if docs:
            # Assuming it's stored as [{"content": "..."}], return the last one's content
            last_doc = docs[-1] if isinstance(docs, list) else docs
            if isinstance(last_doc, dict):
                return last_doc.get("content", "")
        return ""
    
    # ----------- Story Progress (Long-Term) ----------
    def update_story_progress(self, metadata: dict = None):
        self.long_term_story_progress.put_progress(metadata=metadata or {})
    
    def get_story_progress(self) -> dict:
        return self.long_term_story_progress.get_progress() or {}

    # ---------- Unified scene ingestion ----------
    def add_post_scene_bundle(self, scene_bundle: Dict[str, Any], full_scene_text, metadata: Dict[str, Any]):
        if scene_bundle.get("story_summary"):
            self.add_story_summary(scene_bundle["story_summary"], metadata)
        if scene_bundle.get("character_details"):
            self.add_character_detail(scene_bundle["character_details"], metadata)
        if scene_bundle.get("world_details"):
            self.add_world_detail(scene_bundle["world_details"], metadata)

    def add_post_chapter_bundle(self, parts: Dict[str, Dict], metadata: Dict[str, int]):
        if parts.get("character_summary"):
            self.add_character_summary(parts["character_summary"], metadata)
        if parts.get("world_summary"):
            self.add_world_summary(parts["world_summary"], metadata)

    def get_director_context(self, current_chapter_number, query: str, k=5):
        return self.get_context_for_scene(current_chapter_number, query, k)
    
    def close(self):
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
                store.close()
        
        import gc
        gc.collect()

    def cleanup(self):
        self.close()