# src/memory/memory_system.py

from typing import Dict, Any
from .qdrant_store import QdrantStore
from .supabase_store import SupabaseStore

class StoryMemorySystem:
    def __init__(self, user_id: str, story_id: str):
        self.user_id = user_id
        self.story_id = story_id  # must be provided
        #self.story_title = story_title


        # Short-Term: simple dict
        #self.short_term: Dict[str, Any] = {}

        # Supabase (long-term) - all operations automatically scoped by user_id + story_title

        self._long_term_story = None
        self._long_term_characters = None
        self._long_term_worlds = None
        self._long_term_docs = None
        self._long_term_story_progress = None

        self.episodic_story = None
        self.episodic_characters = None
        self.episodic_worlds = None

    @property
    def long_term_story(self):
        if self._long_term_story is None:
            self._long_term_story = SupabaseStore(
                table="story_texts", user_id=self.user_id, story_id=self.story_id
            )
        return self._long_term_story
    
    @property
    def long_term_characters(self):
        if self._long_term_characters is None:
            self._long_term_characters = SupabaseStore(
                table="characters", user_id=self.user_id, story_id=self.story_id
            )
        return self._long_term_characters

    @property
    def long_term_worlds(self):
        if self._long_term_worlds is None:
            self._long_term_worlds = SupabaseStore(
            table="world_elements", user_id=self.user_id, story_id=self.story_id
        )
        return self._long_term_worlds
    
    @property
    def long_term_docs(self):
        if self._long_term_docs is None:
            self._long_term_docs = SupabaseStore(
            table="director_notes", user_id=self.user_id, story_id=self.story_id
        )
        return self._long_term_docs
    
    @property
    def long_term_story_progress(self):
        if self._long_term_story_progress is None:
            self._long_term_story_progress = SupabaseStore(
            table="story_progress", user_id=self.user_id, story_id=self.story_id
        )
        return self._long_term_story_progress


    def qdrant_initialize(self):
        # Qdrant (episodic) - one shared collection, filtered by user_id + story_id
        if self.episodic_story is None:
            base = QdrantStore(collection="episodic_story_memory", user_id=self.user_id, story_id=self.story_id)

            # Use namespaces (payload key) to separate story/characters/world
            self.episodic_story = base.with_namespace("episodic_story")
            self.episodic_characters = base.with_namespace("episodic_characters")
            self.episodic_worlds = base.with_namespace("episodic_worlds")


    # ---------- Short-Term Current Chapter ----------
    # def set_current_chapter(self, chapter_text: str):
    #     self.short_term["current_chapter"] = (
    #         (self.short_term.get("current_chapter", "") + "\n" + chapter_text)
    #         if self.short_term.get("current_chapter")
    #         else chapter_text
    #     )

    # def get_current_chapter(self) -> str:
    #     return self.short_term.get("current_chapter", "Chapter has not started yet.")

    # def reset_current_chapter(self):
    #     self.short_term.pop("current_chapter", None)

    # # ---------- Short-Term Scene Context ----------
    # def clear_scene_context_cache(self):
    #     self.short_term.pop("scene_context", None)
    
    # ---------- Episodic ----------
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
    
    # ---------- Long-Term (Supabase) ----------
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

    # used in main_test.py
    def get_story_cluster(self, chapter_id):
        return self.long_term_story.get_text(chapter_id=chapter_id)

    def get_long_term_characters(self):
        return self.long_term_characters.get_all_characters_or_worlds()
    
    def get_long_term_worlds(self):
        return self.long_term_worlds.get_all_characters_or_worlds()

    # ---------- Director Docs (Long-Term) ----------
    def add_long_term_document(self, text: str, metadata: dict = None):
        self.long_term_docs.put_text(text, metadata=metadata)

    def get_long_term_document(self, name: str) -> str:
        doc = self.long_term_docs.get_text(name)
        return doc if doc else ""
    
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
        for attr in [
            "_long_term_story",
            "_long_term_characters",
            "_long_term_worlds",
            "_long_term_docs",
            "_long_term_story_progress",
            "episodic_story",
            "episodic_characters",
            "episodic_worlds",
        ]:
            store = getattr(self, attr, None)
            if store is not None and hasattr(store, "close"):
                store.close()
            setattr(self, attr, None)

        import gc
        gc.collect()

    def cleanup(self):
        self.close()