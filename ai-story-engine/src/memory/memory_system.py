# src/memory/memory_system.py

from typing import Dict, Any

from networkx import hits
from .qdrant_store import QdrantStore
from .mongodb_store import MongoStore


class StoryMemorySystem:
    def __init__(self):
        """
        Memory system for story generation:
        - Short-term: current working chapter + cached scene context
        - Episodic: summaries/tags for quick context retrieval (Qdrant vectors)
        - Long-term: detailed records (MongoDB text/dicts)
        - Docs: director notes (continuity, decisions, etc.)
        """
        # Short-Term: simple dict
        self.short_term: Dict[str, Any] = {}

        # Episodic memory in Qdrant (for semantic search)
        base = QdrantStore(collection="episodic_story_memory")
        self.episodic_story = base.with_namespace("episodic_story")
        self.episodic_characters = base.with_namespace("episodic_characters")
        self.episodic_world = base.with_namespace("episodic_world")

        # Long-Term memory in MongoDB (raw details, retrievable by queries)
        self.long_term_story = MongoStore(db_name="long_story_memory", collection="story_texts")
        self.long_term_characters = MongoStore(db_name="long_story_memory", collection="characters")
        self.long_term_world = MongoStore(db_name="long_story_memory", collection="world_elements")

        # Director notes / misc docs
        self.long_term_docs = MongoStore(db_name="long_story_memory", collection="director_notes")

    # ---------- Short-Term Current Chapter----------
    def set_current_chapter(self, chapter_text: str):
        if "current_chapter" not in self.short_term:
            self.short_term["current_chapter"] = ""

        if self.short_term["current_chapter"]:
            self.short_term["current_chapter"] += "\n" + chapter_text
        else:
            self.short_term["current_chapter"] = chapter_text

    def get_current_chapter(self) -> str:
        return self.short_term.get("current_chapter", "Chapter has not started yet.")

    def reset_current_chapter(self):

        self.short_term.pop("current_chapter", None)

    # ---------- Short-Term Scene Context ----------
    def clear_scene_context_cache(self):
        self.short_term.pop("scene_context", None)
    
    # ---------- Episodic ----------
    def add_story_summary(self, summary: str, metadata: dict = None):
        self.episodic_story.put(summary, metadata=metadata or {})

    def add_character_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        self.episodic_characters.put_dict_replace_character(data=summary, metadata=metadata or {})

    def add_world_summary(self, summary: Dict[str,str], metadata: dict[str,int] = None):
        self.episodic_world.put_dict_replace_world(data=summary, metadata=metadata or {})

    def search_episodic(self, query: str, metadata: dict = None, k=5):
        return {
            "story": self.episodic_story.search(query, metadata=metadata, k=k),
            "characters": self.episodic_characters.search(query, metadata=metadata, k=k),
            "world": self.episodic_world.search(query, metadata=metadata, k=k),
        }
    
    def search_episodic_story_summary(self, chapter_number):
        hits = self.episodic_story.get_chapter_content(chapter_number=chapter_number)

        #print("\n".join(hits))
        return "\n".join(hits)

    def get_context_for_scene(self, current_chapter_number, query: str, k=10):
        # Episodic: vector-based semantic search only
        episodic_raw = self.search_episodic(query, metadata={"chapter_id": current_chapter_number}, k=k)
        # print("EPISODIC RAW: ", episodic_raw)
        # episodic_hits = []
        # for r in episodic_raw["story"]:
        #     chap = r.metadata.get("chapter")
        #     if chap and chap != current_chapter_number:
        #         episodic_hits.append({
        #             "text": getattr(r, "page_content", None) or getattr(r, "text", ""),
        #             "metadata": {**r.metadata},
        #         })

        # context = {
        #     "episodic": episodic_hits
        # }
        # self.short_term["scene_context"] = context
        return episodic_raw
    
    # ---------- Long-Term (Mongo) ----------
    def add_story_chapter(self, text: str, metadata: dict[str, int] = None):
        self.long_term_story.put_text(text, metadata=metadata or {})

    def add_character_detail(self, scene_bundle, metadata): # Add chapter and scene id since last update in metadata
        self.long_term_characters.put_characters_or_world(
                scene_bundle,
                metadata["scene_id"],
                metadata["chapter_id"]
            )

    def add_world_detail(self, scene_bundle, metadata): # Add chapter and scene id since last update in metadata
        self.long_term_world.put_characters_or_world(
                scene_bundle,
                metadata["scene_id"],
                metadata["chapter_id"]
            )

    def get_long_term_story(self, limit=10):
        return self.long_term_story.get_texts(limit=limit)

    def get_long_term_characters_and_worlds(self):
        return self.long_term_characters.get_all_characters_or_worlds()

    # def get_long_term_world(self):
    #     return self.long_term_world.get_all_characters_or_worlds()

    # ---------- Director Docs (Long-Term) ----------
    def add_long_term_document(self,text: str, metadata: dict = None):
        #doc = {"doc_name": name}
        # if metadata:
        #     doc.update(metadata)
        self.long_term_docs.put_text(text, metadata=metadata)

    def get_long_term_document(self, name: str) -> str:
        doc = self.long_term_docs.get_text(name)
        #print("DOC: ", doc)
        return doc if doc else ""

    # ---------- Formatting helpers ----------
    # def _fmt_hits(self, hits, max_items=3, max_chars=300):
    #     out = []
    #     for h in (hits or [])[:max_items]:
    #         text = h.get("text", str(h))
    #         meta = h.get("metadata", {}) or {}
    #         text = text[:max_chars].strip()
    #         out.append({"text": text, "metadata": meta})
    #     return out

    # def format_context(self, context: dict, max_items=3) -> str:
    #     def block(title, items):
    #         if not items:
    #             return ""
    #         lines = [f"== {title} =="]
    #         for i, item in enumerate(items[:max_items], 1):  # cap at max_items
    #             md = item.get("metadata", {})
    #             md_str = ", ".join(f"{k}:{v}" for k, v in md.items()) if md else "no-meta"
    #             lines.append(f"{i}. {item['text']}  [{md_str}]")
    #         return "\n".join(lines) + "\n"

    #     # episodic only
    #     epi = context.get("episodic", [])

    #     return block("Episodic Memory", epi) or "== Episodic Memory ==\n(none)\n"

    def get_director_context(self, current_chapter_number, query: str, k=5):
        ctx = self.get_context_for_scene(
            current_chapter_number=current_chapter_number,
            query=query,
            k=k
        )
        return ctx
        

    # ---------- Unified scene ingestion ----------
    def add_post_scene_bundle(self, scene_bundle: Dict[str, Any], full_scene_text, metadata: Dict[str, int]):
        #self.add_story_scene(full_scene_text, metadata)
        
        #print("SCENE BUNDLE: ", scene_bundle)
        if "story_summary" in scene_bundle and scene_bundle["story_summary"]:
            self.add_story_summary(scene_bundle["story_summary"], metadata)

        # Add all characters at once    
        if "character_details" in scene_bundle and scene_bundle["character_details"]:
            self.add_character_detail(
                scene_bundle=scene_bundle["character_details"],
                metadata=metadata
            )

        # Add all world entries at once
        if "world_details" in scene_bundle and scene_bundle["world_details"]:
            self.add_world_detail(
                scene_bundle["world_details"],
                metadata
            )
        

    def add_post_chapter_bundle(self, parts: Dict[str, Dict], metadata: Dict[str, int]):
        # Episodic (summaries)
        #self.add_story_summary(parts["story_summary"], metadata)
        if "character_summary" in parts and parts["character_summary"]:
            self.add_character_summary(parts["character_summary"], metadata)
        if "world_summary" in parts and parts["world_summary"]:
            self.add_world_summary(parts["world_summary"], metadata)
