# src/memory/supabase_store.py
from typing import Dict, Any, Optional, List
from supabase import create_client, Client
import os


class SupabaseStore:
    def __init__(self, table: str = "long_form", user_id: str = None, story_id: str = None):
        url: str = os.getenv("SUPABASE_URL")
        key: str = os.getenv("SUPABASE_KEY")
        if not url or not key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

        self.client: Client = create_client(url, key)
        self.table = table
        self.user_id = user_id
        self.story_id = story_id

    # ---------- PUT ----------
    def put_text(self, text: str, metadata: Optional[Dict[str, Any]] = None):
        """Append generic story text to a chapter (scene not used)."""
        chapter_id = (metadata or {}).get("chapter_id", "")

        # Fetch existing text for this user + story + chapter
        existing = (
            self.client.table(self.table)
            .select("text")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .eq("chapter_id", chapter_id)
            .execute()
        )

        old_text = existing.data[0].get("text", "") if existing.data else ""
        new_text = (old_text + " " + text).strip()

        # Upsert row
        self.client.table(self.table).upsert({
            "user_id": self.user_id,
            "story_id": self.story_id,
            "chapter_id": chapter_id,
            "text": new_text,
            "metadata": metadata.copy() if metadata else {}
        }).execute()

    def put_progress(
        self,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Insert or update story progress (chapter_id, scene_id, story_title) 
        for this user + story. Flexible for any table.
        """
        # Extract metadata fields
        chapter_id = (metadata or {}).get("latest_chapter_id", "")
        scene_id = (metadata or {}).get("continue_scene_id", "")
        story_title = (metadata or {}).get("story_title", "")
        word_count = (metadata or {}).get("word_count", 0)

        # Build minimal metadata
        clean_meta = {
            "chapter_id": chapter_id,
            "scene_id": scene_id,
            "story_title": story_title,
            "word_count": word_count,
        }

        # Upsert row
        response = (
            self.client.table(self.table)
            .upsert({
                "user_id": self.user_id,
                "story_id": self.story_id,
                "latest_chapter_id": int(chapter_id) if chapter_id is not None else None,
                "continue_scene_id": int(scene_id) if scene_id is not None else None,
                "word_count": word_count,
                "metadata": clean_meta,
            })
            .execute()
        )

        return response

    def put_characters_or_world(
        self,
        details_dict: Dict[str, str],
        metadata: Dict[str, Any]
    ):
        """Append new details to the existing details string, but always replace metadata with latest."""
        for name, details in details_dict.items():
            existing = (
                self.client.table(self.table)
                .select("details")
                .eq("user_id", self.user_id)
                .eq("story_id", self.story_id)
                .eq("name", name)
                .execute()
            )

            old_details = existing.data[0].get("details", "") if existing.data else ""

            # Flatten dicts if necessary
            if isinstance(details, dict):
                details = " ".join(f"{k}: {v}" for k, v in details.items())

            new_details = (old_details + " " + details).strip()

            self.client.table(self.table).upsert({
                "user_id": self.user_id,
                "story_id": self.story_id,
                "name": name,
                "details": new_details,
                "metadata": metadata
            }).execute()

    # ---------- GET ----------
    def get_texts(self, limit: int = 10) -> List[Dict[str, Any]]:
        result = (
            self.client.table(self.table)
            .select("*")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .limit(limit)
            .execute()
        )
        return result.data or []

    def get_text(self, chapter_id: str) -> Optional[str]:
        result = (
            self.client.table(self.table)
            .select("text")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .eq("chapter_id", chapter_id)
            .execute()
        )
        return result.data[0]["text"] if result.data else None
    
    def get_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest story progress (chapter_id, scene_id, story_title) 
        for this user + story.
        """
        result = (
            self.client.table(self.table)
            .select("latest_chapter_id, continue_scene_id, word_count, metadata")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None


    def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        result = (
            self.client.table(self.table)
            .select("name, details, metadata")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .eq("name", name)
            .execute()
        )
        return result.data[0] if result.data else None

    def get_all_characters_or_worlds(self) -> Dict[str, Any]:
        result = (
            self.client.table(self.table)
            .select("name, details")
            .eq("user_id", self.user_id)
            .eq("story_id", self.story_id)
            .execute()
        )
        return {row["name"]: row["details"] for row in result.data if "name" in row}
