# src/memory/sqlite_store.py
import sqlite3
import json
from typing import Dict, Any, Optional, List
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: str = "story_memory.db", table: str = "long_form", 
                 user_id: str = None, story_id: str = None):
        self.db_path = db_path
        self.table = table
        self.user_id = user_id
        self.story_id = story_id
        
        # Ensure database directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database and tables
        self._init_database()
    
    def _init_database(self):
        """Initialize database with all required tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # Users table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    user_tag TEXT NOT NULL,
                    age INTEGER,
                    stories TEXT DEFAULT '[]', -- JSON array stored as text
                    PRIMARY KEY (user_id),
                    UNIQUE(user_tag)
                )
            """)
            
            # Story texts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS story_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    text TEXT, -- JSON array stored as text
                    metadata TEXT, -- JSON stored as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, chapter_id)
                )
            """)
            
            # Characters table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    details TEXT,
                    metadata TEXT, -- JSON stored as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, name)
                )
            """)
            
            # World elements table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS world_elements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    details TEXT,
                    metadata TEXT, -- JSON stored as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, name)
                )
            """)
            
            # Director notes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS director_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    text TEXT,
                    metadata TEXT, -- JSON serialized as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, story_id, chapter_id)
                )
            """)
            
            # Story progress table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS story_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    latest_chapter_id INTEGER,
                    continue_scene_id INTEGER,
                    word_count INTEGER DEFAULT 0,
                    metadata TEXT, -- JSON stored as text
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id)
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_users_user_tag ON users(user_tag)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_story_texts_user_story ON story_texts(user_id, story_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_user_story ON characters(user_id, story_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_world_elements_user_story ON world_elements(user_id, story_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_director_notes_user_story ON director_notes(user_id, story_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_story_progress_user_story ON story_progress(user_id, story_id)")
            
            conn.commit()

    def _get_connection(self):
        """Get database connection with row factory for dict-like access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- User Management Methods ----------
    def add_user(self, nickname: str, user_tag: str, age: Optional[int], user_id: str, stories: List[Dict] = None):
        """Add a new user to the users table."""
        stories = stories or []
        try:
            with self._get_connection() as conn:
                # Check if user_id already exists
                cursor = conn.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,)
                )
                if cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_id '{user_id}' already exists!"}

                # Check if user_tag already exists
                cursor = conn.execute(
                    "SELECT user_tag FROM users WHERE user_tag = ?",
                    (user_tag,)
                )
                if cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_tag '{user_tag}' already exists!"}

                # Validate user_tag
                if not user_tag.strip():
                    return {"status": "error", "message": "❌ user_tag cannot be empty!"}

                # Insert new user
                conn.execute(
                    """
                    INSERT INTO users (user_id, nickname, user_tag, age, stories)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, nickname, user_tag, age, json.dumps(stories))
                )
                conn.commit()
                return {"status": "success", "user_id": user_id, "nickname": nickname, "user_tag": user_tag}
        except sqlite3.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    def append_story(self, user_id: str, story_title: str, story_id: str):
        """Append a story to the user's stories list."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []
                
                # Check if story already exists
                if any(s["title"] == story_title for s in current_stories):
                    return {"status": "info", "message": f"Story '{story_title}' already exists"}

                # Append new story
                current_stories.append({"title": story_title, "story_id": story_id})

                # Update stories
                conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )
                conn.commit()
                return {"status": "success", "message": f"Story '{story_title}' added", "stories": current_stories}
        except sqlite3.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    def delete_story(self, user_id: str, story_title: str, story_id: str):
        """Delete a story from the user's stories list AND all related tables."""
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []

                # Check if story exists
                story_exists = next(
                    (s for s in current_stories if s["title"] == story_title or s["story_id"] == story_id),
                    None
                )
                if not story_exists:
                    return {"status": "info", "message": f"Story '{story_title}' does not exist"}

                # Remove story from JSON list
                current_stories = [
                    s for s in current_stories
                    if s["title"] != story_title and s["story_id"] != story_id
                ]

                # Update users.stories
                conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )

                # Delete related rows from all other tables
                conn.execute("DELETE FROM story_texts WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                conn.execute("DELETE FROM characters WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                conn.execute("DELETE FROM world_elements WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                conn.execute("DELETE FROM director_notes WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                conn.execute("DELETE FROM story_progress WHERE user_id = ? AND story_id = ?", (user_id, story_id))

                conn.commit()

                return {
                    "status": "success",
                    "message": f"Story '{story_title}' and all related data deleted",
                    "stories": current_stories
                }
        except sqlite3.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}


    def get_user_profile_with_stories(self, user_id: str):
        """Fetch user profile and their stories with progress."""
        try:
            with self._get_connection() as conn:
                # Fetch user profile
                cursor = conn.execute(
                    """
                    SELECT user_id, nickname, user_tag, age, stories
                    FROM users
                    WHERE user_id = ?
                    """,
                    (user_id,)
                )
                user_row = cursor.fetchone()
                if not user_row:
                    return {"status": "error", "message": "❌ User not found"}

                user_data = {
                    "user_id": user_row["user_id"],
                    "nickname": user_row["nickname"],
                    "user_tag": user_row["user_tag"],
                    "age": user_row["age"],
                }

                # Process stories
                story_dicts = json.loads(user_row["stories"]) if user_row["stories"] else []
                stories = []

                for story in story_dicts:
                    title = story.get("title")
                    story_id = story.get("story_id")

                    # Fetch progress for this story
                    cursor = conn.execute(
                        """
                        SELECT latest_chapter_id, continue_scene_id, word_count, metadata
                        FROM story_progress
                        WHERE user_id = ? AND story_id = ?
                        LIMIT 1
                        """,
                        (user_id, story_id)
                    )
                    progress_row = cursor.fetchone()

                    story_data = {
                        "title": title,
                        "story_id": story_id,
                        "latest_chapter_id": 0,
                        "continue_scene_id": 0,
                        "word_count": 0,
                    }

                    if progress_row:
                        story_data["latest_chapter_id"] = progress_row["latest_chapter_id"] or 0
                        story_data["continue_scene_id"] = progress_row["continue_scene_id"] or 0
                        story_data["word_count"] = progress_row["word_count"] or 0

                    stories.append(story_data)

                return {
                    "status": "success",
                    "profile": user_data,
                    "stories": stories if stories else []
                }
        except sqlite3.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    # ---------- PUT methods ----------
    def put_text(self, entry: dict, metadata: Optional[Dict[str, Any]] = None):
        """Append a nested dict to the chapter's JSON array."""
        chapter_id = (metadata or {}).get("chapter_id", "")
        
        with self._get_connection() as conn:
            # Fetch existing text array
            cursor = conn.execute(
                f"SELECT text FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                (self.user_id, self.story_id, chapter_id)
            )
            row = cursor.fetchone()
            
            if row and row['text']:
                old_text = json.loads(row['text'])
                if not isinstance(old_text, list):
                    old_text = [old_text]
            else:
                old_text = []
            
            # Append new entry
            new_text = old_text + [entry]
            
            # Upsert the record
            conn.execute(f"""
                INSERT INTO {self.table} (user_id, story_id, chapter_id, text, metadata)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, story_id, chapter_id) DO UPDATE SET
                text = excluded.text,
                metadata = excluded.metadata
            """, (
                self.user_id, 
                self.story_id, 
                chapter_id, 
                json.dumps(new_text),
                json.dumps(metadata or {})
            ))

    def put_progress(self, metadata: Optional[Dict[str, Any]] = None):
        """Insert or update story progress."""
        metadata = metadata or {}
        chapter_id = metadata.get("latest_chapter_id")
        scene_id = metadata.get("continue_scene_id")
        word_count = metadata.get("word_count", 0)
        story_title = metadata.get("story_title", None)
        
        clean_meta = {
            "chapter_id": chapter_id,
            "scene_id": scene_id,
            "word_count": word_count,
            "story_title": story_title
        }
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO story_progress (user_id, story_id, latest_chapter_id, continue_scene_id, word_count, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, story_id) DO UPDATE SET
                latest_chapter_id = excluded.latest_chapter_id,
                continue_scene_id = excluded.continue_scene_id,
                word_count = excluded.word_count,
                metadata = excluded.metadata,
                updated_at = CURRENT_TIMESTAMP
            """, (
                self.user_id,
                self.story_id,
                int(chapter_id) if chapter_id is not None else None,
                int(scene_id) if scene_id is not None else None,
                word_count,
                json.dumps(clean_meta)
            ))

    def put_characters_or_world(self, details_dict: Dict[str, str], metadata: Dict[str, Any]):
        """Append new details to existing character/world details."""
        with self._get_connection() as conn:
            for name, details in details_dict.items():
                # Get existing details
                cursor = conn.execute(
                    f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                    (self.user_id, self.story_id, name)
                )
                row = cursor.fetchone()
                old_details = row['details'] if row else ""
                
                # Flatten details if it's a dict
                if isinstance(details, dict):
                    details = " ".join(f"{k}: {v}" for k, v in details.items())
                
                new_details = (old_details + " " + details).strip()
                
                # Upsert the record
                conn.execute(f"""
                    INSERT INTO {self.table} (user_id, story_id, name, details, metadata)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id, name) DO UPDATE SET
                    details = excluded.details,
                    metadata = excluded.metadata,
                    updated_at = CURRENT_TIMESTAMP
                """, (
                    self.user_id,
                    self.story_id,
                    name,
                    new_details,
                    json.dumps(metadata)
                ))

    # ---------- GET methods ----------
    def get_text(self, chapter_id: str) -> List[dict]:
        """Get the full JSON array (list of dicts) for a chapter."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT text FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                (self.user_id, self.story_id, chapter_id)
            )
            row = cursor.fetchone()
            
            if row and row['text']:
                return json.loads(row['text'])
            return []

    def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest story progress."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT latest_chapter_id, continue_scene_id, word_count, metadata FROM story_progress WHERE user_id = ? AND story_id = ? LIMIT 1",
                (self.user_id, self.story_id)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'latest_chapter_id': row['latest_chapter_id'],
                    'continue_scene_id': row['continue_scene_id'],
                    'word_count': row['word_count'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None

    def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific character or world element."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT name, details, metadata FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                (self.user_id, self.story_id, name)
            )
            row = cursor.fetchone()
            
            if row:
                return {
                    'name': row['name'],
                    'details': row['details'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None

    def get_all_characters_or_worlds(self) -> Dict[str, Any]:
        """Get all characters or world elements."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                f"SELECT name, details FROM {self.table} WHERE user_id = ? AND story_id = ?",
                (self.user_id, self.story_id)
            )
            return {row['name']: row['details'] for row in cursor.fetchall()}

    def close(self):
        """Close database connection (SQLite handles this automatically)."""
        pass  # SQLite connections are automatically closed when context exits

