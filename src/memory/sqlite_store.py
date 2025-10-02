# src/memory/sqlite_store.py
import aiosqlite
import json
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
from asyncio import Semaphore
#from tenacity import retry, stop_after_attempt, wait_exponential


class SQLiteStore:
    init_lock = asyncio.Lock()
    def __init__(self, db_path: str = "story_memory.db", table: str = "long_form", 
                 user_id: str = None, story_id: str = None):
        self.db_path = db_path
        self.table = table
        self.user_id = user_id
        self.story_id = story_id
        self.semaphore = Semaphore(10)  # Limit to 10 concurrent connections
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        asyncio.get_event_loop().run_until_complete(self._init_database())

    async def _get_connection(self):
        async with self.semaphore:
            try:
                conn = await aiosqlite.connect(self.db_path, timeout=5.0)
                conn.row_factory = aiosqlite.Row
                return conn
            except aiosqlite.Error as e:
                raise Exception(f"Failed to connect to SQLite: {e}")
    
    async def _init_database(self):
        """Initialize database with all required tables."""
        async with self.init_lock:
            async with await aiosqlite.connect(self.db_path) as conn:
                await conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL for concurrent reads
                await conn.execute("PRAGMA synchronous=NORMAL;")
                await conn.execute("PRAGMA foreign_keys = ON")
                
                # Users table
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT NOT NULL,
                        nickname TEXT NOT NULL,
                        user_tag TEXT NOT NULL,
                        age INTEGER,
                        stories TEXT DEFAULT '[]', -- JSON array stored as text
                        PRIMARY KEY (user_id),
                        UNIQUE(user_id, user_tag)
                    )
                """)
                
                # Story texts table
                await conn.execute("""
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
                await conn.execute("""
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
                await conn.execute("""
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
                await conn.execute("""
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
                await conn.execute("""
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
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_users_user_tag ON users(user_tag)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_texts_user_story ON story_texts(user_id, story_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_user_story ON characters(user_id, story_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_world_elements_user_story ON world_elements(user_id, story_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_director_notes_user_story ON director_notes(user_id, story_id)")
                await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_progress_user_story ON story_progress(user_id, story_id)")
                
                await conn.commit()

    # ---------- User Management Methods ----------
    async def add_user(self, nickname: str, user_tag: str, age: Optional[int], user_id: str, stories: List[Dict] = None):
        """Add a new user to the users table."""
        stories = stories or []
        try:
            async with await self._get_connection() as conn:
                # Check if user_id already exists
                cursor = await conn.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,)
                )
                if await cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_id '{user_id}' already exists!"}

                # Check if user_tag already exists
                cursor = await conn.execute(
                    "SELECT user_tag FROM users WHERE user_tag = ?",
                    (user_tag,)
                )
                if await cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_tag '{user_tag}' already exists!"}

                # Validate user_tag
                if not user_tag.strip():
                    return {"status": "error", "message": "❌ user_tag cannot be empty!"}

                # Insert new user
                await conn.execute(
                    """
                    INSERT INTO users (user_id, nickname, user_tag, age, stories)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (user_id, nickname, user_tag, age, json.dumps(stories))
                )
                await conn.commit()
                return {"status": "success", "user_id": user_id, "nickname": nickname, "user_tag": user_tag}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def append_story(self, user_id: str, story_title: str, story_id: str, story_type: str):
        """Append a story to the user's stories list."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []
                
                # Check if story already exists
                if any(s["title"] == story_title for s in current_stories):
                    return {"status": "info", "message": f"Story '{story_title}' already exists"}

                # Append new story
                current_stories.append({"title": story_title, "story_id": story_id, "story_type": story_type})

                # Update stories
                await conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )
                await conn.commit()
                return {"status": "success", "message": f"Story '{story_title}' added", "stories": current_stories}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def delete_story(self, user_id: str, story_title: str, story_id: str):
        """Delete a story from the user's stories list AND all related tables."""
        try:
            async with await self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
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
                await conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )

                # Delete related rows from all other tables
                await conn.execute("DELETE FROM story_texts WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM characters WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM world_elements WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM director_notes WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM story_progress WHERE user_id = ? AND story_id = ?", (user_id, story_id))

                await conn.commit()

                return {
                    "status": "success",
                    "message": f"Story '{story_title}' and all related data deleted",
                    "stories": current_stories
                }
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}



    async def get_user_profile_with_stories(self, user_id: str):
        """Fetch user profile and their stories with progress (hardened)."""
        try:
            async with await self._get_connection() as conn:
                # Fetch user profile
                cursor = await conn.execute(
                    """
                    SELECT user_id, nickname, user_tag, age, stories
                    FROM users
                    WHERE user_id = ?
                    """,
                    (user_id,)
                )
                user_row = await cursor.fetchone()
                if not user_row:
                    return {"status": "error", "message": "❌ User not found"}

                user_data = {
                    "user_id": user_row["user_id"],
                    "nickname": user_row["nickname"],
                    "user_tag": user_row["user_tag"],
                    "age": user_row["age"],
                }

                # Parse stories JSON safely
                try:
                    story_dicts = json.loads(user_row["stories"] or "[]")
                    if not isinstance(story_dicts, list):
                        story_dicts = []
                except (json.JSONDecodeError, TypeError):
                    story_dicts = []

                stories = []
                for story in story_dicts:
                    if not isinstance(story, dict):
                        continue  # skip malformed entries

                    title = story.get("title", "Untitled Story")
                    story_id = story.get("story_id")
                    story_type = story.get("story_type")
                    if not story_id:
                        continue  # skip if story_id missing (invalid story record)

                    # Fetch progress for this story
                    cursor = await conn.execute(
                        """
                        SELECT latest_chapter_id, continue_scene_id, word_count, metadata
                        FROM story_progress
                        WHERE user_id = ? AND story_id = ?
                        LIMIT 1
                        """,
                        (user_id, story_id)
                    )
                    progress_row = await cursor.fetchone()

                    story_data = {
                        "title": title,
                        "story_id": story_id,
                        "story_type": story_type,
                        "latest_chapter_id": 0,
                        "continue_scene_id": 0,
                        "word_count": 0,
                    }

                    if progress_row:
                        story_data.update({
                            "latest_chapter_id": progress_row["latest_chapter_id"] or 0,
                            "continue_scene_id": progress_row["continue_scene_id"] or 0,
                            "word_count": progress_row["word_count"] or 0,
                        })

                    stories.append(story_data)

                return {
                    "status": "success",
                    "profile": user_data,
                    "stories": stories
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}


    # ---------- PUT methods ----------
    #@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def put_text(self, entry: dict, metadata: Optional[Dict[str, Any]] = None):
        chapter_id = (metadata or {}).get("chapter_id", "")
        async with self._get_connection() as conn:
            async with conn.execute('BEGIN'):  # Start transaction
                cursor = await conn.execute(
                    f"SELECT text FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                    (self.user_id, self.story_id, chapter_id)
                )
                row = await cursor.fetchone()
                old_text = json.loads(row['text']) if row and row['text'] else []
                if not isinstance(old_text, list):
                    old_text = [old_text]
                new_text = old_text + [entry]
                await conn.execute(f"""
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
                await conn.commit()  # Commit transaction

    async def put_progress(self, metadata: Optional[Dict[str, Any]] = None):
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
        
        async with await self._get_connection() as conn:
            await conn.execute("""
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

    async def put_characters_or_world(self, details_dict: Dict[str, str], metadata: Dict[str, Any]):
        """Append new details to existing character/world details."""
        async with await self._get_connection() as conn:
            for name, details in details_dict.items():
                # Get existing details
                cursor = await conn.execute(
                    f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                    (self.user_id, self.story_id, name)
                )
                row = await cursor.fetchone()
                old_details = row['details'] if row else ""
                
                # Flatten details if it's a dict
                if isinstance(details, dict):
                    details = " ".join(f"{k}: {v}" for k, v in details.items())
                
                new_details = (old_details + " " + details).strip()
                
                # Upsert the record
                await conn.execute(f"""
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
    async def get_text(self, chapter_id: str) -> List[dict]:
        """Get the full JSON array (list of dicts) for a chapter."""
        async with await self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT text FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                (self.user_id, self.story_id, chapter_id)
            )
            row = await cursor.fetchone()
            
            if row and row['text']:
                return json.loads(row['text'])
            return []

    async def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest story progress."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT latest_chapter_id, continue_scene_id, word_count, metadata FROM story_progress WHERE user_id = ? AND story_id = ? LIMIT 1",
                (self.user_id, self.story_id)
            )
            row = await cursor.fetchone()
            
            if row:
                return {
                    'latest_chapter_id': row['latest_chapter_id'],
                    'continue_scene_id': row['continue_scene_id'],
                    'word_count': row['word_count'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None

    async def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific character or world element."""
        async with await self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT name, details, metadata FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                (self.user_id, self.story_id, name)
            )
            row = await cursor.fetchone()
            
            if row:
                return {
                    'name': row['name'],
                    'details': row['details'],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            return None

    async def get_all_characters_or_worlds(self) -> Dict[str, Any]:
        """Get all characters or world elements."""
        async with await self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT name, details FROM {self.table} WHERE user_id = ? AND story_id = ?",
                (self.user_id, self.story_id)
            )
            return {row['name']: row['details'] for row in cursor.fetchall()}

    async def close(self):
        """Close database connection (SQLite handles this automatically)."""
        pass  # SQLite connections are automatically closed when context exits

