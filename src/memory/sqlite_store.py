import aiosqlite
import json
import asyncio
import base64
from typing import Dict, Any, Optional, List
from asyncio import Semaphore
from contextlib import asynccontextmanager

class SQLiteStore:
    init_lock = asyncio.Lock()
    def __init__(self, table: str = "long_form", 
                 user_id: str = None, story_id: str = None, db_path: str = "story_memory.db"):
        self.db_path = db_path
        self.table = table
        self.user_id = user_id
        self.story_id = story_id
        self.semaphore = Semaphore(10)  # Limit to 10 concurrent connections

    @asynccontextmanager
    async def _get_connection(self):
        async with self.semaphore:  # limits concurrent DB connections
            conn = await aiosqlite.connect(self.db_path, timeout=5.0)
            conn.row_factory = aiosqlite.Row
            try:
                yield conn
            finally:
                await conn.close()
                
    @staticmethod
    async def _init_database(db_path):
        """Initialize database with all required tables."""
        async with aiosqlite.connect(db_path) as conn:
            await conn.execute("PRAGMA journal_mode=WAL;")  # Enable WAL for concurrent reads
            await conn.execute("PRAGMA synchronous=NORMAL;")
            await conn.execute("PRAGMA foreign_keys = ON")
            
            # Users table (user_tag removed)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    age INTEGER,
                    tier INTEGER,
                    no_genre TEXT, -- array stored as text
                    no_themes TEXT, -- array stored as text
                    stories TEXT DEFAULT '[]', -- JSON array stored as text
                    PRIMARY KEY (user_id)
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
                CREATE TABLE IF NOT EXISTS characters_raw (
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
                CREATE TABLE IF NOT EXISTS world_elements_raw (
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
                    chapter_id INTEGER NOT NULL,
                    type TEXT,              -- Added for filtering by note type
                    story_title TEXT,       -- Added for filtering by story title
                    text TEXT,
                    metadata TEXT,          -- Keep JSON metadata for flexibility
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, story_id, chapter_id, type, story_title)
                )
            """)

            # Story progress table with image_data column
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS story_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    latest_chapter_id INTEGER,
                    continue_scene_id INTEGER,
                    word_count INTEGER DEFAULT 0,
                    image_data BLOB,
                    public BOOLEAN,
                    complete BOOLEAN,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usage_mode TEXT,
                    model_used TEXT,
                    story_quality INTEGER,
                    story_like TEXT,
                    story_improve TEXT,
                    interactivity_naturalness INTEGER,
                    story_pacing TEXT,
                    ease_of_use INTEGER,
                    buggy_or_confusing TEXT,
                    additional_feedback TEXT,
                    nps_score INTEGER,
                    reuse_likelihood INTEGER,
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance (idx_users_user_tag removed)
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_texts_user_story ON story_texts(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_user_story ON characters_raw(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_world_elements_user_story ON world_elements_raw(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_director_notes_user_story ON director_notes(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_progress_user_story ON story_progress(user_id, story_id)")
            
            await conn.commit()

    async def add_user(self, nickname: str, age: Optional[int], user_id: str, tier: int, no_genre: List[str] = [], no_themes: List[str] = [], stories: List[Dict] = None):
        """Add a new user to the users table."""
        stories = stories or []
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,)
                )
                if await cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_id '{user_id}' already exists!"}

                await conn.execute(
                    """
                    INSERT INTO users (user_id, nickname, age, tier, no_genre, no_themes, stories)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, nickname, age, tier, json.dumps(no_genre), json.dumps(no_themes), json.dumps(stories))
                )
                await conn.commit()
                return {"status": "success", "user_id": user_id, "nickname": nickname}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def append_story(self, user_id: str, story_title: str, story_id: str, story_type: str):
        """Append a story to the user's stories list."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []
                
                if any(s["title"] == story_title for s in current_stories):
                    return {"status": "info", "message": f"Story '{story_title}' already exists"}

                current_stories.append({"title": story_title, "story_id": story_id, "story_type": story_type})

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

    async def update_user(self, user_id: str, updates: Dict[str, Any]):
        """Update user row with provided values, leaving unspecified fields unchanged."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT user_id FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": f"❌ user_id '{user_id}' not found!"}

                # Prepare fields to update
                fields = []
                values = []
                for field in ["nickname", "age", "tier", "no_genre", "no_themes"]:
                    if field in updates:
                        if field in ["no_genre", "no_themes"]:
                            values.append(json.dumps(updates[field]))
                        else:
                            values.append(updates[field])
                        fields.append(f"{field} = ?")

                if not fields:
                    return {"status": "error", "message": "❌ No valid fields provided for update"}

                query = f"""
                    UPDATE users
                    SET {', '.join(fields)}
                    WHERE user_id = ?
                """
                values.append(user_id)

                await conn.execute(query, values)
                await conn.commit()
                return {"status": "success", "message": f"User '{user_id}' updated successfully"}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def delete_story(self, user_id: str, story_title: str, story_id: str):
        """Delete a story from the user's stories list AND all related tables."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []

                story_exists = next(
                    (s for s in current_stories if s["title"] == story_title or s["story_id"] == story_id),
                    None
                )
                if not story_exists:
                    return {"status": "info", "message": f"Story '{story_title}' does not exist"}

                current_stories = [
                    s for s in current_stories
                    if s["title"] != story_title and s["story_id"] != story_id
                ]

                await conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )

                await conn.execute("DELETE FROM story_texts WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM characters_raw WHERE user_id = ? AND story_id = ?", (user_id, story_id))
                await conn.execute("DELETE FROM world_elements_raw WHERE user_id = ? AND story_id = ?", (user_id, story_id))
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

    async def get_user_profile(self, user_id: str):
        """Fetch user profile."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT user_id, nickname, age, tier, no_genre, no_themes
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
                    "age": user_row["age"],
                    "tier": user_row["tier"],
                    "no_genre": user_row["no_genre"],
                    "no_themes": user_row["no_themes"]
                }

                return {
                    "status": "success",
                    "profile": user_data,
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    

    async def get_user_profile_with_stories(self, user_id: str):
        """Fetch user profile and their stories with progress (fixed image_data decoding)."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT user_id, nickname, age, tier, stories
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
                    "age": user_row["age"],
                    "tier": user_row["tier"]
                }

                try:
                    story_dicts = json.loads(user_row["stories"] or "[]")
                    if not isinstance(story_dicts, list):
                        story_dicts = []
                except (json.JSONDecodeError, TypeError):
                    story_dicts = []

                stories = []
                for story in story_dicts:
                    if not isinstance(story, dict):
                        continue

                    title = story.get("title", "Untitled Story")
                    story_id = story.get("story_id")
                    story_type = story.get("story_type")
                    if not story_id:
                        continue

                    cursor = await conn.execute(
                        """
                        SELECT latest_chapter_id, continue_scene_id, word_count, metadata, image_data, public, complete
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
                        "model": "",
                        "blurb": "",
                        "image_data": None,
                        "public": False,
                        "complete": False
                    }

                    if progress_row:
                        metadata = json.loads(progress_row["metadata"] or "{}")

                        # Handle image_data correctly
                        raw_image = progress_row["image_data"]
                        if raw_image:
                            if isinstance(raw_image, bytes):
                                image_b64 = base64.b64encode(raw_image).decode("utf-8")
                            else:
                                image_b64 = raw_image  # already base64 string
                        else:
                            image_b64 = None

                        story_data.update({
                            "latest_chapter_id": progress_row["latest_chapter_id"] or 0,
                            "continue_scene_id": progress_row["continue_scene_id"] or 0,
                            "word_count": progress_row["word_count"] or 0,
                            "model": metadata.get("model", ""),
                            "blurb": metadata.get("blurb", ""),
                            "image_data": image_b64,
                            "public": bool(progress_row["public"]),
                            "complete": bool(progress_row["complete"])
                        })

                    stories.append(story_data)

                return {
                    "status": "success",
                    "profile": user_data,
                    "stories": stories
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def update_story_public_status(self, user_id: str, story_id: str, public: bool):
        """Update the public status for a story in the story_progress table."""
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT user_id, story_id FROM story_progress WHERE user_id = ? AND story_id = ?
                    """,
                    (user_id, story_id)
                )
                if not await cursor.fetchone():
                    return {"status": "error", "message": f"❌ Story with user_id '{user_id}' and story_id '{story_id}' not found!"}

                await conn.execute(
                    """
                    UPDATE story_progress
                    SET public = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND story_id = ?
                    """,
                    (public, user_id, story_id)
                )
                await conn.commit()
                return {"status": "success", "message": f"Public status updated to {public} for story_id '{story_id}'"}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def put_text(self, entry: dict, metadata: Optional[Dict[str, Any]] = None):
        chapter_id = (metadata or {}).get("chapter_id", "")
        async with self._get_connection() as conn:
            async with conn.execute('BEGIN'):
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
                await conn.commit()
    
    async def put_to_director_notes(self, entry: Any, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        chapter_id = int(metadata.get("chapter_id", 0))
        type_ = metadata.get("type", "default")
        story_title = metadata.get("story_title", "Untitled")

        async with self._get_connection() as conn:
            async with conn.execute('BEGIN'):
                await conn.execute(f"""
                    INSERT INTO {self.table} (
                        user_id, story_id, chapter_id, type, story_title, text, metadata
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id, chapter_id, type, story_title)
                    DO UPDATE SET
                        text = excluded.text,
                        metadata = excluded.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    self.user_id,
                    self.story_id,
                    chapter_id,
                    type_,
                    story_title,
                    json.dumps(entry),
                    json.dumps(metadata)
                ))
                await conn.commit()

    async def put_progress(self, metadata: Optional[Dict[str, Any]] = None):
        """Insert or update story progress while preserving existing metadata values."""
        metadata = metadata or {}
        chapter_id = metadata.get("latest_chapter_id")
        scene_id = metadata.get("continue_scene_id")
        word_count = metadata.get("word_count", 0)
        story_title = metadata.get("story_title")
        tone_temp = metadata.get("tone_temp")
        model = metadata.get("model")
        token_usage = metadata.get("token_usage")  # dict
        blurb = metadata.get("blurb")
        image_data = metadata.get("image_data")  # Bytes
        public = metadata.get("public")
        complete = metadata.get("complete")

        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT metadata, latest_chapter_id, continue_scene_id, word_count, image_data, public, complete
                FROM story_progress
                WHERE user_id = ? AND story_id = ?
            """, (self.user_id, self.story_id))

            existing_row = await cursor.fetchone()

            if existing_row:
                existing_meta = json.loads(existing_row["metadata"] or "{}")

                # Merge old metadata with new (only overwrite if new value is not None)
                merged_meta = {
                    "chapter_id": chapter_id if chapter_id is not None else existing_meta.get("chapter_id"),
                    "scene_id": scene_id if scene_id is not None else existing_meta.get("scene_id"),
                    "word_count": word_count if word_count != 0 else existing_meta.get("word_count", 0),
                    "story_title": story_title if story_title is not None else existing_meta.get("story_title"),
                    "tone_temp": tone_temp if tone_temp is not None else existing_meta.get("tone_temp"),
                    "model": model if model is not None else existing_meta.get("model"),
                    "token_usage": token_usage if token_usage is not None else existing_meta.get("token_usage", {}),
                    "blurb": blurb if blurb is not None else existing_meta.get("blurb")
                }

                await conn.execute("""
                    UPDATE story_progress
                    SET latest_chapter_id = ?,
                        continue_scene_id = ?,
                        word_count = ?,
                        image_data = ?,
                        public = ?,
                        complete = ?,
                        metadata = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND story_id = ?
                """, (
                    merged_meta["chapter_id"],
                    merged_meta["scene_id"],
                    merged_meta["word_count"],
                    image_data if image_data is not None else existing_row["image_data"],
                    public if public is not None else existing_row["public"],
                    complete if complete is not None else existing_row["complete"],
                    json.dumps(merged_meta),
                    self.user_id,
                    self.story_id
                ))

            else:
                # No existing record — insert new
                clean_meta = {
                    "chapter_id": chapter_id,
                    "scene_id": scene_id,
                    "word_count": word_count,
                    "story_title": story_title,
                    "tone_temp": tone_temp,
                    "model": model,
                    "token_usage": token_usage,
                    "blurb": blurb,
                    "public": public,
                    "complete": complete
                }

                await conn.execute("""
                    INSERT INTO story_progress (
                        user_id, story_id, latest_chapter_id, continue_scene_id, word_count, image_data, public, complete, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.user_id,
                    self.story_id,
                    chapter_id,
                    scene_id,
                    word_count,
                    image_data,
                    public,
                    complete,
                    json.dumps(clean_meta)
                ))

            await conn.commit()

    async def put_characters_or_world(self, details_dict: Dict[str, str], metadata: Dict[str, Any]):
        """Append new details to existing character/world details."""
        async with self._get_connection() as conn:
            for name, details in details_dict.items():
                cursor = await conn.execute(
                    f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                    (self.user_id, self.story_id, name)
                )
                row = await cursor.fetchone()
                old_details = row['details'] if row else ""
                
                if isinstance(details, dict):
                    details = " ".join(f"{k}: {v}" for k, v in details.items())
                
                new_details = (old_details + " " + details).strip()
                
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
                await conn.commit()

    async def get_text(self, chapter_id: str) -> List[dict]:
        """Get the full JSON array (list of dicts) for a chapter."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT text FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                (self.user_id, self.story_id, chapter_id)
            )
            row = await cursor.fetchone()
            
            if row and row['text']:
                return json.loads(row['text'])
            return []
        
    async def get_from_director_notes(self, metadata: dict):
        """
        Retrieve text entry by chapter_id, type, and story_title.
        Returns either a list or a single string depending on stored data.
        """
        chapter_id = metadata.get("chapter_id", 0)
        type_ = metadata.get("type", "default")
        story_title = metadata.get("story_title", "Untitled")

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT text FROM {self.table}
                WHERE user_id = ?
                AND story_id = ?
                AND chapter_id = ?
                AND type = ?
                AND story_title = ?
                """,
                (self.user_id, self.story_id, chapter_id, type_, story_title)
            )
            row = await cursor.fetchone()

            if not row or not row["text"]:
                return []

            text = row["text"]

            try:
                decoded = json.loads(text)
                return decoded
            except json.JSONDecodeError:
                return text

    async def get_progress(self) -> Optional[Dict[str, Any]]:
        """Get the latest story progress."""
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                "SELECT latest_chapter_id, continue_scene_id, word_count, image_data, public, complete, metadata FROM story_progress WHERE user_id = ? AND story_id = ? LIMIT 1",
                (self.user_id, self.story_id)
            )
            row = await cursor.fetchone()
            
            if row:
                metadata = json.loads(row['metadata']) if row['metadata'] else {}
                # Ensure token_usage is a dict
                metadata['token_usage'] = metadata.get('token_usage', {}) if isinstance(metadata.get('token_usage'), (dict, list)) else {}
                return {
                    'latest_chapter_id': row['latest_chapter_id'],
                    'continue_scene_id': row['continue_scene_id'],
                    'word_count': row['word_count'],
                    'image_data': row['image_data'],
                    'public': row['public'],
                    'complete': row['complete'],
                    'metadata': metadata
                }
            return None

    async def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        """Get specific character or world element."""
        async with self._get_connection() as conn:
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
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT name, details FROM {self.table} WHERE user_id = ? AND story_id = ?",
                (self.user_id, self.story_id)
            )
            rows = await cursor.fetchall()
            return {row['name']: row['details'] for row in rows}

    async def close(self):
        """Close database connection (SQLite handles this automatically)."""
        pass

    #---------------Analytics Only------------------------
    async def get_all_users(self):
        """Fetch all rows from the users table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM users")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_story_texts(self):
        """Fetch all rows from the story_texts table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM story_texts")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_characters_raw(self):
        """Fetch all rows from the characters_raw table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM characters_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_world_elements_raw(self):
        """Fetch all rows from the world_elements_raw table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM world_elements_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_director_notes(self):
        """Fetch all rows from the director_notes table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM director_notes")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_story_progress(self):
        """Fetch all rows from the story_progress table."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM story_progress")
            rows = await cursor.fetchall()
            # Decode binary image data to base64 for easy export
            results = []
            for row in rows:
                record = dict(row)
                # if record.get("image_data"):
                #     record["image_data"] = base64.b64encode(record["image_data"]).decode("utf-8")
                results.append(record)
            return results

    async def get_all_feedback(self):
        """Fetch all rows from the feedback table."""
        print("Getting feedback...")
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM feedback")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_data(self):
        """
        Fetch all data from every table in a structured format.
        Returns a dictionary for easy JSON serialization or analytics use.
        """
        users = await self.get_all_users()
        story_texts = await self.get_all_story_texts()
        characters = await self.get_all_characters_raw()
        world_elements = await self.get_all_world_elements_raw()
        director_notes = await self.get_all_director_notes()
        story_progress = await self.get_all_story_progress()
        feedback = await self.get_all_feedback()

        return {
            "users": users,
            "story_texts": story_texts,
            "characters_raw": characters,
            "world_elements_raw": world_elements,
            "director_notes": director_notes,
            "story_progress": story_progress,
            "feedback": feedback
        }

    #--------------Feedback----------------------
    async def insert_feedback(self, feedback: Dict[str, Any]):
        """Insert one feedback record into the feedback table."""
        try:
            async with self._get_connection() as conn:
                # Define columns (excluding id and submitted_at which auto-generate)
                columns = [
                    "usage_mode", "model_used", "story_quality", "story_like", "story_improve",
                    "interactivity_naturalness", "story_pacing", "ease_of_use", 
                    "buggy_or_confusing", "additional_feedback", "nps_score", "reuse_likelihood"
                ]
                
                placeholders = ", ".join(["?"] * len(columns))
                sql = f"INSERT INTO feedback ({', '.join(columns)}) VALUES ({placeholders})"
                
                values = [feedback.get(col, None) for col in columns]
                
                await conn.execute(sql, values)  # ✅ Async execute
                await conn.commit()              # ✅ Async commit
                
                return {"status": "success", "message": "Feedback inserted successfully"}
                
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}