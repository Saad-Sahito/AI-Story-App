import aiosqlite
import json
import asyncio
import ast
import base64
from datetime import datetime, timedelta, UTC
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
        self.semaphore = Semaphore(20)  # Limit to 10 concurrent connections

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

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT NOT NULL PRIMARY KEY,
                    nickname TEXT NOT NULL,
                    age INTEGER,
                    tier INTEGER,
                    monthly_word_count INTEGER DEFAULT 0,
                    stories TEXT DEFAULT '[]',
                    signup_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_reset_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    lqs_customer_id TEXT,
                    lqs_subscription_id TEXT,
                    lqs_variant_id TEXT,
                    subscription_status TEXT DEFAULT 'inactive',
                    subscription_renewal_date TIMESTAMP,
                    cancel_at_period_end BOOLEAN DEFAULT 0
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS story_texts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    chapter_id TEXT NOT NULL,
                    text TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, chapter_id)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS characters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    prev_names TEXT,
                    details TEXT,
                    significance TEXT,
                    chapter_id INTEGER NOT NULL,
                    act_id INTEGER NOT NULL,
                    scene_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, name)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS world_elements (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    prev_names TEXT,
                    details TEXT,
                    significance TEXT,
                    chapter_id INTEGER NOT NULL,
                    act_id INTEGER NOT NULL,
                    scene_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, name)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS director_notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    chapter_id INTEGER NOT NULL,
                    act_id INTEGER NOT NULL,
                    type TEXT,
                    text TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, story_id, chapter_id, act_id, type)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS story_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    story_title TEXT,
                    flow_type TEXT,
                    target_medium TEXT,
                    total_acts INTEGER,
                    target_length INTEGER,
                    story_length INTEGER,
                    last_chapter_id INTEGER,
                    genre_list TEXT,
                    sub_genre_list TEXT,
                    themes_list TEXT,
                    pov TEXT,
                    prose_style TEXT,
                    tense TEXT,
                    blurb TEXT,
                    tone_temp FLOAT,
                    min_age INTEGER,
                    author_type TEXT,
                    latest_phase INTEGER,
                    current_act_id INTEGER,
                    latest_chapter_id INTEGER,
                    continue_scene_id INTEGER,
                    story_word_count INTEGER DEFAULT 0,
                    chapter_word_count INTEGER DEFAULT 0,
                    status TEXT,
                    author_token_usage TEXT,
                    director_token_usage TEXT,
                    writer_token_usage TEXT,
                    ingestor_token_usage TEXT,
                    utility_token_usage TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id),
                    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
                )
            """)
            
            # Feedback table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    usage_mode TEXT, 
                    story_quality INTEGER, 
                    story_like TEXT, 
                    story_improve TEXT, 
                    interactivity_meaningfulness INTEGER, 
                    story_pacing TEXT, 
                    ease_of_use INTEGER, 
                    buggy_or_confusing TEXT, 
                    additional_feedback TEXT, 
                    nps_score INTEGER, 
                    reuse_likelihood INTEGER, 
                    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_texts_user_story ON story_texts(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_characters_user_story ON characters_raw(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_world_elements_user_story ON world_elements_raw(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_director_notes_user_story ON director_notes(user_id, story_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_story_progress_user_story ON story_progress(user_id, story_id)")
            
            await conn.commit()

    # Helper function to get shared story data


    async def _add_user(self, nickname: str, age: Optional[int], user_id: str, tier: int, stories: List[Dict] = None):
        stories = stories or []
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("SELECT user_id FROM users WHERE user_id = ?", (user_id,))
                if await cursor.fetchone():
                    return {"status": "error", "message": f"❌ user_id '{user_id}' already exists!"}

                await conn.execute("""
                    INSERT INTO users (user_id, nickname, age, tier, stories)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, nickname, age, tier, json.dumps(stories)))
                await conn.commit()
                return {"status": "success", "user_id": user_id, "nickname": nickname}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def _append_story(self, user_id: str, story_id: str):
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
                
                if any(s["story_id"] == story_id for s in current_stories):
                    return {"status": "info", "message": f"Story '{story_id}' already exists"}

                current_stories.append({"story_id": story_id})

                await conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )
                await conn.commit()
                return {"status": "success", "message": f"Story '{story_id}' added"}
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def _update_user(self, user_id: str, updates: Dict[str, Any]):
        #print(f"\n[UPDATE_USER] user_id={user_id}, updates={updates}, db_path={self.db_path}")
        try:
            async with self._get_connection() as conn:
                # Pre-check
                pre = await conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                pre_row = await pre.fetchone()
                #print(f"[PRE] User exists: {bool(pre_row)} | Data: {dict(pre_row) if pre_row else None}")

                allowed = {"nickname", "age", "tier"}
                to_update = {k: v for k, v in updates.items() if k in allowed}
                if not to_update:
                    return {"status": "error", "message": "No valid fields"}

                set_parts = [f"{k} = ?" for k in to_update]
                values = list(to_update.values()) + [user_id]
                query = f"UPDATE users SET {', '.join(set_parts)} WHERE user_id = ?"

                #print(f"[EXEC] {query} | {values}")

                cursor = await conn.execute(query, values)
                await conn.commit()
                #print(f"[COMMIT] rowcount={cursor.rowcount}")

                # Post-check
                post = await conn.execute("SELECT nickname, age, tier FROM users WHERE user_id = ?", (user_id,))
                # post_row = await post.fetchone()
                #print(f"[POST] Updated row: {dict(post_row) if post_row else None}")

                if cursor.rowcount == 0:
                    return {"status": "error", "message": "User not found or no changes"}
                return {"status": "success", "updated": list(to_update.keys())}
            
        except Exception as e:
            print(f"[EXCEPTION update_user] {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def _delete_story(self, user_id: str, story_title: str, story_id: str):
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    "SELECT stories FROM users WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "User not found"}

                current_stories = json.loads(row['stories']) if row['stories'] else []

                story_exists = next(
                    (s for s in current_stories if s["title"] == story_title or s["story_id"] == story_id),
                    None
                )
                if not story_exists:
                    return {"status": "info", "message": f"Story '{story_title}' does not exist"}

                # Remove the story from the list
                current_stories = [
                    s for s in current_stories
                    if s["title"] != story_title and s["story_id"] != story_id
                ]

                # Only update the stories field — do NOT delete any other data
                await conn.execute(
                    """
                    UPDATE users
                    SET stories = ?
                    WHERE user_id = ?
                    """,
                    (json.dumps(current_stories), user_id)
                )

                await conn.commit()

                return {
                    "status": "success",
                    "message": f"Story '{story_title}' removed from user's story list",
                    "stories": current_stories
                }
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"Database error: {str(e)}"}

    async def _get_user_profile(self, user_id: str):
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT *
                    FROM users
                    WHERE user_id = ?
                """, (user_id,))
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                user_data = dict(row)
                user_data["stories"] = json.loads(user_data["stories"]) if user_data["stories"] else []

                return {"status": "success", "profile": user_data}

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def _get_user_profile_with_stories(self, user_id: str):
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(
                    """
                    SELECT user_id, nickname, age, tier, monthly_word_count, stories
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
                    "monthly_word_count": user_row["monthly_word_count"]
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

                    story_id = story.get("story_id")
                    if not story_id:
                        continue

                    # Get user-specific row only
                    cursor = await conn.execute(
                        """
                        SELECT * FROM story_progress
                        WHERE user_id = ? AND story_id = ?
                        LIMIT 1
                        """,
                        (user_id, story_id)
                    )
                    row = await cursor.fetchone()

                    if not row:  # No row for this user → skip (no shared fallback)
                        continue

                    story_data = {
                        "title": row["story_title"] or "",
                        "story_id": story_id,
                        "flow_type": row["flow_type"] or "Standard",
                        "target_medium": row["target_medium"] or "classic",
                        "author_type": row["author_type"] or "Basic",
                        "latest_phase": row["latest_phase"] or 0,
                        "min_age": row["min_age"] or 15,
                        "pov": row["pov"],
                        "prose_style": row["prose_style"],
                        "tense": row["tense"],
                        "genre": json.loads(row["genre_list"]) if row["genre_list"] else [],
                        "sub_genre": json.loads(row["sub_genre_list"]) if row["sub_genre_list"] else [],
                        "themes": json.loads(row["themes_list"]) if row["themes_list"] else [],
                        "total_acts": row["total_acts"] or 3,
                        "target_length": row["target_length"] or 0,
                        "updated_at": row["updated_at"],
                        "status": row["status"] or "Ongoing"
                    }

                    stories.append(story_data)

                return {
                    "status": "success",
                    "profile": user_data,
                    "stories": stories
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}
        
    async def update_story_progress(self, metadata: Optional[Dict[str, Any]] = None):
        print("Updating Story Progress...")
        print(metadata)
        metadata = metadata or {}

        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT * FROM story_progress
                WHERE user_id = ? AND story_id = ?
            """, (self.user_id, self.story_id))
            
            existing_row = await cursor.fetchone()
            if not existing_row:
                # Assuming row should exist; if insert logic is elsewhere, you may want to handle creation here.
                raise ValueError("Story progress row does not exist for this user/story")

            # Helper to merge token usage (accumulative per user)
            def process_token_usage(new_value: Optional[dict], existing_value: Optional[str]):
                default = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                try:
                    existing = json.loads(existing_value) if existing_value else default
                    if not isinstance(existing, dict):
                        existing = default
                except json.JSONDecodeError:
                    existing = default
                
                if new_value is not None:
                    total = {
                        "prompt_tokens": existing["prompt_tokens"] + new_value.get("prompt_tokens", 0),
                        "completion_tokens": existing["completion_tokens"] + new_value.get("completion_tokens", 0),
                    }
                    total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
                    return json.dumps(total)
                return json.dumps(existing)

            # Determine new values (use metadata if provided, otherwise keep existing)
            story_title = metadata.get("story_title", existing_row["story_title"])
            flow_type = metadata.get("flow_type", existing_row["flow_type"])
            target_medium = metadata.get("target_medium", existing_row["target_medium"])
            total_acts = metadata.get("total_acts", existing_row["total_acts"])
            target_length = metadata.get("target_length", existing_row["target_length"])
            story_length = metadata.get("story_length", existing_row["story_length"])
            last_chapter_id = metadata.get("last_chapter_id", existing_row["last_chapter_id"])
            genre = metadata.get("genre", json.loads(existing_row["genre_list"] or "[]"))
            sub_genre = metadata.get("sub_genre", json.loads(existing_row["sub_genre_list"] or "[]"))
            themes = metadata.get("themes", json.loads(existing_row["themes_list"] or "[]"))
            pov = metadata.get("pov", existing_row["pov"])
            prose_style = metadata.get("prose_style", existing_row["prose_style"])
            tense = metadata.get("tense", existing_row["tense"])
            blurb = metadata.get("blurb", existing_row["blurb"])
            tone_temp = metadata.get("tone_temp", existing_row["tone_temp"])
            min_age = metadata.get("min_age", existing_row["min_age"])

            author_type = metadata.get("author_type", existing_row["author_type"])
            latest_phase = metadata.get("latest_phase", existing_row["latest_phase"])
            current_act_id = metadata.get("current_act_id", existing_row["current_act_id"])
            latest_chapter_id = metadata.get("latest_chapter_id", existing_row["latest_chapter_id"])
            continue_scene_id = metadata.get("continue_scene_id", existing_row["continue_scene_id"])
            story_word_count = metadata.get("story_word_count", existing_row["story_word_count"])
            chapter_word_count = metadata.get("chapter_word_count", existing_row["chapter_word_count"])
            status = metadata.get("status", existing_row["status"])

            updated_author_token_usage = process_token_usage(metadata.get("author_token_usage"), existing_row["author_token_usage"])
            updated_director_token_usage = process_token_usage(metadata.get("director_token_usage"), existing_row["director_token_usage"])
            updated_writer_token_usage = process_token_usage(metadata.get("writer_token_usage"), existing_row["writer_token_usage"])
            updated_ingestor_token_usage = process_token_usage(metadata.get("ingestor_token_usage"), existing_row["ingestor_token_usage"])
            updated_utility_token_usage = process_token_usage(metadata.get("utility_token_usage"), existing_row["utility_token_usage"])

            await conn.execute("""
                UPDATE story_progress
                SET story_title = ?,
                    flow_type = ?,
                    target_medium = ?,
                    total_acts = ?,
                    target_length = ?,
                    story_length = ?,
                    last_chapter_id = ?,
                    genre_list = ?,
                    sub_genre_list = ?,
                    themes_list = ?,
                    pov = ?,
                    prose_style = ?,
                    tense = ?,
                    blurb = ?,
                    tone_temp = ?,
                    min_age = ?,
                    author_type = ?,
                    latest_phase = ?,
                    current_act_id = ?,
                    latest_chapter_id = ?,
                    continue_scene_id = ?,
                    story_word_count = ?,
                    chapter_word_count = ?,
                    status = ?,
                    author_token_usage = ?,
                    director_token_usage = ?,
                    writer_token_usage = ?,
                    ingestor_token_usage = ?,
                    utility_token_usage = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ? AND story_id = ?
            """, (
                story_title,
                flow_type,
                target_medium,
                total_acts,
                target_length,
                story_length,
                last_chapter_id,
                json.dumps(genre),
                json.dumps(sub_genre),
                json.dumps(themes),
                pov,
                prose_style,
                tense,
                blurb,
                tone_temp,
                min_age,
                author_type,
                latest_phase,
                current_act_id,
                latest_chapter_id,
                continue_scene_id,
                story_word_count,
                chapter_word_count,
                status,
                updated_author_token_usage,
                updated_director_token_usage,
                updated_writer_token_usage,
                updated_ingestor_token_usage,
                updated_utility_token_usage,
                self.user_id,
                self.story_id
            ))

            await conn.commit()

    async def get_story_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest story progress (fully per-user, no shared fallback).
        """
        if not self.user_id or not self.story_id:
            raise ValueError("user_id and story_id must be provided")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT * FROM story_progress 
                    WHERE user_id = ? AND story_id = ?
                    LIMIT 1
                """, (self.user_id, self.story_id))
                
                row = await cursor.fetchone()
                if not row:
                    return None

                result = {
                    "author_type": row["author_type"] or "Basic",
                    "latest_phase": row["latest_phase"] or 0,
                    "current_act_id": row["current_act_id"] or 1,
                    "latest_chapter_id": row["latest_chapter_id"] or 0,
                    "last_chapter_id": row["last_chapter_id"] or 0,
                    "continue_scene_id": row["continue_scene_id"] or 0,
                    "story_word_count": row["story_word_count"] or 0,
                    "chapter_word_count": row["chapter_word_count"] or 0,
                    "status": row["status"] or "Ongoing",
                    "author_token_usage": row["author_token_usage"],
                    "director_token_usage": row["director_token_usage"],
                    "writer_token_usage": row["writer_token_usage"],
                    "ingestor_token_usage": row["ingestor_token_usage"],
                    "utility_token_usage": row["utility_token_usage"],
                    "total_acts": row["total_acts"] or 3,
                    "target_length": row["target_length"] or 0,
                    "pov": row["pov"],
                    "tense": row["tense"],
                    "prose_style": row["prose_style"],
                    "genre": json.loads(row["genre_list"]) if row["genre_list"] else [],
                    "sub_genre": json.loads(row["sub_genre_list"]) if row["sub_genre_list"] else [],
                    "themes": json.loads(row["themes_list"]) if row["themes_list"] else [],
                    "flow_type": row["flow_type"] or "novel",
                    "target_medium": row["target_medium"] or "classic",
                    "story_title": row["story_title"],
                    "tone_temp": row["tone_temp"],
                    "blurb": row["blurb"],
                    "min_age": row["min_age"] or 15
                }
                
                return result
        
        except aiosqlite.Error as e:
            raise aiosqlite.Error(f"Database error in get_story_progress: {str(e)}")
        
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
        act_id = int(metadata.get("act_id", 0))
        type_ = metadata.get("type", "default")

        async with self._get_connection() as conn:
            async with conn.execute('BEGIN'):
                await conn.execute("""
                    INSERT INTO director_notes (
                        user_id, story_id, chapter_id, act_id, type, text, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id, chapter_id, act_id, type) DO UPDATE SET
                        text = excluded.text,
                        metadata = excluded.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    self.user_id,
                    self.story_id,
                    chapter_id,
                    act_id,
                    type_,
                    json.dumps(entry),
                    json.dumps(metadata)
                ))
                await conn.commit()

    async def put_characters_or_world(self, details_dict: Dict[str, Any], metadata: Dict[str, Any]):
        async with self._get_connection() as conn:
            act_id = metadata.get("act_id", 0)
            chapter_id = metadata.get("chapter_id", 0)
            scene_id = metadata.get("scene_id", 0)

            for _, entry in details_dict.items():
                # Support both 'name'/'new_name' and 'previous_names'/'prev_names'
                raw_new_name = entry.get("new_name") or entry.get("name")
                raw_prev_names = entry.get("prev_names") or entry.get("previous_names") or []
                new_details = str(entry.get("details", "")).strip()
                new_significance = str(entry.get("significance", "")).strip()

                if not raw_new_name and not new_details:
                    continue

                new_name = str(raw_new_name).strip() if raw_new_name else ""
                if not new_name:
                    continue

                # Normalize prev_names → comma-separated string
                prev_names_str = ",".join([n.strip() for n in raw_prev_names if n and str(n).strip()]) if raw_prev_names else ""

                # ── Step 1: Try to find if this name already exists as current name ──
                row = await (await conn.execute(
                    f"SELECT name, prev_names, details, significance FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                    (self.user_id, self.story_id, new_name)
                )).fetchone()

                if row:
                    # Case A: Name already exists → append details + significance + update prev_names
                    current_prev = row[1] or ""
                    current_details = row[2] or ""
                    current_significance = row[3] or ""

                    # Append new details
                    merged_details = " ".join([s for s in [current_details, new_details] if s]).strip()
                    
                    # Append new significance
                    merged_significance = " ".join([s for s in [current_significance, new_significance] if s]).strip()

                    # Merge prev_names (avoid duplicates)
                    existing_prev = {p.strip() for p in current_prev.split(",") if p.strip()}
                    new_prev = {p.strip() for p in prev_names_str.split(",") if p.strip()}
                    all_prev = existing_prev.union(new_prev)
                    if new_name in all_prev:
                        all_prev.remove(new_name)
                    final_prev_names = ",".join(sorted(all_prev)) if all_prev else ""

                    await conn.execute(f"""
                        UPDATE {self.table} SET
                            prev_names = ?,
                            details = ?,
                            significance = ?,
                            act_id = ?,
                            chapter_id = ?,
                            scene_id = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND story_id = ? AND name = ?
                    """, (
                        final_prev_names,
                        merged_details,
                        merged_significance,
                        act_id, chapter_id, scene_id,
                        self.user_id, self.story_id, new_name
                    ))
                    continue

                # ── Step 2: Check if new_name exists in anyone's prev_names → promote it ──
                cursor = await conn.execute(f"""
                    SELECT name, prev_names, details, significance FROM {self.table}
                    WHERE user_id = ? AND story_id = ? AND prev_names LIKE ?
                """, (self.user_id, self.story_id, f"%,{new_name},%"))

                # Also check edges: starts with, ends with, or exact
                cursor2 = await conn.execute(f"""
                    SELECT name, prev_names, details, significance FROM {self.table}
                    WHERE user_id = ? AND story_id = ?
                    AND (prev_names = ? OR prev_names LIKE ? OR prev_names LIKE ?)
                """, (
                    self.user_id, self.story_id,
                    new_name,
                    f"{new_name},%",
                    f"%,{new_name}"
                ))
                
                # Combine results
                candidate = await cursor.fetchone()
                if not candidate:
                    candidate = await cursor2.fetchone()

                if candidate:
                    old_current_name, old_prev_names_str, old_details, old_significance = candidate
                    old_details = old_details or ""
                    old_significance = old_significance or ""

                    # Merge details and significance
                    merged_details = " ".join([s for s in [old_details, new_details] if s]).strip()
                    merged_significance = " ".join([s for s in [old_significance, new_significance] if s]).strip()

                    # Update prev_names: add old current name to prev, set new_name as current
                    prev_set = {p.strip() for p in (old_prev_names_str or "").split(",") if p.strip()}
                    if old_current_name:
                        prev_set.add(old_current_name.strip())
                    if new_name in prev_set:
                        prev_set.remove(new_name)

                    # Merge incoming prev_names
                    if prev_names_str:
                        incoming_prev = {p.strip() for p in prev_names_str.split(",") if p.strip()}
                        prev_set = prev_set.union(incoming_prev)
                    
                    final_prev_names = ",".join(sorted(prev_set)) if prev_set else ""

                    # Promote: change name to new_name
                    await conn.execute(f"""
                        UPDATE {self.table} SET
                            name = ?,
                            prev_names = ?,
                            details = ?,
                            significance = ?,
                            act_id = ?,
                            chapter_id = ?,
                            scene_id = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND story_id = ? AND name = ?
                    """, (
                        new_name,
                        final_prev_names,
                        merged_details,
                        merged_significance,
                        act_id, chapter_id, scene_id,
                        self.user_id, self.story_id, old_current_name
                    ))
                    continue

                # ── Step 3: Completely new name → insert fresh ──
                final_prev_names = prev_names_str
                if final_prev_names and new_name in {p.strip() for p in final_prev_names.split(",")}:
                    prev_parts = [p.strip() for p in final_prev_names.split(",") if p.strip() != new_name]
                    final_prev_names = ",".join(prev_parts)

                await conn.execute(f"""
                    INSERT INTO {self.table}
                        (user_id, story_id, name, prev_names, details, significance, act_id, chapter_id, scene_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id, name) DO UPDATE SET
                        prev_names = excluded.prev_names,
                        details = trim({self.table}.details || ' ' || excluded.details),
                        significance = trim({self.table}.significance || ' ' || excluded.significance),
                        act_id = excluded.act_id,
                        chapter_id = excluded.chapter_id,
                        scene_id = excluded.scene_id,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    self.user_id, self.story_id, new_name,
                    final_prev_names, new_details or "", new_significance or "",
                    act_id, chapter_id, scene_id
                ))

            await conn.commit()


    async def get_text(self, chapter_id: str) -> Dict[str, Any]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT text, metadata FROM {self.table} WHERE user_id = ? AND story_id = ? AND chapter_id = ?",
                (self.user_id, self.story_id, chapter_id)
            )
            row = await cursor.fetchone()
            
            if row and row['text']:
                return {
                    "text": json.loads(row['text']),
                    "metadata": json.loads(row['metadata'] or '{}')
                }
        
        # Return default empty structure if not found or text is empty
        return {
            "text": [],
            "metadata": {}
        }
    
    async def get_from_director_notes(self, metadata: dict):
        chapter_id = metadata.get("chapter_id", 0)
        act_id = metadata.get("act_id", 0)
        type_ = metadata.get("type", "default")

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT text FROM {self.table}
                WHERE user_id = ?
                AND story_id = ?
                AND chapter_id = ?
                AND act_id = ?
                AND type = ?
                """,
                (self.user_id, self.story_id, chapter_id, act_id, type_)
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
    
    

    async def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT name, prev_names, details, significance, act_id, chapter_id, scene_id
                FROM {self.table}
                WHERE user_id = ? AND story_id = ? AND name = ?
                """,
                (self.user_id, self.story_id, name)
            )
            row = await cursor.fetchone()

            if row:
                return {
                    "name": row["name"],
                    "previous_names": row["prev_names"],
                    "details": row["details"][:500],
                    "significance": row["significance"][:500],
                    "act_id": row["act_id"],
                    "chapter_id": row["chapter_id"],
                    "scene_id": row["scene_id"]
                }

            return None

    async def get_all_characters_or_worlds_by_chapter(self, chapter_number: int) -> Dict[str, str]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT name, details, significance
                FROM {self.table}
                WHERE user_id = ? AND story_id = ? AND chapter_id = ?
                """,
                (self.user_id, self.story_id, chapter_number)
            )
            rows = await cursor.fetchall()

            # Return dictionary mapping name → details
            return {row["name"]: "Detailes: " + row["details"][:500] + "\nSignificance: " + row["significance"] for row in rows} if rows else {}


    async def get_all_characters_or_worlds(self) -> Dict[str, Any]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT name, details, significance FROM {self.table} WHERE user_id = ? AND story_id = ?",
                (self.user_id, self.story_id)
            )
            rows = await cursor.fetchall()
            return {row["name"]: "Detailes: " + row["details"][:500] + "\nSignificance: " + row["significance"] for row in rows} if rows else {}
    
    async def get_all_character_or_world_names(self) -> List[str]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT name 
                FROM {self.table}
                WHERE user_id = ? AND story_id = ?
                """,
                (self.user_id, self.story_id)
            )
            rows = await cursor.fetchall()
            return [row["name"] for row in rows] if rows else []


    async def close(self):
        pass

#----------- Analytics Only Calls --------------
    async def _get_all_users(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM users")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_story_texts(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM story_texts")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_characters_raw(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM characters_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_world_elements_raw(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM world_elements_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_director_notes(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM director_notes")
            rows = await cursor.fetchall()
            notes = [dict(row) for row in rows]
            for note in notes:
                try:
                    note['text'] = json.loads(note['text'])
                except json.JSONDecodeError:
                    try:
                        note['text'] = ast.literal_eval(note['text'])
                    except (ValueError, SyntaxError):
                        pass  # leave as string if both fail
            return notes

    async def _get_all_story_progress(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT id, user_id, story_id, author_type, latest_phase, current_act_id, latest_chapter_id, 
                       continue_scene_id, story_word_count, status, author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
                       created_at, updated_at
                FROM story_progress
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_feedback(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM feedback")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def _get_all_data(self):
        users = await self.get_all_users()
        story_texts = await self.get_all_story_texts()
        characters = await self.get_all_characters_raw()
        world_elements = await self.get_all_world_elements_raw()
        director_notes = await self.get_all_director_notes()
        story_progress = await self.get_all_story_progress()
        feedback = await self.get_all_feedback()

        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM shared_story_data")
            rows = await cursor.fetchall()
            shared_story_data = [dict(row) for row in rows]
            for record in shared_story_data:
                record["genre_list"] = json.loads(record["genre_list"]) if record["genre_list"] else []
                record["sub_genre_list"] = json.loads(record["sub_genre_list"]) if record["sub_genre_list"] else []
                record["themes_list"] = json.loads(record["themes_list"]) if record["themes_list"] else []

        return {
            "users": users,
            "story_texts": story_texts,
            "characters_raw": characters,
            "world_elements_raw": world_elements,
            "director_notes": director_notes,
            "story_progress": story_progress,
            "shared_story_data": shared_story_data,
            "feedback": feedback
        }

# ---------- Feedback --------------------
    async def insert_feedback(self, feedback: Dict[str, Any]):
        try:
            async with self._get_connection() as conn:
                columns = [
                    "usage_mode", "story_quality", "story_like", "story_improve",
                    "interactivity_meaningfulness", "story_pacing", "ease_of_use", 
                    "buggy_or_confusing", "additional_feedback", "nps_score", "reuse_likelihood"
                ]
                
                placeholders = ", ".join(["?"] * len(columns))
                sql = f"INSERT INTO feedback ({', '.join(columns)}) VALUES ({placeholders})"
                
                values = [feedback.get(col, None) for col in columns]
                
                await conn.execute(sql, values)
                await conn.commit()
                
                return {"status": "success", "message": "Feedback inserted successfully"}
                
        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}