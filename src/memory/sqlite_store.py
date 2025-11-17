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
                CREATE TABLE IF NOT EXISTS characters_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    details TEXT,
                    chapter_id INTEGER NOT NULL,
                    act_id INTEGER NOT NULL,
                    scene_id INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, story_id, name)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS world_elements_raw (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    details TEXT,
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
                    story_title TEXT NOT NULL,
                    text TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE (user_id, story_id, chapter_id, act_id, type, story_title)
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS shared_story_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    story_id TEXT NOT NULL,
                    story_title TEXT,
                    story_type TEXT,
                    total_acts INTEGER,
                    target_length INTEGER,
                    genre_list TEXT,
                    sub_genre_list TEXT,
                    themes_list TEXT,
                    pov TEXT,
                    prose_style TEXT,
                    tense TEXT,
                    blurb TEXT,
                    tone_temp FLOAT,
                    image_data BLOB,
                    public BOOLEAN,
                    min_age INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(story_id)
                )
            """)
            # narrative_voice TEXT,
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS story_progress (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    story_id TEXT NOT NULL,
                    current_act_id INTEGER,
                    latest_chapter_id INTEGER,
                    continue_scene_id INTEGER,
                    story_word_count INTEGER DEFAULT 0,
                    chapter_word_count INTEGER DEFAULT 0,
                    complete BOOLEAN,
                    author_token_usage TEXT,
                    director_token_usage TEXT,
                    writer_token_usage TEXT,
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
    async def _get_shared_story_data(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve shared story data for a given story_id."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT story_id, story_title, story_type, total_acts, target_length, genre_list, sub_genre_list, themes_list,
                       pov, tense, prose_style,
                       blurb, tone_temp, image_data, public, min_age, created_at, updated_at
                FROM shared_story_data
                WHERE story_id = ?
            """, (story_id,))
            row = await cursor.fetchone()
            if row:
                result = dict(row)
                result["genre_list"] = json.loads(result["genre_list"]) if result["genre_list"] else []
                result["sub_genre_list"] = json.loads(result["sub_genre_list"]) if result["sub_genre_list"] else []
                result["themes_list"] = json.loads(result["themes_list"]) if result["themes_list"] else []
                return result
            return None

    # Helper function to upsert shared story data
    async def _upsert_shared_story_data(self, story_id: str, metadata: Dict[str, Any]):
        """Insert or update shared story data, preserving existing values where not specified."""
        async with self._get_connection() as conn:
            existing = await self._get_shared_story_data(story_id)
            if existing:
                # Only update fields that are explicitly provided (not None)
                merged_data = {
                    "story_title": metadata.get("story_title") if metadata.get("story_title") is not None else existing["story_title"],
                    "story_type": metadata.get("story_type") if metadata.get("story_type") is not None else (existing["story_type"] or "classic"),
                    "total_acts": metadata.get("total_acts") if metadata.get("total_acts") is not None else (existing["total_acts"] or 3),
                    "target_length": metadata.get("target_length") if metadata.get("target_length") is not None else (existing["target_length"] or 0),
                    "genre_list": metadata.get("genre") if metadata.get("genre") is not None else existing["genre_list"],
                    "sub_genre_list": metadata.get("sub_genre") if metadata.get("sub_genre") is not None else existing["sub_genre_list"],
                    "themes_list": metadata.get("themes") if metadata.get("themes") is not None else existing["themes_list"],
                    "pov": metadata.get("pov") if metadata.get("pov") is not None else existing["pov"],
                    #"narrative_voice": metadata.get("narrative_voice") if metadata.get("narrative_voice") is not None else existing["narrative_voice"],
                    "tense": metadata.get("tense") if metadata.get("tense") is not None else existing["tense"],
                    "prose_style": metadata.get("prose_style") if metadata.get("prose_style") is not None else existing["prose_style"],
                    "blurb": metadata.get("blurb") if metadata.get("blurb") is not None else existing["blurb"],
                    "tone_temp": metadata.get("tone_temp") if metadata.get("tone_temp") is not None else existing["tone_temp"],
                    #"model": metadata.get("model") if metadata.get("model") is not None else existing["model"],
                    "image_data": metadata.get("image_data") if metadata.get("image_data") is not None else existing["image_data"],
                    "public": metadata.get("public") if metadata.get("public") is not None else existing["public"],
                    "min_age": metadata.get("min_age") if metadata.get("min_age") is not None else existing["min_age"]
                }
            else:
                # Creating new record - use provided values or defaults
                merged_data = {
                    "story_title": metadata.get("story_title"),
                    "story_type": metadata.get("story_type", "classic"),
                    "total_acts": metadata.get("total_acts", 3),
                    "target_length": metadata.get("target_length", 0),
                    "genre_list": metadata.get("genre", []),
                    "sub_genre_list": metadata.get("sub_genre", []),
                    "themes_list": metadata.get("themes", []),
                    "pov": metadata.get("pov"),
                    "tense": metadata.get("tense"),
                    #"narrative_voice": metadata.get("narrative_voice"),
                    "prose_style": metadata.get("prose_style"),
                    "blurb": metadata.get("blurb"),
                    "tone_temp": metadata.get("tone_temp"),
                    #"model": metadata.get("model"),
                    "image_data": metadata.get("image_data"),
                    "public": metadata.get("public", False),
                    "min_age": metadata.get("min_age", 15)
                }

            await conn.execute("""
            INSERT INTO shared_story_data (
                story_id, story_title, story_type, total_acts, target_length, genre_list, sub_genre_list, themes_list, pov, tense, prose_style,
                blurb, tone_temp, image_data, public, min_age
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(story_id) DO UPDATE SET
                story_title = excluded.story_title,
                story_type = excluded.story_type,
                total_acts = excluded.total_acts,
                target_length = excluded.target_length,
                genre_list = excluded.genre_list,
                sub_genre_list = excluded.sub_genre_list,
                themes_list = excluded.themes_list,
                pov = excluded.pov,
                tense = excluded.tense,
                prose_style = excluded.prose_style,
                blurb = excluded.blurb,
                tone_temp = excluded.tone_temp,
                image_data = excluded.image_data,
                public = excluded.public,
                min_age = excluded.min_age,
                updated_at = CURRENT_TIMESTAMP
        """, (
                story_id,
                merged_data["story_title"],
                merged_data["story_type"],
                merged_data["total_acts"],
                merged_data["target_length"],
                json.dumps(merged_data["genre_list"]),
                json.dumps(merged_data["sub_genre_list"]),
                json.dumps(merged_data["themes_list"]),
                merged_data["pov"],
                merged_data["prose_style"],
                merged_data["tense"],
               # merged_data["narrative_voice"],
                merged_data["blurb"],
                merged_data["tone_temp"],
                #merged_data["model"],
                merged_data["image_data"],
                merged_data["public"],
                merged_data["min_age"]
            ))
            await conn.commit()

    # User helpers
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def add_user(self, nickname: str, age: Optional[int], user_id: str, tier: int, stories: List[Dict] = None):
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

    async def append_story(self, user_id: str, story_title: str, story_id: str, story_type: str):
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
                post_row = await post.fetchone()
                #print(f"[POST] Updated row: {dict(post_row) if post_row else None}")

                if cursor.rowcount == 0:
                    return {"status": "error", "message": "User not found or no changes"}
                return {"status": "success", "updated": list(to_update.keys())}
        except Exception as e:
            print(f"[EXCEPTION update_user] {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}

    async def delete_story(self, user_id: str, story_title: str, story_id: str):
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

    async def get_user_profile(self, user_id: str):
        await self.ensure_word_count_fresh(user_id)
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

    async def increment_monthly_word_count(self, words_added: int):
        if not isinstance(words_added, int) or words_added < 0:
            return {"status": "error", "message": "❌ words_added must be a non-negative integer"}

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT monthly_word_count FROM users WHERE user_id = ?
                """, (self.user_id,))
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                new_value = row["monthly_word_count"] + words_added
                await conn.execute("""
                    UPDATE users SET monthly_word_count = ? WHERE user_id = ?
                """, (new_value, self.user_id))
                await conn.commit()

                return {
                    "status": "success",
                    "message": f"✅ Added {words_added} words (new total: {new_value})",
                    "new_word_count": new_value
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def get_monthly_word_count(self):
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT tier, monthly_word_count, last_reset_date
                    FROM users
                    WHERE user_id = ?
                """, (self.user_id,))
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                return {
                    # "status": "success",
                    "tier": row["tier"],
                    "monthly_word_count": row["monthly_word_count"],
                    "last_reset_date": row["last_reset_date"]
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    # Subscription management
    async def update_subscription(self, user_id: str, data: Dict[str, Any]):
        try:
            async with self._get_connection() as conn:
                await conn.execute("""
                    UPDATE users
                    SET lqs_customer_id = ?,
                        lqs_subscription_id = ?,
                        lqs_variant_id = ?,
                        subscription_status = ?,
                        subscription_renewal_date = ?,
                        cancel_at_period_end = ?
                    WHERE user_id = ?
                """, (
                    data.get("customer_id"),
                    data.get("subscription_id"),
                    data.get("variant_id"),
                    data.get("status"),
                    data.get("renewal_date"),
                    int(data.get("cancel_at_period_end", False)),
                    user_id
                ))
                await conn.commit()
                return {"status": "success", "message": f"✅ Subscription updated for {user_id}"}
        except Exception as e:
            return {"status": "error", "message": f"❌ Error updating subscription: {str(e)}"}

    async def is_subscription_active(self, user_id: str) -> bool:
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT subscription_status FROM users WHERE user_id = ?
            """, (user_id,))
            row = await cursor.fetchone()
            return bool(row and row["subscription_status"] in ["active", "trialing"])

    # Monthly reset system
    async def ensure_word_count_fresh(self, user_id: str):
        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT last_reset_date, subscription_renewal_date
                    FROM users
                    WHERE user_id = ?
                """, (user_id,))
                row = await cursor.fetchone()
                if not row:
                    return {"status": "error", "message": "❌ User not found"}

                now = datetime.now(UTC)
                last_reset = datetime.fromisoformat(row["last_reset_date"])

                renewal_date = (
                    datetime.fromisoformat(row["subscription_renewal_date"])
                    if row["subscription_renewal_date"]
                    else None
                )

                should_reset = (
                    (renewal_date and now >= renewal_date) or
                    ((now - last_reset) >= timedelta(days=30))
                )

                if should_reset:
                    await self._reset_word_count(conn, user_id)
                    return {"status": "success", "message": "✅ Word count reset (renewal or 30 days passed)"}

                return {"status": "ok", "message": "No reset needed"}

        except Exception as e:
            return {"status": "error", "message": f"❌ Error checking reset: {str(e)}"}

    async def _reset_word_count(self, conn, user_id: str, new_value: Optional[int] = None):
        new_value = new_value or 0
        now = datetime.now(UTC).isoformat()
        await conn.execute("""
            UPDATE users
            SET monthly_word_count = ?,
                last_reset_date = ?
            WHERE user_id = ?
        """, (new_value, now, user_id))
        await conn.commit()

    async def get_user_profile_with_stories(self, user_id: str):
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

                    title = story.get("title", "Untitled Story")
                    story_id = story.get("story_id")
                    story_type = story.get("story_type")
                    if not story_id:
                        continue

                    # Get user-specific progress
                    cursor = await conn.execute(
                        """
                        SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                               story_word_count, complete, author_token_usage, director_token_usage, writer_token_usage
                        FROM story_progress
                        WHERE user_id = ? AND story_id = ?
                        LIMIT 1
                        """,
                        (user_id, story_id)
                    )
                    progress_row = await cursor.fetchone()

                    # Get shared story data
                    shared_data = await self._get_shared_story_data(story_id)
                    story_progress = int((progress_row["story_word_count"]/shared_data["target_length"]) * 100)
                    story_data = {
                        "title": title,
                        "story_id": story_id,
                        "story_type": story_type or "classic",
                        "current_act_id": 1,
                        "latest_chapter_id": 0,
                        "continue_scene_id": 0,
                        "story_word_count": 0,
                        "complete": False,
                        #"token_usage": 0,
                        #"model": "",
                        "blurb": "",
                        "image_data": None,
                        "public": False,
                        "min_age": 15,
                        "pov": None,
                        #"narrative_voice": None,
                        "prose_style": None,
                        "tense": None,
                        "genre": [],
                        "themes": [],
                        #"tone_temp": None,
                        "total_acts": 3,
                        "story_progress": story_progress,
                        #"target_length": 0
                    }

                    if progress_row:
                        story_data.update({
                            "current_act_id": progress_row["current_act_id"] or 1,
                            "latest_chapter_id": progress_row["latest_chapter_id"] or 0,
                            "continue_scene_id": progress_row["continue_scene_id"] or 0,
                            "story_word_count": progress_row["story_word_count"] or 0,
                            "complete": bool(progress_row["complete"]),
                            #"token_usage": json.loads(progress_row["token_usage"]) if progress_row["token_usage"] else 0
                        })

                    if shared_data:
                        story_data.update({
                            #"model": shared_data["model"] or "",
                            "blurb": shared_data["blurb"] or "",
                            "image_data": shared_data["image_data"],
                            "public": bool(shared_data["public"]),
                            "min_age": shared_data["min_age"],
                            "pov": shared_data["pov"],
                            "tense": shared_data["tense"],
                            "prose_style": shared_data["prose_style"],
                           # "narrative_voice": shared_data["narrative_voice"],
                            "genre": shared_data["genre_list"],
                            "sub_genre": shared_data["sub_genre_list"],
                            "themes": shared_data["themes_list"],
                            #"tone_temp": shared_data["tone_temp"],
                            "total_acts": shared_data["total_acts"],
                            #"target_length": shared_data["target_length"] or 0,
                            "story_type": shared_data["story_type"] or story_type or "classic"
                        })

                    stories.append(story_data)

                return {
                    "status": "success",
                    "profile": user_data,
                    "stories": stories
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"❌ Database error: {str(e)}"}

    async def update_story_public_status(self, story_id: str, public: bool):
        try:
            async with self._get_connection() as conn:
                shared_data = await self._get_shared_story_data(story_id)
                if not shared_data:
                    await self._upsert_shared_story_data(story_id, {"public": public})
                else:
                    await conn.execute("""
                        UPDATE shared_story_data
                        SET public = ?,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE story_id = ?
                    """, (public, story_id))
                
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
        act_id = int(metadata.get("act_id", 0))
        type_ = metadata.get("type", "default")
        story_title = metadata.get("story_title", "Untitled")

        async with self._get_connection() as conn:
            async with conn.execute('BEGIN'):
                await conn.execute("""
                    INSERT INTO director_notes (
                        user_id, story_id, chapter_id, act_id, type, story_title, text, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, story_id, chapter_id, act_id, type, story_title) DO UPDATE SET
                        text = excluded.text,
                        metadata = excluded.metadata,
                        updated_at = CURRENT_TIMESTAMP
                """, (
                    self.user_id,
                    self.story_id,
                    chapter_id,
                    act_id,
                    type_,
                    story_title,
                    json.dumps(entry),
                    json.dumps(metadata)
                ))
                await conn.commit()

    async def put_characters_or_world(self, details_dict: Dict[str, Any], metadata: Dict[str, Any]):
        async with self._get_connection() as conn:
            for _, entry in details_dict.items():
                old_name = entry.get("old_name")
                new_name = entry.get("new_name")
                details = entry.get("details", "")

                act_id = metadata.get("act_id", 0)
                chapter_id = metadata.get("chapter_id", 0)
                scene_id = metadata.get("scene_id", 0)

                # --- Case 1: Renamed ---
                if old_name != new_name:
                    # Fetch old details (if any)
                    cursor = await conn.execute(
                        f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                        (self.user_id, self.story_id, old_name)
                    )
                    old_row = await cursor.fetchone()
                    old_details = old_row['details'] if old_row else ""

                    # Fetch new details (if already exists)
                    cursor = await conn.execute(
                        f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                        (self.user_id, self.story_id, new_name)
                    )
                    new_row = await cursor.fetchone()
                    existing_new_details = new_row['details'] if new_row else ""

                    # Merge all three sources
                    merged_details = " ".join(
                        part.strip() for part in [old_details, existing_new_details, details] if part
                    )

                    # Delete old record (rename migration)
                    await conn.execute(
                        f"DELETE FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                        (self.user_id, self.story_id, old_name)
                    )

                    # Upsert merged under new name
                    await conn.execute(f"""
                        INSERT INTO {self.table} 
                        (user_id, story_id, name, details, act_id, chapter_id, scene_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, story_id, name) DO UPDATE SET
                            details = excluded.details,
                            act_id = excluded.act_id,
                            chapter_id = excluded.chapter_id,
                            scene_id = excluded.scene_id,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        self.user_id,
                        self.story_id,
                        new_name,
                        merged_details,
                        act_id,
                        chapter_id,
                        scene_id
                    ))

                # --- Case 2: Same name, just append ---
                else:
                    cursor = await conn.execute(
                        f"SELECT details FROM {self.table} WHERE user_id = ? AND story_id = ? AND name = ?",
                        (self.user_id, self.story_id, new_name)
                    )
                    row = await cursor.fetchone()
                    old_details = row['details'] if row else ""

                    new_details = " ".join(
                        part.strip() for part in [old_details, details] if part
                    )

                    await conn.execute(f"""
                        INSERT INTO {self.table} 
                        (user_id, story_id, name, details, act_id, chapter_id, scene_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(user_id, story_id, name) DO UPDATE SET
                            details = excluded.details,
                            act_id = excluded.act_id,
                            chapter_id = excluded.chapter_id,
                            scene_id = excluded.scene_id,
                            updated_at = CURRENT_TIMESTAMP
                    """, (
                        self.user_id,
                        self.story_id,
                        new_name,
                        new_details,
                        act_id,
                        chapter_id,
                        scene_id
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
        story_title = metadata.get("story_title", "Untitled")

        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT text FROM {self.table}
                WHERE user_id = ?
                AND story_id = ?
                AND chapter_id = ?
                AND act_id = ?
                AND type = ?
                AND story_title = ?
                """,
                (self.user_id, self.story_id, chapter_id, act_id, type_, story_title)
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

    async def update_story_progress(self, metadata: Optional[Dict[str, Any]] = None):
        metadata = metadata or {}
        
        # User-specific fields
        current_act_id = metadata.get("current_act_id")
        chapter_id = metadata.get("latest_chapter_id")
        scene_id = metadata.get("continue_scene_id")
        story_word_count = metadata.get("story_word_count")
        chapter_word_count = metadata.get("chapter_word_count")
        complete = metadata.get("complete")
        author_token_usage = metadata.get("author_token_usage")
        director_token_usage = metadata.get("director_token_usage")
        writer_token_usage = metadata.get("writer_token_usage")
        
        # Shared fields
        shared_metadata = {
            "story_title": metadata.get("story_title"),
            "story_type": metadata.get("story_type"),
            "total_acts": metadata.get("total_acts"),
            "target_length": metadata.get("target_length"),
            "genre": metadata.get("genre"),
            "themes": metadata.get("themes"),
            "pov": metadata.get("pov"),
            #"narrative_voice": metadata.get("narrative_voice"),
            "prose_style": metadata.get("prose_style"),
            "tense": metadata.get("tense"),
            "blurb": metadata.get("blurb"),
            "tone_temp": metadata.get("tone_temp"),
            "image_data": metadata.get("image_data"),
            "public": metadata.get("public"),
            "min_age": metadata.get("min_age")
        }

        async with self._get_connection() as conn:
            # Update shared_story_data
            await self._upsert_shared_story_data(self.story_id, shared_metadata)
            
            # Update story_progress
            cursor = await conn.execute("""
                SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                    story_word_count, chapter_word_count, complete, 
                    author_token_usage, director_token_usage, writer_token_usage
                FROM story_progress
                WHERE user_id = ? AND story_id = ?
            """, (self.user_id, self.story_id))
            
            existing_row = await cursor.fetchone()
            
            # Helper function to handle token usage
            def process_token_usage(new_value, existing_value):
                # Default token usage structure
                default = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                
                # Parse existing value from database (stored as JSON string)
                if existing_value:
                    try:
                        existing = json.loads(existing_value) if isinstance(existing_value, str) else existing_value
                        if not isinstance(existing, dict):
                            existing = default
                    except json.JSONDecodeError:
                        existing = default
                else:
                    existing = default
                
                # If new_value is provided, merge it with existing
                if new_value is not None:
                    if isinstance(new_value, dict):
                        # Validate and merge new_value
                        result = existing.copy()
                        result.update({
                            "prompt_tokens": new_value.get("prompt_tokens", existing["prompt_tokens"]),
                            "completion_tokens": new_value.get("completion_tokens", existing["completion_tokens"]),
                            "total_tokens": new_value.get("total_tokens", existing["total_tokens"])
                        })
                        return json.dumps(result)
                    elif isinstance(new_value, str):
                        # If new_value is a JSON string, parse and merge
                        try:
                            parsed = json.loads(new_value)
                            result = existing.copy()
                            result.update({
                                "prompt_tokens": parsed.get("prompt_tokens", existing["prompt_tokens"]),
                                "completion_tokens": parsed.get("completion_tokens", existing["completion_tokens"]),
                                "total_tokens": parsed.get("total_tokens", existing["total_tokens"])
                            })
                            return json.dumps(result)
                        except json.JSONDecodeError:
                            return json.dumps(existing)
                # If no new_value, return existing as JSON
                return json.dumps(existing)

            # Process token usage fields
            updated_author_token_usage = process_token_usage(
                author_token_usage,
                existing_row["author_token_usage"] if existing_row else None
            )
            updated_director_token_usage = process_token_usage(
                director_token_usage,
                existing_row["director_token_usage"] if existing_row else None
            )
            updated_writer_token_usage = process_token_usage(
                writer_token_usage,
                existing_row["writer_token_usage"] if existing_row else None
            )

            if existing_row:
                await conn.execute("""
                    UPDATE story_progress
                    SET current_act_id = ?,
                        latest_chapter_id = ?,
                        continue_scene_id = ?,
                        story_word_count = ?,
                        chapter_word_count = ?,
                        complete = ?,
                        author_token_usage = ?,
                        director_token_usage = ?,
                        writer_token_usage = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ? AND story_id = ?
                """, (
                    current_act_id if current_act_id is not None else existing_row["current_act_id"] or 1,
                    chapter_id if chapter_id is not None else existing_row["latest_chapter_id"] or 0,
                    scene_id if scene_id is not None else existing_row["continue_scene_id"] or 0,
                    story_word_count if story_word_count is not None else existing_row["story_word_count"] or 0,
                    chapter_word_count if chapter_word_count is not None else existing_row["chapter_word_count"] or 0,
                    complete if complete is not None else existing_row["complete"] or False,
                    updated_author_token_usage,
                    updated_director_token_usage,
                    updated_writer_token_usage,
                    self.user_id,
                    self.story_id
                ))
            else:
                await conn.execute("""
                    INSERT INTO story_progress (
                        user_id, story_id, current_act_id, latest_chapter_id, 
                        continue_scene_id, story_word_count, chapter_word_count, 
                        complete, author_token_usage, director_token_usage, writer_token_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    self.user_id,
                    self.story_id,
                    current_act_id or 1,
                    chapter_id or 0,
                    scene_id or 0,
                    story_word_count or 0,
                    chapter_word_count or 0,
                    complete or False,
                    updated_author_token_usage,
                    updated_director_token_usage,
                    updated_writer_token_usage
                ))
            
            await conn.commit()

    async def get_story_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest story progress, combining user-specific and shared story data.

        Returns:
            Dict containing:
                - current_act_id: Current act number (int, default 1)
                - latest_chapter_id: Current chapter number (int, default 0)
                - continue_scene_id: Current scene number (int, default 0)
                - story_word_count: Total words written (int, default 0)
                - chapter_word_count: Chapter words written (int, default 0)
                - complete: Is story finished (bool, default False)
                - author_token_usage: Tokens used by author role (int, default 0)
                - director_token_usage: Tokens used by director role (int, default 0)
                - writer_token_usage: Tokens used by writer role (int, default 0)
                - total_acts: Total acts planned (int, default 3)
                - target_length: Target word count (int, default 0)
                - pov: Point of view (str, default None)
                - genre: Story genre (list, default [])
                - story_type: Story type (str, default "classic")
                - story_title: Story title (str, default None)
                - tone_temp: Temperature for tone (float, default None)
                - model: LLM model (str, default None)
                - blurb: Story blurb (str, default None)
                - image_data: Cover image (blob, default None)
                - public: Is story public (bool, default False)
            Returns None if no progress or shared data exists for the user_id and story_id.
        
        Raises:
            ValueError: If user_id or story_id is not provided.
            aiosqlite.Error: If a database error occurs.
        """
        if not self.user_id or not self.story_id:
            raise ValueError("user_id and story_id must be provided")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                        story_word_count, chapter_word_count, complete, 
                        author_token_usage, director_token_usage, writer_token_usage
                    FROM story_progress 
                    WHERE user_id = ? AND story_id = ?
                    LIMIT 1
                """, (self.user_id, self.story_id))
                
                progress_row = await cursor.fetchone()
                shared_data = await self._get_shared_story_data(self.story_id)
                
                if not progress_row and not shared_data:
                    return None
                
                result = {
                    "current_act_id": 1,
                    "latest_chapter_id": 0,
                    "continue_scene_id": 0,
                    "story_word_count": 0,
                    "chapter_word_count": 0,
                    "complete": False,
                    "author_token_usage": 0,
                    "director_token_usage": 0,
                    "writer_token_usage": 0,
                    "total_acts": 3,
                    "target_length": 0,
                    "pov": None,
                   # "narrative_voice": None,
                    "tense": None,
                    "prose_style": None,
                    "genre": [],
                    "themes": [],
                    "story_type": "classic",
                    "story_title": None,
                    "tone_temp": None,
                    #"model": None,
                    "blurb": None,
                    "image_data": None,
                    "public": False,
                    "min_age": 15
                }
                
                if progress_row:
                    result.update({
                        "current_act_id": progress_row["current_act_id"] if progress_row["current_act_id"] is not None else 1,
                        "latest_chapter_id": progress_row["latest_chapter_id"] if progress_row["latest_chapter_id"] is not None else 0,
                        "continue_scene_id": progress_row["continue_scene_id"] if progress_row["continue_scene_id"] is not None else 0,
                        "story_word_count": progress_row["story_word_count"] if progress_row["story_word_count"] is not None else 0,
                        "chapter_word_count": progress_row["chapter_word_count"] if progress_row["chapter_word_count"] is not None else 0,
                        "complete": bool(progress_row["complete"]),
                        "author_token_usage": progress_row["author_token_usage"] if progress_row["author_token_usage"] is not None else 0,
                        "director_token_usage": progress_row["director_token_usage"] if progress_row["director_token_usage"] is not None else 0,
                        "writer_token_usage": progress_row["writer_token_usage"] if progress_row["writer_token_usage"] is not None else 0
                    })
                
                if shared_data:
                    result.update({
                        "total_acts": shared_data["total_acts"] if shared_data["total_acts"] is not None else 3,
                        "target_length": shared_data["target_length"] if shared_data["target_length"] is not None else 0,
                        "pov": shared_data["pov"],
                        "tense": shared_data["tense"],
                        #"narrative_voice": shared_data["narrative_voice"],
                        "prose_style": shared_data["prose_style"],
                        "genre": shared_data["genre_list"] or [],
                        "sub_genre": shared_data["sub_genre_list"] or [],
                        "themes": shared_data["themes_list"] or [],
                        "story_type": shared_data["story_type"] or "classic",
                        "story_title": shared_data["story_title"],
                        "tone_temp": shared_data["tone_temp"],
                        #"model": shared_data["model"],
                        "blurb": shared_data["blurb"],
                        "image_data": shared_data["image_data"],
                        "public": bool(shared_data["public"]),
                        "min_age": shared_data["min_age"],
                    })
                
                return result
        
        except aiosqlite.Error as e:
            raise aiosqlite.Error(f"Database error in get_story_progress: {str(e)}")

    async def get_act_progress_summary(self) -> Dict[str, Any]:
        """
        Get a summary of act-based progress for display/debugging.
        
        Returns:
            Dict with current_act_id, total_acts, chapters_completed, 
            word_count, target_length, completion_percentage
        """
        progress = await self.get_story_progress()
        
        if not progress:
            return {
                "current_act_id": 1,
                "total_acts": 3,
                "chapters_completed": 0,
                "story_word_count": 0,
                "chapter_word_count": 0,
                "target_length": 0,
                "completion_percentage": 0,
                "is_complete": False,
                "genre": [],
                "pov": None,
                "story_type": "classic"
            }
        
        story_word_count = progress["story_word_count"]
        target_length = progress["target_length"]
        completion_pct = 0
        if target_length > 0:
            completion_pct = min(100, int((story_word_count / target_length) * 100))
        
        return {
            "current_act_id": progress["current_act_id"],
            "total_acts": progress["total_acts"],
            "chapters_completed": progress["latest_chapter_id"] or 0,
            "story_word_count": story_word_count,
            "target_length": target_length,
            "completion_percentage": completion_pct,
            "is_complete": progress["complete"],
            "genre": progress["genre"],
            "pov": progress["pov"],
            "story_type": progress["story_type"]
        }

    async def increment_chapter(self, word_count_delta: int = 0, scene_id: int = 1):
        progress = await self.get_story_progress()
        
        if not progress:
            raise ValueError("No story progress found")
        
        current_chapter = progress["latest_chapter_id"] or 0
        current_word_count = progress["story_word_count"]
        
        await self.update_story_progress({
            "latest_chapter_id": current_chapter + 1,
            "continue_scene_id": scene_id,
            "story_word_count": current_word_count + word_count_delta
        })

    async def increment_act(self, new_act_number: int):
        progress = await self.get_story_progress()
        
        if not progress:
            raise ValueError("No story progress found")
        
        total_acts = progress["total_acts"]
        
        if new_act_number > total_acts:
            await self.update_story_progress({
                "current_act_id": new_act_number,
                "complete": True
            })
        else:
            await self.update_story_progress({
                "current_act_id": new_act_number
            })

    async def mark_story_complete(self):
        await self.update_story_progress({"complete": True})

    async def get_character_or_world(self, name: str) -> Optional[Dict[str, Any]]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT name, details, act_id, chapter_id, scene_id
                FROM {self.table}
                WHERE user_id = ? AND story_id = ? AND name = ?
                """,
                (self.user_id, self.story_id, name)
            )
            row = await cursor.fetchone()

            if row:
                return {
                    "name": row["name"],
                    "details": row["details"],
                    "act_id": row["act_id"],
                    "chapter_id": row["chapter_id"],
                    "scene_id": row["scene_id"]
                }

            return None

    async def get_all_characters_or_worlds_by_chapter(self, chapter_number: int) -> Dict[str, str]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT name, details
                FROM {self.table}
                WHERE user_id = ? AND story_id = ? AND chapter_id = ?
                """,
                (self.user_id, self.story_id, chapter_number)
            )
            rows = await cursor.fetchall()

            # Return dictionary mapping name → details
            return {row["name"]: row["details"] for row in rows} if rows else {}


    async def get_all_characters_or_worlds(self) -> Dict[str, Any]:
        async with self._get_connection() as conn:
            cursor = await conn.execute(
                f"SELECT name, details FROM {self.table} WHERE user_id = ? AND story_id = ?",
                (self.user_id, self.story_id)
            )
            rows = await cursor.fetchall()
            return {row["name"]: row["details"] for row in rows}
    
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
    async def get_all_users(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM users")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_story_texts(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM story_texts")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_characters_raw(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM characters_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_world_elements_raw(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM world_elements_raw")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_director_notes(self):
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

    async def get_all_story_progress(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT id, user_id, story_id, current_act_id, latest_chapter_id, 
                       continue_scene_id, story_word_count, complete, author_token_usage, director_token_usage, writer_token_usage, 
                       created_at, updated_at
                FROM story_progress
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_feedback(self):
        async with self._get_connection() as conn:
            cursor = await conn.execute("SELECT * FROM feedback")
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_all_data(self):
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
                if record.get("image_data"):
                    record["image_data"] = base64.b64encode(record["image_data"]).decode("utf-8")

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