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
                CREATE TABLE IF NOT EXISTS world_elements_raw (
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
                    image_data BLOB,
                    public BOOLEAN,
                    min_age INTEGER,
                    author_id TEXT,
                    overall_rating FLOAT,
                    total_ratings INTEGER,
                    views INTEGER,
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
                    rating INTEGER,
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
    async def _get_shared_story_data(self, story_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve shared story data for a given story_id."""
        async with self._get_connection() as conn:
            cursor = await conn.execute("""
                SELECT story_id, story_title, story_type, total_acts, target_length, story_length, last_chapter_id, last_chapter_id, genre_list, sub_genre_list, themes_list,
                       pov, tense, prose_style,
                       blurb, tone_temp, image_data, public, min_age, author_id, created_at, updated_at
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
                    "last_chapter_id": metadata.get("last_chapter_id") if metadata.get("last_chapter_id") is not None else (existing["last_chapter_id"] or 0),
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
                    "min_age": metadata.get("min_age") if metadata.get("min_age") is not None else existing["min_age"],
                    "author_id": metadata.get("author_id") if metadata.get("author_id") is not None else existing["author_id"]
                }
            else:
                # Creating new record - use provided values or defaults
                merged_data = {
                    "story_title": metadata.get("story_title"),
                    "story_type": metadata.get("story_type", "classic"),
                    "total_acts": metadata.get("total_acts", 3),
                    "target_length": metadata.get("target_length", 0),
                    "last_chapter_id": metadata.get("last_chapter_id", 0),
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
                    "min_age": metadata.get("min_age", 15),
                    "author_id": metadata.get("author_id", None)
                }

            await conn.execute("""
            INSERT INTO shared_story_data (
                story_id, story_title, story_type, total_acts, target_length, last_chapter_id, genre_list, sub_genre_list, themes_list, pov, tense, prose_style,
                blurb, tone_temp, image_data, public, min_age, author_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(story_id) DO UPDATE SET
                story_title = excluded.story_title,
                story_type = excluded.story_type,
                total_acts = excluded.total_acts,
                target_length = excluded.target_length,
                last_chapter_id = excluded.last_chapter_id,
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
                author_id = excluded.author_id,
                updated_at = CURRENT_TIMESTAMP
        """, (
                story_id,
                merged_data["story_title"],
                merged_data["story_type"],
                merged_data["total_acts"],
                merged_data["target_length"],
                merged_data["last_chapter_id"],
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
                merged_data["min_age"],
                merged_data["author_id"]
            ))
            await conn.commit()

    # User helpers
    # async def __get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
    #     async with self._get_connection() as conn:
    #         cursor = await conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
    #         row = await cursor.fetchone()
    #         return dict(row) if row else None

    async def _increment_story_views(self, story_id: str):
        async with self._get_connection() as conn:
            await conn.execute("""
                UPDATE shared_story_data
                SET views = COALESCE(views, 0) + 1
                WHERE story_id = ?
            """, (story_id,))
            await conn.commit()

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

    async def _append_story(self, user_id: str, story_title: str, story_id: str, story_type: str, self_created: bool = False):
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

                current_stories.append({"title": story_title, "story_id": story_id, "story_type": story_type, "self_created": self_created})

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
        
    async def __update_shared_story_length(self, story_id: str, word_count: int):
        """
        Updates the story_length (in words) for a shared story.
        Only allows updates if the story exists and is public or belongs to someone (safe to update).
        
        Args:
            story_id (str): The unique story_id
            word_count (int): Current total word count of the completed/shared story
        
        Returns:
            dict: {"status": "success"} or {"status": "error", "message": "..."}
        """
        if not isinstance(word_count, int) or word_count < 0:
            return {"status": "error", "message": "word_count must be a non-negative integer"}

        query = """
            UPDATE shared_story_data
            SET story_length = ?, updated_at = CURRENT_TIMESTAMP
            WHERE story_id = ?
        """
        async with self._get_connection() as conn:
            cursor = await conn.execute(query, (word_count, story_id))
            await conn.commit()

        if cursor.rowcount == 0:
            return {"status": "error", "message": "Story not found"}

        return {"status": "success", "message": "Story length updated"}
    
    async def _update_story_public_status(self, user_id: str, story_id: str, public: bool):
        """
        Allows only the author of the story to change its public status.
        
        Args:
            user_id (str): The user attempting the change
            story_id (str): The story to update
            public (bool): New public status (True = visible in gallery)
        
        Returns:
            dict: success or error message
        """
        try:
            async with self._get_connection() as conn:
                # First: verify ownership
                row = await conn.execute("""
                    SELECT author_id FROM shared_story_data 
                    WHERE story_id = ?
                """, (story_id,))

                result = await row.fetchone()
                author_id = result['author_id']

                if author_id != user_id:
                    return {"status": "error", "message": "You can only change public status of your own stories"}

                # Owner confirmed → update public status
                await conn.execute("""
                    UPDATE shared_story_data
                    SET public = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE story_id = ?
                """, (int(public), story_id))

                await conn.commit()

                story_progress = await self.get_story_progress()
                word_count = story_progress['story_word_count']

                # if story_progress and story_progress.get("story_id") == story_id:
                await self.__update_shared_story_length(
                    story_id=story_id,
                    word_count=word_count
                )

                return {
                    "status": "success",
                    "message": f"Story is now {'public' if public else 'private'}"
                }

        except aiosqlite.Error as e:
            return {"status": "error", "message": f"Database error: {str(e)}"}
        except Exception as e:
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

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

    async def update_self_story_progress(self, metadata: Optional[Dict[str, Any]] = None):
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
        ingestor_token_usage = metadata.get("ingestor_token_usage")
        utility_token_usage = metadata.get("utility_token_usage")
        async with self._get_connection() as conn:            
            # Update story_progress
            cursor = await conn.execute("""
                SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                    story_word_count, chapter_word_count, complete, 
                    author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
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
            updated_ingestor_token_usage = process_token_usage(
                ingestor_token_usage,
                existing_row["ingestor_token_usage"] if existing_row else None
            )
            updated_utility_token_usage = process_token_usage(
                utility_token_usage,
                existing_row["utility_token_usage"] if existing_row else None
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
                        ingestor_token_usage = ?,
                        utility_token_usage = ?,
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
                    updated_ingestor_token_usage,
                    updated_utility_token_usage,
                    self.user_id,
                    self.story_id
                ))

            else:
                await conn.execute("""
                    INSERT INTO story_progress (
                        user_id, story_id, current_act_id, latest_chapter_id, 
                        continue_scene_id, story_word_count, chapter_word_count, 
                        complete, author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    updated_writer_token_usage,
                    updated_ingestor_token_usage,
                    updated_utility_token_usage
                ))
            
            await conn.commit()

            return {'status': 'success', 'message': 'Updated Story State'}

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
        ingestor_token_usage = metadata.get("ingestor_token_usage")
        utility_token_usage = metadata.get("utility_token_usage")
        
        # Shared fields
        shared_metadata = {
            "story_title": metadata.get("story_title"),
            "story_type": metadata.get("story_type"),
            "total_acts": metadata.get("total_acts"),
            "target_length": metadata.get("target_length"),
            "last_chapter_id": chapter_id,
            "genre": metadata.get("genre"),
            "sub_genre": metadata.get("sub_genre"),
            "themes": metadata.get("themes"),
            "pov": metadata.get("pov"),
            "prose_style": metadata.get("prose_style"),
            "tense": metadata.get("tense"),
            "blurb": metadata.get("blurb"),
            "tone_temp": metadata.get("tone_temp"),
            "image_data": metadata.get("image_data"),
            "public": metadata.get("public"),
            "min_age": metadata.get("min_age"),
            "author_id": metadata.get("author_id")
        }

        async with self._get_connection() as conn:
            # Update shared_story_data
            await self._upsert_shared_story_data(self.story_id, shared_metadata)
            
            # Update story_progress
            cursor = await conn.execute("""
                SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                    story_word_count, chapter_word_count, complete, 
                    author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
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
            updated_ingestor_token_usage = process_token_usage(
                ingestor_token_usage,
                existing_row["ingestor_token_usage"] if existing_row else None
            )
            updated_utility_token_usage = process_token_usage(
                utility_token_usage,
                existing_row["utility_token_usage"] if existing_row else None
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
                        ingestor_token_usage = ?,
                        utility_token_usage = ?,
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
                    updated_ingestor_token_usage,
                    updated_utility_token_usage,
                    self.user_id,
                    self.story_id
                ))
            else:
                await conn.execute("""
                    INSERT INTO story_progress (
                        user_id, story_id, current_act_id, latest_chapter_id, 
                        continue_scene_id, story_word_count, chapter_word_count, 
                        complete, author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    updated_writer_token_usage,
                    updated_ingestor_token_usage,
                    updated_utility_token_usage
                ))
            
            await conn.commit()
    
    async def get_self_story_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest story progress only
        """
        if not self.user_id or not self.story_id:
            raise ValueError("user_id and story_id must be provided")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                        story_word_count, chapter_word_count, complete, 
                        author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
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
                    "complete": False,
                    "author_token_usage": 0,
                    "director_token_usage": 0,
                    "writer_token_usage": 0,
                    "ingestor_token_usage": 0,
                    "utility_token_usage": 0
                }
                
                if progress_row:
                    result.update({
                        "current_act_id": progress_row["current_act_id"] if progress_row["current_act_id"] is not None else 1,
                        "latest_chapter_id": progress_row["latest_chapter_id"] if progress_row["latest_chapter_id"] is not None else 0,
                        "continue_scene_id": progress_row["continue_scene_id"] if progress_row["continue_scene_id"] is not None else 0,
                        "story_word_count": progress_row["story_word_count"] if progress_row["story_word_count"] is not None else 0,
                        "complete": bool(progress_row["complete"]),
                        "author_token_usage": progress_row["author_token_usage"] if progress_row["author_token_usage"] is not None else 0,
                        "director_token_usage": progress_row["director_token_usage"] if progress_row["director_token_usage"] is not None else 0,
                        "writer_token_usage": progress_row["writer_token_usage"] if progress_row["writer_token_usage"] is not None else 0,
                        "ingestor_token_usage": progress_row["ingestor_token_usage"] if progress_row["ingestor_token_usage"] is not None else 0,
                        "utility_token_usage": progress_row["utility_token_usage"] if progress_row["utility_token_usage"] is not None else 0
                    })
                
                return result
        
        except aiosqlite.Error as e:
            raise aiosqlite.Error(f"Database error in get_self_story_progress: {str(e)}")
        
    async def get_story_progress(self) -> Optional[Dict[str, Any]]:
        """
        Get the latest story progress, combining user-specific and shared story data.
        """
        if not self.user_id or not self.story_id:
            raise ValueError("user_id and story_id must be provided")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute("""
                    SELECT current_act_id, latest_chapter_id, continue_scene_id, 
                        story_word_count, chapter_word_count, complete, rating,
                        author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
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
                    "last_chapter_id": 0,
                    "continue_scene_id": 0,
                    "story_word_count": 0,
                    "chapter_word_count": 0,
                    "complete": False,
                    "author_token_usage": 0,
                    "director_token_usage": 0,
                    "writer_token_usage": 0,
                    "ingestor_token_usage": 0,
                    "utility_token_usage": 0,
                    "total_acts": 3,
                    "target_length": 0,
                    "pov": None,
                    "tense": None,
                    "prose_style": None,
                    "rating": 0,
                    "genre": [],
                    "themes": [],
                    "story_type": "classic",
                    "story_title": None,
                    "tone_temp": None,
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
                        "writer_token_usage": progress_row["writer_token_usage"] if progress_row["writer_token_usage"] is not None else 0,
                        "ingestor_token_usage": progress_row["ingestor_token_usage"] if progress_row["ingestor_token_usage"] is not None else 0,
                        "utility_token_usage": progress_row["utility_token_usage"] if progress_row["utility_token_usage"] is not None else 0,
                        "rating": progress_row["rating"] if progress_row["rating"] is not None else 0
                    })
                
                if shared_data:
                    result.update({
                        "total_acts": shared_data["total_acts"] if shared_data["total_acts"] is not None else 3,
                        "target_length": shared_data["target_length"] if shared_data["target_length"] is not None else 0,
                        "last_chapter_id": shared_data["last_chapter_id"] if shared_data["last_chapter_id"] is not None else 0,
                        "pov": shared_data["pov"],
                        "tense": shared_data["tense"],
                        "prose_style": shared_data["prose_style"],
                        "genre": shared_data["genre_list"] or [],
                        "sub_genre": shared_data["sub_genre_list"] or [],
                        "themes": shared_data["themes_list"] or [],
                        "story_type": shared_data["story_type"] or "classic",
                        "story_title": shared_data["story_title"],
                        "tone_temp": shared_data["tone_temp"],
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

#----------- Public Calls ----------------
    async def _get_public_stories_page(self, min_age: int, page_index: int = 1, page_size: int = 9):
        print("\n--- DEBUG: GET PUBLIC STORIES PAGE ---")
        print(f"Input min_age      = {min_age} (type: {type(min_age)})")
        print(f"Page index         = {page_index}")
        print(f"Page size          = {page_size}")

        offset = (page_index - 1) * page_size
        print(f"Calculated offset  = {offset}")

        async with self._get_connection() as conn:
            query = """
                SELECT 
                    s.story_id,
                    s.story_title,
                    s.story_type,
                    s.story_length,
                    s.genre_list,
                    s.sub_genre_list,
                    s.themes_list,
                    s.pov,
                    s.prose_style,
                    s.blurb,
                    s.image_data,
                    s.min_age,
                    s.author_id,
                    s.overall_rating,
                    s.total_ratings,
                    s.views,
                    s.created_at,
                    s.updated_at,
                    u.nickname AS author_nickname
                FROM shared_story_data s
                JOIN users u ON s.author_id = u.user_id
                WHERE s.public = 1
                AND s.min_age <= ?
                ORDER BY s.created_at DESC
                LIMIT ? OFFSET ?
            """

            print("\n--- DEBUG: Executing SQL ---")
            print(query)
            print(f"SQL Params: min_age={min_age}, limit={page_size}, offset={offset}")

            async with conn.execute(query, (min_age, page_size, offset)) as cursor:
                columns = [desc[0] for desc in cursor.description]
                print("\n--- DEBUG: Columns Returned ---")
                print(columns)

                rows = await cursor.fetchall()
                print(f"\n--- DEBUG: Number of rows returned = {len(rows)}")

                # If no rows returned, test why
                if len(rows) == 0:
                    print("\n--- DEBUG: Checking all public rows with raw min_age values ---")
                    async with conn.execute(
                        "SELECT story_id, min_age, typeof(min_age), public FROM shared_story_data WHERE public = 1"
                    ) as debug_cur:
                        debug_rows = await debug_cur.fetchall()
                        for dr in debug_rows:
                            print(f"Row: story_id={dr[0]}, min_age={dr[1]}, typeof={dr[2]}, public={dr[3]}")
                    print("--- END DEBUG (NO MATCHING ROWS) ---\n")

        # Process results normally
        stories = []
        for row in rows:
            story_dict = dict(zip(columns, row))

            # Print the raw DB row for diagnosis
            print("\n--- DEBUG: RAW ROW DATA ---")
            for k, v in story_dict.items():
                print(f"{k}: {repr(v)} (type: {type(v)})")

            # Remove non-public fields
            hidden_fields = ['total_acts', 'tone_temp', 'public', 'tense', 'target_length']
            for key in hidden_fields:
                story_dict.pop(key, None)

            stories.append(story_dict)

        print("\n--- END DEBUG ---\n")
        return {"status": "success", "stories": stories}


    
    async def __recalculate_overall_rating(self, story_id: str):
        """
        Recalculates and updates the overall_rating and total_ratings in shared_story_data
        based on all non-null ratings in story_progress for this story_id.
        """
        # Get all ratings that are not NULL or 0
        query = """
            SELECT rating FROM story_progress
            WHERE story_id = ? AND rating IS NOT NULL AND rating > 0
        """
        async with self._get_connection() as conn:
            async with conn.execute(query, (story_id,) ) as cursor:
                rows = await cursor.fetchall()

            ratings = [row[0] for row in rows]
            total_ratings = len(ratings)

            if not ratings:
                new_rating = None
            else:
                new_rating = round(sum(ratings) / len(ratings), 2)

            update_query = """
                UPDATE shared_story_data
                SET overall_rating = ?,
                    total_ratings = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE story_id = ?
            """
            await conn.execute(update_query, (new_rating, total_ratings, story_id))
            await conn.commit()

    async def _rate_user_story(self, user_id: str, story_id: str, rating: int):
        """
        Updates the rating (1–5) for a user's progress on a specific story.
        Then triggers recalculation of the overall rating.
        """
        print(rating)
        if not 1 <= rating <= 5:
            return {"status": "error", "message": "Rating must be between 1 and 5"}

        update_query = """
            UPDATE story_progress
            SET rating = ?, updated_at = CURRENT_TIMESTAMP
            WHERE user_id = ? AND story_id = ?
        """
        async with self._get_connection() as conn:
            result = await conn.execute(update_query, (rating, user_id, story_id))
            await conn.commit()

        if result.rowcount == 0:
            return {"status": "error", "message": "Story progress not found for this user"}

        # Recalculate overall rating after update
        await self.__recalculate_overall_rating(story_id)

        return {"status": "success", "message": "Rating saved"}


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
                SELECT id, user_id, story_id, current_act_id, latest_chapter_id, 
                       continue_scene_id, story_word_count, complete, author_token_usage, director_token_usage, writer_token_usage, ingestor_token_usage, utility_token_usage
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