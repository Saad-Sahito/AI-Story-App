from fastapi import HTTPException
import asyncio
import time
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket
import json
import os
from pathlib import Path
from contextlib import asynccontextmanager
from setup.shared_redis_pool import REDIS_POOL, get_redis_client
from setup.main_setup import MainSetup
from src.memory.user_management import add_user, delete_story, get_user_profile_with_stories
from src.memory.shared_resources import SHARED_QDRANT
from src.memory.storage_delete import delete_all_qdrant_collections, delete_sqlite_db
from setup.story_types.interactive_setup import get_shared_interactive_setup, close_shared_interactive_setup
from setup.story_types.classic_setup import get_shared_classic_setup, close_shared_classic_setup
from src.story_engines.interactive_adventure.agents import shared_scene_planner as interactive_scene_planner_module
from src.story_engines.classic_narrative.agents import shared_scene_planner as classic_scene_planner_module
from src.memory.sqlite_store import SQLiteStore

# Initialize api_backend BEFORE lifespan
print("🟡 Initializing APIBackend...")
try:
    mainsetup = MainSetup()
    print("✅ APIBackend initialized")
except Exception as e:
    print("❌ APIBackend initialization error: ", e)

# Background cleanup task
cleanup_task = None

async def cleanup_sessions_periodically():
    """Background task to clean up inactive sessions every hour."""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            cleaned_count = mainsetup.cleanup_inactive_sessions(max_age_seconds=3600)
            if cleaned_count > 0:
                print(f"✅ Cleaned up {cleaned_count} inactive sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Error in session cleanup: {e}")




# Base path = directory where this file is located
BASE_DIR = Path(__file__).resolve().parent

# Data folder path
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)  # ✅ makes 'data' if it doesn't exist

# Full path to the database file
DB_PATH = DATA_DIR / "story_memory.db"

# Example: creating the file (if you just want to ensure it exists)
if not DB_PATH.exists():
    DB_PATH.touch()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up AI Story App...")
    try:
        await get_shared_interactive_setup()
        await get_shared_classic_setup()
        await SQLiteStore._init_database(DB_PATH)
        print("✅ Shared story setups initialized")
    except Exception as e:
        print(f"❌ Error initializing story setups: {e}")
        raise

    if interactive_scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE is None:
        print("❌ ERROR: SHARED_SCENE_PLANNER_SERVICE was not initialized!")
    else:
        print("✅ SHARED_SCENE_PLANNER_SERVICE is properly initialized")
    if classic_scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE is None:
        print("❌ ERROR: SHARED_SCENE_PLANNER_SERVICE was not initialized!")
    else:
        print("✅ SHARED_SCENE_PLANNER_SERVICE is properly initialized")
    if SHARED_QDRANT is None:
        print("❌ ERROR: SHARED_QDRANT was not initialized!")
    else:
        print("✅ SHARED_QDRANT is properly initialized")

    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    print("✅ Background cleanup task started")

    try:
        yield
    finally:
        print("🛑 Shutting down AI Story App...")
        if cleanup_task:
            cleanup_task.cancel()
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

        try:
            await close_shared_interactive_setup()
            await close_shared_classic_setup()
            print("✅ Shared story setups closed")
        except Exception as e:
            print(f"❌ Error closing story setups: {e}")

        if SHARED_QDRANT:
            try:
                await SHARED_QDRANT.close()
                print("✅ SHARED_QDRANT closed")
            except Exception as e:
                print(f"❌ Error closing SHARED_QDRANT: {e}")

        try:
            async with await get_redis_client() as client:
                cursor = 0
                patterns = ["session:*", "classic_session:*", "interactive_session:*", "active_ws:*", "director_running:*"]
                deleted_keys = 0
                for pattern in patterns:
                    cursor = 0
                    while True:
                        cursor, keys = await client.scan(cursor, match=pattern, count=100)
                        for key in keys:
                            lock_key = f"lock:{key}"
                            try:
                                async with redis_lock(client, lock_key, timeout=30):
                                    await client.delete(key)
                                    deleted_keys += 1
                            except redis.RedisError as e:
                                print(f"❌ Failed to delete key {key}: {e}")
                        if cursor == 0:
                            break
                print(f"✅ Cleaned up {deleted_keys} Redis keys")
        except redis.RedisError as e:
            print(f"❌ Error cleaning up Redis sessions: {e}")
        finally:
            await REDIS_POOL.disconnect()
            print("✅ Redis connection pool disconnected")
        print("✅ Shutdown complete")

app = FastAPI(
    title="Whimsera App",
    description="AI Story Generation App",
    lifespan=lifespan
)

# Async redis_lock (reused from interactive_setup.py)
from contextlib import asynccontextmanager

@asynccontextmanager
async def redis_lock(client, lock_key, timeout=10):
    lock_value = str(time.time())
    acquired = await client.set(lock_key, lock_value, nx=True, ex=timeout)
    if acquired:
        try:
            yield
        finally:
            lua = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                await client.eval(lua, 1, lock_key, lock_value)
            except redis.RedisError:
                pass
    else:
        raise HTTPException(status_code=503, detail="Could not acquire lock")

@app.get("/")
async def root():
    return {"message": "status ok"}

# ---------------- Story Management API Routes ----------------
@app.post("/premise")
async def api_create_premise(initial_story_data: dict):
    return await mainsetup.create_premise(initial_story_data=initial_story_data)

@app.websocket("/ws/next_chapter/{user_id}/{story_id}")
async def websocket_next_chapter(websocket: WebSocket, user_id: str, story_id: str, story_type: str):
    try:
        print(f"🔵 Calling api_backend.handle_story_websocket...")
        await mainsetup.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id, story_type=story_type)
        print(f"🟢 api_backend.handle_story_websocket completed successfully")
    except Exception as e:
        print(f"🔴 Error in websocket_next_chapter: {e}")
        import traceback
        traceback.print_exc()

@app.post("/stories/initialize_story")
async def api_initialize_story(user_id: str, story_type: str, story_title: str = "" ):
    return await mainsetup.initialize_story(user_id=user_id, story_title=story_title, story_type=story_type)

@app.put("/stories/{story_id}")
async def api_continue_story(user_id: str, story_id: str, story_type: str):
    return await mainsetup.continue_story(user_id=user_id, story_id=story_id, story_type=story_type)

@app.get("/stories/progress/{user_id}/{story_id}")
async def api_get_story_progress(user_id: str, story_id: str, story_type: str):
    return await mainsetup.get_story_progress_for_user(user_id=user_id, story_id=story_id, story_type=story_type)

@app.patch("/stories/logout/{user_id}/{story_id}")
async def api_logout_story(user_id: str, story_id: str, story_type: str):
    return mainsetup.logout_story(user_id=user_id, story_id=story_id, story_type=story_type)

@app.get("/stories/cluster/{user_id}/{story_id}")
async def api_story_cluster(user_id: str, story_id: str, story_type: str, chapter_number: int):
    return mainsetup.get_story_cluster(user_id=user_id, story_id=story_id, story_type=story_type, chapter_number=chapter_number)

# ---------------- User Session Management Routes ----------------
@app.patch("/users/{user_id}/session")
def api_logout(user_id: str):
    return mainsetup.logout(user_id)

# gets all active users in redis pool (for app manager use)
@app.get("/users/active")
async def api_get_active_users():
    """Retrieve all session data for active users from Redis (classic + interactive)."""
    try:
        client = await get_redis_client()
        users = {}
        cursor = 0
        patterns = ["classic_session:*", "interactive_session:*"]

        for pattern in patterns:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                for key in keys:
                    try:
                        parts = key.split(":", 2)  # e.g., classic_session:user_id[:story_id]
                        if len(parts) < 2:
                            continue

                        prefix = parts[0]  # "classic_session" or "interactive_session"
                        user_id = parts[1]

                        if user_id not in users:
                            users[user_id] = {
                                "classic_session": {"last_active": None, "stories": {}},
                                "interactive_session": {"last_active": None, "stories": {}}
                            }

                        data = await client.get(key)
                        if not data:
                            print(f"🔍 No data for key {key}")
                            continue

                        session_data = json.loads(data)

                        if len(parts) == 2:  # user-level key
                            users[user_id][prefix]["last_active"] = session_data.get("last_active")
                            users[user_id][prefix]["stories"] = session_data.get("stories", {})
                        elif len(parts) == 3:  # story-specific key
                            story_id = parts[2]
                            users[user_id][prefix]["stories"][story_id] = session_data

                    except json.JSONDecodeError as e:
                        print(f"❌ Invalid JSON for key {key}: {e}")
                        continue
                    except Exception as e:
                        print(f"❌ Error processing key {key}: {e}")
                        continue

                if cursor == 0:
                    break

        return {
            "status": "success",
            "users": users,
            "count": len(users)
        }

    except redis.RedisError as e:
        print(f"❌ Redis error in api_get_active_users: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve active users")
    finally:
        import gc
        gc.collect()


# ---------------- User Data Management Routes ----------------
@app.post("/users")
async def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
    return await add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)

@app.patch("/users/{user_id}/stories/{story_title}")
async def api_delete_story(user_id: str, story_title: str):
    story_title_normalized = story_title.lower().replace(" ", "_")
    story_id = f"{story_title_normalized}_{user_id}"
    await mainsetup.logout_story(user_id, story_id)
    return await delete_story(user_id, story_title)

@app.get("/users/{user_id}/profile")
async def api_get_user_profile_data_and_stories(user_id: str):
    return await get_user_profile_with_stories(user_id=user_id)

#-----------------------------------------------------------------
# CAUTION: Deletes entire app storage (admin only)
@app.patch("/storage")
async def api_del_storage():
    sql_path = "/home/saadn/whimsera_app/data/story_memory.db"
    val1 = delete_sqlite_db(sql_path)
    val2 = await delete_all_qdrant_collections()
    return val1, val2