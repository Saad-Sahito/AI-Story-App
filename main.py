from fastapi import HTTPException
import asyncio
import time
import redis.asyncio as redis
from fastapi import FastAPI, WebSocket
import json
from contextlib import asynccontextmanager
from setup.shared_redis_pool import REDIS_POOL, get_redis_client
from setup.main_setup import MainSetup
from src.memory.user_management import add_user, delete_story, get_user_profile_with_stories
from src.memory.shared_resources import SHARED_QDRANT
from src.memory.storage_delete import delete_all_qdrant_collections, delete_sqlite_db
from setup.story_types.interactive_setup import get_shared_interactive_setup, close_shared_interactive_setup
from setup.story_types.classic_setup import get_shared_classic_setup, close_shared_classic_setup
from src.story_engines.interactive_adventure.agents import shared_scene_planner as scene_planner_module

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




@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up AI Story App...")

    # Initialize shared resources
    try:
        await get_shared_interactive_setup()
        await get_shared_classic_setup()
        print("✅ Shared story setups initialized")
    except Exception as e:
        print(f"❌ Error initializing story setups: {e}")

    # Verify shared clients
    if scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE is None:
        print("❌ ERROR: SHARED_SCENE_PLANNER_SERVICE was not initialized!")
    else:
        print("✅ SHARED_SCENE_PLANNER_SERVICE is properly initialized")
    if SHARED_QDRANT is None:
        print("❌ ERROR: SHARED_QDRANT was not initialized!")
    else:
        print("✅ SHARED_QDRANT is properly initialized")

    # Start background cleanup task
    global cleanup_task
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    print("✅ Background cleanup task started")

    yield

    # Shutdown
    print("🛑 Shutting down AI Story App...")

    # Cancel background cleanup task
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass

    # Clean up shared story setups
    try:
        await close_shared_interactive_setup()
        await close_shared_classic_setup()
        print("✅ Shared story setups closed")
    except Exception as e:
        print(f"❌ Error closing story setups: {e}")

    # Clean up shared Qdrant client
    if SHARED_QDRANT:
        try:
            await SHARED_QDRANT.close()  # Assuming AsyncQdrantClient
            print("✅ SHARED_QDRANT closed")
        except Exception as e:
            print(f"❌ Error closing SHARED_QDRANT: {e}")

    # Clean up all Redis sessions
    try:
        async with await get_redis_client() as client:
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match="session:*", count=100)
                for key in keys:
                    user_id = key.decode().split(":")[1]
                    async with await redis_lock(client, f"lock:{key}"):  # Use async redis_lock
                        await MainSetup.logout(user_id)  # Make logout async (see below)
                if cursor == 0:
                    break
            print(f"✅ Cleaned up {len(keys)} Redis session keys")
    except redis.RedisError as e:
        print(f"❌ Error cleaning up Redis sessions: {e}")
    finally:
        # Disconnect Redis pool
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
async def api_logout_story(user_id: str, story_id: str):
    return mainsetup.logout_story(user_id=user_id, story_id=story_id)

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
    """Retrieve all session data for active users from Redis."""
    try:
        client = await get_redis_client()
        users = {}
        cursor = 0
        while True:
            cursor, keys = await client.scan(cursor, match="session:*", count=100)
            for key in keys:
                try:
                    parts = key.split(":", 2)
                    user_id = parts[1]
                    if user_id not in users:
                        users[user_id] = {"last_active": None, "stories": {}}
                    data = client.get(key)
                    if not data:
                        print(f"🔍 No data for key {key}")
                        continue
                    session_data = json.loads(data)
                    if len(parts) == 2:  # User-level key: session:user_id
                        users[user_id]["last_active"] = session_data.get("last_active")
                        users[user_id]["stories"] = session_data.get("stories", {})
                    elif len(parts) == 3:  # Story-specific key: session:user_id:story_id
                        story_id = parts[2]
                        users[user_id]["stories"][story_id] = session_data
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
def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
    return add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)

@app.patch("/users/{user_id}/stories/{story_title}")
def api_delete_story(user_id: str, story_title: str):
    story_title_normalized = story_title.lower().replace(" ", "_")
    story_id = f"{story_title_normalized}_{user_id}"
    mainsetup.logout_story(user_id, story_id)
    return delete_story(user_id, story_title)

@app.get("/users/{user_id}/profile")
def api_get_user_profile_data_and_stories(user_id: str):
    return get_user_profile_with_stories(user_id=user_id)

#-----------------------------------------------------------------
# CAUTION: Deletes entire app storage (admin only)
@app.patch("/storage")
def api_del_storage():
    sql_path = "/home/saadn/whimsera_app/data/story_memory.db"
    val1 = delete_sqlite_db(sql_path)
    val2 = delete_all_qdrant_collections()
    return val1, val2