from fastapi import HTTPException, Query
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
from pydantic import BaseModel
from contextlib import asynccontextmanager
from fastapi import Request, WebSocketDisconnect
from fastapi.responses import JSONResponse
from jose import jwt, JWTError, jwk
import httpx
import os
from jose.utils import base64url_decode




# Initialize api_backend BEFORE lifespan
print("🟡 Initializing APIBackend...")
try:
    mainsetup = MainSetup()
    print("✅ APIBackend initialized")
except Exception as e:
    print("❌ APIBackend initialization error: ", e)

# Background cleanup task
cleanup_task = None

from fastapi.middleware.cors import CORSMiddleware




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
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://whimsera.com",
        "https://www.whimsera.com",
        "http://localhost:3000",  # for local development
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Async redis_lock (reused from interactive_setup.py)



SUPABASE_PROJECT_URL = os.getenv("SUPABASE_URL", "https://YOUR_PROJECT_ID.supabase.co")
JWKS_URL = f"{SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json"
jwks_cache = None
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
if not SUPABASE_JWT_SECRET:
    raise ValueError("SUPABASE_JWT_SECRET environment variable is required for HS256 verification")

async def get_jwks():
    global jwks_cache
    if jwks_cache is None:
        async with httpx.AsyncClient() as client:
            resp = await client.get(JWKS_URL)
            resp.raise_for_status()
            jwks_cache = resp.json()
    return jwks_cache

async def verify_supabase_jwt(token: str):
    """Verifies Supabase JWT using the project's JWT Secret (HS256 symmetric signing)."""
    try:
        print(f"Verifying JWT with HS256 (partial): {token[:10]}...")
        
        # Verify signature and claims, including audience
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",  # Match the JWT's aud claim
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": False,
                "verify_aud": True  # Explicitly enable audience verification
            }
        )
        
        print(f"JWT payload: {payload}")
        return payload
        
    except JWTError as e:
        detail = str(e)
        if "Signature" in detail:
            detail = "Invalid token signature"
        elif "exp" in detail.lower():
            detail = "Token expired"
        elif "audience" in detail.lower():
            detail = "Invalid audience"
        else:
            detail = "Invalid token payload"
        
        print(f"JWT verification error: {detail}")
        raise HTTPException(status_code=401, detail=detail)

@app.middleware("http")
async def supabase_auth_middleware(request: Request, call_next):
    print(f"🔍 Middleware invoked for: {request.method} {request.url.path}")
    if request.method == "OPTIONS":
        print(f"✅ Skipping auth for OPTIONS request: {request.url.path}")
        return await call_next(request)
    whitelist = ["/users/active", "/storage", "/docs", "/openapi.json"]
    if request.url.path == "/" or any(request.url.path.startswith(path) for path in whitelist):
        print(f"✅ Skipping auth for whitelisted route: {request.url.path}")
        return await call_next(request)
    print(f"🔎 Processing non-whitelisted route: {request.url.path}")
    auth_header = request.headers.get("Authorization")
    print(f"🔎 Authorization header: {auth_header}")
    if not auth_header or not auth_header.startswith("Bearer "):
        print("❌ Missing or invalid Authorization header")
        return JSONResponse(status_code=401, content={"detail": "Missing or invalid Authorization header"})
    token = auth_header.split(" ")[1]
    print(f"🔎 Extracted token: {token[:10]}...")
    try:
        payload = await verify_supabase_jwt(token)
        print(f"✅ JWT verified, user: {payload.get('sub')}")
    except Exception as e:
        print(f"❌ JWT verification failed: {str(e)}")
        return JSONResponse(status_code=401, content={"detail": str(e)})
    supabase_user_id = payload.get("sub")
    if not supabase_user_id:
        print("❌ Invalid token payload: No 'sub' claim")
        return JSONResponse(status_code=401, content={"detail": "Invalid token payload"})
    try:
        if request.method in ("POST", "PUT", "PATCH"):
            body = await request.json()
            user_id = body.get("user_id") or body.get("initial_story_data", {}).get("user_id")
        else:
            user_id = request.query_params.get("user_id") or request.path_params.get("user_id")
    except Exception as e:
        print(f"❌ Error reading request body or query: {str(e)}")
        user_id = request.query_params.get("user_id") or request.path_params.get("user_id")
    if user_id and user_id != supabase_user_id:
        print(f"❌ User ID mismatch: {user_id} != {supabase_user_id}")
        return JSONResponse(status_code=403, content={"detail": "User ID does not match authenticated user"})
    print(f"✅ Authenticated Supabase user: {supabase_user_id}")
    request.state.supabase_user = payload
    return await call_next(request)


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
async def websocket_next_chapter(websocket: WebSocket, user_id: str, story_id: str, story_type: str = Query(...)):
    try:
        print(f"🔵 Calling api_backend.handle_story_websocket...")
        token = (
            websocket.headers.get("Authorization", "").replace("Bearer ", "")
            or websocket.query_params.get("token")
        )
        print(f"Received token: {token[:10]}...")  # Log partial token
        if not token:
            await websocket.close(code=4001)
            print("❌ Missing Supabase token in WebSocket connection")
            return
        
        payload = await verify_supabase_jwt(token)
        supabase_user_id = payload.get("sub")
        print(f"User ID from URL: {user_id}, Supabase user ID: {supabase_user_id}")
        
        if not supabase_user_id or supabase_user_id != user_id:
            await websocket.close(code=4003)
            print(f"❌ WebSocket auth failed: user_id mismatch ({user_id} != {supabase_user_id})")
            return
        
        await mainsetup.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id, story_type=story_type)
        print(f"🟢 api_backend.handle_story_websocket completed successfully")
    except WebSocketDisconnect:
        print(f"⚪ WebSocket disconnected: {user_id}")
    except Exception as e:
        print(f"🔴 Error in websocket_next_chapter: {e}")
        import traceback
        traceback.print_exc()
        try:
            await websocket.close(code=1011)
        except Exception:
            pass
        
@app.post("/stories/initialize_story")
async def api_initialize_story(
    user_id: str = Query(...),
    story_type: str = Query(...),
    story_title: str = Query("")
):
    return await mainsetup.initialize_story(user_id=user_id, story_title=story_title, story_type=story_type)

class ContinueStoryRequest(BaseModel):
    user_id: str
    story_type: str

@app.put("/stories/{story_id}")
async def api_continue_story(story_id: str, request: ContinueStoryRequest):
    return await mainsetup.continue_story(user_id=request.user_id, story_id=story_id, story_type=request.story_type)

@app.get("/stories/progress/{user_id}/{story_id}")
async def api_get_story_progress(user_id: str, story_id: str, story_type: str):
    return await mainsetup.get_story_progress_for_user(user_id=user_id, story_id=story_id, story_type=story_type)

@app.patch("/stories/logout/{user_id}/{story_id}")
async def api_logout_story(user_id: str, story_id: str, story_type: str):
    return await mainsetup.logout_story(user_id=user_id, story_id=story_id, story_type=story_type)

@app.get("/stories/cluster/{user_id}/{story_id}")
async def api_story_cluster(user_id: str, story_id: str, story_type: str, chapter_number: int):
    return await mainsetup.get_story_cluster(user_id=user_id, story_id=story_id, story_type=story_type, chapter_number=chapter_number)

# ---------------- User Session Management Routes ----------------
@app.patch("/users/{user_id}/session")
async def api_logout(user_id: str):
    return await mainsetup.logout(user_id)

# gets all active users in redis pool (for app manager use)
# @app.get("/users/active")
# async def api_get_active_users():
#     """Retrieve all session data for active users from Redis (classic + interactive)."""
#     try:
#         client = await get_redis_client()
#         users = {}
#         cursor = 0
#         patterns = ["classic_session:*", "interactive_session:*"]

#         for pattern in patterns:
#             cursor = 0
#             while True:
#                 cursor, keys = await client.scan(cursor, match=pattern, count=100)
#                 for key in keys:
#                     try:
#                         parts = key.split(":", 2)  # e.g., classic_session:user_id[:story_id]
#                         if len(parts) < 2:
#                             continue

#                         prefix = parts[0]  # "classic_session" or "interactive_session"
#                         user_id = parts[1]

#                         if user_id not in users:
#                             users[user_id] = {
#                                 "classic_session": {"last_active": None, "stories": {}},
#                                 "interactive_session": {"last_active": None, "stories": {}}
#                             }

#                         data = await client.get(key)
#                         if not data:
#                             print(f"🔍 No data for key {key}")
#                             continue

#                         session_data = json.loads(data)

#                         if len(parts) == 2:  # user-level key
#                             users[user_id][prefix]["last_active"] = session_data.get("last_active")
#                             users[user_id][prefix]["stories"] = session_data.get("stories", {})
#                         elif len(parts) == 3:  # story-specific key
#                             story_id = parts[2]
#                             users[user_id][prefix]["stories"][story_id] = session_data

#                     except json.JSONDecodeError as e:
#                         print(f"❌ Invalid JSON for key {key}: {e}")
#                         continue
#                     except Exception as e:
#                         print(f"❌ Error processing key {key}: {e}")
#                         continue

#                 if cursor == 0:
#                     break

#         return {
#             "status": "success",
#             "users": users,
#             "count": len(users)
#         }

#     except redis.RedisError as e:
#         print(f"❌ Redis error in api_get_active_users: {e}")
#         raise HTTPException(status_code=500, detail="Failed to retrieve active users")
#     finally:
#         import gc
#         gc.collect()


# ---------------- User Data Management Routes ----------------


class UserCreate(BaseModel):
    nickname: str
    user_tag: str
    age: int
    stories: list = []
    user_id: str | None = None

@app.post("/users")
async def api_add_user(user: UserCreate):
    return await add_user(
        nickname=user.nickname,
        user_tag=user.user_tag,
        age=user.age,
        user_id=user.user_id,
        stories=user.stories
    )

# @app.post("/users")
# async def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
#     return await add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)

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
# @app.patch("/storage")
# async def api_del_storage():
#     sql_path = "/home/saadn/whimsera_app/data/story_memory.db"
#     val1 = delete_sqlite_db(sql_path)
#     val2 = await delete_all_qdrant_collections()
#     return val1, val2