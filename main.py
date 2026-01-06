
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from jose import jwt, JWTError
from typing import Dict, Any, List
import httpx
import os
from pathlib import Path
from contextlib import asynccontextmanager
import asyncio
import time
import json
import redis.asyncio as redis
from pydantic import BaseModel, ConfigDict
from src.setup.main_setup import MainSetup
from src.setup.shared_redis_pool import get_redis_client, REDIS_POOL
from src.memory.user_management import add_user, delete_story, get_user_profile_with_stories, get_user_profile, update_user_settings
from src.memory.feedback import post_user_feedback
from src.memory.shared_resources import SHARED_QDRANT

from src.memory.sqlite_store import SQLiteStore



# Background cleanup task
cleanup_task = None
try:
    mainsetup = MainSetup()
    #print("✅ APIBackend initialized")
except Exception as e:
    print("❌ APIBackend initialization error: ", e)
async def cleanup_sessions_periodically():
    """Background task to clean up inactive sessions every hour."""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            cleaned_count = await mainsetup.cleanup_inactive_sessions()
            if cleaned_count > 0:
                print(f"✅ Cleaned up {cleaned_count} inactive sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Error in session cleanup: {e}")

# Base path and data directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "story_memory.db"
if not DB_PATH.exists():
    DB_PATH.touch()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up AI Story App...")
    try:
        await SQLiteStore._init_database(DB_PATH)
        print("✅ Shared story setups initialized")
    except Exception as e:
        print(f"❌ Error initializing story setups: {e}")
        raise


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

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://whimsera.com",
        "https://pre-alpha.whimsera.com",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:4000",
        "http://127.0.0.1:4000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
    allow_headers=["*"],
)

# Supabase Configuration
SUPABASE_PROJECT_URL = os.getenv("SUPABASE_URL")
JWKS_URL = f"{SUPABASE_PROJECT_URL}/auth/v1/.well-known/jwks.json"
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
        #print(f"Verifying JWT with HS256 (partial): {token[:10]}...")
        payload = jwt.decode(
            token,
            SUPABASE_JWT_SECRET,
            algorithms=["HS256"],
            audience="authenticated",
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_nbf": False,
                "verify_aud": True,
            }
        )
        #print(f"JWT payload: {payload}")
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
        #print(f"JWT verification error: {detail}")
        raise HTTPException(status_code=401, detail=detail)

from fastapi import Depends

async def require_admin(request: Request):
    """
    Ensures that the Supabase-authenticated user has the 'admin' role
    inside raw_app_meta_data.
    """
    payload = getattr(request.state, "supabase_user", None)
    if not payload:
        raise HTTPException(status_code=401, detail="Missing authentication context")

    app_metadata = payload.get("app_metadata") or payload.get("raw_app_meta_data") or {}
    role = app_metadata.get("role")

    if role != "admin":
        raise HTTPException(status_code=403, detail="Access denied: Admin role required")

    return payload

# Authentication Middleware
@app.middleware("http")
async def supabase_auth_middleware(request: Request, call_next):
    #print(f"🔍 Middleware invoked for: {request.method} {request.url.path}")

    # Handle OPTIONS requests for all routes
    if request.method == "OPTIONS":
        #print(f"✅ Handling OPTIONS request for: {request.url.path}")
        return JSONResponse(
            status_code=200,
            content={"message": "OK"},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "http://127.0.0.1:3000"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )

    # Skip authentication for whitelisted routes
    whitelist = ["/users/active", "/docs", "/openapi.json", "/community/stories"]
    if request.url.path == "/" or any(request.url.path.startswith(path) for path in whitelist):
        #print(f"✅ Skipping auth for whitelisted route: {request.url.path}")
        response = await call_next(request)
        response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "http://127.0.0.1:3000")
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD"
        response.headers["Access-Control-Allow-Headers"] = "*"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    # Authenticate non-whitelisted routes
    auth_header = request.headers.get("Authorization")
    #print(f"🔎 Authorization header: {auth_header}")

    if not auth_header or not auth_header.startswith("Bearer "):
        print("❌ Missing or invalid Authorization header")
        return JSONResponse(
            status_code=401,
            content={"detail": "Missing or invalid Authorization header"},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "http://127.0.0.1:3000"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )

    token = auth_header.split(" ")[1]
    print(f"🔎 Extracted token: {token[:10]}...")
    try:
        payload = await verify_supabase_jwt(token)
        print(f"✅ JWT verified, user: {payload.get('sub')}")
    except Exception as e:
        print(f"❌ JWT verification failed: {str(e)}")
        return JSONResponse(
            status_code=401,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "http://127.0.0.1:3000"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )

    supabase_user_id = payload.get("sub")
    if not supabase_user_id:
        print("❌ Invalid token payload: No 'sub' claim")
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid token payload"},
            headers={
                "Access-Control-Allow-Origin": request.headers.get("Origin", "http://127.0.0.1:3000"),
                "Access-Control-Allow-Methods": "GET, POST, PUT, PATCH, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )

    request.state.supabase_user = payload
    
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = request.headers.get("Origin", "http://127.0.0.1:3000")
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, PATCH, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

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

#---------Health Check Call---------------

@app.get("/")
async def root():
    return {"message": "status ok"}

#--------------Story Management API Routes------------------


# @app.websocket("/ws/next_chapter/{user_id}/{story_id}")
# async def websocket_next_chapter(websocket: WebSocket, user_id: str, story_id: str, story_type: str = Query(...)):
#     try:
#         #print(f"🔵 Calling api_backend.handle_story_websocket...")
#         token = (
#             websocket.headers.get("Authorization", "").replace("Bearer ", "")
#             or websocket.query_params.get("token")
#         )
#         #print(f"Received token: {token[:10]}...")
#         if not token:
#             await websocket.close(code=4001)
#             print("❌ Missing Supabase token in WebSocket connection")
#             return
#         payload = await verify_supabase_jwt(token)
#         supabase_user_id = payload.get("sub")
#         #print(f"User ID from URL: {user_id}, Supabase user ID: {supabase_user_id}")
#         if not supabase_user_id or supabase_user_id != user_id:
#             await websocket.close(code=4003)
#             print(f"❌ WebSocket auth failed: user_id mismatch ({user_id} != {supabase_user_id})")
#             return
#         await mainsetup.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id, story_type=story_type)
#         #print(f"🟢 api_backend.handle_story_websocket completed successfully")
#     #except WebSocketDisconnect:
#         #print(f"⚪ WebSocket disconnected: {user_id}")
#     except Exception as e:
#         #print(f"🔴 Error in websocket_next_chapter: {e}")
#         import traceback
#         traceback.print_exc()
#         try:
#             await websocket.close(code=1011)
#         except Exception:
#             pass


#-------------------------Start Session Methods-------------------------------
@app.post("/stories/initialize_story")
async def api_initialize_story(
    user_id: str = Query(...),
):
    return await mainsetup.initialize_story(user_id=user_id)

class ContinueStoryRequest(BaseModel):
    user_id: str
    story_id: str

@app.post("/stories/{story_id}")
async def api_continue_story(request: ContinueStoryRequest):
    print(request.story_id)
    return await mainsetup.continue_story(user_id=request.user_id, story_id=request.story_id)


class UserStoryRequest(BaseModel):
    user_id: str
    story_id: str

class UserStoryLastPhaseRequest(BaseModel):
    user_id: str
    story_id: str
    last_phase: int

class DocumentRequest(BaseModel):
    user_id: str
    story_id: str
    document_dict: dict

class ListDocumentRequest(BaseModel):
    user_id: str
    story_id: str
    document_list_dict: List[dict]

class StringRequest(BaseModel):
    user_id: str
    story_id: str
    document_str: str
#---------------------------------------------Story Author Calls------------------------------------------------------------

#---------------------------CREATION: Common Story Phase Methods----------------------
@app.post("/creation/add_story_seed")
async def api_add_story_seed(body: DocumentRequest):
    print("Story Seed: ", body.document_dict)
    return await mainsetup.add_story_seed(
        user_id=body.user_id,
        story_id=body.story_id,
        story_seed=body.document_dict
    )

@app.post("/creation/generate_world_foundation") # Phase 2
async def api_generate_world_foundation(request: UserStoryRequest):
    return await mainsetup.generate_world_foundation(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/generate_narrative_agents") # Phase 4
async def api_generate_narrative_agents(request: UserStoryRequest):
    return await mainsetup.generate_narrative_agents(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/connect_agents_to_plot") # Phase 6
async def api_connect_agents_to_plot(request: UserStoryRequest):
    return await mainsetup.connect_agents_to_plot(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/connect_world_to_conflict") # Phase 7
async def api_connect_world_to_conflict(request: UserStoryRequest):
    return await mainsetup.connect_world_to_conflict(user_id=request.user_id, story_id=request.story_id)

#-----Variable Phase Common Method------


@app.post("/creation/create_story_tracker") # Multi Phase
async def api_create_story_tracker(request: UserStoryLastPhaseRequest):
    return await mainsetup.create_story_tracker(user_id=request.user_id, story_id=request.story_id, last_phase=request.last_phase)


#-----------------------------Compact Flow Phase Methods-----------------------------
@app.post("/creation/generate_compact_plot") # Phase 3
async def api_generate_compact_plot(request: UserStoryRequest):
    return await mainsetup.generate_compact_plot(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/generate_simplified_conflict") # Phase 5
async def api_generate_simplified_conflict(request: UserStoryRequest):
    return await mainsetup.generate_simplified_conflict(user_id=request.user_id, story_id=request.story_id)


#-----------------------------Standard/Epic Flow Phase Methods-------------------------------
@app.post("/creation/generate_minimal_plot_outline") # Phase 3
async def api_generate_minimal_plot_outline(request: UserStoryRequest):
    return await mainsetup.generate_minimal_plot_outline(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/generate_conflict_layers") # Phase 5
async def api_generate_conflict_layers(request: UserStoryRequest):
    return await mainsetup.generate_conflict_layers(user_id=request.user_id, story_id=request.story_id)

#--------Variable Phase Common Method----------
@app.post("/creation/validate_story_elements") # Multi Phase 
async def api_validate_story_elements(request: UserStoryLastPhaseRequest):
    return await mainsetup.validate_story_elements(user_id=request.user_id, story_id=request.story_id, last_phase=request.last_phase)


#------------------------Standard Only Phase Methods--------------------------
@app.post("/creation/expand_plot_outline") # Phase 8
async def api_expand_plot_outline(request: UserStoryRequest):
    return await mainsetup.expand_plot_outline(user_id=request.user_id, story_id=request.story_id)


#------------------------Epic Only Phase Methods--------------------------
@app.post("/creation/generate_character_backstories") # Phase 8
async def api_generate_character_backstories(request: UserStoryRequest):
    return await mainsetup.generate_character_backstories(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/expand_world_detail") # Phase 9
async def api_expand_world_detail(request: UserStoryRequest):
    return await mainsetup.expand_world_detail(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/generate_subplot_architecture") # Phase 10
async def api_generate_subplot_architecture(request: UserStoryRequest):
    return await mainsetup.generate_subplot_architecture(user_id=request.user_id, story_id=request.story_id)

@app.post("/creation/expand_plot_with_enhancements") # Phase 11
async def api_expand_plot_with_enhancements(request: UserStoryRequest):
    return await mainsetup.expand_plot_with_enhancements(user_id=request.user_id, story_id=request.story_id)



#---------------------------CHANGE: Common Story Phase Methods----------------------
@app.post("/change/add_story_seed") # Phase 1
async def api_change_add_story_seed(request: DocumentRequest):
    print(request.document_dict)
    return await mainsetup.change_add_story_seed(
        user_id=request.user_id,
        story_id=request.story_id,
        story_seed=request.document_dict
    )

@app.post("/change/generate_world_foundation") # Phase 2
async def api_change_generate_world_foundation(request: DocumentRequest):
    return await mainsetup.change_generate_world_foundation(user_id=request.user_id, story_id=request.story_id, world_foundation=request.document_dict)

@app.post("/change/generate_narrative_agents") # Phase 4
async def api_change_generate_narrative_agents(request: ListDocumentRequest):
    return await mainsetup.change_generate_narrative_agents(user_id=request.user_id, story_id=request.story_id, narrative_agents=request.document_list_dict)

@app.post("/change/connect_agents_to_plot") # Phase 6
async def api_change_connect_agents_to_plot(request: ListDocumentRequest):
    return await mainsetup.change_connect_agents_to_plot(user_id=request.user_id, story_id=request.story_id, connected_agents=request.document_list_dict)

@app.post("/change/connect_world_to_conflict") # Phase 7
async def api_change_connect_world_to_conflict(request: DocumentRequest):
    return await mainsetup.change_connect_world_to_conflict(user_id=request.user_id, story_id=request.story_id, integrated_world=request.document_dict)

#-----Variable Phase Common Method------
class ChangeStoryTrackerRequest(BaseModel):
    user_id: str
    story_id: str
    last_phase: int
    story_tracker: dict

@app.post("/change/create_story_tracker") # Multi Phase
async def api_change_create_story_tracker(request: ChangeStoryTrackerRequest):
    return await mainsetup.change_create_story_tracker(user_id=request.user_id, story_id=request.story_id, last_phase=request.last_phase, story_tracker=request.story_tracker)


#-----------------------------Compact Flow Phase Methods-----------------------------
@app.post("/change/generate_compact_plot") # Phase 3
async def api_change_generate_compact_plot(request: DocumentRequest):
    return await mainsetup.change_generate_compact_plot(user_id=request.user_id, story_id=request.story_id, compact_plot=request.document_dict)

@app.post("/change/generate_simplified_conflict") # Phase 5
async def api_change_generate_simplified_conflict(request: DocumentRequest):
    return await mainsetup.change_generate_simplified_conflict(user_id=request.user_id, story_id=request.story_id, conflict_matrix=request.document_dict)


#-----------------------------Standard/Epic Flow Phase Methods-------------------------------
@app.post("/change/generate_minimal_plot_outline") # Phase 3
async def api_change_generate_minimal_plot_outline(request: DocumentRequest):
    return await mainsetup.change_generate_minimal_plot_outline(user_id=request.user_id, story_id=request.story_id, minimal_plot=request.document_dict)

@app.post("/change/generate_conflict_layers") # Phase 5
async def api_change_generate_conflict_layers(request: DocumentRequest):
    return await mainsetup.change_generate_conflict_layers(user_id=request.user_id, story_id=request.story_id, conflict_matrix=request.document_dict)

#--------Variable Phase Common Method----------
# @app.post("/change/validate_story_elements") # Multi Phase 
# async def api_change_validate_story_elements(request: UserStoryLastPhaseRequest):
#     return await mainsetup.change_validate_story_elements(user_id=request.user_id, story_id=request.story_id, last_phase=request.last_phase)


#------------------------Standard Only Phase Methods--------------------------
@app.post("/change/expand_plot_outline") # Phase 8
async def api_change_expand_plot_outline(request: DocumentRequest):
    return await mainsetup.change_expand_plot_outline(user_id=request.user_id, story_id=request.story_id, expanded_plot=request.document_dict)


#------------------------Epic Only Phase Methods--------------------------
@app.post("/change/generate_character_backstories") # Phase 8
async def api_change_generate_character_backstories(request: ListDocumentRequest):
    return await mainsetup.change_generate_character_backstories(user_id=request.user_id, story_id=request.story_id, backstories=request.document_list_dict)

@app.post("/change/expand_world_detail") # Phase 9
async def api_expand_world_detail(request: DocumentRequest):
    return await mainsetup.change_expand_world_detail(user_id=request.user_id, story_id=request.story_id, world_guide=request.document_dict)

@app.post("/change/generate_subplot_architecture") # Phase 10
async def api_change_generate_subplot_architecture(request: DocumentRequest):
    return await mainsetup.change_generate_subplot_architecture(user_id=request.user_id, story_id=request.story_id, subplot_arch=request.document_dict)

@app.post("/change/expand_plot_with_enhancements") # Phase 11
async def api_change_expand_plot_with_enhancements(request: DocumentRequest):
    return await mainsetup.change_expand_plot_with_enhancements(user_id=request.user_id, story_id=request.story_id, expanded_plot=request.document_dict)





#---------------------------------------------Basic Author Calls------------------------------------------------------------
#---------------------------CREATION: Story Phase Methods----------------------
class OneSentenceGenerationRequest(BaseModel):
    user_id: str
    story_id: str
    user_context: str
    target_medium: str

@app.post("/creation/one_sentence_generation")
async def api_one_sentence_generation(request: OneSentenceGenerationRequest):
    return await mainsetup.one_sentence_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        user_context=request.user_context,
        target_medium=request.target_medium
        )

class OneParagraphGenerationRequest(BaseModel):
    user_id: str
    story_id: str
    tone: str

@app.post("/creation/one_paragraph_generation")
async def api_one_paragraph_generation(request: OneParagraphGenerationRequest):
    return await mainsetup.one_paragraph_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        tone=request.tone
        )

class OnePageGenerationRequest(BaseModel):
    user_id: str
    story_id: str
    target_audience_age: int
    target_length: int

@app.post("/creation/one_page_generation")
async def api_one_page_generation(request: OnePageGenerationRequest):
    return await mainsetup.one_page_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        target_audience_age=request.target_audience_age,
        target_length=request.target_length
        )

@app.post("/creation/protagonist_generation")
async def api_protagonist_generation(request: UserStoryRequest):
    return await mainsetup.protagonist_generation(user_id=request.user_id, story_id=request.story_id)

#---------------------------CHANGE: Story Phase Methods----------------------
@app.post("/change/one_sentence_generation")
async def api_change_one_sentence_generation(request: DocumentRequest):
    return await mainsetup.change_one_sentence_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        one_sentence_form=request.document_dict
        )

@app.post("/change/one_paragraph_generation")
async def api_change_one_paragraph_generation(request: DocumentRequest):
    return await mainsetup.change_one_paragraph_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        one_paragraph_form=request.document_dict
        )

@app.post("/change/one_page_generation")
async def api_change_one_page_generation(request: DocumentRequest):
    return await mainsetup.change_one_page_generation(
        user_id=request.user_id,
        story_id=request.story_id,
        one_page_form=request.document_dict
        )

@app.post("/change/protagonist_generation")
async def api_change_protagonist_generation(request: DocumentRequest):
    return await mainsetup.change_protagonist_generation(
        user_id=request.user_id, 
        story_id=request.story_id, 
        protagonist_form=request.document_dict
        )







@app.post("/stories/progress/{user_id}/{story_id}/update")
async def api_update_story_progress(request: DocumentRequest):
    return await mainsetup.update_story_progress_for_user(user_id=request.user_id, story_id=request.story_id, metadata=request.document_dict)

@app.get("/stories/progress/{user_id}/{story_id}/update-request")
async def api_update_story_progress(user_id: str, story_id: str, story_context: str):
    return await mainsetup.update_request_story_progress_for_user(user_id=user_id, story_id=story_id, story_context=story_context)

@app.get("/stories/progress/{user_id}/{story_id}")
async def api_get_story_progress(user_id: str, story_id: str):
    return await mainsetup.get_story_progress_for_user(user_id=user_id, story_id=story_id.strip())

@app.get("/stories/director_notes/{user_id}/{story_id}/{type}")
async def api_get_director_notes(user_id: str, story_id: str, type: str):
    return await mainsetup.get_director_notes(user_id=user_id, story_id=story_id.strip(), type=type)

# @app.patch("/stories/logout/{user_id}/{story_id}")
# async def api_logout_story(user_id: str, story_id: str, story_type: str):
#     return await mainsetup.logout_story(user_id=user_id, story_id=story_id, story_type=story_type)


class ImageRequest(BaseModel):
    story_type: str

@app.post("/stories/image-gen/{user_id}/{story_id}")
async def api_gen_story_image_cover(user_id: str, story_id: str, request_body: ImageRequest):
    return await mainsetup.get_book_cover_image(user_id=user_id, story_id=story_id, story_type=request_body.story_type)

#-----------------User Session Management Routes--------------------------
@app.patch("/users/{user_id}/session")
async def api_logout(user_id: str):
    return await mainsetup.logout(user_id)

#-------------------User Data Management Routes---------------------------
class UserCreate(BaseModel):
    nickname: str
    age: int
    tier: int = 1
    stories: list = []
    user_id: str | None = None

@app.post("/users")
async def api_add_user(user: UserCreate, request: Request):
    print(f"📥 Received POST /users request")
    print(f"Request headers: {request.headers}")
    print(f"Request body: {user}")
    # Verify user_id matches authenticated user
    supabase_user_id = request.state.supabase_user.get("sub")
    if user.user_id and user.user_id != supabase_user_id:
        raise HTTPException(status_code=403, detail="User ID does not match authenticated user")
    print(f"Adding user: {user}")
    result = await add_user(
        nickname=user.nickname,
        age=user.age,
        tier=user.tier,
        user_id=supabase_user_id,
        stories=user.stories
    )
    print(f"User added: {result}")
    return result

class DeleteStoryRequest(BaseModel):
    story_title: str
    story_type: str

@app.patch("/users/{user_id}/stories/{story_id}/delete")
async def api_delete_story(
    user_id: str,
    story_id: str,
    data: DeleteStoryRequest = Body(...),
):
    try:
        await mainsetup.logout_story(user_id=user_id, story_id=story_id, story_type=data.story_type)
    except:
        pass
    return await delete_story(user_id=user_id, story_title=data.story_title, story_id=story_id)

@app.get("/users/{user_id}/profile/data")
async def api_get_user_profile_data(user_id: str):
    return await get_user_profile(user_id=user_id)

@app.get("/users/{user_id}/profile")
async def api_get_user_profile_data_and_stories(user_id: str):
    return await get_user_profile_with_stories(user_id=user_id)

@app.post("/users/{user_id}/profile/setting")
async def api_update_user_settings(user_id: str, user_data: Dict[str, Any]):
    return await update_user_settings(user_id=user_id, user_data=user_data)




#gets all active users in redis pool (for app manager use)
@app.get("/users/active")
async def api_get_active_users(_: dict = Depends(require_admin)):
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

#---------------Feedback------------------
@app.post("/feedback")
async def api_post_feedback(
    form: Dict[str, Any],
    request: Request
):
    try:
        # ✅ Get authenticated user from middleware
        supabase_user = request.state.supabase_user
        if not supabase_user:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        user_id = supabase_user.get("sub")
        
        # ✅ Call the feedback function (add user_id if needed)
        result = await post_user_feedback(form, user_id=user_id)
        
        if result["status"] == "success":
            return {"message": "Feedback submitted successfully!", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail=result["message"])
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Feedback error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")