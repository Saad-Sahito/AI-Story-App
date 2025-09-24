# main.py or startup.py - Initialize shared instances on startup

import asyncio
from fastapi import FastAPI, WebSocket
import traceback
from contextlib import asynccontextmanager
from connection.api_backend import SESSIONS, APIBackend
from src.memory.user_management import add_user, append_story, delete_story, get_user_profile_with_stories
import src.agents.scene_creation_subgraph.shared_scene_planner as scene_planner_module
from src.memory.shared_resources import SHARED_QDRANT



# Background cleanup task
cleanup_task = None

async def cleanup_sessions_periodically():
    """Background task to clean up inactive sessions every hour"""
    while True:
        try:
            await asyncio.sleep(3600)  # 1 hour
            cleaned_count = APIBackend.cleanup_inactive_sessions(max_age_seconds=3600)
            if cleaned_count > 0:
                print(f"✅ Cleaned up {cleaned_count} inactive sessions")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"❌ Error in session cleanup: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting up AI Story App...")
    api_backend = APIBackend()
    print("✅ Shared instances initialized")
    
    # Verify shared clients
    if scene_planner_module.SHARED_SCENE_PLANNER_SERVICE is None:
        print("❌ ERROR: SHARED_SCENE_PLANNER_SERVICE was not initialized!")
    else:
        print("✅ SHARED_SCENE_PLANNER_SERVICE is properly initialized")
    # if SHARED_SUPABASE is None:
    #     print("❌ ERROR: SHARED_SUPABASE was not initialized!")
    # else:
    #     print("✅ SHARED_SUPABASE is properly initialized")
    if SHARED_QDRANT is None:
        print("❌ ERROR: SHARED_QDRANT was not initialized!")
    else:
        print("✅ SHARED_QDRANT is properly initialized")
    
    cleanup_task = asyncio.create_task(cleanup_sessions_periodically())
    print("✅ Background cleanup task started")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down AI Story App...")
    if cleanup_task:
        cleanup_task.cancel()
        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass
    
    # Clean up shared clients
    # if SHARED_SUPABASE:
    #     try:
    #         SHARED_SUPABASE.close()
    #     except Exception:
    #         pass
    if SHARED_QDRANT:
        try:
            SHARED_QDRANT.close()
        except Exception:
            pass
    
    # Clean up sessions
    user_ids = list(SESSIONS.keys())
    for user_id in user_ids:
        APIBackend.logout(user_id)
    
    print("✅ All sessions and shared clients cleaned up")
    print("✅ Shutdown complete")

app = FastAPI(
    title="AI Interactive Story App",
    description="Optimized for memory efficiency with shared instances",
    lifespan=lifespan
)

try:
    api_backend = APIBackend()
    print(">>> APIBackend initialized successfully")
except Exception as e:
    print(">>> ERROR during APIBackend init:", e)
    traceback.print_exc()




# This is the health check endpoint
@app.get("/")
async def root():
    return {"message": "status ok"}

# ---------------- Story Management API Routes ----------------
# Create Premise
@app.post("/premise")
async def api_create_premise(initial_story_data: dict):
    return await api_backend.create_premise(initial_story_data=initial_story_data)

@app.websocket("/ws/next_chapter")
async def websocket_next_chapter(websocket: WebSocket):
    # Just delegate everything to the handler
    await api_backend.handle_story_websocket(websocket)

# initialize story
@app.post("/stories/initialize_story")
def api_initialize_story(user_id: str, story_title: str = ""):
    return api_backend.initialize_story(user_id=user_id, story_title=story_title)

# Continue story
@app.put("/stories/{story_id}")
def api_continue_story(user_id: str, story_id: str):
    return api_backend.continue_story(user_id=user_id, story_id=story_id)

# Get story progress
# @app.get("/stories/{story_id}/progress")
# def api_get_story_progress(user_id: str, story_id: str):
#     return get_progress(user_id=user_id, story_id=story_id)



# ---------------- User Session Management Routes ----------------
# Use DELETE to end a user's session (logout)
@app.delete("/users/{user_id}/session")
def api_logout(user_id: str):
    return api_backend.logout(user_id)


# ---------------- User Data Management Routes ----------------
# Add User
@app.post("/users")
def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
    return add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)

# Append Story to user data
@app.put("/users/{user_id}/stories/{story_title}")
def api_append_story(user_id: str, story_title: str):
    return append_story(user_id, story_title)

# Use DELETE to remove a story associated with a user
@app.delete("/users/{user_id}/stories/{story_title}")
def api_delete_story(user_id: str, story_title: str):
    return delete_story(user_id, story_title)

# Use GET to retrieve a list of a user's stories
# @app.get("/users/{user_id}/stories")
# def api_get_user_stories(user_id: str):
#     return get_user_stories(user_id)

# Get user profile data except for stories
# @app.get("/users/{user_id}/profile")
# def api_get_user_profile_data(user_id: str):
#     return get_user_profile_data(user_id)


# Get user profile data along with stories
@app.get("/users/{user_id}/profile")
def api_get_user_profile_data_and_stories(user_id: str):
    return get_user_profile_with_stories(user_id=user_id)





