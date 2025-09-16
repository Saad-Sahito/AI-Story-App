# src/main.py
import asyncio
import uuid
import time
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager

# Import your helper functions / classes
from llm_client.llm_client import LLMClient
from memory.memory_system import StoryMemorySystem
from agents.story_author import StoryAuthor
from agents.director_agent import DirectorGraph
from agents.scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph

# auth_supabase.py
import os
import jwt
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# --- Config ---
SESSION_TTL = 1800  # 30 min in seconds

# --- Global shared objects ---
llm_client = LLMClient()
SESSIONS = {}  # your global session dict

# --- Auth / Supabase JWT ---
security = HTTPBearer()

# Get this from your Supabase project settings (API → JWT secret)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
ALGORITHM = "HS256"  # Supabase default

def verify_supabase_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> str:
    """
    Verifies a Supabase JWT and extracts the user_id (sub).
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SUPABASE_JWT_SECRET, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: no subject",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return user_id
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Supabase token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# --- FastAPI app ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    async def cleanup_loop():
        try:
            while True:
                cleanup_sessions()
                await asyncio.sleep(600)  # every 10 minutes
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(cleanup_loop())
    print("Cleanup loop started")

    yield  # <-- App runs here

    task.cancel()
    print("Cleanup loop stopped")

app = FastAPI(lifespan=lifespan)

# ---------------- Session Management ----------------
def touch_session(user_id: str):
    """Update last_active timestamp on each request."""
    if user_id in SESSIONS:
        SESSIONS[user_id]["last_active"] = time.time()

def cleanup_sessions():
    import time
    now = time.time()
    expired = []
    for user_id, session in list(SESSIONS.items()):
        if now - session["last_access"] > SESSION_TTL:
            expired.append(user_id)
    for user_id in expired:
        del SESSIONS[user_id]
        print(f"Session {user_id} expired and removed")

# ---------------- User Setup Handlers ----------------
# Common function to set session data
def setup_user_SESSION(user_id: str, story_id: str, memory_system: StoryMemorySystem = None, story_author: StoryAuthor = None):
    # Create per-user instances
    sceneplanner = ScenePlannerGraph(llm_client=llm_client)
    director = DirectorGraph(llm_client=llm_client, memory_system=memory_system, sceneplanner=sceneplanner)

    # Store in SESSIONS
    SESSIONS[user_id] = {
        "story_id": story_id,
        "memory_system": memory_system,
        "story_author": story_author,
        "sceneplanner": sceneplanner,
        "director": director,
        "waiting_for_user_choice": False,
        "last_active": time.time(),
    }

# When user continues a story, we may want to load their last progress
def continue_story(user_id: str, story_id: str) -> dict:
    """Start a new session for a user."""
    if not user_id:
        raise ValueError("Please enter a valid User ID.")
    
    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)

    setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system)

    return {"message": f"Session started for {user_id}", "story_id": story_id}
    
# When user starts a new story we initialize everything
def initialize_story(user_id: str, story_title: str = "") -> dict:
    """Initialize story session for a user."""
    if not user_id:
        raise ValueError("Please enter a valid User ID.")

    story_id = str(uuid.uuid4())

    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
    story_author = StoryAuthor(llm_client=llm_client, memory_system=memory_system)
    memory_system.update_story_progress(metadata={"latest_chapter_id": 1, "continue_scene_id": 1, "story_title": story_title, "word_count": 0})

    setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system, story_author=story_author)

    return {"message": f"Story initialized for {user_id}", "story_id": story_id}

# ------------- Story Flow Handlers -------------
# If new story is started, this is called after initialize_story
async def create_premise(user_id: str, initial_story_data: dict) -> dict:
    touch_session(user_id)
    data = SESSIONS.get(user_id)
    if not data:
        return {"error": "Please initialize your story first!"}
    
    title = initial_story_data.get("title", "")
    setting = initial_story_data.get("setting", "")
    pov = initial_story_data.get("pov", "")
    length = initial_story_data.get("length", "")
    guide_prose = initial_story_data.get("guide_prose", "")
    additional_themes = initial_story_data.get("additional_themes", "")
    genre = initial_story_data.get("genre", "")
    tone = initial_story_data.get("tone", "")

    form_string = f"""
    Title: {title}
    Setting: {setting}
    POV: {pov}
    Length: {length}
    Guide Prose: {guide_prose}
    Additional Themes: {additional_themes}
    Genre: {genre}
    Tone: {tone}
    """

    premise = data["story_author"].set_story_premise(form_string, title)
    if premise == "An error occured, Please try again.":
        return {"error": premise}

    data["state"]["premise"] = premise
    #synopsis = create_synopsis(user_id)
    return {"premise": premise }   #"synopsis": synopsis


# def create_synopsis(user_id: str) -> str:
#     data = SESSIONS.get(user_id)
#     synopsis = data["story_author"].set_story_synopsis(
#         data["state"]["premise"], data["state"]["story_title"]
#     )
#     data["state"]["synopsis"] = synopsis
#     return synopsis



async def handle_scene_chunk(user_id: str, chunk: str):
    data = SESSIONS.get(user_id)
    if chunk.strip().startswith("<DECISION_POINT>"):
        data["waiting_for_user_choice"] = True
        return {"decision_point": chunk.strip()}

    return {"scene_chunk": chunk}


async def handle_user_choice(user_id: str, choice: str):
    touch_session(user_id)
    data = SESSIONS.get(user_id)
    if not choice.strip():
        return {"error": "Empty choice."}

    data["waiting_for_user_choice"] = False
    data["sceneplanner"].receive_user_input(choice)
    return {"message": "Choice received."}


async def generate_next_chapter(user_id: str):
    touch_session(user_id)
    data = SESSIONS.get(user_id)
    if not data:
        return {"error": "Initialize story first"}
    output_chunks = []

    async def director_runner():
        state = await data["director"].run(
            #initial_state,
            scene_chunk_callback=lambda chunk: output_chunks.append(
                asyncio.run(handle_scene_chunk(user_id, chunk))
            ),
        )
        #data["state"]["chapter_id"] = state.get("current_chapter_id")
        output_chunks.append(
            {"chapter_complete": True, "word_count": state.word_count}
        )

    await director_runner()
    return output_chunks


# ---------------- API Routes ----------------
@app.post("/initialize")
def api_initialize(user_id: str = Depends(verify_supabase_token), story_title: str = "", story_id: str = None, background_tasks: BackgroundTasks = None):
    background_tasks.add_task(cleanup_sessions)  # cleanup on each call
    return initialize_story(user_id, story_title, story_id)


@app.post("/premise")
async def api_premise(user_id: str = Depends(verify_supabase_token), initial_story_data: dict = None, background_tasks: BackgroundTasks = None):
    background_tasks.add_task(cleanup_sessions)
    return await create_premise(user_id=user_id, initial_story_data=initial_story_data)


@app.post("/next_chapter")
async def api_next_chapter(user_id: str = Depends(verify_supabase_token), background_tasks: BackgroundTasks = None):
    background_tasks.add_task(cleanup_sessions)
    return await generate_next_chapter(user_id)


@app.post("/choice")
async def api_choice(user_id: str = Depends(verify_supabase_token), choice: str = None, background_tasks: BackgroundTasks = None):
    background_tasks.add_task(cleanup_sessions)
    return await handle_user_choice(user_id, choice)


@app.post("/logout")
def api_logout(user_id: str = Depends(verify_supabase_token)):
    """Explicitly clear a session when user logs out."""
    if user_id in SESSIONS:
        del SESSIONS[user_id]
        return {"message": f"Session for {user_id} deleted."}
    return {"message": "No active session for this user."}

@app.post("/touch")
def api_touch(user_id: str = Depends(verify_supabase_token)):
    """Touch the session to keep it alive."""
    touch_session(user_id)
    return {"message": "Session touched."}






