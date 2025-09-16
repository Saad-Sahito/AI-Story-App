import uuid
import asyncio
import time
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from llm_client.llm_client import LLMClient
from memory.memory_system import StoryMemorySystem
from agents.story_author import StoryAuthor
from agents.director_agent import DirectorGraph
from agents.scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph

# --- Globals ---
llm_client = LLMClient()
# key: user_id, value: {story_id: {...story session data...}, last_active: timestamp}
SESSIONS = {}
app = FastAPI()


# ---------------- Session Management ----------------
def setup_user_SESSION(user_id: str, story_id: str, memory_system=None, story_author=None):
    sceneplanner = ScenePlannerGraph(llm_client=llm_client)
    director = DirectorGraph(llm_client=llm_client, memory_system=memory_system, sceneplanner=sceneplanner)

    if user_id not in SESSIONS:
        SESSIONS[user_id] = {"last_active": time.time()}

    SESSIONS[user_id][story_id] = {
        "memory_system": memory_system,
        "story_author": story_author,
        "sceneplanner": sceneplanner,
        "director": director,
        "user_input_future": None,
    }
    SESSIONS[user_id]["last_active"] = time.time()


# ---------------- Story Flow ----------------
def initialize_story(user_id: str, story_title: str = ""):
    story_id = str(uuid.uuid4())
    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
    memory_system.qdrant_initialize()
    story_author = StoryAuthor(llm_client=llm_client, memory_system=memory_system)
    
    setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system, story_author=story_author)
    return {"message": f"Story initialized for {user_id}", "story_id": story_id}


def continue_story(user_id: str, story_id: str) -> dict:
    if not user_id:
        raise ValueError("Please enter a valid User ID.")

    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
    story_author = StoryAuthor(llm_client=llm_client, memory_system=memory_system)
    memory_system.qdrant_initialize()
    story_progress_data = memory_system.get_story_progress()
    if not story_progress_data:
        raise ValueError("No existing story found for this user and story ID.")

    setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system, story_author=story_author)
    story_text = (
        memory_system.get_long_term_story(chapter_id=story_progress_data.get("latest_chapter_id"))
        or "No story text for this chapter found."
    )
    return {"message": f"Session started for {user_id} and {story_id}", "story_text": story_text}


# ---------------- API Routes ----------------
@app.post("/premise")
async def api_create_premise(initial_story_data: dict):
    user_id = initial_story_data["user_id"]
    story_id = initial_story_data["story_id"]

    user_data = SESSIONS.get(user_id)
    if not user_data:
        return {"error": "Invalid user ID"}

    story_data = user_data.get(story_id)
    if not story_data:
        return {"error": "Invalid story ID"}

    filtered_data = {k: v for k, v in initial_story_data.items() if k not in ["story_id", "user_id"]}

    form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])

    premise = await asyncio.to_thread(
        story_data["story_author"].set_story_premise,
        form_string,
        initial_story_data.get("title", ""),
    )
    story_data["memory_system"].update_story_progress(
        metadata={
            "latest_chapter_id": 1,
            "continue_scene_id": 1,
            "story_title": initial_story_data["title"],
            "word_count": 0,
        }
    )
    return {"premise": premise}


async def handle_user_choice(user_id: str, story_id: str, choice: str):
    user_data = SESSIONS.get(user_id)
    story_data = user_data.get(story_id) if user_data else None
    if not story_data or not choice.strip():
        return {"error": "Invalid session or empty choice."}

    story_data["sceneplanner"].receive_user_input(choice)
    return {"message": f"Choice '{choice}' received"}


@app.post("/next_chapter")
async def api_next_chapter(user_id: str, story_id: str):
    user_data = SESSIONS.get(user_id)
    if not user_data:
        return {"error": "Invalid user ID"}

    story_data = user_data.get(story_id)
    if not story_data:
        return {"error": "Invalid story ID"}

    queue = asyncio.Queue()

    def scene_chunk_callback(chunk: str):
        queue.put_nowait(chunk)

    async def run_director():
        await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
        queue.put_nowait(json.dumps({"chapter_complete": True}) + "\n")
        queue.put_nowait(None)

    async def event_stream():
        asyncio.create_task(run_director())
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/initialize_story")
def api_initialize(user_id: str, story_title: str = ""):
    return initialize_story(user_id=user_id, story_title=story_title)


@app.post("/continue_story")
def api_continue_story(user_id: str, story_id: str = ""):
    return continue_story(user_id=user_id, story_id=story_id)


@app.post("/choice")
async def api_choice(user_id: str, story_id: str, choice: str):
    user_data = SESSIONS.get(user_id)
    if not user_data:
        return {"error": "Invalid user ID"}

    story_data = user_data.get(story_id)
    if not story_data or not choice.strip():
        return {"error": "Invalid story ID or empty choice."}

    scene_planner = story_data["sceneplanner"]
    future = scene_planner._user_input_future
    if future and not future.done():
        future.set_result(choice)

    return {"message": f"Choice '{choice}' received"}


@app.post("/logout")
def api_logout(user_id: str):
    if user_id in SESSIONS:
        del SESSIONS[user_id]
        return {"message": "Session cleared."}
    return {"message": "No active session."}
