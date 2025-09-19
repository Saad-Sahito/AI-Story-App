
from fastapi import FastAPI, WebSocket
from connection.main_test import MainTest
from src.memory.user_management import add_user, append_story, delete_story, get_user_stories, get_progress
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# --- Add this block ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
main_test = MainTest()

# ---------------- Story Management API Routes ----------------
@app.post("/premise")
async def api_create_premise(initial_story_data: dict):
    return await main_test.create_premise(initial_story_data=initial_story_data)

@app.websocket("/ws/next_chapter")
async def websocket_next_chapter(websocket: WebSocket):
    # Just delegate everything to the handler
    await main_test.handle_story_websocket(websocket)

@app.post("/initialize_story")
def api_initialize(user_id: str, story_title: str = ""):
    return main_test.initialize_story(user_id=user_id, story_title=story_title)

@app.post("/continue_story")
def api_continue_story(user_id: str, story_id: str = ""):
    return main_test.continue_story(user_id=user_id, story_id=story_id)

@app.post("/get_story_progress")
def api_get_story_progress(user_id: str, story_id: str):
    return get_progress(user_id=user_id, story_id=story_id)

# ---------------- User Session Management Routes ----------------
@app.post("/logout")
def api_logout(user_id: str):
    return main_test.logout(user_id)


# ---------------- User Data Management Routes ----------------
@app.post("/add_user")
def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
    return add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)


@app.post("/append_story")
def api_append_story(user_id: str, story_title: str):
    return append_story(user_id, story_title)


@app.post("/delete_story")
def api_delete_story(user_id: str, story_title: str):
    return delete_story(user_id, story_title)

@app.post("/get_user_stories")
def api_get_user_stories(user_id: str):
    return get_user_stories(user_id)