
from fastapi import FastAPI, WebSocket
from connection.main_test import MainTest
from src.memory.user_management import add_user, append_story, delete_story, get_user_stories, get_progress, get_user_profile_data, get_user_profile_with_stories
#from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# --- Add this block (for local use only) ---
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["http://localhost:8080"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
main_test = MainTest()


# This is the health check endpoint
@app.get("/")
async def root():
    return {"status": "ok"}




# ---------------- Story Management API Routes ----------------
@app.post("/premise")
async def api_create_premise(initial_story_data: dict):
    return await main_test.create_premise(initial_story_data=initial_story_data)

@app.websocket("/ws/next_chapter")
async def websocket_next_chapter(websocket: WebSocket):
    # Just delegate everything to the handler
    await main_test.handle_story_websocket(websocket)

# Use POST to initialize/create a new story
@app.post("/stories")
def api_initialize_story(user_id: str, story_title: str = ""):
    return main_test.initialize_story(user_id=user_id, story_title=story_title)

# Use PUT to update an existing story
@app.put("/stories/{story_id}")
def api_continue_story(user_id: str, story_id: str):
    return main_test.continue_story(user_id=user_id, story_id=story_id)

# Use GET to retrieve the progress of a specific story
@app.get("/stories/{story_id}/progress")
def api_get_story_progress(user_id: str, story_id: str):
    return get_progress(user_id=user_id, story_id=story_id)



# ---------------- User Session Management Routes ----------------
# Use DELETE to end a user's session (logout)
@app.delete("/users/{user_id}/session")
def api_logout(user_id: str):
    return main_test.logout(user_id)


# ---------------- User Data Management Routes ----------------
# Use POST to create a new user
@app.post("/users")
def api_add_user(nickname: str, user_tag: str, age: int, stories: list = [], user_id: str = None):
    return add_user(nickname=nickname, user_tag=user_tag, age=age, user_id=user_id, stories=stories)

# Use PUT to update a story associated with a user
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