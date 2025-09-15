# src/main.py
from nicegui import ui
import asyncio
import re
import uuid

# Import your helper functions / classes
from llm_client.llm_client import LLMClient
from memory.memory_system import StoryMemorySystem
from agents.story_author import StoryAuthor
from agents.director_agent import DirectorGraph
from agents.scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph

# --- Global Backend Objects (shared) ---
llm_client = LLMClient()  # this is safe to share

# --- Top Inputs for User Identification ---
ui.label("User & Story Setup").classes("text-lg font-bold mt-4 text-gray-700")
user_id_input = ui.input(label="User ID")
story_id_input = ui.input(label="Story ID (leave blank to auto-generate)")
start_btn = ui.button("Start / Initialize Story")

# --- UI Elements (rest of your app) ---
story_title_input = ui.input(label="Story Title")
setting_input = ui.input(label="Setting")
main_character_input = ui.input(label="Main Character")
genre_input = ui.input(label="Genre")
tone_input = ui.input(label="Tone")
submit_btn = ui.button("Create Story Synopsis")

ui.label('Story Synopsis').classes('text-lg font-bold mt-4 text-gray-700')
story_output = ui.markdown("")

ui.label('Chapter Output').classes('text-lg font-bold mt-4 text-gray-700')
scene_output = ui.markdown("")
decision_output = ui.markdown("")
choice_input = ui.textarea(label="Your Decision")
choice_btn = ui.button("Submit Decision", on_click=lambda: None)
next_chapter_btn = ui.button("Generate Next Chapter", on_click=lambda: None)

# --- Session-specific backend instances ---
session_data = {}

def initialize_story():
    """Called when user clicks Start / Initialize Story"""
    user_id = user_id_input.value.strip()
    if not user_id:
        story_output.content = "Please enter a valid User ID."
        story_output.update()
        return

    story_id = story_id_input.value.strip() or str(uuid.uuid4())

    # Create per-user instances
    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id, story_title=story_title_input.value)
    story_author = StoryAuthor(llm_client=llm_client, memory_system=memory_system)
    sceneplanner = ScenePlannerGraph(llm_client=llm_client)
    director = DirectorGraph(llm_client=llm_client, memory_system=memory_system, sceneplanner=sceneplanner)

    # Store in session_data dictionary keyed by user_id
    session_data[user_id] = {
        "story_id": story_id,
        "memory_system": memory_system,
        "story_author": story_author,
        "sceneplanner": sceneplanner,
        "director": director,
        "state": {
            "premise": None,
            "chapter_id": 1,
            "story_title": story_title_input.value,
            "current_scene": None,
            "synopsis": None,
            "waiting_for_user_choice": False,
            "word_count": 0
        }
    }

    story_output.content = f"Story initialized! User ID: {user_id}, Story ID: {story_id}"
    story_output.update()

start_btn.on_click(initialize_story)

# --- Handlers (modified to use session_data per user) ---
async def create_premise(user_id):
    data = session_data.get(user_id)
    if not data:
        story_output.content = "Please initialize your story first!"
        story_output.update()
        return

    form_string = f"""
    Title: {story_title_input.value}
    Setting: {setting_input.value}
    Main Character: {main_character_input.value}
    Genre: {genre_input.value}
    Tone: {tone_input.value}
    """
    premise = data["story_author"].set_story_premise(form_string, data["state"]["story_title"])
    if premise == "An error occured, Please try again.":
        story_output.content = premise
        story_output.update()
    else:
        data["state"]["premise"] = premise
        story_output.content = premise
        story_output.update()
        create_synopsis(user_id, )

def create_synopsis(user_id):
    data = session_data.get(user_id)
    synopsis = data["story_author"].set_story_synopsis(data["state"]["premise"], data["state"]["story_title"])
    data["state"]["synopsis"] = synopsis
    next_chapter_btn.visible = True

def count_words_regex(text: str) -> int:
    return len(re.findall(r'\b\w+\b', text))

async def handle_scene_chunk(user_id, chunk: str):
    data = session_data.get(user_id)
    if chunk.strip().startswith("<DECISION_POINT>"):
        data["state"]["waiting_for_user_choice"] = True
        decision_output.content = chunk.strip()
        decision_output.visible = True
        choice_input.visible = True
        choice_btn.visible = True
        return
    data["state"]["word_count"] += count_words_regex(chunk)
    scene_output.content += "\n\n" + chunk
    scene_output.update()

async def handle_user_choice(user_id):
    data = session_data.get(user_id)
    if not choice_input.value.strip():
        return
    choice = choice_input.value.strip()
    choice_input.value = ""
    decision_output.content = ""
    decision_output.visible = False
    choice_input.visible = False
    choice_btn.visible = False
    data["state"]["waiting_for_user_choice"] = False
    data["sceneplanner"].receive_user_input(choice)

async def generate_next_chapter(user_id):
    data = session_data.get(user_id)
    chapter_id = data["state"]["chapter_id"]
    story_title = data["state"]["story_title"]
    scene_output.content = ""
    decision_output.content = ""
    scene_output.update()
    decision_output.update()
    next_chapter_btn.visible = False

    initial_state = {
        "messages": [],
        "current_chapter_id": chapter_id,
        "story_title": story_title,
        "story_id": data["story_id"],
        "user_id": user_id,
    }

    async def director_runner():
        await data["director"].run(initial_state, scene_chunk_callback=lambda chunk: handle_scene_chunk(user_id, chunk))
        data["state"]["chapter_id"] += 1
        next_chapter_btn.visible = True
        scene_output.content += f"\n\n**Chapter Complete!**\nStory Word Count: {data['state']['word_count']}\n"
        scene_output.update()

    asyncio.create_task(director_runner())

# --- Bind UI actions ---
submit_btn.on_click(lambda: asyncio.create_task(create_premise(user_id_input.value.strip())))
next_chapter_btn.on_click(lambda: asyncio.create_task(generate_next_chapter(user_id_input.value.strip())))
choice_btn.on_click(lambda: asyncio.create_task(handle_user_choice(user_id_input.value.strip())))

# Hide elements initially
next_chapter_btn.visible = False
choice_input.visible = False
choice_btn.visible = False
decision_output.visible = False

# --- Run NiceGUI app ---
import os
port = 8080 # int(os.environ.get("PORT", 8080))
ui.run(title="Interactive Story App", host="localhost", port=port)
