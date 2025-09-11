# src/main.py
from nicegui import ui, app
import asyncio
import re

# Import your helper functions / classes
from llm_client.llm_client import LLMClient
from memory.memory_system import StoryMemorySystem
from agents.story_author import StoryAuthor
from agents.director_agent import DirectorGraph
from agents.scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph

# Initialize director and sceneplanner instances
llm_client = LLMClient()
memory_system = StoryMemorySystem()
story_author = StoryAuthor(llm_client=llm_client, memory_system=memory_system)
sceneplanner = ScenePlannerGraph(llm_client=llm_client)
director = DirectorGraph(llm_client=llm_client, memory_system=memory_system, sceneplanner=sceneplanner)

# Global state
state = {
    "premise": None,
    "chapter_id": 1,
    "current_scene": None,
    "synopsis": None,
    "waiting_for_user_choice": False,
    "word_count": 0
}


# --- UI Elements ---
story_title_input = ui.input(label="Story Title")
setting_input = ui.input(label="Setting")
main_character_input = ui.input(label="Main Character")
genre_input = ui.input(label="Genre")
tone_input = ui.input(label="Tone")

submit_btn = ui.button("Create Story Synopsis")

# --- Story Output Section ---
ui.label('Story Synopsis').classes('text-lg font-bold mt-4 text-gray-700')
story_output = ui.markdown("")

# --- Scene Output Section ---
ui.label(f'Chapter {state["chapter_id"]}').classes('text-lg font-bold mt-4 text-gray-700')
scene_output = ui.markdown("")

decision_output = ui.markdown("")

choice_input = ui.textarea(label="Your Decision")
choice_btn = ui.button("Submit Decision", on_click=lambda: None)
next_chapter_btn = ui.button("Generate Next Chapter", on_click=lambda: None)


# --- Handlers ---
async def create_premise():
    form_string = f"""
    Title: {story_title_input.value}
    Setting: {setting_input.value}
    Main Character: {main_character_input.value}
    Genre: {genre_input.value}
    Tone: {tone_input.value}
    """
    #premise = await asyncio.to_thread(story_author.set_story_premise, form_string)
    premise = story_author.set_story_premise(form_string)
    if premise == "An error occured, Please try again.":
        story_output.content = premise
        story_output.update()
    else:
        state["premise"] = premise
        story_output.content = premise
        story_output.update()
        create_synopsis()

def create_synopsis():
    synopsis = story_author.set_story_synopsis(state["premise"])
    state["synopsis"] = synopsis

    # story_output.content = synopsis
    # story_output.update()

    next_chapter_btn.visible = True

def count_words_regex(text: str) -> int:
    words = re.findall(r'\b\w+\b', text)
    return len(words)

async def handle_scene_chunk(chunk: str):
    #print(f"[UI] Received chunk: {repr(chunk)}")
    #print("CHUNK",chunk)
    if chunk.strip().startswith("<DECISION_POINT>"):
        state["waiting_for_user_choice"] = True
        decision_output.content = chunk.strip()  # Show decision prompt here
        decision_output.visible = True
        choice_input.visible = True
        choice_btn.visible = True
        return

    # Append story text separately
    state["word_count"] += count_words_regex(chunk)
    scene_output.content += "\n\n" + chunk
    scene_output.update()


async def handle_user_choice():
    if not choice_input.value.strip():
        return
    choice = choice_input.value.strip()
    choice_input.value = ""

    # Hide decision UI
    decision_output.content = ""
    decision_output.visible = False
    choice_input.visible = False
    choice_btn.visible = False
    state["waiting_for_user_choice"] = False

    # Send choice back into ScenePlannerGraph
    sceneplanner.receive_user_input(choice)


async def generate_next_chapter():
    chapter_id = state["chapter_id"]
    scene_output.content = ""   # Clear previous chapter output
    decision_output.content = ""  # Clear any pending decision text
    scene_output.update()
    decision_output.update()
    next_chapter_btn.visible = False

    initial_state = {
        "messages": [],
        "current_chapter_id": chapter_id
    }

    async def director_runner():
        await director.run(initial_state, scene_chunk_callback=handle_scene_chunk)
        state["chapter_id"] += 1
        next_chapter_btn.visible = True
        scene_output.content = scene_output.content + "\n\n**Chapter Complete!**\nStory Word Count: " + str(state["word_count"]) + "\n"
        scene_output.update()

    asyncio.create_task(director_runner())


# --- Bind UI actions ---
submit_btn.on_click(lambda: asyncio.create_task(create_premise()))
next_chapter_btn.on_click(lambda: asyncio.create_task(generate_next_chapter()))
choice_btn.on_click(handle_user_choice)  # NiceGUI can handle async directly

# Hide elements initially
next_chapter_btn.visible = False
choice_input.visible = False
choice_btn.visible = False
decision_output.visible = False

ui.run(title="Interactive Story App", port=8080)
