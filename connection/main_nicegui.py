from nicegui import ui
import asyncio
import json
import httpx
import websockets


# API_URL = "https://ai-story-app.onrender.com"
# WS_URL = "wss://ai-story-app.onrender.com/ws/next_chapter"
API_URL = "http://127.0.0.1:8000"
WS_URL = "ws://127.0.0.1:8000/ws/next_chapter"


# --- UI Elements ---
user_id_input = ui.input(label="User ID (optional)")
story_id_input = ui.input(label="Story ID (optional)")
story_title_input = ui.input(label="Story Title")
setting_input = ui.input(label="Setting")
pov_input = ui.input(label="Point of View")
length_input = ui.input(label="Length")
guide_prose_input = ui.input(label="Guide Prose")
themes_input = ui.input(label="Additional Themes")
genre_input = ui.input(label="Genre")
tone_input = ui.input(label="Tone")

initialize_btn = ui.button("Initialize Story")
continue_btn = ui.button("Continue Story")
premise_btn = ui.button("Create Premise")
next_chapter_btn = ui.button("Generate Next Chapter")
choice_input = ui.textarea(label="Your Choice")
choice_btn = ui.button("Submit Choice")

story_output = ui.markdown("")
chapter_output = ui.markdown("")

# hide buttons/inputs initially
premise_btn.visible = False
next_chapter_btn.visible = False
choice_input.visible = False
choice_btn.visible = False


# --- Backend Calls ---
async def initialize_story():
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{API_URL}/initialize_story",
            params={"user_id": user_id_input.value, "story_title": story_title_input.value},
        )
        if resp.is_success:
            story_output.content = "✅ Story initialized"
            premise_btn.visible = True
            story_output.update()
            premise_btn.update()


async def continue_story():
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(
            f"{API_URL}/continue_story",
            params={"user_id": user_id_input.value, "story_id": story_id_input.value},
        )
        if resp.is_success:
            data = resp.json()
            story_output.content = f"✅ Story continued\n\n{data.get('story_text')}"
            premise_btn.visible = True
            story_output.update()
            next_chapter_btn.visible = True
            next_chapter_btn.update()


async def create_premise():
    payload = {
        "title": story_title_input.value,
        "setting": setting_input.value,
        "pov": pov_input.value,
        "length": length_input.value,
        "guide_prose": guide_prose_input.value,
        "additional_themes": themes_input.value,
        "genre": genre_input.value,
        "tone": tone_input.value,
        "user_id": user_id_input.value,
        "story_id": story_id_input.value,
    }
    async with httpx.AsyncClient(timeout=None) as client:
        resp = await client.post(f"{API_URL}/premise", json=payload)
        if resp.is_success:
            data = resp.json()
            story_output.content = f"📖 Premise:\n\n{data.get('premise')}"
            next_chapter_btn.visible = True
            story_output.update()
            next_chapter_btn.update()


# --- WebSocket for live chapter generation ---
ws_connection = None
async def generate_next_chapter():
    global ws_connection

    ws_connection = await websockets.connect(WS_URL)

    # Send user_id and story_id first
    await ws_connection.send(
        json.dumps({"user_id": user_id_input.value, "story_id": story_id_input.value})
    )

    chapter_output.content = ""
    chapter_output.update()

    async for message in ws_connection:
        try:
            chunk = json.loads(message)
            print("✅ WS received:", chunk)
        except Exception as e:
            print("❌ Invalid WS chunk:", message, e)
            continue

        # --- Scene chunks ---
        if "scene_chunk" in chunk:
            chapter_output.content += f"\n\n{chunk['scene_chunk']}"
            chapter_output.update()

        # --- Decision point ---
        if "decision_point" in chunk:
            chapter_output.content += f"\n\n👉 **Decision:** {chunk['decision_point']}"
            chapter_output.update()

            # show choice box
            choice_input.visible = True
            choice_btn.visible = True
            choice_input.value = ""  # FIX: clear old input
            choice_input.update()
            choice_btn.update()

            # wait for user choice (pause until submit_choice sets future)
            future = asyncio.get_event_loop().create_future()
            choice_input.user_future = future
            await future  # FIX: correctly pause director stream

            # hide inputs again
            choice_input.visible = False
            choice_btn.visible = False
            choice_input.update()
            choice_btn.update()

        # --- Chapter complete ---
        if "chapter_complete" in chunk:
            chapter_output.content += "\n\n✅ **Chapter complete!**"
            chapter_output.update()
            break


async def submit_choice():
    """Send the user's choice back to the backend and unblock generator."""
    global ws_connection
    choice = choice_input.value.strip()
    if not choice or not ws_connection:
        return

    await ws_connection.send(json.dumps({"choice": choice}))

    # unblock generate_next_chapter()
    if hasattr(choice_input, "user_future") and choice_input.user_future:
        if not choice_input.user_future.done():
            choice_input.user_future.set_result(True)  # FIX: resume generator
        choice_input.user_future = None


# --- Bind Buttons ---
initialize_btn.on_click(lambda: asyncio.create_task(initialize_story()))
continue_btn.on_click(lambda: asyncio.create_task(continue_story()))
premise_btn.on_click(lambda: asyncio.create_task(create_premise()))
next_chapter_btn.on_click(lambda: asyncio.create_task(generate_next_chapter()))
choice_btn.on_click(lambda: asyncio.create_task(submit_choice()))

ui.run(title="Interactive Story UI", host="localhost", port=8080)
