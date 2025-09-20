
import asyncio
import time
#import json
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
#from fastapi.responses import StreamingResponse
from src.llm_client.llm_client import LLMClient
from src.memory.memory_system import StoryMemorySystem
from src.agents.story_author import StoryAuthor
from src.agents.director_agent import DirectorGraph
from src.agents.scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph
from src.memory.user_management import append_story


SESSIONS = {}

class MainTest:
    def __init__(self):
        # --- Globals ---
        self.llm_client = LLMClient()
        # key: user_id, value: {story_id: {...story session data...}, last_active: timestamp}
        #self.SESSIONS = {}
        # self.app = FastAPI()


    # ---------------- Session Management ----------------
    def setup_user_SESSION(self, user_id: str, story_id: str, memory_system=None, story_author=None):
        sceneplanner = ScenePlannerGraph(llm_client=self.llm_client)
        director = DirectorGraph(llm_client=self.llm_client, memory_system=memory_system, sceneplanner=sceneplanner)

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
    def initialize_story(self, user_id: str, story_title: str = ""):
        story_title_normalized = story_title.replace(" ", "_")
        story_id = story_title_normalized + "_" + user_id 
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        memory_system.qdrant_initialize()

        story_author = StoryAuthor(llm_client=self.llm_client, memory_system=memory_system)
        append_story(user_id=user_id, story_title=story_title, story_id=story_id)
        self.setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system, story_author=story_author)
        return {"status": "success", "message": f"Story initialized for {user_id}", "story_id": story_id}


    def continue_story(self,user_id: str, story_id: str) -> dict: # story_progress_data
        if not user_id:
            raise HTTPException(status_code=403, detail="Please enter a valid User ID.")

        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        #story_author = StoryAuthor(llm_client=self.llm_client, memory_system=memory_system)
        memory_system.qdrant_initialize()
        story_progress_data = memory_system.get_story_progress()
        
        if not story_progress_data:
            raise HTTPException(status_code=405, detail="No existing story found for this user and story ID.")
        
        if user_id not in SESSIONS:
            self.setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system)
        elif story_id not in SESSIONS[user_id]:
            self.setup_user_SESSION(user_id=user_id, story_id=story_id, memory_system=memory_system)


        story_text = (
            memory_system.get_long_term_story(chapter_id=story_progress_data.get("latest_chapter_id"))
            or "No story text for this chapter found."
        )
        return {"status": "success", "message": f"Session started for {user_id} and {story_id}", "story_text": story_text}

    # async def handle_user_choice(self, user_id: str, story_id: str, choice: str):
    #     user_data = SESSIONS.get(user_id)
    #     story_data = user_data.get(story_id) if user_data else None
    #     if not story_data or not choice.strip():
    #         return {"error": "Invalid session or empty choice."}

    #     story_data["sceneplanner"].receive_user_input(choice)
    #     return {"status": "success", "message": f"Choice '{choice}' received"}

    async def create_premise(self, initial_story_data: dict):
        user_id = initial_story_data["user_id"]
        story_id = initial_story_data["story_id"]

        user_data = SESSIONS.get(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")

        story_data = user_data.get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")

        filtered_data = {k: v for k, v in initial_story_data.items() if k not in ["story_id", "user_id"]}

        form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])

        # call set_story_premise in a thread to avoid blocking
        await asyncio.to_thread(
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
        return {"status": "success", "premise": "Premise set."}

    async def handle_story_websocket(self, websocket: WebSocket):
        """
        Handles WebSocket story session:
        - Sends scene chunks and decision points to client
        - Receives user choices and feeds them back into sceneplanner
        """
        await websocket.accept()
        try:
            # Expect init {user_id, story_id}
            init_data = await websocket.receive_json()
            user_id = init_data.get("user_id")
            story_id = init_data.get("story_id")

            # Validate session and story
            user_data = SESSIONS.get(user_id)
            if not user_data:
                await websocket.send_json({"error": "Invalid user ID"})
                await websocket.close()
                return

            story_data = user_data.get(story_id)
            if not story_data:
                await websocket.send_json({"error": "Invalid story ID"})
                await websocket.close()
                return

            queue = asyncio.Queue()

            def scene_chunk_callback(chunk: dict):
                """Receive already-formatted dicts from scene writer/planner."""
                queue.put_nowait(chunk)

            async def send_loop():
                """Send story chunks and decisions to client"""
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    await websocket.send_json(item)   # <-- send as proper JSON


            async def run_director():
                # Run the director, which will call our callback
                await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
                # After run ends, send chapter complete
                await queue.put({"chapter_complete": True})
                await queue.put(None)


            # Launch director
            asyncio.create_task(run_director())




            async def recv_loop():
                """Listen for user choices from client"""
                while True:
                    try:
                        msg = await websocket.receive_json()
                    except WebSocketDisconnect:
                        break

                    if "choice" in msg:
                        choice = msg["choice"].strip()
                        if not choice:
                            continue
                        # Fulfill the future in sceneplanner
                        sceneplanner = story_data["sceneplanner"]
                        future = getattr(sceneplanner, "_user_input_future", None)
                        if future and not future.done():
                            future.set_result(choice)
                        print(f"✅ Received choice: {choice}")

            # Run both loops concurrently
            await asyncio.gather(send_loop(), recv_loop())

        except WebSocketDisconnect:
            print("❌ Client disconnected")
        except Exception as e:
            print("⚠️ Error in handle_story_websocket:", e)
            await websocket.close()


    def logout(user_id: str):
        if user_id in SESSIONS:
            del SESSIONS[user_id]
            return {"status": "success", "message": "Session cleared."}
        return {"status": "success", "message": "No active session."}


# # @app.post("/next_chapter")
# # async def api_next_chapter(user_id: str, story_id: str):
# #     user_data = SESSIONS.get(user_id)
# #     if not user_data:
# #         return {"error": "Invalid user ID"}

# #     story_data = user_data.get(story_id)
# #     if not story_data:
# #         return {"error": "Invalid story ID"}

# #     queue = asyncio.Queue()

# #     def scene_chunk_callback(chunk: str):
# #         queue.put_nowait(chunk)

# #     async def run_director():
# #         await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
# #         queue.put_nowait(json.dumps({"chapter_complete": True}) + "\n")
# #         queue.put_nowait(None)

# #     async def event_stream():
# #         asyncio.create_task(run_director())
# #         while True:
# #             item = await queue.get()
# #             if item is None:
# #                 break
# #             yield item

# #     return StreamingResponse(event_stream(), media_type="text/event-stream")

# @app.websocket("/ws/next_chapter")
# async def websocket_next_chapter(websocket: WebSocket):
#     await websocket.accept()
#     try:
#         # Expect init {user_id, story_id}
#         init_data = await websocket.receive_json()
#         user_id = init_data.get("user_id")
#         story_id = init_data.get("story_id")

#         user_data = SESSIONS.get(user_id)
#         if not user_data:
#             await websocket.send_json({"error": "Invalid user ID"})
#             await websocket.close()
#             return

#         story_data = user_data.get(story_id)
#         if not story_data:
#             await websocket.send_json({"error": "Invalid story ID"})
#             await websocket.close()
#             return

#         queue = asyncio.Queue()

#         def scene_chunk_callback(chunk: str):
#             queue.put_nowait(json.dumps({"scene_chunk": chunk}))

#         async def run_director():
#             # Run the director (produces scene chunks)
#             await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
#             await queue.put(json.dumps({"chapter_complete": True}))
#             await queue.put(None)

#         # Run director in background
#         asyncio.create_task(run_director())

#         async def send_loop():
#             """Send story chunks to client"""
#             while True:
#                 item = await queue.get()
#                 if item is None:
#                     break
#                 await websocket.send_text(item)

#         async def recv_loop():
#             """Listen for user choices from client"""
#             while True:
#                 try:
#                     msg = await websocket.receive_json()
#                 except WebSocketDisconnect:
#                     break

#                 if "choice" in msg:
#                     choice = msg["choice"].strip()
#                     if not choice:
#                         continue
#                     sceneplanner = story_data["sceneplanner"]
#                     # Use the same future pattern you had before
#                     future = getattr(sceneplanner, "_user_input_future", None)
#                     if future and not future.done():
#                         future.set_result(choice)
#                     print(f"✅ Received choice: {choice}")

#         # Run send + receive in parallel
#         await asyncio.gather(send_loop(), recv_loop())

#     except WebSocketDisconnect:
#         print("❌ Client disconnected")
#     except Exception as e:
#         print("⚠️ Error:", e)
#         await websocket.close()


# @app.post("/initialize_story")
# def api_initialize(user_id: str, story_title: str = ""):
#     return initialize_story(user_id=user_id, story_title=story_title)


# @app.post("/continue_story")
# def api_continue_story(user_id: str, story_id: str = ""):
#     return continue_story(user_id=user_id, story_id=story_id)


# # @app.post("/choice")
# # async def api_choice(user_id: str, story_id: str, choice: str):
# #     user_data = SESSIONS.get(user_id)
# #     if not user_data:
# #         return {"error": "Invalid user ID"}

# #     story_data = user_data.get(story_id)
# #     if not story_data or not choice.strip():
# #         return {"error": "Invalid story ID or empty choice."}

# #     scene_planner = story_data["sceneplanner"]
# #     future = scene_planner._user_input_future
# #     if future and not future.done():
# #         future.set_result(choice)

# #     return {"message": f"Choice '{choice}' received"}


# @app.post("/logout")
# def api_logout(user_id: str):
#     if user_id in SESSIONS:
#         del SESSIONS[user_id]
#         return {"status": "success", "message": "Session cleared."}
#     return {"status": "success", "message": "No active session."}
