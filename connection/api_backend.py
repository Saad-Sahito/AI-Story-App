
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
from starlette.websockets import WebSocketState


SESSIONS = {}

class APIBackend:
    def __init__(self):
        self.llm_client = LLMClient()

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
        #del SESSIONS[user_id][story_id]["sceneplanner"]
        SESSIONS[user_id]["last_active"] = time.time()


    # ---------------- Story Flow ----------------
    def initialize_story(self, user_id: str, story_title: str = ""):
        story_title_normalized = story_title.lower().replace(" ", "_")
        story_id = story_title_normalized + "_" + user_id 
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        memory_system.qdrant_initialize()

        story_author = StoryAuthor(llm_client=self.llm_client, memory_system=memory_system)
        #append_story(user_id=user_id, story_title=story_title, story_id=story_id)
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
            memory_system.get_story_cluster(chapter_id=story_progress_data.get("latest_chapter_id"))
            or "No story text for this chapter found."
        )
        return {"status": "success", "message": f"Session started for {user_id} and {story_id}", "story_cluster": story_text}    #, "story_text": story_text



    async def create_premise(self, initial_story_data: dict):
        tone_dict = {
            0: "Light",
            20: "Humorous",
            40: "Epic",
            60: "Serious",
            80: "Dark",
            100: "Gritty"
        }
        length_dict = { 
            0: "Flash Fiction (1,000 - 2,500 words)",
            20: "Short Story (2,500 - 7,500 words)",
            40: "Novelette (7,500 - 20,000 words)",
            60: "Novella (20,000 - 40,000 words)",
            80: "Novel Chapter (40,000 - 70,000 words)",
            100: "Epic / Series (70,000 - 100,000+ words)"
        }

        user_id = initial_story_data["user_id"]
        story_id = initial_story_data["story_id"]

        user_data = SESSIONS.get(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")

        story_data = user_data.get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")

        # ✅ Map numeric Tone to string
        if "Tone" in initial_story_data:
            tone_value = initial_story_data["Tone"]
            mapped_tone = tone_dict.get(tone_value)
            if mapped_tone is None:
                mapped_tone = tone_dict[min(tone_dict.keys(), key=lambda k: abs(k - tone_value))]
            initial_story_data["Tone"] = mapped_tone

        # ✅ Map numeric Length to string
        if "Length" in initial_story_data:
            length_value = initial_story_data["Length"]
            mapped_length = length_dict.get(length_value)
            if mapped_length is None:
                mapped_length = length_dict[min(length_dict.keys(), key=lambda k: abs(k - length_value))]
            initial_story_data["Length"] = mapped_length

        # Filter out story_id and user_id
        filtered_data = {k: v for k, v in initial_story_data.items() if k not in ["story_id", "user_id"]}

        # Build premise string
        form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])

        # Call set_story_premise in a thread to avoid blocking
        await asyncio.to_thread(
            story_data["story_author"].set_story_premise,
            form_string,
            initial_story_data.get("Title", ""),
        )

        story_data["memory_system"].update_story_progress(
            metadata={
                "latest_chapter_id": 1,
                "continue_scene_id": 1,
                "story_title": initial_story_data["Title"],  # fixed capitalization
                "word_count": 0,
            }
        )
        del story_data["story_author"]
        return {"status": "success", "premise": "Premise set."}


    async def handle_story_websocket(self, websocket: WebSocket):
        await websocket.accept()
        try:
            # Receive init data
            init_data = await websocket.receive_json()
            user_id = init_data.get("user_id")
            story_id = init_data.get("story_id")

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
                """Receive dicts from sceneplanner"""
                queue.put_nowait(chunk)

            async def send_loop():
                try:
                    while True:
                        item = await queue.get()
                        if item is None:
                            break
                        await websocket.send_json(item)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    print("❌ send_loop error:", e)

            async def recv_loop():
                try:
                    while True:
                        msg = await websocket.receive_json()
                        if "choice" in msg:
                            choice = msg["choice"].strip()
                            if not choice:
                                continue

                            sceneplanner = story_data.get("sceneplanner")
                            if sceneplanner and hasattr(sceneplanner, "user_input_queue"):
                                # Add this check to make sure queue exists
                                if sceneplanner.user_input_queue is not None:
                                    sceneplanner.user_input_queue.put_nowait(choice)
                                    print(f"✅ Received choice: {choice}")
                                else:
                                    print("⚠️ ScenePlanner user_input_queue is None!")
                            else:
                                print("⚠️ ScenePlanner has no user_input_queue!")

                except WebSocketDisconnect:
                    print("❌ Client disconnected")
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    print("❌ recv_loop error:", e)


            async def run_director():
                await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
                await queue.put({"chapter_complete": True})
                await queue.put(None)

            # Start loops
            director_task = asyncio.create_task(run_director())
            send_task = asyncio.create_task(send_loop())
            recv_task = asyncio.create_task(recv_loop())

            # Run concurrently and exit if one fails
            # done, pending = await asyncio.wait(
            #     [director_task, send_task, recv_task],
            #     return_when=asyncio.FIRST_EXCEPTION
            # )
            # Run all tasks until they complete or are canceled naturally
            await asyncio.gather(director_task, send_task, recv_task, return_exceptions=True)

            # Cancel any still-running tasks
            # for task in pending:
            #     task.cancel()
            # await asyncio.gather(*pending, return_exceptions=True)

        except Exception as e:
            print("⚠️ Error in handle_story_websocket:", e)
        finally:
            try:
                await websocket.close()
            except RuntimeError:
                # Already closed, ignore
                pass



    def logout(user_id: str):
        if user_id in SESSIONS:
            for story_id, story_data in list(SESSIONS[user_id].items()):
                if story_id == "last_active": 
                    continue
                story_data["memory_system"].cleanup()
            del SESSIONS[user_id]
            import gc
            gc.collect()

            return {"status": "success", "message": "Session cleared."}












    # async def handle_user_choice(self, user_id: str, story_id: str, choice: str):
    #     user_data = SESSIONS.get(user_id)
    #     story_data = user_data.get(story_id) if user_data else None
    #     if not story_data or not choice.strip():
    #         return {"error": "Invalid session or empty choice."}

    #     story_data["sceneplanner"].receive_user_input(choice)
    #     return {"status": "success", "message": f"Choice '{choice}' received"}