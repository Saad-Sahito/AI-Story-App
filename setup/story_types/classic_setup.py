import json
import asyncio
import time
import redis
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
from memory_profiler import profile
from setup.shared_redis_pool import get_redis_client
from src.memory.memory_system import StoryMemorySystem
from src.story_engines.classic_narrative.agents.story_author import StoryAuthor
from src.story_engines.classic_narrative.agents.director_agent import DirectorGraph
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from typing import Optional
from asyncio import Lock

SESSION_TTL = 3600  # 1 hour expiration for inactive sessions

load_dotenv()

from contextlib import asynccontextmanager

@asynccontextmanager
async def redis_lock(client, lock_key, timeout=10):
    lock_value = str(time.time())
    acquired = await client.set(lock_key, lock_value, nx=True, ex=timeout)
    if acquired:
        try:
            yield
        finally:
            lua = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                await client.eval(lua, 1, lock_key, lock_value)
            except redis.RedisError:
                pass
    else:
        raise HTTPException(status_code=503, detail="Could not acquire lock")

class ClassicStorySetup:
    def __init__(self):
        #self.llm_client = get_shared_client()
        if scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE is None:
            scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE = scene_planner_module.SharedScenePlannerService()
            print("Initialized classic shared scene planner service")
    async def _get_session(self, user_id: str, story_id: str = None):
        client = get_redis_client()
        async with client:
            try:
                if story_id:
                    key = f"session:{user_id}:{story_id}"
                    lock_key = f"lock:{key}"
                    async with redis_lock(client, lock_key):
                        data = await client.get(key)
                        if not data:
                            print(f"❌ No session found for {key}")
                            return None
                        session_data = json.loads(data)
                        if 'memory_system_params' in session_data:
                            params = session_data['memory_system_params']
                            if not all(k in params for k in ['user_id', 'story_id']):
                                print(f"❌ Invalid memory_system_params: {params}")
                                return None
                            session_data['memory_system'] = StoryMemorySystem(
                                user_id=params['user_id'],
                                story_id=params['story_id']
                            )
                            session_data['memory_system'].qdrant_initialize()
                            session_data['director'] = DirectorGraph(memory_system=session_data['memory_system'])
                            session_data['user_input_queue'] = asyncio.Queue()
                        async with client.pipeline() as pipe:
                            pipe.expire(key, SESSION_TTL)
                            await pipe.execute()
                        print(f"🔍 Retrieved story_session for {key}: {session_data}")
                        return session_data
                else:
                    # ... similar changes for user-level session
                    key = f"session:{user_id}"
                    lock_key = f"lock:{key}"
                    async with redis_lock(client, lock_key):
                        data = await client.get(key)
                        if not data:
                            print(f"❌ No user session found for {key}")
                            return {"last_active": time.time(), "stories": {}}
                        user_session = json.loads(data)
                        user_session["stories"] = {}
                        story_keys = await client.keys(f"session:{user_id}:*")
                        for story_key in story_keys:
                            story_id = story_key.decode().split(":")[2]
                            story_lock_key = f"lock:{story_key}"
                            async with redis_lock(client, story_lock_key):
                                story_data = await client.get(story_key)
                                if story_data:
                                    story_session = json.loads(story_data)
                                    if 'memory_system_params' in story_session:
                                        params = story_session['memory_system_params']
                                        if not all(k in params for k in ['user_id', 'story_id']):
                                            print(f"❌ Invalid memory_system_params for {story_key}: {params}")
                                            continue
                                        story_session['memory_system'] = StoryMemorySystem(
                                            user_id=params['user_id'],
                                            story_id=params['story_id']
                                        )
                                        story_session['memory_system'].qdrant_initialize()
                                        story_session['director'] = DirectorGraph(memory_system=story_session['memory_system'])
                                        story_session['user_input_queue'] = asyncio.Queue()
                                    user_session["stories"][story_id] = story_session
                                    async with client.pipeline() as pipe:
                                        pipe.expire(story_key, SESSION_TTL)
                                        await pipe.execute()
                        async with client.pipeline() as pipe:
                            pipe.expire(key, SESSION_TTL)
                            await pipe.execute()
                        print(f"🔍 Retrieved user_session for {user_id}: {user_session}")
                        return user_session
            except redis.RedisError as e:
                print(f"❌ Redis error in _get_session: {e}")
                raise HTTPException(status_code=500, detail="Failed to access session storage")
            except Exception as e:
                print(f"❌ Unexpected error in _get_session: {e}")
                import traceback
                traceback.print_exc()
                return None

    async def _set_session(self, user_id: str, story_id: str = None, data: dict = None):
        client = get_redis_client()
        async with client:
            try:
                key = f"session:{user_id}:{story_id}" if story_id else f"session:{user_id}"
                lock_key = f"lock:{key}"
                async with redis_lock(client, lock_key):
                    serializable_data = data.copy() if data else {}
                    if 'memory_system' in serializable_data:
                        memory_system = serializable_data['memory_system']
                        if memory_system is not None:
                            try:
                                serializable_data['memory_system_params'] = {
                                    'user_id': memory_system.user_id,
                                    'story_id': memory_system.story_id
                                }
                            except AttributeError as e:
                                print(f"❌ AttributeError in _set_session for memory_system: {e}")
                                serializable_data['memory_system_params'] = {}
                        del serializable_data['memory_system']
                    if 'director' in serializable_data:
                        del serializable_data['director']
                    if 'user_input_queue' in serializable_data:
                        del serializable_data['user_input_queue']
                    print(f"🔍 Serializing data for key {key}: {serializable_data}")
                    async with client.pipeline() as pipe:
                        pipe.set(key, json.dumps(serializable_data))
                        pipe.expire(key, SESSION_TTL)
                        await pipe.execute()
            except redis.RedisError as e:
                print(f"❌ Redis error in _set_session: {e}")
                raise HTTPException(status_code=500, detail="Failed to store session data")
            except TypeError as e:
                print(f"❌ Serialization error in _set_session: {e}")
                print(f"🔍 Problematic data: {serializable_data}")
                raise HTTPException(status_code=500, detail=f"Invalid session data format: {str(e)}")

    @profile
    def setup_user_session(self, user_id: str, story_id: str, memory_system=None):
        """Set up a user session with Redis."""
        client = get_redis_client()
        user_key = f"session:{user_id}"
        user_lock_key = f"lock:{user_key}"
        with redis_lock(client, user_lock_key):
            user_session = self._get_session(user_id) or {"last_active": time.time(), "stories": {}}
            story_session = {
                "memory_system_params": {
                    "user_id": user_id,
                    "story_id": story_id
                } if memory_system else {},
                "last_active": time.time(),
            }
            user_session["stories"][story_id] = story_session
            self._set_session(user_id, data=user_session)
        self._set_session(user_id, story_id, data=story_session)

    @profile
    async def initialize_story(self, user_id: str, story_title: str = ""):
        """Initialize a new story and store session in Redis."""
        story_title_normalized = story_title.lower().replace(" ", "_")
        story_id = f"{story_title_normalized}_{user_id}"
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        memory_system.qdrant_initialize()
        self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
        return {"status": "success", "message": f"Story initialized for {user_id}", "story_id": story_id}

    @profile
    async def continue_story(self, user_id: str, story_id: str) -> dict:
        """Continue an existing story, loading session from Redis."""
        client = get_redis_client()
        async with client:  # Use context manager
            user_key = f"session:{user_id}"
            user_lock_key = f"lock:{user_key}"
            with redis_lock(client, user_lock_key):
                if not user_id:
                    raise HTTPException(status_code=403, detail="Please enter a valid User ID.")
                memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                memory_system.qdrant_initialize()
                story_progress_data = memory_system.get_story_progress()
                if not story_progress_data:
                    raise HTTPException(status_code=405, detail="No existing story found for this user and story ID.")
                self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
                user_session = self._get_session(user_id)  # Reload session
                print(f"🔍 continue_story user_session[stories][{story_id}]: {user_session['stories'][story_id]}")
                story_text = (
                    user_session["stories"][story_id]['memory_system'].get_story_cluster(chapter_id=story_progress_data.get("latest_chapter_id"))
                    or "No story text for this chapter found."
                )
                return {"status": "success", "message": f"Session started for {user_id} and {story_id}", "story_cluster": story_text}

    @profile
    async def create_premise(self, initial_story_data: dict):
        """Create story premise and update session in Redis."""
        client = get_redis_client()
        tone_dict = {
            0: "Light", 20: "Humorous", 40: "Epic",
            60: "Serious", 80: "Dark", 100: "Gritty"
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
        async with client:  # Use context manager
            user_key = f"session:{user_id}"
            user_lock_key = f"lock:{user_key}"
            with redis_lock(client, user_lock_key):
                user_data = self._get_session(user_id)
                if not user_data:
                    raise HTTPException(status_code=403, detail="Invalid user ID")
                story_data = user_data["stories"].get(story_id)
                if not story_data:
                    raise HTTPException(status_code=405, detail="Invalid story ID")
                story_author = StoryAuthor(memory_system=story_data['memory_system'])
                if "Tone" in initial_story_data:
                    tone_value = initial_story_data["Tone"]
                    mapped_tone = tone_dict.get(tone_value)
                    if mapped_tone is None:
                        mapped_tone = tone_dict[min(tone_dict.keys(), key=lambda k: abs(k - tone_value))]
                    initial_story_data["Tone"] = mapped_tone
                if "Length" in initial_story_data:
                    length_value = initial_story_data["Length"]
                    mapped_length = length_dict.get(length_value)
                    if mapped_length is None:
                        mapped_length = length_dict[min(length_dict.keys(), key=lambda k: abs(k - length_value))]
                    initial_story_data["Length"] = mapped_length
                filtered_data = {k: v for k, v in initial_story_data.items() if k not in ["story_id", "user_id"]}
                form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])
                await story_author.set_story_premise(
                    form_string,
                    initial_story_data.get("Title", ""),
                )
                del story_author
                story_data["memory_system"].update_story_progress(
                    metadata={
                        "latest_chapter_id": 1,
                        "continue_scene_id": 1,
                        "story_title": initial_story_data["Title"],
                        "word_count": 0,
                    }
                )
                # Create a serializable copy of story_data
                serializable_story_data = {
                    "memory_system_params": story_data.get("memory_system_params", {}),
                    "last_active": time.time(),
                }
                # Update user_data stories
                user_data["stories"][story_id] = serializable_story_data
                # Save sessions
                self._set_session(user_id, story_id, serializable_story_data)
                self._set_session(user_id, data=user_data)
        from src.memory.user_management import append_story
        append_story(user_id=user_id, story_title=initial_story_data.get("Title", ""), story_id=story_id, story_type=initial_story_data.get("story_type", ""))
        return {"status": "success", "premise": "Premise set."}

    async def handle_story_websocket(self, websocket: WebSocket, user_id: str, story_id: str):
        """Handle WebSocket for story progression, using Redis sessions."""
        await websocket.accept()
        client = await get_redis_client()  # Now async
        async with client:
            try:
                print(f"🔍 DEBUG: Connected WS for user_id={user_id}, story_id={story_id}")
                # Check for existing WebSocket connection
                ws_key = f"active_ws:{user_id}:{story_id}"
                async with redis_lock(client, f"lock:{ws_key}"):
                    if await client.get(ws_key):
                        print(f"❌ Another WebSocket connection active for {user_id}/{story_id}")
                        await websocket.send_json({"error": "Another connection is active for this story"})
                        await websocket.close()
                        return
                    await client.set(ws_key, "1", ex=SESSION_TTL)

                init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
                user_id = init_data.get("user_id")
                story_id = init_data.get("story_id")
                print(f"🔍 DEBUG: Received init_data: user_id={user_id}, story_id={story_id}")
                if not user_id or not story_id:
                    print(f"❌ Invalid init_data: user_id={user_id}, story_id={story_id}")
                    await websocket.send_json({"error": "Missing user_id or story_id"})
                    await websocket.close()
                    return

                user_key = f"interactive_session:{user_id}"
                user_lock_key = f"lock:{user_key}"
                async with redis_lock(client, user_lock_key):
                    user_data = await self._get_session(user_id)  # Now async
                    if not user_data:
                        print(f"❌ No user session for user_id={user_id}")
                        await websocket.send_json({"error": "Invalid user ID"})
                        await websocket.close()
                        return
                    story_data = user_data["stories"].get(story_id)
                    if not story_data:
                        print(f"❌ No story session for story_id={story_id}")
                        await websocket.send_json({"error": "Invalid story ID"})
                        await websocket.close()
                        return

                    # Check if director is already running
                    director_key = f"director_running:{user_id}:{story_id}"
                    async with redis_lock(client, f"lock:{director_key}"):
                        if await client.get(director_key):
                            print(f"❌ Director already running for {user_id}/{story_id}")
                            await websocket.send_json({"error": "Story progression already active"})
                            await websocket.close()
                            return
                        await client.set(director_key, "1", ex=SESSION_TTL)

                    serializable_story_data = {
                        "memory_system_params": story_data.get("memory_system_params", {}),
                        "last_active": time.time(),
                    }
                    user_data["stories"][story_id] = serializable_story_data
                    story_key = f"interactive_session:{user_id}:{story_id}"
                    async with client.pipeline() as pipe:
                        pipe.set(user_key, json.dumps(user_data))
                        pipe.expire(user_key, SESSION_TTL)
                        pipe.set(story_key, json.dumps(serializable_story_data))
                        pipe.expire(story_key, SESSION_TTL)
                        await pipe.execute()
                    print(f"✅ DEBUG: Saved story session for {user_id}/{story_id}")
                    print(f"🔍 DEBUG: Story data initialized: {serializable_story_data}")

                queue = asyncio.Queue()

                def scene_chunk_callback(chunk: dict):
                    print(f"🔍 DEBUG: scene_chunk_callback: {chunk}")
                    queue.put_nowait(chunk)

                async def refresh_ttl_loop():
                    try:
                        while True:
                            await asyncio.sleep(30)  # Refresh every 30 seconds
                            async with client.pipeline() as pipe:
                                pipe.expire(user_key, SESSION_TTL)
                                pipe.expire(story_key, SESSION_TTL)
                                pipe.expire(f"input_queue:{user_id}:{story_id}", SESSION_TTL)
                                pipe.expire(f"continue_input_queue:{user_id}:{story_id}", SESSION_TTL)
                                await pipe.execute()
                            print(f"🔍 DEBUG: Refreshed TTLs for {user_id}/{story_id}")
                    except asyncio.CancelledError:
                        print("🔍 DEBUG: refresh_ttl_loop cancelled")
                        return
                    except Exception as e:
                        print(f"❌ refresh_ttl_loop error: {e}")
                        import traceback
                        traceback.print_exc()

                async def send_loop():
                    try:
                        while True:
                            item = await queue.get()
                            if item is None:
                                print("🔍 DEBUG: send_loop received None, exiting")
                                break
                            print(f"🔍 DEBUG: Sending to WebSocket: {item}")
                            await websocket.send_json(item)
                    except asyncio.CancelledError:
                        print("🔍 DEBUG: send_loop cancelled")
                        return
                    except Exception as e:
                        print(f"❌ send_loop error: {e}")
                        import traceback
                        traceback.print_exc()

                async def recv_loop():
                    try:
                        while True:
                            msg = await websocket.receive_json()
                            print(f"🔍 DEBUG: Received WebSocket message: {msg}")
                            if "choice" in msg:
                                choice = msg["choice"].strip()
                                if not choice:
                                    print(f"⚠️ WARNING: Empty choice received for {user_id}/{story_id}")
                                    continue
                                queue_key = f"input_queue:{user_id}:{story_id}"
                                queue_lock_key = f"lock:{queue_key}"
                                async with redis_lock(client, queue_lock_key):
                                    async with client.pipeline() as pipe:
                                        pipe.rpush(queue_key, choice)
                                        pipe.expire(queue_key, SESSION_TTL)
                                        await pipe.execute()
                                    print(f"✅ DEBUG: Pushed choice '{choice}' to Redis queue {queue_key}")
                            elif "continue_chapter" in msg:
                                continue_chapter = msg["continue_chapter"]
                                queue_key = f"continue_input_queue:{user_id}:{story_id}"
                                queue_lock_key = f"lock:{queue_key}"
                                async with redis_lock(client, queue_lock_key):
                                    async with client.pipeline() as pipe:
                                        pipe.rpush(queue_key, continue_chapter)
                                        pipe.expire(queue_key, SESSION_TTL)
                                        await pipe.execute()
                                    print(f"✅ DEBUG: Pushed continue chapter '{continue_chapter}' to Redis queue {queue_key}")
                    except WebSocketDisconnect:
                        print("❌ Client disconnected")
                    except asyncio.CancelledError:
                        print("🔍 DEBUG: recv_loop cancelled")
                        return
                    except Exception as e:
                        print(f"❌ recv_loop error: {e}")
                        import traceback
                        traceback.print_exc()

                async def run_director():
                    try:
                        print(f"🔍 DEBUG: Starting director.run for {user_id}/{story_id}")
                        await story_data["director"].run(scene_chunk_callback=scene_chunk_callback)
                        print("✅ Director run completed")
                        await queue.put({"chapter_complete": True})
                        await queue.put(None)
                    except Exception as e:
                        print(f"❌ ERROR in run_director: {e}")
                        import traceback
                        traceback.print_exc()
                        await queue.put({"error": f"Director failed: {str(e)}"})
                        await queue.put(None)

                director_task = asyncio.create_task(run_director())
                send_task = asyncio.create_task(send_loop())
                recv_task = asyncio.create_task(recv_loop())
                ttl_task = asyncio.create_task(refresh_ttl_loop())
                try:
                    await asyncio.gather(director_task, send_task, recv_task, ttl_task, return_exceptions=False)
                except Exception as e:
                    print(f"❌ ERROR in asyncio.gather: {e}")
                    import traceback
                    traceback.print_exc()
                    await websocket.send_json({"error": f"WebSocket task failed: {str(e)}"})
                finally:
                    director_task.cancel()
                    send_task.cancel()
                    recv_task.cancel()
                    ttl_task.cancel()
                    # Clean up Redis flags
                    async with redis_lock(client, f"lock:{ws_key}"):
                        await client.delete(ws_key)
                    async with redis_lock(client, f"lock:{director_key}"):
                        await client.delete(director_key)
                    # Close queues/resources
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for init_data")
                await websocket.send_json({"error": "Timeout waiting for initial data"})
            except Exception as e:
                print(f"❌ Error in handle_story_websocket: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": str(e)})
            finally:
                try:
                    await websocket.close()
                    print("🔍 DEBUG: WebSocket closed")
                    # Ensure Redis flags are cleaned up even on error
                    async with redis_lock(client, f"lock:{ws_key}"):
                        await client.delete(ws_key)
                    async with redis_lock(client, f"lock:{director_key}"):
                        await client.delete(director_key)
                except RuntimeError:
                    print("🔍 DEBUG: WebSocket already closed")

    async def get_story_cluster(self, user_id: str, story_id: str, chapter_number: int):
        user_data = self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")
        try:
            return {"status": "success", "data": await story_data['memory_system'].get_story_cluster(chapter_id=chapter_number)}  # Await
        except:
            return {"status": "error"}

    async def get_story_progress_for_user(self, user_id: str, story_id: str) -> dict:
        user_data = self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")
        try:
            return {"status": "success", "data": await story_data['memory_system'].get_story_progress()}  # Await
        except:
            return {"status": "error"}
    
    async def close(self):
        """Clean up resources used by ClassicStorySetup."""
        try:
            # 1. Close Redis pool (if exists)
            from setup.shared_redis_pool import REDIS_POOL
            if REDIS_POOL:
                await REDIS_POOL.disconnect()
                print("✅ Redis pool closed")

            # 2. Clean up memory systems (if needed)
            # Example: explicitly close Qdrant connections
            if hasattr(self, "_active_memory_systems"):
                for ms in self._active_memory_systems:
                    try:
                        await ms.close()  # if your StoryMemorySystem has cleanup
                    except Exception as e:
                        print(f"⚠️ Failed to close memory system: {e}")

            # 3. Cancel any lingering background tasks
            if hasattr(self, "_tasks"):
                for task in self._tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                print("✅ Background tasks cancelled")

            print("✅ ClassicStorySetup closed successfully")

        except Exception as e:
            print(f"❌ Error while closing ClassicStorySetup: {e}")




SHARED_CLASSIC_STORY_SETUP: Optional[ClassicStorySetup] = None
_classic_setup_lock = Lock()

async def get_shared_classic_setup() -> ClassicStorySetup:
    global SHARED_CLASSIC_STORY_SETUP
    try:
        async with _classic_setup_lock:
            if SHARED_CLASSIC_STORY_SETUP is None:
                SHARED_CLASSIC_STORY_SETUP = ClassicStorySetup()
            return SHARED_CLASSIC_STORY_SETUP
    except Exception as e:
        raise Exception(f"Failed to initialize ClassicStorySetup: {e}")

async def close_shared_classic_setup():
    global SHARED_CLASSIC_STORY_SETUP
    async with _classic_setup_lock:
        if SHARED_CLASSIC_STORY_SETUP is not None:
            try:
                await SHARED_CLASSIC_STORY_SETUP.close()
            except Exception as e:
                print(f"Error closing ClassicStorySetup: {e}")
            SHARED_CLASSIC_STORY_SETUP = None