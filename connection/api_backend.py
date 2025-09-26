
# src/api_backend.py - UPDATED FOR FULL REDIS COMPATIBILITY AND DIRECTOR DEBUGGING

import json
import asyncio
import time
import redis
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
from memory_profiler import profile
from src.llm_client.llm_client import get_shared_client
from src.memory.memory_system import StoryMemorySystem
from src.agents.story_author import StoryAuthor
from src.agents.director_agent import DirectorGraph
import src.agents.scene_creation_subgraph.shared_scene_planner as scene_planner_module

# Initialize Redis connection pool
REDIS_POOL = redis.ConnectionPool(
    host='localhost',
    port=6379,
    db=0,
    decode_responses=True,
    max_connections=10,
    retry_on_timeout=True
)

SESSION_TTL = 3600  # 1 hour expiration for inactive sessions

# Global shared instances
SHARED_LLM_CLIENT = None

load_dotenv()

def get_redis_client():
    """Get a Redis client, reconnecting if necessary."""
    try:
        client = redis.Redis(connection_pool=REDIS_POOL)
        client.ping()
        return client
    except redis.ConnectionError as e:
        print(f"❌ Redis connection error, reconnecting: {e}")
        client = redis.Redis(connection_pool=REDIS_POOL)
        client.ping()
        return client
    except redis.RedisError as e:
        print(f"❌ Redis error in get_redis_client: {e}")
        raise HTTPException(status_code=500, detail="Failed to connect to session storage")

class APIBackend:
    def __init__(self):
        self.llm_client = get_shared_client()
        if scene_planner_module.SHARED_SCENE_PLANNER_SERVICE is None:
            scene_planner_module.SHARED_SCENE_PLANNER_SERVICE = scene_planner_module.SharedScenePlannerService()
            print("Initialized shared scene planner service")

    def _get_session(self, user_id: str, story_id: str = None):
        """Retrieve session data from Redis, reinitializing non-serializable objects."""
        client = get_redis_client()
        try:
            if story_id:
                # Fetch story-specific session
                key = f"session:{user_id}:{story_id}"
                data = client.get(key)
                if not data:
                    print(f"❌ No session found for {key}")
                    return None
                session_data = json.loads(data)
                # Reinitialize non-serializable objects
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
                    session_data['story_author'] = (
                        StoryAuthor(memory_system=session_data['memory_system'])
                        if session_data.get('story_author_needed', False)
                        else None
                    )
                client.expire(key, SESSION_TTL)
                print(f"🔍 Retrieved story_session for {key}: {session_data}")
                return session_data
            else:
                # Fetch user-level session and merge story-specific data
                key = f"session:{user_id}"
                data = client.get(key)
                if not data:
                    print(f"❌ No user session found for {key}")
                    return None
                user_session = json.loads(data)
                # Fetch all story-specific sessions for this user
                story_keys = client.keys(f"session:{user_id}:*")
                for story_key in story_keys:
                    story_id = story_key.split(":")[2]
                    story_data = client.get(story_key)
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
                            story_session['story_author'] = (
                                StoryAuthor(memory_system=story_session['memory_system'])
                                if story_session.get('story_author_needed', False)
                                else None
                            )
                        user_session[story_id] = story_session
                        client.expire(story_key, SESSION_TTL)
                client.expire(key, SESSION_TTL)
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

    def _set_session(self, user_id: str, story_id: str = None, data: dict = None):
        """Store session data in Redis, ensuring all objects are serializable."""
        client = get_redis_client()
        try:
            key = f"session:{user_id}:{story_id}" if story_id else f"session:{user_id}"
            serializable_data = data.copy() if data else {}

            # Remove non-serializable objects and store parameters
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
            
            if 'story_author' in serializable_data:
                serializable_data['story_author_needed'] = serializable_data['story_author'] is not None
                del serializable_data['story_author']

            print(f"🔍 Serializing data for key {key}: {serializable_data}")

            client.set(key, json.dumps(serializable_data), ex=SESSION_TTL)
        except redis.RedisError as e:
            print(f"❌ Redis error in _set_session: {e}")
            raise HTTPException(status_code=500, detail="Failed to store session data")
        except TypeError as e:
            print(f"❌ Serialization error in _set_session: {e}")
            print(f"🔍 Problematic data: {serializable_data}")
            raise HTTPException(status_code=500, detail=f"Invalid session data format: {str(e)}")

    @profile
    def setup_user_session(self, user_id: str, story_id: str, memory_system=None, story_author=None):
        """Set up a user session with Redis."""
        user_session = self._get_session(user_id) or {"last_active": time.time()}
        story_session = {
            "memory_system_params": {
                "user_id": user_id,
                "story_id": story_id
            } if memory_system else {},
            "story_author_needed": story_author is not None,
            "last_active": time.time(),
        }
        user_session[story_id] = story_session
        self._set_session(user_id, data=user_session)
        self._set_session(user_id, story_id, data=story_session)

    @profile
    def initialize_story(self, user_id: str, story_title: str = ""):
        """Initialize a new story and store session in Redis."""
        story_title_normalized = story_title.lower().replace(" ", "_")
        story_id = f"{story_title_normalized}_{user_id}"
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        memory_system.qdrant_initialize()

        story_author = StoryAuthor(memory_system=memory_system)
        self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system, story_author=story_author)
        return {"status": "success", "message": f"Story initialized for {user_id}", "story_id": story_id}

    @profile
    def continue_story(self, user_id: str, story_id: str) -> dict:
        """Continue an existing story, loading session from Redis."""
        if not user_id:
            raise HTTPException(status_code=403, detail="Please enter a valid User ID.")

        user_session = self._get_session(user_id)
        if not user_session or story_id not in user_session:
            # Fallback: Initialize new memory_system if session is missing
            memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
            memory_system.qdrant_initialize()
            story_progress_data = memory_system.get_story_progress()
            if not story_progress_data:
                raise HTTPException(status_code=405, detail="No existing story found for this user and story ID.")
            self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
            user_session = self._get_session(user_id)  # Reload session
        else:
            # Ensure story_session has memory_system
            if 'memory_system' not in user_session[story_id]:
                params = user_session[story_id].get('memory_system_params', {})
                if not params:
                    raise HTTPException(status_code=500, detail="Invalid session data: missing memory_system_params")
                user_session[story_id]['memory_system'] = StoryMemorySystem(
                    user_id=params['user_id'], 
                    story_id=params['story_id']
                )
                user_session[story_id]['memory_system'].qdrant_initialize()
                user_session[story_id]['director'] = DirectorGraph(memory_system=user_session[story_id]['memory_system'])
                user_session[story_id]['user_input_queue'] = asyncio.Queue()
                user_session[story_id]['story_author'] = (
                    StoryAuthor(memory_system=user_session[story_id]['memory_system'])
                    if user_session[story_id].get('story_author_needed', False)
                    else None
                )
            story_progress_data = user_session[story_id]['memory_system'].get_story_progress()

        print(f"🔍 continue_story user_session[{story_id}]: {user_session[story_id]}")
        story_text = (
            user_session[story_id]['memory_system'].get_story_cluster(chapter_id=story_progress_data.get("latest_chapter_id"))
            or "No story text for this chapter found."
        )
        return {"status": "success", "message": f"Session started for {user_id} and {story_id}", "story_cluster": story_text}

    @profile
    async def create_premise(self, initial_story_data: dict):
        """Create story premise and update session in Redis."""
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

        user_data = self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")

        story_data = user_data.get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")

        # Ensure story_author is initialized if needed
        if story_data.get('story_author') is None and story_data.get('story_author_needed', False):
            if 'memory_system' not in story_data:
                params = story_data.get('memory_system_params', {})
                if not params:
                    raise HTTPException(status_code=500, detail="Invalid session data: missing memory_system_params")
                story_data['memory_system'] = StoryMemorySystem(
                    user_id=params['user_id'], 
                    story_id=params['story_id']
                )
                story_data['memory_system'].qdrant_initialize()
            story_data['story_author'] = StoryAuthor(memory_system=story_data['memory_system'])

        print(f"🔍 create_premise story_data: {story_data}")

        # Map numeric values to strings
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

        await story_data["story_author"].set_story_premise(
            form_string,
            initial_story_data.get("Title", ""),
        )

        # Ensure memory_system is initialized
        if 'memory_system' not in story_data:
            params = story_data.get('memory_system_params', {})
            if not params:
                raise HTTPException(status_code=500, detail="Invalid session data: missing memory_system_params")
            story_data['memory_system'] = StoryMemorySystem(
                user_id=params['user_id'], 
                story_id=params['story_id']
            )
            story_data['memory_system'].qdrant_initialize()

        story_data["memory_system"].update_story_progress(
            metadata={
                "latest_chapter_id": 1,
                "continue_scene_id": 1,
                "story_title": initial_story_data["Title"],
                "word_count": 0,
            }
        )

        # Clean up story_author to save memory
        story_data["story_author"] = None
        self._set_session(user_id, story_id, story_data)
        from src.memory.user_management import append_story
        append_story(user_id, initial_story_data.get("Title", ""), story_id)

        return {"status": "success", "premise": "Premise set."}

    @profile
    async def handle_story_websocket(self, websocket: WebSocket, user_id: str, story_id: str):
        """Handle WebSocket for story progression, using Redis sessions."""
        await websocket.accept()
        try:
            print(f"🔍 DEBUG: Connected WS for user_id={user_id}, story_id={story_id}")
            init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            user_id = init_data.get("user_id")
            story_id = init_data.get("story_id")
            print(f"🔍 DEBUG: Received init_data: user_id={user_id}, story_id={story_id}")

            if not user_id or not story_id:
                print(f"❌ Invalid init_data: user_id={user_id}, story_id={story_id}")
                await websocket.send_json({"error": "Missing user_id or story_id"})
                await websocket.close()
                return

            user_data = self._get_session(user_id)
            if not user_data:
                print(f"❌ No user session for user_id={user_id}")
                await websocket.send_json({"error": "Invalid user ID"})
                await websocket.close()
                return
            story_data = user_data.get(story_id)
            if not story_data:
                print(f"❌ No story session for story_id={story_id}")
                await websocket.send_json({"error": "Invalid story ID"})
                await websocket.close()
                return

            # Ensure director is initialized
            if 'memory_system' not in story_data:
                params = story_data.get('memory_system_params', {})
                if not params:
                    print(f"❌ Invalid session data: missing memory_system_params for {user_id}:{story_id}")
                    await websocket.send_json({"error": "Invalid session data: missing memory_system_params"})
                    await websocket.close()
                    return
                story_data['memory_system'] = StoryMemorySystem(
                    user_id=params['user_id'], 
                    story_id=params['story_id']
                )
                story_data['memory_system'].qdrant_initialize()
            if 'director' not in story_data:
                story_data['director'] = DirectorGraph(memory_system=story_data['memory_system'])

            # Save the updated story_data to Redis (without non-serializable objects)
            serializable_story_data = story_data.copy()
            if 'director' in serializable_story_data:
                del serializable_story_data['director']
            if 'memory_system' in serializable_story_data:
                serializable_story_data['memory_system_params'] = {
                    'user_id': story_data['memory_system'].user_id,
                    'story_id': story_data['memory_system'].story_id
                }
                del serializable_story_data['memory_system']
            if 'story_author' in serializable_story_data:
                serializable_story_data['story_author_needed'] = story_data['story_author'] is not None
                del serializable_story_data['story_author']
            self._set_session(user_id, story_id, serializable_story_data)
            print(f"✅ DEBUG: Saved story session for {user_id}/{story_id}")

            print(f"🔍 DEBUG: Story data initialized: {serializable_story_data}")

            queue = asyncio.Queue()

            def scene_chunk_callback(chunk: dict):
                print(f"🔍 DEBUG: scene_chunk_callback: {chunk}")
                queue.put_nowait(chunk)

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
                client = get_redis_client()
                try:
                    while True:
                        msg = await websocket.receive_json()
                        print(f"🔍 DEBUG: Received WebSocket message: {msg}")
                        if "choice" in msg:
                            choice = msg["choice"].strip()
                            if not choice:
                                print(f"⚠️ WARNING: Empty choice received for {user_id}/{story_id}")
                                continue
                            # Push choice to Redis List
                            queue_key = f"input_queue:{user_id}:{story_id}"
                            client.rpush(queue_key, choice)
                            client.expire(queue_key, SESSION_TTL)
                            print(f"✅ DEBUG: Pushed choice '{choice}' to Redis queue {queue_key}")
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

            try:
                await asyncio.gather(director_task, send_task, recv_task, return_exceptions=False)
            except Exception as e:
                print(f"❌ ERROR in asyncio.gather: {e}")
                import traceback
                traceback.print_exc()
                await websocket.send_json({"error": f"WebSocket task failed: {str(e)}"})

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
                import gc
                gc.collect()
            except RuntimeError:
                print("🔍 DEBUG: WebSocket already closed")


    @staticmethod
    @profile
    def logout(user_id: str):
        """Clear all sessions for a user from Redis."""
        client = get_redis_client()
        try:
            # Use SCAN instead of KEYS for memory efficiency
            cursor = 0
            keys_to_delete = []
            while True:
                cursor, keys = client.scan(cursor, match=f"session:{user_id}*", count=100)
                for key in keys:
                    parts = key.split(":", 2)
                    if len(parts) == 3:  # Story-specific key: session:user_id:story_id
                        _, user_id, story_id = parts
                        story_data = APIBackend()._get_session(user_id, story_id)
                        if story_data and story_data.get("memory_system"):
                            story_data["memory_system"].cleanup()
                    keys_to_delete.append(key)
                if cursor == 0:
                    break
            if keys_to_delete:
                client.delete(*keys_to_delete)
                print(f"✅ Deleted {len(keys_to_delete)} keys for user {user_id}")
            else:
                print(f"🔍 No sessions found for user {user_id}")
            import gc
            gc.collect()
            return {"status": "success", "message": "Session cleared."}
        except redis.RedisError as e:
            print(f"❌ Redis error in logout: {e}")
            raise HTTPException(status_code=500, detail="Failed to clear session")

    @staticmethod
    @profile
    def cleanup_inactive_sessions(max_age_seconds: int = 3600):
        """Clean up inactive sessions from Redis using SCAN."""
        client = get_redis_client()
        try:
            current_time = time.time()
            removed = 0
            cursor = 0
            while True:
                cursor, keys = client.scan(cursor, match="session:*", count=100)
                for key in keys:
                    data = client.get(key)
                    if not data:
                        continue
                    session_data = json.loads(data)
                    if current_time - session_data.get("last_active", 0) > max_age_seconds:
                        user_id = key.split(":")[1]
                        APIBackend.logout(user_id)
                        removed += 1
                if cursor == 0:
                    break
            return removed
        except redis.RedisError as e:
            print(f"❌ Redis error in cleanup_inactive_sessions: {e}")
            raise HTTPException(status_code=500, detail="Failed to clean up sessions")
