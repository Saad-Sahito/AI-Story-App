import json
import asyncio
import time
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from dotenv import load_dotenv
from setup.shared_redis_pool import get_redis_client
from src.memory.memory_system import StoryMemorySystem
from src.story_engines.classic_narrative.agents.story_author import StoryAuthor
from src.story_engines.classic_narrative.agents.director_agent import DirectorGraph
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from typing import Optional
from asyncio import Lock
from contextlib import asynccontextmanager

SESSION_TTL = 3600  # 1 hour expiration for inactive sessions
BASE_SESSION_KEY = "classic_session"
load_dotenv()

@asynccontextmanager
async def redis_lock(client, lock_key, timeout=30, retries=10, retry_delay=1.0):
    lock_value = str(time.time())
    acquired = False
    try:
        for attempt in range(retries):
            try:
                acquired = await client.set(lock_key, lock_value, nx=True, ex=timeout)
                if acquired:
                    break
                ttl = await client.ttl(lock_key)
                if ttl == -1:
                    print(f"🔍 Detected stale lock with no TTL for {lock_key}, removing")
                    await client.delete(lock_key)
                elif ttl == -2:
                    print(f"🔍 Lock {lock_key} does not exist, retrying")
                else:
                    print(f"🔍 Lock acquisition failed for {lock_key} at {time.time()}, attempt {attempt + 1}/{retries}, TTL={ttl}")
                await asyncio.sleep(retry_delay)
            except redis.RedisError as e:
                print(f"❌ Redis error during lock acquisition for {lock_key}: {e}")
                await asyncio.sleep(retry_delay)
        if not acquired:
            raise HTTPException(status_code=503, detail=f"Could not acquire lock for {lock_key} after {retries} attempts")
        
        yield
        
    finally:
        if acquired:
            lua = """
            if redis.call("get", KEYS[1]) == ARGV[1] then
                return redis.call("del", KEYS[1])
            else
                return 0
            end
            """
            try:
                result = await client.eval(lua, 1, lock_key, lock_value)
                if result == 1:
                    pass
                else:
                    print(f"🔍 Lock {lock_key} not released: different lock value or already expired")
            except redis.RedisError as e:
                print(f"❌ Failed to release lock for {lock_key}: {e}")
                try:
                    current_value = await client.get(lock_key)
                    if current_value == lock_value.encode():
                        await client.delete(lock_key)
                        print(f"✅ Forcibly released stale lock for {lock_key}")
                    else:
                        print(f"🔍 Lock {lock_key} not forcibly released: different lock value")
                except redis.RedisError as e:
                    print(f"❌ Failed to forcibly release lock for {lock_key}: {e}")

class ClassicStorySetup:
    def __init__(self):
        if scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE is None:
            scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE = scene_planner_module.SharedScenePlannerService()
            print("Initialized classic shared scene planner service")

    async def _get_session(self, user_id: str, story_id: str = None):
        client = await get_redis_client()
        try:
            if story_id:
                key = f"{BASE_SESSION_KEY}:{user_id}:{story_id}"
                lock_key = f"lock:{key}"
                init_lock_key = f"init_lock:{key}"
                
                async with redis_lock(client, lock_key, timeout=30, retries=10, retry_delay=1.0):
                    data = await client.get(key)
                    if not data:
                        print(f"❌ No session found for {key}")
                        raise HTTPException(status_code=404, detail=f"No session found for {key}")
                    try:
                        session_data = json.loads(data)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error for {key}: {e}")
                        raise HTTPException(status_code=500, detail="Invalid session data format")
                    
                    if not session_data.get("memory_system_initialized", False):
                        print(f"🔍 Memory system not initialized for {key}, attempting initialization")
                        
                        async with redis_lock(client, init_lock_key, timeout=60, retries=20, retry_delay=2.0):
                            fresh_data = await client.get(key)
                            if fresh_data:
                                try:
                                    fresh_session = json.loads(fresh_data)
                                    if fresh_session.get("memory_system_initialized", False):
                                        session_data = fresh_session
                                        params = session_data['memory_system_params']
                                        session_data['memory_system'] = StoryMemorySystem(
                                            user_id=params['user_id'],
                                            story_id=params['story_id']
                                        )
                                        await session_data['memory_system'].qdrant_initialize()
                                        session_data['director'] = DirectorGraph(memory_system=session_data['memory_system'])
                                        session_data['user_input_queue'] = asyncio.Queue()
                                        async with client.pipeline() as pipe:
                                            pipe.expire(key, SESSION_TTL)
                                            await pipe.execute()
                                        return session_data
                                except json.JSONDecodeError:
                                    pass
                            
                            params = session_data['memory_system_params']
                            if not all(k in params for k in ['user_id', 'story_id']):
                                print(f"❌ Invalid memory_system_params: {params}")
                                raise HTTPException(status_code=400, detail="Invalid session data")
                            
                            print(f"🔄 Initializing memory system for {key}")
                            session_data['memory_system'] = StoryMemorySystem(
                                user_id=params['user_id'],
                                story_id=params['story_id']
                            )
                            
                            try:
                                max_retries = 3
                                for attempt in range(max_retries):
                                    try:
                                        await session_data['memory_system'].qdrant_initialize()
                                        print(f"✅ Qdrant initialized successfully for {key}")
                                        break
                                    except Exception as e:
                                        if attempt < max_retries - 1:
                                            print(f"⚠️ Qdrant initialization attempt {attempt + 1} failed: {e}, retrying...")
                                            await asyncio.sleep(2 ** attempt)
                                        else:
                                            print(f"❌ Qdrant initialization failed after {max_retries} attempts: {e}")
                                            raise
                            except Exception as e:
                                print(f"❌ Qdrant initialization error: {e}")
                                raise HTTPException(status_code=500, detail=f"Failed to initialize memory system: {str(e)}")
                            
                            session_data['director'] = DirectorGraph(memory_system=session_data['memory_system'])
                            session_data['user_input_queue'] = asyncio.Queue()
                            session_data["memory_system_initialized"] = True
                            
                            serializable_data = {
                                "memory_system_params": session_data.get("memory_system_params", {}),
                                "last_active": time.time(),
                                "memory_system_initialized": True,
                            }
                            async with client.pipeline() as pipe:
                                pipe.set(key, json.dumps(serializable_data))
                                pipe.expire(key, SESSION_TTL)
                                await pipe.execute()
                            print(f"✅ Memory system initialized and saved for {key}")
                    else:
                        params = session_data['memory_system_params']
                        session_data['memory_system'] = StoryMemorySystem(
                            user_id=params['user_id'],
                            story_id=params['story_id']
                        )
                        await session_data['memory_system'].qdrant_initialize()
                        session_data['director'] = DirectorGraph(memory_system=session_data['memory_system'])
                        session_data['user_input_queue'] = asyncio.Queue()
                    
                    async with client.pipeline() as pipe:
                        pipe.expire(key, SESSION_TTL)
                        await pipe.execute()
                    print(f"🔍 Retrieved story_session for {key}")
                    return session_data
            else:
                key = f"{BASE_SESSION_KEY}:{user_id}"
                lock_key = f"lock:{key}"
                async with redis_lock(client, lock_key, timeout=30, retries=10, retry_delay=1.0):
                    data = await client.get(key)
                    if not data:
                        print(f"No user session found for {key}")
                        return {"last_active": time.time(), "stories": {}}
                    try:
                        user_session = json.loads(data)
                    except json.JSONDecodeError as e:
                        print(f"❌ JSON decode error for {key}: {e}")
                        raise HTTPException(status_code=500, detail="Invalid session data format")
                    user_session["stories"] = {}
                    cursor = 0
                    while True:
                        cursor, story_keys = await client.scan(cursor, match=f"{BASE_SESSION_KEY}:{user_id}:*", count=100)
                        for story_key in story_keys:
                            story_id_from_key = story_key.split(":")[-1]
                            story_data = await client.get(story_key)
                            if story_data:
                                try:
                                    story_session = json.loads(story_data)
                                except json.JSONDecodeError as e:
                                    print(f"❌ JSON decode error for {story_key}: {e}")
                                    continue
                                
                                if not story_session.get("memory_system_initialized", False):
                                    init_lock_key = f"init_lock:{story_key}"
                                    try:
                                        async with redis_lock(client, init_lock_key, timeout=60, retries=20, retry_delay=2.0):
                                            fresh_story_data = await client.get(story_key)
                                            if fresh_story_data:
                                                fresh_story_session = json.loads(fresh_story_data)
                                                if fresh_story_session.get("memory_system_initialized", False):
                                                    story_session = fresh_story_session
                                                else:
                                                    params = story_session['memory_system_params']
                                                    if not all(k in params for k in ['user_id', 'story_id']):
                                                        print(f"❌ Invalid memory_system_params for {story_key}: {params}")
                                                        continue
                                                    story_session['memory_system'] = StoryMemorySystem(
                                                        user_id=params['user_id'],
                                                        story_id=params['story_id']
                                                    )
                                                    await story_session['memory_system'].qdrant_initialize()
                                                    story_session['director'] = DirectorGraph(memory_system=story_session['memory_system'])
                                                    story_session['user_input_queue'] = asyncio.Queue()
                                                    story_session["memory_system_initialized"] = True
                                                    
                                                    serializable_data = {
                                                        "memory_system_params": story_session.get("memory_system_params", {}),
                                                        "last_active": time.time(),
                                                        "memory_system_initialized": True,
                                                    }
                                                    await client.set(story_key, json.dumps(serializable_data))
                                    except Exception as e:
                                        print(f"⚠️ Failed to initialize memory for {story_key}: {e}")
                                        continue
                                else:
                                    params = story_session['memory_system_params']
                                    story_session['memory_system'] = StoryMemorySystem(
                                        user_id=params['user_id'],
                                        story_id=params['story_id']
                                    )
                                    await story_session['memory_system'].qdrant_initialize()
                                    story_session['director'] = DirectorGraph(memory_system=story_session['memory_system'])
                                    story_session['user_input_queue'] = asyncio.Queue()
                                
                                user_session["stories"][story_id_from_key] = story_session
                        if cursor == 0:
                            break
                    async with client.pipeline() as pipe:
                        pipe.expire(key, SESSION_TTL)
                        await pipe.execute()
                    print(f"🔍 Retrieved user_session for {user_id}")
                    return user_session
        except redis.RedisError as e:
            print(f"❌ Redis error in _get_session: {e}")
            raise HTTPException(status_code=500, detail="Failed to access session storage")
        except HTTPException:
            raise
        except Exception as e:
            print(f"❌ Unexpected error in _get_session: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Unexpected error in session retrieval: {str(e)}")

    async def _set_session(self, user_id: str, story_id: str = None, data: dict = None):
        client = await get_redis_client()
        try:
            key = f"{BASE_SESSION_KEY}:{user_id}:{story_id}" if story_id else f"{BASE_SESSION_KEY}:{user_id}"
            lock_key = f"lock:{key}"
            async with redis_lock(client, lock_key):
                serializable_data = data.copy() if data else {}
                if story_id:
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
                        serializable_data["memory_system_initialized"] = data.get("memory_system_initialized", True)
                    if 'director' in serializable_data:
                        del serializable_data['director']
                    if 'user_input_queue' in serializable_data:
                        del serializable_data['user_input_queue']
                else:
                    if 'stories' in serializable_data:
                        serializable_stories = {}
                        for sid, story_data in serializable_data['stories'].items():
                            story_copy = story_data.copy()
                            if 'memory_system' in story_copy:
                                if story_copy['memory_system'] is not None:
                                    try:
                                        story_copy['memory_system_params'] = {
                                            'user_id': story_copy['memory_system'].user_id,
                                            'story_id': story_copy['memory_system'].story_id
                                        }
                                    except AttributeError as e:
                                        print(f"❌ AttributeError in _set_session for story {sid} memory_system: {e}")
                                        story_copy['memory_system_params'] = {}
                                del story_copy['memory_system']
                            if 'director' in story_copy:
                                del story_copy['director']
                            if 'user_input_queue' in story_copy:
                                del story_copy['user_input_queue']
                            story_copy['memory_system_initialized'] = story_data.get("memory_system_initialized", True)
                            serializable_stories[sid] = story_copy
                        serializable_data['stories'] = serializable_stories
                print(f"🔍 Serializing data for key {key}: {serializable_data}")
                async with client.pipeline() as pipe:
                    pipe.set(key, json.dumps(serializable_data))
                    pipe.expire(key, SESSION_TTL)
                    await pipe.execute()
                print(f"✅ Writing to Redis key={key}, data={serializable_data}")
        except redis.RedisError as e:
            print(f"❌ Redis error in _set_session: {e}")
            raise HTTPException(status_code=500, detail="Failed to store session data")
        except TypeError as e:
            print(f"❌ Serialization error in _set_session: {e}")
            print(f"🔍 Problematic data: {serializable_data}")
            raise HTTPException(status_code=500, detail=f"Invalid session data format: {str(e)}")

    async def setup_user_session(self, user_id: str, story_id: str, memory_system=None):
        """Set up a user session with Redis."""
        client = await get_redis_client()
        user_key = f"{BASE_SESSION_KEY}:{user_id}"
        user_lock_key = f"lock:{user_key}"
        user_session = await self._get_session(user_id) or {"last_active": time.time(), "stories": {}}
        async with redis_lock(client, user_lock_key):
            story_session = {
                "memory_system_params": {
                    "user_id": user_id,
                    "story_id": story_id
                } if memory_system else {},
                "last_active": time.time(),
            }
            user_session["stories"][story_id] = story_session
        await self._set_session(user_id, data=user_session)
        await self._set_session(user_id, story_id, data=story_session)

    async def initialize_story(self, user_id: str, story_title: str = ""):
        """Initialize a new story and store session in Redis."""
        story_title_normalized = story_title.lower().replace(" ", "_").replace(":", "_")
        story_id = f"{story_title_normalized}_{user_id}"
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        await memory_system.qdrant_initialize()
        await self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
        return {"status": "success", "message": f"Story initialized for {user_id}", "story_id": story_id}

    async def continue_story(self, user_id: str, story_id: str) -> dict:
        """Continue an existing story, loading session from Redis."""
        client = await get_redis_client()
        user_key = f"{BASE_SESSION_KEY}:{user_id}"
        user_lock_key = f"lock:{user_key}"
        async with redis_lock(client, user_lock_key):
            if not user_id:
                raise HTTPException(status_code=403, detail="Please enter a valid User ID.")
            memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
            await memory_system.qdrant_initialize()
            story_progress_data = await memory_system.get_story_progress()
            if not story_progress_data:
                raise HTTPException(status_code=405, detail="No existing story found for this user and story ID.")
        await self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
        user_session = await self._get_session(user_id)
        story_text = (
            await user_session["stories"][story_id]['memory_system'].get_story_cluster(chapter_id=story_progress_data.get("latest_chapter_id"))
            or "No story text for this chapter found."
        )
        return {"status": "success", "message": f"Session started for {user_id} and {story_id}", "story_cluster": story_text}

    async def create_premise(self, initial_story_data: dict, model: str):
        """Create story premise and update session in Redis."""
        client = await get_redis_client()
        tone_dict = {
            0: "Playful", 20: "Lighthearted", 40: "Adventurous",
            60: "Dramatic", 80: "Serious", 100: "Intense"
        }
        tone_temp_dict = {
            0: 0.8, 20: 0.7, 40: 0.75,
            60: 0.7, 80: 0.65, 100: 0.75
        }
        length_dict = {
            0: "Short Long Story (7,500 - 15,000 words)",
            20: "Novelette (15,000 - 25,000 words)",
            40: "Novella (25,000 - 40,000 words)",
            60: "Novel Chapter (40,000 - 60,000 words)",
            80: "Full Novel (60,000 - 90,000 words)",
            100: "Epic / Series (90,000 - 150,000+ words)"
        }
        user_id = initial_story_data["user_id"]
        story_id = initial_story_data["story_id"]
        user_key = f"{BASE_SESSION_KEY}:{user_id}"
        user_lock_key = f"lock:{user_key}"
        user_data = await self._get_session(user_id=user_id)
        async with redis_lock(client, user_lock_key):
            if not user_data:
                raise HTTPException(status_code=403, detail="Invalid user ID")
            story_data = user_data["stories"].get(story_id)
            if not story_data:
                raise HTTPException(status_code=405, detail="Invalid story ID")
            story_author = StoryAuthor(memory_system=story_data['memory_system'])
            if "Tone" in initial_story_data:
                tone_value = int(initial_story_data["Tone"])
                mapped_tone = tone_dict.get(tone_value)
                mapped_tone_temp = tone_temp_dict.get(tone_value, 0.7)
                if mapped_tone is None:
                    mapped_tone = tone_dict[min(tone_dict.keys(), key=lambda k: abs(k - tone_value))]
                    mapped_tone_temp = tone_temp_dict.get(min(tone_temp_dict.keys(), key=lambda k: abs(k - tone_value)), 0.7)
                initial_story_data["Tone"] = mapped_tone
            if "Length" in initial_story_data:
                length_value = int(initial_story_data["Length"])
                mapped_length = length_dict.get(length_value)
                if mapped_length is None:
                    mapped_length = length_dict[min(length_dict.keys(), key=lambda k: abs(k - length_value))]
                initial_story_data["Length"] = mapped_length
            if "age" in initial_story_data:
                age = int(initial_story_data["age"])
            filtered_data = {k: v for k, v in initial_story_data.items() if k not in ["story_id", "user_id", "age"]}
            form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])
            form_string += f"\nThe target audience is {age} years old."
            tokens, blurb, image_data = await story_author.set_story_premise(
                form_string,
                initial_story_data.get("Title", ""),
                model=model
            )
            del story_author
            await story_data["memory_system"].update_story_progress(
                metadata={
                    "latest_chapter_id": 1,
                    "continue_scene_id": 1,
                    "story_title": initial_story_data["Title"],
                    "word_count": 0,
                    "tone_temp": mapped_tone_temp,
                    "model": model,
                    "token_usage": tokens,
                    "blurb": blurb,
                    "image_data": image_data
                }
            )
            serializable_story_data = {
                "memory_system_params": story_data.get("memory_system_params", {}),
                "last_active": time.time(),
            }
            user_data["stories"][story_id] = serializable_story_data
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        from src.memory.user_management import append_story
        await append_story(user_id=user_id, story_title=initial_story_data.get("Title", ""), story_id=story_id, story_type=initial_story_data.get("story_type", ""))
        return {"status": "success", "blurb": blurb, "image_data": image_data}

    async def handle_story_websocket(self, websocket: WebSocket, user_id: str, story_id: str):
        await websocket.accept()
        client = await get_redis_client()
        ws_key = f"{BASE_SESSION_KEY}_active_ws:{user_id}:{story_id}"
        director_key = f"director_running:{user_id}:{story_id}"
        SESSION_TTL = 3600

        disconnect_event = asyncio.Event()

        try:
            async with redis_lock(client, f"lock:{ws_key}", timeout=30, retries=10, retry_delay=1.0):
                if await client.get(ws_key):
                    await websocket.send_json({"error": "Another connection is active for this story"})
                    await websocket.close(code=1000)
                    return
                await client.set(ws_key, "1", ex=SESSION_TTL)

            try:
                init_data = await asyncio.wait_for(websocket.receive_json(), timeout=10.0)
            except asyncio.TimeoutError:
                await websocket.send_json({"error": "Timeout waiting for initial data"})
                await websocket.close(code=1001)
                return

            received_user_id = init_data.get("user_id")
            received_story_id = init_data.get("story_id")
            if not received_user_id or not received_story_id or received_user_id != user_id or received_story_id != story_id:
                await websocket.send_json({"error": "Invalid or mismatched user_id/story_id"})
                await websocket.close(code=1000)
                return

            user_key = f"{BASE_SESSION_KEY}:{user_id}"
            user_data = await self._get_session(user_id)
            async with redis_lock(client, f"lock:{user_key}", timeout=30, retries=10, retry_delay=1.0):
                if not user_data:
                    await websocket.send_json({"error": "Invalid user ID"})
                    await websocket.close(code=1000)
                    return
                story_data = user_data["stories"].get(story_id)
                if not story_data:
                    await websocket.send_json({"error": "Invalid story ID"})
                    await websocket.close(code=1000)
                    return

                async with redis_lock(client, f"lock:{director_key}", timeout=30, retries=10, retry_delay=1.0):
                    if await client.get(director_key):
                        await websocket.send_json({"error": "Story progression already active"})
                        await websocket.close(code=1000)
                        return
                    await client.set(director_key, "1", ex=SESSION_TTL)

                if not story_data.get("memory_system_initialized", False):
                    await story_data["memory_system"].qdrant_initialize()
                    story_data["memory_system_initialized"] = True
                    story_data["director"] = DirectorGraph(memory_system=story_data["memory_system"])

                    story_key = f"{BASE_SESSION_KEY}:{user_id}:{story_id}"
                    serializable_story_data = {
                        "memory_system_params": story_data.get("memory_system_params", {}),
                        "last_active": time.time(),
                        "memory_system_initialized": True,
                    }
                    user_data["stories"][story_id] = serializable_story_data
                    async with client.pipeline() as pipe:
                        pipe.set(story_key, json.dumps(serializable_story_data))
                        pipe.expire(story_key, SESSION_TTL)
                        pipe.set(user_key, json.dumps(user_data))
                        pipe.expire(user_key, SESSION_TTL)
                        await pipe.execute()

            queue = asyncio.Queue()

            def scene_chunk_callback(chunk: dict):
                queue.put_nowait(chunk)

            async def refresh_ttl_loop():
                try:
                    while not disconnect_event.is_set():
                        await asyncio.sleep(30)
                        async with client.pipeline() as pipe:
                            pipe.expire(user_key, SESSION_TTL)
                            pipe.expire(f"{BASE_SESSION_KEY}:{user_id}:{story_id}", SESSION_TTL)
                            await pipe.execute()
                except asyncio.CancelledError:
                    return

            async def send_loop():
                try:
                    while not disconnect_event.is_set():
                        item = await queue.get()
                        if item is None:
                            break
                        await websocket.send_json(item)
                except asyncio.CancelledError:
                    return
                except Exception as e:
                    print(f"❌ send_loop error: {e}")

            async def recv_loop():
                try:
                    while not disconnect_event.is_set():
                        try:
                            msg = await websocket.receive_json()
                        except asyncio.TimeoutError:
                            continue
                        except WebSocketDisconnect:
                            print("❌ Client disconnected during recv_loop")
                            disconnect_event.set()
                            break
                        if "choice" in msg:
                            choice = msg["choice"].strip()
                            if not choice:
                                continue
                            queue_key = f"input_queue:{user_id}:{story_id}"
                            async with redis_lock(client, f"lock:{queue_key}"):
                                async with client.pipeline() as pipe:
                                    pipe.rpush(queue_key, choice)
                                    pipe.expire(queue_key, SESSION_TTL)
                                    await pipe.execute()
                        elif "continue_chapter" in msg:
                            choice = msg["continue_chapter"]
                            if choice is None:
                                continue
                            if isinstance(choice, str) and choice.isdigit():
                                choice = int(choice)
                            queue_key = f"continue_input_queue:{user_id}:{story_id}"
                            async with redis_lock(client, f"lock:{queue_key}"):
                                async with client.pipeline() as pipe:
                                    pipe.rpush(queue_key, choice)
                                    pipe.expire(queue_key, SESSION_TTL)
                                    await pipe.execute()
                except Exception as e:
                    print(f"❌ recv_loop crashed: {e}")
                    disconnect_event.set()
                except asyncio.CancelledError:
                    return

            async def run_director():
                try:
                    await story_data["director"].run(
                        scene_chunk_callback=scene_chunk_callback,
                        stop_event=disconnect_event
                    )
                    await queue.put({"chapter_complete": True})
                except Exception as e:
                    await queue.put({"error": f"Director failed: {str(e)}"})
                finally:
                    await queue.put(None)

            director_task = asyncio.create_task(run_director())
            send_task = asyncio.create_task(send_loop())
            recv_task = asyncio.create_task(recv_loop())
            ttl_task = asyncio.create_task(refresh_ttl_loop())

            try:
                done, pending = await asyncio.wait(
                    [director_task, send_task, recv_task, ttl_task],
                    return_when=asyncio.FIRST_COMPLETED
                )
                for finished in done:
                    print(f"✅ Task completed first: {finished.get_coro().__name__}")
                disconnect_event.set()
            finally:
                for t in [director_task, send_task, recv_task, ttl_task]:
                    t.cancel()

        except Exception as e:
            print(f"❌ Error in handle_story_websocket: {e}")
            import traceback; traceback.print_exc()
            try:
                await websocket.send_json({"error": f"Server error: {str(e)}"})
            except Exception:
                pass
        finally:
            try:
                async with redis_lock(client, f"lock:{ws_key}"):
                    await client.delete(ws_key)
                async with redis_lock(client, f"lock:{director_key}"):
                    await client.delete(director_key)
            finally:
                await websocket.close(code=1000)

    async def get_story_cluster(self, user_id: str, story_id: str, chapter_number: int):
        user_data = await self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")
        try:
            return {"status": "success", "data": await story_data['memory_system'].get_story_cluster(chapter_id=chapter_number)}
        except:
            return {"status": "error"}

    async def get_story_progress_for_user(self, user_id: str, story_id: str) -> dict:
        user_data = await self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=403, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=405, detail="Invalid story ID")
        try:
            return {"status": "success", "data": await story_data['memory_system'].get_story_progress()}
        except:
            return {"status": "error"}
    
    async def close(self):
        try:
            from setup.shared_redis_pool import REDIS_POOL
            if REDIS_POOL:
                await REDIS_POOL.disconnect()
                print("✅ Redis pool closed")
            if hasattr(self, "_active_memory_systems"):
                for ms in self._active_memory_systems:
                    try:
                        await ms.close()
                    except Exception as e:
                        print(f"⚠️ Failed to close memory system: {e}")
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

    async def logout(self, user_id: str):
        client = await get_redis_client()
        try:
            user_key = f"{BASE_SESSION_KEY}:{user_id}"
            lock_key = f"lock:{user_key}"
            async with redis_lock(client, lock_key):
                cursor = 0
                story_keys = []
                while True:
                    cursor, keys = await client.scan(cursor, match=f"{BASE_SESSION_KEY}:{user_id}:*", count=100)
                    story_keys.extend(keys)
                    if cursor == 0:
                        break
                for story_key in story_keys:
                    parts = story_key.split(":", 2)
                    if len(parts) == 3:
                        story_id = parts[2]
                        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                        await memory_system.cleanup()
                keys_to_delete = story_keys + [user_key] if await client.exists(user_key) else story_keys
                if keys_to_delete:
                    await client.delete(*keys_to_delete)
                    print(f"✅ Deleted {len(keys_to_delete)} keys for user {user_id}")
                else:
                    print(f"🔍 No sessions found for user {user_id}")
            import gc
            gc.collect()
            return {"status": "success", "message": "Session cleared."}
        except redis.RedisError as e:
            print(f"❌ Redis error in logout: {e}")
            raise HTTPException(status_code=500, detail="Failed to clear session")
        
    async def logout_story(self, user_id: str, story_id: str):
        client = await get_redis_client()
        try:
            user_key = f"{BASE_SESSION_KEY}:{user_id}"
            user_lock_key = f"lock:{user_key}"
            story_key = f"{BASE_SESSION_KEY}:{user_id}:{story_id}"
            async with redis_lock(client, user_lock_key):
                memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                await memory_system.cleanup()
                if await client.exists(story_key):
                    await client.delete(story_key)
                else:
                    print(f"🔍 No story session found for {story_key}")
                data = await client.get(user_key)
                if data:
                    user_session = json.loads(data)
                    if "stories" in user_session and story_id in user_session["stories"]:
                        del user_session["stories"][story_id]
                    async with client.pipeline() as pipe:
                        pipe.set(user_key, json.dumps(user_session))
                        pipe.expire(user_key, SESSION_TTL)
                        await pipe.execute()
                    print(f"✅ Removed story {story_id} from user session {user_id}")
            import gc
            gc.collect()
            return {"status": "success", "message": f"Story {story_id} session cleared for user {user_id}."}
        except redis.RedisError as e:
            print(f"❌ Redis error in logout_story: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear story session for {story_id}")
        except Exception as e:
            print(f"❌ Unexpected error in logout_story: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Failed to clear story session: {str(e)}")

    async def cleanup_inactive_sessions(self, max_age_seconds: int = 3600):
        client = await get_redis_client()
        try:
            current_time = time.time()
            removed = 0
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=f"{BASE_SESSION_KEY}:*", count=100)
                for key in keys:
                    data = await client.get(key)
                    if not data:
                        continue
                    session_data = json.loads(data)
                    if current_time - session_data.get("last_active", 0) > max_age_seconds:
                        async with redis_lock(client, f"lock:{key}"):
                            fresh_data = await client.get(key)
                            if fresh_data:
                                fresh_session = json.loads(fresh_data)
                                if current_time - fresh_session.get("last_active", 0) > max_age_seconds:
                                    parts = key.split(":", 2)
                                    user_id = parts[1]
                                    if len(parts) == 3:
                                        story_id = parts[2]
                                        await self.logout_story(user_id, story_id)
                                        removed += 1
                                    elif len(parts) == 2:
                                        await self.logout(user_id)
                                        removed += 1
                if cursor == 0:
                    break
            return removed
        except redis.RedisError as e:
            print(f"❌ Redis error in cleanup_inactive_sessions: {e}")
            raise HTTPException(status_code=500, detail="Failed to clean up sessions")

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