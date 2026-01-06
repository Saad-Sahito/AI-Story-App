# src/setup/main_setup.py
import time
from dotenv import load_dotenv
from fastapi import WebSocket
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect, HTTPException
import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from src.utilities.story_helpers import StoryHelpers
from src.setup.shared_redis_pool import get_redis_client
from src.plot_engine.story_author import StoryAuthor
from src.plot_engine.basic_author import SnowflakeWorkflow
from src.memory.memory_system import StoryMemorySystem
from contextlib import asynccontextmanager



load_dotenv()
BASE_SESSION_KEY = "Base"
SESSION_TTL = 3600  # 1 hour expiration for inactive sessions

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

class MainSetup:

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
                                        session_data['author'] = StoryAuthor()
                                        session_data['basic_author'] = SnowflakeWorkflow()
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
                                        # print(f"✅ Qdrant initialized successfully for {key}")
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
                            
                            session_data['author'] = StoryAuthor()
                            session_data['basic_author'] = SnowflakeWorkflow()
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
                        session_data['author'] = StoryAuthor()
                        session_data['basic_author'] = SnowflakeWorkflow()
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
                                                    story_session['author'] = StoryAuthor()
                                                    story_session['basic_author'] = SnowflakeWorkflow()
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
                                    story_session['author'] = StoryAuthor()
                                    story_session['basic_author'] = SnowflakeWorkflow()
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
                    if 'author' in serializable_data:
                        del serializable_data['author']
                    if 'basic_author' in serializable_data:
                        del serializable_data['basic_author']
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
                            if 'author' in story_copy:
                                del story_copy['author']
                            if 'basic_author' in story_copy:
                                del story_copy['basic_author']
                            if 'user_input_queue' in story_copy:
                                del story_copy['user_input_queue']
                            story_copy['memory_system_initialized'] = story_data.get("memory_system_initialized", True)
                            serializable_stories[sid] = story_copy
                        serializable_data['stories'] = serializable_stories
                # print(f"🔍 Serializing data for key {key}: {serializable_data}")
                async with client.pipeline() as pipe:
                    pipe.set(key, json.dumps(serializable_data))
                    pipe.expire(key, SESSION_TTL)
                    await pipe.execute()
                # print(f"✅ Writing to Redis key={key}, data={serializable_data}")
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

    async def continue_story(self, user_id: str, story_id: str):
        """Continue an existing story, loading session from Redis."""
        client = await get_redis_client()
        user_key = f"{BASE_SESSION_KEY}:{user_id}"
        user_lock_key = f"lock:{user_key}"
        async with redis_lock(client, user_lock_key):
            if not user_id:
                raise HTTPException(status_code=433, detail="Please enter a valid User ID.")
            memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
            await memory_system.qdrant_initialize()
            # story_progress_data = await memory_system.get_story_progress()
            # if not story_progress_data:
            #     raise HTTPException(status_code=455, detail="No existing story found for this user and story ID.")
        await self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
        # user_session = await self._get_session(user_id)
        # story_text = (
        #     await user_session["stories"][story_id]['memory_system'].get_story_cluster(chapter_id=story_progress_data.get("latest_chapter_id"))
        #     or "No story text for this chapter found."
        # )
        return {"status": "success", "message": f"Session continued for {user_id} and {story_id}"}
    
    async def initialize_story(self, user_id: str):
        """Initialize a new story and store session in Redis."""
        timestamp = int(time.time())
        story_id = f"{user_id}_{timestamp}"
        memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
        await memory_system.qdrant_initialize()
        await self.setup_user_session(user_id=user_id, story_id=story_id, memory_system=memory_system)
        return {"status": "success", "message": f"Story initialized for user: {user_id}, story: {story_id}", "story_id": story_id}
    

#---------------------------------------------Story Author Calls------------------------------------------------------------
    # Phase 1
    async def add_story_seed(self, user_id: str, story_id: str, story_seed: dict):
        """
        Create story seed Phase 1.
        """

        # Extract metadata
        target_medium = story_seed.get("target_medium", "novel")
        target_length = story_seed.get("target_length")
        if target_length < 40000:
            flow_type = "Compact"
        elif target_length >= 40000 and target_length <=80000:
            flow_type = "Standard"
        else:
            flow_type = "Epic"
        # Get user session - _get_session handles its own locking
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        # Clean user context (remove system fields)
        user_context = {
            k: v for k, v in story_seed.items() 
            if k not in ["story_id", "user_id"]
        }

        story_seed = await story_data["author"].create_minimal_seed(
            user_context=user_context
        )
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(story_seed.model_dump()),
            metadata={"type": "story_seed", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 1,
                "story_title": story_seed.title,
                "total_acts": story_seed.act_count,
                "author_token_usage": None,
                "utility_token_usage": None,
                "flow_type": flow_type,
                "target_medium": target_medium,
                "target_length": story_seed.target_length,
                "pov": story_seed.pov,
                "prose_style": story_seed.prose_style,
                "tense": "past",
                "genre": story_seed.genre,
                "sub_genre": story_seed.sub_genre,
                "themes": story_seed.themes,
                "min_age": story_seed.target_audience_age,
                "status": "Ongoing",
                "author_type": "Advanced"
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
        await append_story(
            user_id=user_id,
            story_id=story_id
        )
        
        return {
            "status": "success", "data": story_seed
        }

    # Phase 2
    async def generate_world_foundation(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)

        world_foundation, world_tokens, utility_tokens = await story_data['author'].world_builder.generate_world_foundation(seed)
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(world_foundation.model_dump()),
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 2,
                "author_token_usage": world_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": world_foundation
        }
    
    # Compact Flow Only (Phase 3)
    async def generate_compact_plot(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)

        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)

        compact_plot, plot_tokens, utility_tokens = await story_data['author']._generate_compact_plot(
            seed, world_foundation
        )
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(compact_plot.model_dump()),
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 3,
                "author_token_usage": plot_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": world_foundation
        }
    
    # Phase 4
    async def generate_narrative_agents(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        if not minimal_plot_str:
            minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        compact_plot = json.loads(minimal_plot_str)
        narrative_agents, agent_tokens, utility_tokens = await story_data['author'].agent_genesis.generate_narrative_agents(
            seed, world_foundation, compact_plot['required_character_roles']
        )
        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json([a.model_dump() for a in narrative_agents]),
            metadata={"type": "narrative_agents", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 4,
                "author_token_usage": agent_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": narrative_agents
        }
    
    # Compact Flow Only (Phase 5)
    async def generate_simplified_conflict(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)
        compact_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        compact_plot = json.loads(compact_plot_str)
        narrative_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "narrative_agents", "story_id": story_id}
        )
        narrative_agents = json.loads(narrative_agents_str)

        conflict_matrix, conflict_tokens, utility_tokens = await story_data['author']._generate_simplified_conflict(
            compact_plot, seed, narrative_agents, world_foundation
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(conflict_matrix.model_dump()),
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 5,
                "author_token_usage": conflict_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": conflict_matrix
        }
    
    # phase 6
    async def connect_agents_to_plot(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        compact_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        compact_plot = json.loads(compact_plot_str)
        narrative_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "narrative_agents", "story_id": story_id}
        )
        narrative_agents = json.loads(narrative_agents_str)
        conflict_matrix = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )

        connected_agents, agents_tokens, utility_tokens = await story_data['author'].agent_genesis.connect_agents_to_plot(
            narrative_agents, conflict_matrix, compact_plot
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json([a.model_dump() for a in connected_agents]),
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 6,
                "author_token_usage": agents_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": connected_agents
        }
    
    # Phase 7
    async def connect_world_to_conflict(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        conflict_matrix = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )

        integrated_world, integrate_tokens, utility_tokens = await story_data['author'].world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(integrated_world.model_dump()),
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 7,
                "author_token_usage": integrate_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": connected_agents
        }
    
    # For Standard/Epic Flow (Phase 3)
    async def generate_minimal_plot_outline(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)

        minimal_plot, plot_tokens, utility_tokens = await story_data['author'].generate_minimal_plot_outline(
            seed, world_foundation
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(minimal_plot.model_dump()),
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 3,
                "author_token_usage": plot_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": minimal_plot
        }

    # For Standard/Epic Flow (Phase 5)
    async def generate_conflict_layers(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        world_foundation_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        world_foundation = json.loads(world_foundation_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)
        narrative_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "narrative_agents", "story_id": story_id}
        )
        narrative_agents = json.loads(narrative_agents_str)

        conflict_matrix, conflict_tokens, utility_tokens = await story_data['author'].conflict_architect.generate_conflict_layers(
            minimal_plot, seed, narrative_agents, world_foundation
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(conflict_matrix.model_dump()),
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 5,
                "author_token_usage": conflict_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": conflict_matrix
        }

    # For Standard Flow Only (Phase 8)
    async def expand_plot_outline(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        conflict_matrix_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        conflict_matrix = json.loads(conflict_matrix_str)
        integrated_world_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        integrated_world = json.loads(integrated_world_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)

        expanded_plot, expand_tokens, utility_tokens = await story_data['author'].expand_plot_outline(
            minimal_plot, connected_agents, integrated_world,
            conflict_matrix, seed
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(expanded_plot.model_dump()),
            metadata={"type": "expanded_plot", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 8,
                "author_token_usage": expand_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": expanded_plot
        }
    
    # For Epic Flow Only (Phase 8)
    async def generate_character_backstories(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        integrated_world_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        integrated_world = json.loads(integrated_world_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)

        backstories, backstory_tokens, utility_tokens = await story_data['author'].agent_genesis.generate_character_backstories(
            connected_agents=connected_agents,
            world=integrated_world,
            plot=minimal_plot,
            seed=seed,
            importance_threshold="supporting"
        )

        # Merge backstories into connected_agents
        # backstory_dict = {b['name']: b for b in backstories}
        # enriched_agents = []
        # for agent in connected_agents:
        #     agent_dict = agent.model_dump() if hasattr(agent, 'model_dump') else agent
        #     if agent_dict["name"] in backstory_dict:
        #         agent_dict["detailed_backstory"] = backstory_dict[agent_dict["name"]].model_dump()
        #     enriched_agents.append(agent_dict)

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json([b.model_dump() for b in backstories]),
            metadata={"type": "character_backstories", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 8,
                "author_token_usage": backstory_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": backstories
        }
    
    # For Epic Flow Only (Phase 9)
    async def expand_world_detail(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        integrated_world_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        integrated_world = json.loads(integrated_world_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)

        world_guide, world_guide_tokens, utility_tokens = await story_data['author'].world_builder.expand_world_detail(
            integrated_world=integrated_world,
            plot=minimal_plot,
            seed=seed,
            connected_agents=connected_agents
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(world_guide.model_dump()),
            metadata={"type": "world_guide", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 9,
                "author_token_usage": world_guide_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": world_guide
        }
    
    # For Epic Flow Only (Phase 10)
    async def generate_subplot_architecture(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)

        subplot_arch, subplot_tokens, utility_tokens = await story_data['author'].generate_subplot_architecture(
            plot=minimal_plot,
            connected_agents=connected_agents,
            seed=seed,
            act_count=seed['act_count']
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(subplot_arch.model_dump()),
            metadata={"type": "subplot_architecture", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 10,
                "author_token_usage": subplot_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": subplot_arch
        }
    
    # For Epic Flow Only (Phase 11)
    async def expand_plot_with_enhancements(self, user_id: str, story_id: str):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        conflict_matrix_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        conflict_matrix = json.loads(conflict_matrix_str)
        integrated_world_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        integrated_world = json.loads(integrated_world_str)
        minimal_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        minimal_plot = json.loads(minimal_plot_str)
        backstories_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "character_backstories", "story_id": story_id}
        )
        backstories = json.loads(backstories_str)
        world_guide_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "world_guide", "story_id": story_id}
        )
        world_guide = json.loads(world_guide_str)
        subplot_arch_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "subplot_architecture", "story_id": story_id}
        )
        subplot_arch = json.loads(subplot_arch_str)

        expanded_plot, expand_tokens, utility_tokens = await story_data['author'].expand_plot_with_enhancements(
            minimal_plot=minimal_plot,
            connected_agents=connected_agents,
            integrated_world=integrated_world,
            conflict_matrix=conflict_matrix,
            seed=seed,
            backstories=backstories,
            world_guide=world_guide,
            subplot_architecture=subplot_arch
        )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(expanded_plot.model_dump()),
            metadata={"type": "expanded_plot", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 11,
                "author_token_usage": expand_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": expanded_plot
        }

    # For Standard/Epic Flow Only
    async def validate_story_elements(self, user_id: str, story_id: str, last_phase: int):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        conflict_matrix = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        integrated_world_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        integrated_world = json.loads(integrated_world_str)
        expanded_plot_str = await story_data['author'].get_long_term_document(
            metadata={"type": "expanded_plot", "story_id": story_id}
        )
        expanded_plot = json.loads(expanded_plot_str)

        quality_report, quality_tokens, utility_tokens = await story_data['author'].validate_story_elements(
            seed, expanded_plot, connected_agents, integrated_world, conflict_matrix
        )
            
        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(quality_report.model_dump()),
            metadata={"type": "quality_report", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "last_phase": last_phase+1,
                "author_token_usage": quality_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": quality_report
        }

    async def create_story_tracker(self, user_id: str, story_id: str, last_phase: int):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        seed_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "story_seed", "story_id": story_id}
        )
        seed = json.loads(seed_str)
        compact_plot_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        compact_plot = json.loads(compact_plot_str)
        connected_agents_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        connected_agents = json.loads(connected_agents_str)
        conflict_matrix = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )

        story_tracker, tracker_tokens, utility_tokens = await story_data['author']._create_story_tracker(
            act_count=seed['act_count'],
            final_plot=compact_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix
            )

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(story_tracker.model_dump()),
            metadata={"type": "story_tracker", "story_id": story_id}
        )
        await story_data["memory_system"].update_story_progress(
            metadata={
                "last_phase": last_phase+1,
                "author_token_usage": tracker_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        return {
            "status": "success", "data": story_tracker
        }

    #------------------------Change Phases Methods---------------------------
    # Phase 1
    async def change_add_story_seed(self, user_id: str, story_id: str, story_seed: dict):
        """
        Create story seed, blurb, and cover image at story initialization.
        Works for BOTH classic and interactive stories.
        """

        # Extract metadata
        target_medium = story_seed.get("target_medium", "novel")
        target_length = story_seed.get("target_length")
        if target_length < 40000:
            flow_type = "Compact"
        elif target_length >= 40000 and target_length <=80000:
            flow_type = "Standard"
        else:
            flow_type = "Epic"
        # Get user session - _get_session handles its own locking
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(story_seed)),
            metadata={"type": "story_seed", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "story_title": story_seed['title'],
                "total_acts": story_seed['act_count'],
                "flow_type": flow_type,
                "target_medium": target_medium,
                "target_length": story_seed['target_length'],
                "pov": story_seed['pov'],
                "prose_style": story_seed['prose_style'],
                "tense": "past",
                "genre": story_seed['genre'],
                "sub_genre": story_seed['sub_genre'],
                "themes": story_seed['themes'],
                "min_age": story_seed['target_audience_age']
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success"
        }

    # Phase 2
    async def change_generate_world_foundation(self, user_id: str, story_id: str, world_foundation: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(world_foundation),
            metadata={"type": "world_foundation", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)

        return {
            "status": "success"
        }
    
    # Compact Flow Only (Phase 3)
    async def change_generate_compact_plot(self, user_id: str, story_id: str, compact_plot: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(compact_plot),
            metadata={"type": "compact_plot", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)

        return {
            "status": "success"
        }
    
    # Phase 4
    async def change_generate_narrative_agents(self, user_id: str, story_id: str, narrative_agents: List[dict]):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json([json.dumps(a) for a in narrative_agents]),
            metadata={"type": "narrative_agents", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)

        return {
            "status": "success"
        }
    
    # Compact Flow Only (Phase 5)
    async def change_generate_simplified_conflict(self, user_id: str, story_id: str, conflict_matrix: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
       
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(conflict_matrix)),
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)

        return {
            "status": "success"
        }
    
    # phase 6
    async def change_connect_agents_to_plot(self, user_id: str, story_id: str, connected_agents: List[dict]):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json([json.dumps(a) for a in connected_agents]),
            metadata={"type": "connected_agents", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # Phase 7
    async def change_connect_world_to_conflict(self, user_id: str, story_id: str, integrated_world: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(integrated_world)),
            metadata={"type": "integrated_world", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # For Standard/Epic Flow (Phase 3)
    async def change_generate_minimal_plot_outline(self, user_id: str, story_id: str, minimal_plot: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(minimal_plot)),
            metadata={"type": "minimal_plot", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }

    # For Standard/Epic Flow (Phase 5)
    async def change_generate_conflict_layers(self, user_id: str, story_id: str, conflict_matrix: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(conflict_matrix)),
            metadata={"type": "conflict_matrix", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }

    # For Standard Flow Only (Phase 8)
    async def change_expand_plot_outline(self, user_id: str, story_id: str, expanded_plot: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(expanded_plot)),
            metadata={"type": "expanded_plot", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # For Epic Flow Only (Phase 8)
    async def change_generate_character_backstories(self, user_id: str, story_id: str, backstories: List[dict]):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json([json.dumps(b) for b in backstories]),
            metadata={"type": "character_backstories", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # For Epic Flow Only (Phase 9)
    async def change_expand_world_detail(self, user_id: str, story_id: str, world_guide: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(world_guide)),
            metadata={"type": "world_guide", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # For Epic Flow Only (Phase 10)
    async def change_generate_subplot_architecture(self, user_id: str, story_id: str, subplot_arch: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(subplot_arch)),
            metadata={"type": "subplot_architecture", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }
    
    # For Epic Flow Only (Phase 11)
    async def change_expand_plot_with_enhancements(self, user_id: str, story_id: str, expanded_plot: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(expanded_plot)),
            metadata={"type": "expanded_plot", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }

    async def change_create_story_tracker(self, user_id: str, story_id: str, story_tracker: dict):
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        progress = await story_data["memory_system"].get_story_progress()

        await story_data['memory_system'].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(story_tracker)),
            metadata={"type": "story_tracker", "story_id": story_id}
        )
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        return {
            "status": "success"
        }



#---------------------------------------------Basic Author Calls------------------------------------------------------------
    # Phase 1
    async def step_1_one_sentence(self, user_id: str, story_id: str, user_context: str, target_medium: str):
        """
        Create single sentence describing the story, Phase 1.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")

        one_sentence_form, author_tokens, utility_tokens= await story_data["basic_author"].step_1_one_sentence(
            user_context, target_medium
        )
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(one_sentence_form.model_dump()),
            metadata={"type": "step_1_one_sentence", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 1,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
                "target_medium": target_medium,
                "genre": one_sentence_form.genre,
                "sub_genre": one_sentence_form.sub_genre,
                "themes": one_sentence_form.themes,
                "status": "Ongoing",
                "author_type": "Basic",
                "flow_type": "Not relevant"
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
        await append_story(
            user_id=user_id,
            story_id=story_id
        )
        
        return {
            "status": "success", "data": one_sentence_form
        }
    
    # Phase 2
    async def step_2_one_paragraph(self, user_id: str, story_id: str, tone: str):
        """
        Create single paragraph describing the story, Phase 2.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        one_sentence_form_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_1_one_sentence", "story_id": story_id}
        )
        one_sentence_form = json.loads(one_sentence_form_str)
        one_sentence_form['tone'] = tone
        one_paragraph_form, author_tokens, utility_tokens = await story_data["basic_author"].step_2_one_paragraph(
            one_sentence_context=one_sentence_form
        )
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(one_paragraph_form.model_dump()),
            metadata={"type": "step_2_one_paragraph", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 2,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
                "target_medium": one_paragraph_form.target_medium,
                "genre": one_paragraph_form.genre,
                "sub_genre": one_paragraph_form.sub_genre,
                "themes": one_paragraph_form.themes,
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": one_paragraph_form
        }
    
    # Phase 3
    async def step_3_character_summaries(self, user_id: str, story_id: str, target_audience_age: int, target_length: int):
        """
        Create single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        one_paragraph_form_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_2_one_paragraph", "story_id": story_id}
        )
        one_paragraph_form = json.loads(one_paragraph_form_str)
        one_paragraph_form['target_audience_age'] = target_audience_age
        one_paragraph_form['target_length'] = target_length
        one_page_form, author_tokens, utility_tokens = await story_data["basic_author"].step_3_character_summaries(
            one_paragraph_context=one_paragraph_form
        )

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(one_page_form.model_dump()),
            metadata={"type": "step_3_character_summaries", "story_id": story_id}
        )
        story_title = one_page_form.title if one_page_form.title else 'untitled'
        print(story_title)
        print(target_length)
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 3,
                "story_title": story_title,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
                "min_age": target_audience_age,
                "target_length": target_length,
            }
        )

        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": one_page_form
        }

    # Phase 4
    async def step_4_one_page_plot(self, user_id: str, story_id: str, target_audience_age: int, target_length: int):
        """
        Create single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        one_paragraph_form_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_2_one_paragraph", "story_id": story_id}
        )
        one_paragraph_form = json.loads(one_paragraph_form_str)
        one_paragraph_form['target_audience_age'] = target_audience_age
        one_paragraph_form['target_length'] = target_length
        one_page_form, author_tokens, utility_tokens = await story_data["basic_author"].step_4_one_page_plot(
            one_paragraph_context=one_paragraph_form
        )

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(one_page_form.model_dump()),
            metadata={"type": "step_4_one_page_plot", "story_id": story_id}
        )
        story_title = one_page_form.title if one_page_form.title else 'untitled'
        print(story_title)
        print(target_length)
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 4,
                "story_title": story_title,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
                "target_medium": one_page_form.target_medium,
                "genre": one_page_form.genre,
                "sub_genre": one_page_form.sub_genre,
                "themes": one_page_form.themes,
                "total_acts": one_page_form.act_count   
            }
        )

        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": one_page_form
        }

    # Phase 5
    async def step_5_four_page_outline(self, user_id: str, story_id: str):
        """
        Create single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        one_page_form_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_4_one_page_plot", "story_id": story_id}
        )
        one_page_form = json.loads(one_page_form_str)
        character_summaries_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_3_character_summaries", "story_id": story_id}
        )
        character_summaries = json.loads(character_summaries_str)

        four_page_form, author_tokens, utility_tokens = await story_data["basic_author"].step_5_four_page_outline(
            one_page_context=one_page_form,
            characters=character_summaries
        )

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(four_page_form)),
            metadata={"type": "step_5_four_page_outline", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 5,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": four_page_form
        } 
    
    # Phase 6
    async def step_6_character_charts(self, user_id: str, story_id: str):
        """
        Create single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        four_page_form_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_5_four_page_outline", "story_id": story_id}
        )
        four_page_form = json.loads(four_page_form_str)
        character_summaries_str = await story_data["memory_system"].get_long_term_document(
            metadata={"type": "step_3_character_summaries", "story_id": story_id}
        )
        character_summaries = json.loads(character_summaries_str)
              
        character_charts, author_tokens, utility_tokens = await story_data["basic_author"].step_6_character_charts(
            characters=character_summaries,
            outline=four_page_form
        )

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(character_charts)),
            metadata={"type": "step_6_character_charts", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "latest_phase": 6,
                "author_token_usage": author_tokens,
                "utility_token_usage": utility_tokens,
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": four_page_form
        } 

    #------------------------Change Phases Methods---------------------------
    # Phase 1
    async def change_one_sentence_generation(self, user_id: str, story_id: str, one_sentence_form: dict):
        """
        Change single sentence describing the story, Phase 1.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(one_sentence_form)),
            metadata={"type": "one_sentence_form", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "target_medium": one_sentence_form['target_medium'],
                "genre": one_sentence_form['genre'],
                "sub_genre": one_sentence_form['sub_genre'],
                "themes": one_sentence_form['themes'],
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params"),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success"
            }
    
    # Phase 2
    async def change_one_paragraph_generation(self, user_id: str, story_id: str, one_paragraph_form: dict):
        """
        Change single paragraph describing the story, Phase 2.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.load(one_paragraph_form)),
            metadata={"type": "one_paragraph_form", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "target_medium": one_paragraph_form['target_medium'],
                "genre": one_paragraph_form['genre'],
                "sub_genre": one_paragraph_form['sub_genre'],
                "themes": one_paragraph_form['themes'],
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success", "data": one_paragraph_form
        }

    # Phase 3
    async def change_one_page_generation(self, user_id: str, story_id: str, one_page_form: dict):
        """
        Change single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")

        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(one_page_form)),
            metadata={"type": "one_page_form", "story_id": story_id}
        )
        # Update story progress in memory system
        await story_data["memory_system"].update_story_progress(
            metadata={
                "target_medium": one_page_form['target_medium'],
                "genre": one_page_form['genre'],
                "sub_genre": one_page_form['sub_genre'],
                "themes": one_page_form['themes'],
                "min_age": one_page_form['target_audience_age'],
                "target_length": one_page_form['target_length'],
                "total_acts": one_page_form['act_count'],
                "story_title": one_page_form['title']
            }
        )
        
        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success"
        }

    # Phase 4
    async def change_protagonist_generation(self, user_id: str, story_id: str, protagonist_form: dict):
        """
        Create single page describing the story, Phase 3.
        """        
        user_data = await self._get_session(user_id=user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        
        story_data = user_data["stories"].get(story_id)
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        
        await story_data["memory_system"].add_long_term_document(
            text=StoryHelpers.compress_json(json.dumps(protagonist_form)),
            metadata={"type": "protagonist_form", "story_id": story_id}
        )

        serializable_story_data = {
            "memory_system_params": story_data.get("memory_system_params", {}),
            "last_active": time.time(),
        }
        user_data["stories"][story_id] = serializable_story_data
        
        await self._set_session(user_id, story_id, serializable_story_data)
        await self._set_session(user_id, data=user_data)
        
        return {
            "status": "success"
        } 





#---------------------------Fetch progress and docs methods-------------------------------
    async def update_story_progress_for_user(self, user_id: str, story_id: str, metadata: dict) -> dict:
        user_data = await self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id.strip())
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        try:
            await story_data["memory_system"].update_story_progress(metadata)
            return {"status": "success"}
        except:
            return {"status": "error"}
        
    async def get_story_progress_for_user(self, user_id: str, story_id: str) -> dict:
        user_data = await self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id.strip())
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        try:
            progress = await story_data["memory_system"].get_story_progress()
            return {"status": "success", "data": progress}
        except:
            return {"status": "error"}
        

    async def get_director_notes(self, user_id: str, story_id: str, type: str) -> dict:
        user_data = await self._get_session(user_id)
        if not user_data:
            raise HTTPException(status_code=433, detail="Invalid user ID")
        story_data = user_data["stories"].get(story_id.strip())
        if not story_data:
            raise HTTPException(status_code=455, detail="Invalid story ID")
        try:
            progress = await story_data["memory_system"].get_story_progress()
            story_data = await story_data['memory_system'].get_long_term_document(metadata={"type": type, "story_title": progress['story_title']})
            return {"status": "success", "data": json.loads(story_data)}
        except:
            return {"status": "error"}


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