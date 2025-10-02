# src/setup/main_setup.py
import json
import time
import redis
from fastapi import HTTPException
from dotenv import load_dotenv
from setup.shared_redis_pool import get_redis_client
from src.memory.memory_system import StoryMemorySystem
from fastapi import WebSocket
from setup.story_types.interactive_setup import get_shared_interactive_setup
from setup.story_types.classic_setup import get_shared_classic_setup

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


class MainSetup:
    async def initialize_story(self, user_id: str, story_type: str, story_title: str = ""):
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            await setup.initialize_story(user_id=user_id, story_title=story_title)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            await setup.initialize_story(user_id=user_id, story_title=story_title)

    async def continue_story(self, user_id: str, story_id: str, story_type: str) -> dict:
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.continue_story(user_id=user_id, story_id=story_id)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.continue_story(user_id=user_id, story_id=story_id)

    async def create_premise(self, initial_story_data: dict):
        story_type = initial_story_data['story_type']
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.create_premise(initial_story_data=initial_story_data)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.create_premise(initial_story_data=initial_story_data)

    async def handle_story_websocket(self, websocket: WebSocket, user_id: str, story_id: str, story_type: str):
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            await setup.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            await setup.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id)

    async def get_story_progress_for_user(self, user_id: str, story_id: str, story_type: str) -> dict:
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.get_story_progress_for_user(user_id=user_id, story_id=story_id)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.get_story_progress_for_user(user_id=user_id, story_id=story_id)

    async def get_story_cluster(self, user_id: str, story_id: str, chapter_number: int, story_type: str) -> dict:
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.get_story_cluster(user_id=user_id, story_id=story_id, chapter_number=chapter_number)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.get_story_cluster(user_id=user_id, story_id=story_id, chapter_number=chapter_number)
    @staticmethod
    async def logout(user_id: str):
        """Clear all sessions for a user from Redis."""
        client = get_redis_client()
        async with client:
            try:
                user_key = f"session:{user_id}"
                lock_key = f"lock:{user_key}"
                async with redis_lock(client, lock_key):
                    # Use SCAN to find story keys
                    cursor = 0
                    story_keys = []
                    while True:
                        cursor, keys = await client.scan(cursor, match=f"session:{user_id}:*", count=100)
                        story_keys.extend(keys)
                        if cursor == 0:
                            break
                    # Cleanup memory for each story
                    for story_key in story_keys:
                        parts = story_key.split(":", 2)
                        if len(parts) == 3:
                            story_id = parts[2]
                            memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                            memory_system.cleanup()
                    # Delete all keys
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
        
    @staticmethod
    async def logout_story(user_id: str, story_id: str):
        """Clear a specific story session for a user from Redis."""
        client = get_redis_client()
        async with client:
            try:
                user_key = f"session:{user_id}"
                user_lock_key = f"lock:{user_key}"
                story_key = f"session:{user_id}:{story_id}"
                async with redis_lock(client, user_lock_key):
                    # Cleanup memory
                    memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                    memory_system.cleanup()
                    # Delete story key if exists
                    if await client.exists(story_key):
                        await client.delete(story_key)
                        print(f"✅ Deleted story session for {story_key}")
                    else:
                        print(f"🔍 No story session found for {story_key}")
                    # Update user session to remove story reference
                    data = await client.get(user_key)
                    if data:
                        user_session = json.loads(data)
                        if "stories" in user_session and story_id in user_session["stories"]:
                            del user_session["stories"][story_id]
                        async with client.pipeline() as pipe:
                            pipe.set(user_key, json.dumps(user_session))
                            pipe.expire(user_key, SESSION_TTL)
                            pipe.execute()
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

    @staticmethod
    async def cleanup_inactive_sessions(max_age_seconds: int = 3600):
        """Clean up inactive sessions from Redis using SCAN."""
        client = get_redis_client()
        async with client:
            try:
                current_time = time.time()
                removed = 0
                cursor = 0
                while True:
                    cursor, keys = await client.scan(cursor, match="session:*", count=100)
                    for key in keys:
                        data = await client.get(key)
                        if not data:
                            continue
                        session_data = json.loads(data)
                        if current_time - session_data.get("last_active", 0) > max_age_seconds:
                            async with redis_lock(client, f"lock:{key}"):
                                fresh_data = await client.get(key)  # Re-fetch
                                if fresh_data:
                                    fresh_session = json.loads(fresh_data)
                                    if current_time - fresh_session.get("last_active", 0) > max_age_seconds:
                                        parts = key.split(":", 2)
                                        user_id = parts[1]
                                        if len(parts) == 3:  # Story-specific session
                                            story_id = parts[2]
                                            MainSetup.logout_story(user_id, story_id)
                                            removed += 1
                                        elif len(parts) == 2:  # User session
                                            MainSetup.logout(user_id)
                                            removed += 1
                    if cursor == 0:
                        break
                return removed
            except redis.RedisError as e:
                print(f"❌ Redis error in cleanup_inactive_sessions: {e}")
                raise HTTPException(status_code=500, detail="Failed to clean up sessions")