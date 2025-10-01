import json
import time
import redis
from fastapi import HTTPException
from dotenv import load_dotenv
from shared_redis_pool import get_redis_client
from src.memory.memory_system import StoryMemorySystem
from contextlib import contextmanager
from fastapi import WebSocket
import interactive_setup as interactive_setup_module

SESSION_TTL = 3600  # 1 hour expiration for inactive sessions

# Global shared instances
#SHARED_LLM_CLIENT = None
load_dotenv()

@contextmanager
def redis_lock(client, lock_key, timeout=10):
    """Context manager for acquiring a Redis lock using SETNX."""
    acquired = client.set(lock_key, "locked", nx=True, ex=timeout)
    if acquired:
        try:
            yield
        finally:
            client.delete(lock_key)
    else:
        raise HTTPException(status_code=503, detail="Could not acquire lock, please try again")

class MainSetup:
    async def initialize_story(self, user_id: str, story_type: str, story_title: str = ""):
        if story_type == 'interactive':
            if interactive_setup_module.SHARED_INTERACTIVE_SETUP is None:
                interactive_setup_module.SHARED_INTERACTIVE_SETUP = interactive_setup_module.InteractiveSetup()
                print("Initialized shared InteractiveSetup")
            interactive_setup_module.SHARED_INTERACTIVE_SETUP.initialize_story(user_id=user_id, story_title=story_title)

    async def continue_story(self, user_id: str, story_id: str, story_type: str) -> dict:
        if story_type == 'interactive':
            if interactive_setup_module.SHARED_INTERACTIVE_SETUP is None:
                interactive_setup_module.SHARED_INTERACTIVE_SETUP = interactive_setup_module.InteractiveSetup()
                print("Initialized shared InteractiveSetup")
            interactive_setup_module.SHARED_INTERACTIVE_SETUP.continue_story(user_id=user_id, story_id=story_id)

    async def create_premise(self, initial_story_data: dict):
        story_type = initial_story_data['story_type']
        if story_type == 'interactive':
            if interactive_setup_module.SHARED_INTERACTIVE_SETUP is None:
                interactive_setup_module.SHARED_INTERACTIVE_SETUP = interactive_setup_module.InteractiveSetup()
                print("Initialized shared InteractiveSetup")
            interactive_setup_module.SHARED_INTERACTIVE_SETUP.create_premise(initial_story_data=initial_story_data)

    async def handle_story_websocket(self, websocket: WebSocket, user_id: str, story_id: str, story_type: str):
        if story_type == 'interactive':
            if interactive_setup_module.SHARED_INTERACTIVE_SETUP is None:
                interactive_setup_module.SHARED_INTERACTIVE_SETUP = interactive_setup_module.InteractiveSetup()
                print("Initialized shared InteractiveSetup")
            interactive_setup_module.SHARED_INTERACTIVE_SETUP.handle_story_websocket(websocket=websocket, user_id=user_id, story_id=story_id)

    @staticmethod
    def logout(user_id: str):
        """Clear all sessions for a user from Redis."""
        client = get_redis_client()
        try:
            user_key = f"session:{user_id}"
            lock_key = f"lock:{user_key}"
            with redis_lock(client, lock_key):
                # Use SCAN to find story keys
                cursor = 0
                story_keys = []
                while True:
                    cursor, keys = client.scan(cursor, match=f"session:{user_id}:*", count=100)
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
                keys_to_delete = story_keys + [user_key] if client.exists(user_key) else story_keys
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
    def logout_story(user_id: str, story_id: str):
        """Clear a specific story session for a user from Redis."""
        client = get_redis_client()
        try:
            user_key = f"session:{user_id}"
            user_lock_key = f"lock:{user_key}"
            story_key = f"session:{user_id}:{story_id}"
            with redis_lock(client, user_lock_key):
                # Cleanup memory
                memory_system = StoryMemorySystem(user_id=user_id, story_id=story_id)
                memory_system.cleanup()
                # Delete story key if exists
                if client.exists(story_key):
                    client.delete(story_key)
                    print(f"✅ Deleted story session for {story_key}")
                else:
                    print(f"🔍 No story session found for {story_key}")
                # Update user session to remove story reference
                data = client.get(user_key)
                if data:
                    user_session = json.loads(data)
                    if "stories" in user_session and story_id in user_session["stories"]:
                        del user_session["stories"][story_id]
                    with client.pipeline() as pipe:
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