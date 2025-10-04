import asyncio
import httpx
import websockets
import json
import random
import string
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

# Import the FastAPI app from main.py
from main import app  # Ensure main.py is in the same directory or adjust the import path

API_BASE = "http://localhost:8000"
WS_BASE = "ws://localhost:8000"
CONCURRENT_USERS = 15
SERVER_PORT = 8000

async def cleanup_redis():
    """Clear Redis sessions before tests."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            resp = await client.patch(f"{API_BASE}/storage")
            print(f"✅ Cleaned up Redis sessions: {resp.status_code}")
            if resp.status_code != 200:
                print(f"❌ Cleanup failed: {resp.text}")
        except Exception as e:
            print(f"❌ Failed to clean up Redis: {e}")

async def simulate_user(user_index: int):
    """Simulate a user interacting with the FastAPI app."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            user_id = f"user{user_index}_{''.join(random.choices(string.ascii_lowercase, k=5))}"
            nickname = f"User{user_index}"
            user_tag = f"tag{user_index}"
            age = random.randint(18, 40)

            # Add user
            resp = await client.post(
                f"{API_BASE}/users",
                params={"nickname": nickname, "user_tag": user_tag, "age": age, "user_id": user_id}
            )
            print(f"[User {user_id}] Add user -> {resp.status_code}")
            if resp.status_code != 200:
                print(f"[User {user_id}] Add user failed: {resp.text}")
                return
            await asyncio.sleep(1.0)

            # Initialize story
            story_type = random.choice(["classic", "interactive"])
            story_title = f"MyStory{user_index}"
            story_id = f"{story_title.lower()}_{user_id}"
            print(f"[User {user_id}] Story_id: {story_id}")

            resp = await client.post(
                f"{API_BASE}/stories/initialize_story",
                params={"user_id": user_id, "story_type": story_type, "story_title": story_title}
            )
            print(f"[User {user_id}] Initialize story -> {resp.status_code}")
            if resp.status_code != 200:
                print(f"[User {user_id}] Initialize story failed: {resp.text}")
                return
            await asyncio.sleep(1.0)

            # Create premise
            premise = {
                "POV": "First-person",
                "Tone": 1,
                "Genre": ["Fantasy"],
                "Title": story_title,
                "Length": 3,
                "Setting": "A mystical forest",
                "user_id": user_id,
                "story_id": story_id,
                "story_type": story_type,
                "Guide Prose": ["Keep it lighthearted"],
                "Additional Themes": ["Friendship", "Adventure"]
            }

            resp = await client.post(f"{API_BASE}/premise", json=premise)
            print(f"[User {user_id}] Create premise -> {resp.status_code}")
            if resp.status_code != 200:
                print(f"[User {user_id}] Create premise failed: {resp.text}")
                return
            await asyncio.sleep(1.0)

            # WebSocket interaction
            ws_url = f"{WS_BASE}/ws/next_chapter/{user_id}/{story_id}?story_type={story_type}"
            try:
                async with websockets.connect(ws_url, ping_timeout=20) as websocket:
                    print(f"[User {user_id}] Connected WebSocket -> OK")
                    try:
                        # Send initial message
                        await websocket.send(json.dumps({"user_id": user_id, "story_id": story_id}))
                        for i in range(3):
                            msg = await asyncio.wait_for(websocket.recv(), timeout=15.0)
                            print(f"[User {user_id}] WS received [{i+1}/3]: {msg}")
                            await websocket.send(json.dumps({"choice": f"Continue the adventure {i+1}"}))
                            await asyncio.sleep(1.0)
                        # Allow server to process final messages
                        await asyncio.sleep(2.0)
                    except asyncio.TimeoutError:
                        print(f"[User {user_id}] WS timeout")
                        await websocket.close(code=1001)
                    except Exception as e:
                        print(f"[User {user_id}] WS error: {e}")
                        await websocket.close(code=1000)
            except websockets.exceptions.ConnectionClosed as e:
                print(f"[User {user_id}] WS connection closed: {e}")
            except Exception as e:
                print(f"[User {user_id}] WS connection failed: {e}")

        except Exception as e:
            print(f"[User {user_index}] Error: {e}")
            import traceback
            traceback.print_exc()

async def run_server():
    """Run the FastAPI server in the same process."""
    config = uvicorn.Config(app, host="0.0.0.0", port=SERVER_PORT, log_level="info")
    server = uvicorn.Server(config)
    try:
        await server.serve()
    except asyncio.CancelledError:
        print("✅ Server shutdown initiated")
        await server.shutdown()
    except Exception as e:
        print(f"❌ Server error: {e}")

async def main():
    """Run the server and tests, ensuring proper shutdown."""
    # Start the server in the background
    server_task = asyncio.create_task(run_server())
    # Wait for the server to start
    #await asyncio.sleep(2.0)
    
    # Clean up Redis before tests
    #await cleanup_redis()
    await asyncio.sleep(1.0)

    # Run user simulations
    tasks = [simulate_user(i) for i in range(CONCURRENT_USERS)]
    try:
        await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"❌ Error in user simulations: {e}")

    # Allow server to process remaining requests
    await asyncio.sleep(5.0)

    # Shut down the server
    server_task.cancel()
    try:
        await server_task
    except asyncio.CancelledError:
        print("✅ Server task cancelled")

if __name__ == "__main__":
    asyncio.run(main())