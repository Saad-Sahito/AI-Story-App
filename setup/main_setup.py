# src/setup/main_setup.py
from dotenv import load_dotenv
from fastapi import WebSocket
from setup.story_types.interactive_setup import get_shared_interactive_setup
from setup.story_types.classic_setup import get_shared_classic_setup

SESSION_TTL = 3600  # 1 hour expiration for inactive sessions

load_dotenv()



class MainSetup:
    async def initialize_story(self, user_id: str, story_type: str, story_title: str = ""):
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.initialize_story(user_id=user_id, story_title=story_title)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.initialize_story(user_id=user_id, story_title=story_title)

    async def continue_story(self, user_id: str, story_id: str, story_type: str) -> dict:
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.continue_story(user_id=user_id, story_id=story_id)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.continue_story(user_id=user_id, story_id=story_id)

    async def create_premise(self, initial_story_data: dict, model: str):
        story_type = initial_story_data['story_type']
        if story_type == 'interactive':
            setup = await get_shared_interactive_setup()
            return await setup.create_premise(initial_story_data=initial_story_data, model=model)
        elif story_type == 'classic':
            setup = await get_shared_classic_setup()
            return await setup.create_premise(initial_story_data=initial_story_data, model=model)

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

    async def logout(self, user_id: str):
        """Clear all sessions for a user from Redis."""
        try:
            setup_interactive = await get_shared_interactive_setup()
            res_int = await setup_interactive.logout(user_id=user_id)
            setup_classic = await get_shared_classic_setup()
            res_cls = await setup_classic.logout(user_id=user_id)
            return res_int, res_cls
        except:
            return {"error": "One of or both sessions uninitialized"}

    async def logout_story(self, user_id: str, story_id: str, story_type: str):
        """Clear a specific story session for a user from Redis."""
        if story_type == "interactive":
            setup_interactive = await get_shared_interactive_setup()
            return await setup_interactive.logout_story(user_id=user_id, story_id=story_id)
        elif story_type == "classic":
            setup_classic = await get_shared_classic_setup()
            return await setup_classic.logout_story(user_id=user_id, story_id=story_id)
            
    async def cleanup_inactive_sessions(self):
        """Clean up inactive sessions from Redis using SCAN."""
        try:
            setup_interactive = await get_shared_interactive_setup()
            res_int = await setup_interactive.cleanup_inactive_sessions()
            setup_classic = await get_shared_classic_setup()
            res_cls = await setup_classic.cleanup_inactive_sessions()
            return {"interactive sessions cleaned: ", res_int}, {"classic sessions cleaned: ", res_cls}
        except:
            return {"error": "One of or both sessions uninitialized"}