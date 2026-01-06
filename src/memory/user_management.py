from typing import List, Dict, Optional, Any
from .shared_resources import get_sqlite_store

async def add_user(nickname: str, age: Optional[int], user_id: str, tier: int, stories: List[Dict] = None):
    """
    Add a new user to the users table.
    """
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._add_user(nickname=nickname, age=age, tier=tier, user_id=user_id, stories=stories)
    await store.close()
    return result

async def append_story(user_id: str, story_id: str):
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._append_story(user_id=user_id, story_id=story_id)
    await store.close()
    return result

async def delete_story(user_id: str, story_title: str, story_id: str):
    """
    Delete a story from the user's stories list.
    """
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._delete_story(user_id, story_title, story_id)
    await store.close()
    return result

async def get_user_profile(user_id: str):
    """
    Fetch user profile (age, nickname, user_tag) and their stories with progress.
    """
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._get_user_profile(user_id)
    await store.close()
    return result

async def get_user_profile_with_stories(user_id: str):
    """
    Fetch user profile (age, nickname, user_tag) and their stories with progress.
    """
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._get_user_profile_with_stories(user_id)
    print(result)
    await store.close()
    return result

async def update_user_settings(user_id: str, user_data: Dict[str, Any]):
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = await store._update_user(user_id=user_id, updates=user_data)
    await store.close()
    return result

async def update_user_story_public_status(user_id: str, story_id: str, public: bool = True):
    store = await get_sqlite_store(table="users", user_id=user_id, story_id=story_id)
    result = await store._update_story_public_status(user_id=user_id, story_id=story_id, public=public)
    await store.close()
    return result

async def user_rate_story(user_id: str, story_id: str, rating: int):
    """
    User rate story.
    """
    store = await get_sqlite_store(table="story_progress", user_id=user_id, story_id=story_id)
    result = await store._rate_user_story(user_id=user_id, story_id=story_id, rating=rating)
    await store.close()
    return result

async def update_story_view_state(user_id: str, story_id: str, act_id: int, chapter_id: int, scene_id: int):
    store = await get_sqlite_store(table="story_progress", user_id=user_id, story_id=story_id)
    result = await store.update_self_story_progress(metadata={
        'current_act_id': act_id,
        'latest_chapter_id': chapter_id,
        'continue_scene_id': scene_id
    })
    await store.close()
    return result