from typing import List, Dict, Optional, Any
from .shared_resources import get_sqlite_store

async def get_all_users():
    """
    Add a new user to the users table.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_users()
    await store.close()
    return result

async def get_all_story_texts():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_story_texts()
    await store.close()
    return result

async def get_all_characters_raw():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_characters_raw()
    await store.close()
    return result

async def get_all_world_elements_raw():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_world_elements_raw()
    await store.close()
    return result

async def get_all_director_notes():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_director_notes()
    await store.close()
    return result

async def get_all_story_progress():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_story_progress()
    await store.close()
    return result

async def get_all_data():
    """
    Append a story to the user's stories list.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.get_all_data()
    await store.close()
    return result