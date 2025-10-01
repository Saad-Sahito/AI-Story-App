from typing import List, Dict, Optional
from .shared_resources import get_sqlite_store

def add_user(nickname: str, user_tag: str, age: Optional[int], user_id: str, stories: List[Dict] = None):
    """
    Add a new user to the users table.
    """
    store = get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = store.add_user(nickname, user_tag, age, user_id, stories)
    store.close()
    return result

def append_story(user_id: str, story_title: str, story_id: str, story_type: str):
    """
    Append a story to the user's stories list.
    """
    store = get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = store.append_story(user_id=user_id, story_title=story_title, story_id=story_id, story_type=story_type)
    store.close()
    return result

def delete_story(user_id: str, story_title: str, story_id: str):
    """
    Delete a story from the user's stories list.
    """
    store = get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = store.delete_story(user_id, story_title, story_id)
    store.close()
    return result

def get_user_profile_with_stories(user_id: str):
    """
    Fetch user profile (age, nickname, user_tag) and their stories with progress.
    """
    store = get_sqlite_store(table="users", user_id=user_id, story_id=None)
    result = store.get_user_profile_with_stories(user_id)
    store.close()
    return result