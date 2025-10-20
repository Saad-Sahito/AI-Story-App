
from .shared_resources import get_sqlite_store

async def post_user_feedback(form, user_id: str = None):
    """
    Add a new user to the users table.
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store.insert_feedback(form)
    await store.close()
    return result