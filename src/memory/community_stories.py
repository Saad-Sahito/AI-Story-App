# from typing import List, Dict, Optional, Any
from .shared_resources import get_sqlite_store

async def get_all_public_stories_page(min_age: int, page_index: int = 1, page_size: int = 9):
    """
    Get all public stories
    """
    store = await get_sqlite_store(table=None, user_id=None, story_id=None)
    result = await store._get_public_stories_page(min_age=min_age, page_index=page_index, page_size=page_size)
    print(result)
    await store.close()
    return result    

async def setup_community_story(story_id: str, user_id: str):
    store = await get_sqlite_store(table=None, user_id=user_id, story_id=story_id)
    present = await store.get_self_story_progress()
    if present != None:
        return {'status': 'error', 'message': 'Story already present in your dashboard'}
    shared_story_data = await store._get_shared_story_data(story_id=story_id)

    try:
        if shared_story_data['public'] == 1:
            await store.update_self_story_progress(metadata={
                'story_word_count': shared_story_data.get('story_length'),
                'current_act_id' : 1,
                'latest_chapter_id': 1,
                'continue_scene_id': 1,
                'complete': 1
            })
            await store._append_story(user_id=user_id, story_title=shared_story_data.get('story_title'), story_id=story_id, story_type=shared_story_data.get('story_type'), self_created=False)
            await store._increment_story_views(story_id=story_id)
            return {'status': 'success', 'message': 'Story successfully added to user dashboard'}
        else:
            return {'status': 'error', 'message': 'Story not public'}
    except:
        return {'status': 'error', 'message': 'Failed to load story'}
    
