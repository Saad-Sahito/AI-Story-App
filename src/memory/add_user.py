from fastapi import FastAPI, HTTPException
from supabase import create_client, Client
import uuid
import os
from dotenv import load_dotenv
from memory_system import StoryMemorySystem

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
memory_system = StoryMemorySystem()
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def add_user(nickname: str, user_tag: str, age: int, user_id: str, stories: list):
    # 1. Validate user_tag
    if not user_tag.strip():
        raise HTTPException(status_code=400, detail="❌ user_tag cannot be empty!")

    # 2. Check if user_tag already exists
    existing_tag = supabase.table("users").select("user_tag").eq("user_tag", user_tag).execute()
    if existing_tag.data and len(existing_tag.data) > 0:
        raise HTTPException(status_code=410, detail=f"❌ user_tag '{user_tag}' already exists!")

    # 3. Generate unique user_id
    while True:
        new_user_id = user_id
        existing_id = supabase.table("users").select("user_id").eq("user_id", new_user_id).execute()
        if not existing_id.data or len(existing_id.data) == 0:
            break  # unique user_id found

    # 4. Insert new user
    response = supabase.table("users").insert({
        "user_id": new_user_id,
        "nickname": nickname,
        "user_tag": user_tag,
        "age": age,
        "stories": stories
    }).execute()

    if response.error:
        raise HTTPException(status_code=500, detail=f"❌ Error adding user: {response.error}")

    return {"status": "success", "user_id": new_user_id, "nickname": nickname, "user_tag": user_tag}

def append_story(user_id: str, story_title: str):
    # 1. Fetch current stories
    result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="❌ User not found")

    current_stories = result.data[0].get("stories", [])
    if story_title in current_stories:
        return {"status": "info", "message": f"Story '{story_title}' already exists"}

    # 2. Append new story
    current_stories.append(story_title)
    try:
        update_response = supabase.table("users").update({"stories": current_stories}).eq("user_id", user_id).execute()
        if update_response.data is None:
            raise HTTPException(status_code=500, detail="❌ Error updating stories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error updating stories: {e}")

    return {"status": "success", "message": f"Story '{story_title}' added", "stories": current_stories}



def delete_story(user_id: str, story_title: str):
    result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="❌ User not found")

    current_stories = result.data[0].get("stories", [])
    if story_title not in current_stories:
        return {"status": "info", "message": f"Story '{story_title}' does not exist"}

    current_stories.remove(story_title)

    try:
        update_response = supabase.table("users").update({"stories": current_stories}).eq("user_id", user_id).execute()
        if update_response.data is None:
            raise HTTPException(status_code=500, detail="❌ Error updating stories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error updating stories: {e}")

    return {"status": "success", "message": f"Story '{story_title}' deleted", "stories": current_stories}

def get_progress(user_id: str, story_id: str):
        """
        Get the latest story progress (chapter_id, scene_id, story_title) 
        for this user + story.
        """
        result = (
            supabase.table("story_progress")
            .select("latest_chapter_id, continue_scene_id, word_count, metadata")
            .eq("user_id", user_id)
            .eq("story_id", story_id)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

def get_user_stories(user_id: str):
    result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="❌ User not found")
    
    titles = result.data[0].get("stories", [])

    if not titles:  # cleaner than == []
        return {"status": "info", "message": "No stories found"}

    stories = []
    for title in titles:
        story_id = f"{user_id}_{title.replace(' ', '_').lower()}"
        progress = get_progress(user_id, story_id)

        story_data = {
            "title": title,
            "latest_chapter_id": 0,
            "continue_scene_id": 0,
            "word_count": 0,
        }

        if progress:
            story_data["latest_chapter_id"] = progress.get("latest_chapter_id", 0)
            story_data["continue_scene_id"] = progress.get("continue_scene_id", 0)
            story_data["word_count"] = progress.get("word_count", 0)
            story_data["title"] = f"{title} (Chapter {progress.get('latest_chapter_id', 0)})"

        stories.append(story_data)

    return {"status": "success", "stories": stories}
