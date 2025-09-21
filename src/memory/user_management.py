from fastapi import HTTPException
from supabase import create_client, Client
from fastapi.responses import JSONResponse
import uuid
import os
from dotenv import load_dotenv


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


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


def add_user(nickname: str, user_tag: str, age: int, user_id: str, stories: list):
    try:
        # 1. Validate user_tag
        if not user_tag.strip():
            return JSONResponse(status_code=400, content={"error": "❌ user_tag cannot be empty!"})

        # 2. Check if user_tag already exists
        existing_tag = supabase.table("users").select("user_tag").eq("user_tag", user_tag).execute()
        if existing_tag.data and len(existing_tag.data) > 0:
            return JSONResponse(status_code=410, content={"error": f"❌ user_tag '{user_tag}' already exists!"})

        # 3. Check if user_id already exists
        existing_id = supabase.table("users").select("user_id").eq("user_id", user_id).execute()
        if existing_id.data and len(existing_id.data) > 0:
            return JSONResponse(status_code=420, content={"error": f"❌ user_id '{user_id}' already exists!"})

        # 4. Insert new user
        response = supabase.table("users").insert({
            "user_id": user_id,
            "nickname": nickname,
            "user_tag": user_tag,
            "age": age,
            "stories": stories
        }).execute()

        # Safely check error
        if hasattr(response, "error") and response.error:
            return JSONResponse(status_code=500, content={"error": str(response.error)})

        return {"status": "success", "user_id": user_id, "nickname": nickname, "user_tag": user_tag}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})



def append_story(user_id: str, story_title: str, story_id: str):
    result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="❌ User not found")

    current_stories = result.data[0].get("stories", [])

    # Check if story already exists
    if any(s["title"] == story_title for s in current_stories):
        return {"status": "info", "message": f"Story '{story_title}' already exists"}

    # Append new story dict
    current_stories.append({"title": story_title, "story_id": story_id})

    try:
        update_response = (
            supabase.table("users")
            .update({"stories": current_stories})
            .eq("user_id", user_id)
            .execute()
        )
        if update_response.data is None:
            raise HTTPException(status_code=500, detail="❌ Error updating stories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error updating stories: {e}")

    return {"status": "success", "message": f"Story '{story_title}' added", "stories": current_stories}


def delete_story(user_id: str, story_title: str, story_id: str):
    result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
    if not result.data or len(result.data) == 0:
        raise HTTPException(status_code=404, detail="❌ User not found")

    current_stories = result.data[0].get("stories", [])

    # Find the story by title or id
    story_exists = next((s for s in current_stories if s["title"] == story_title or s["story_id"] == story_id), None)
    if not story_exists:
        return {"status": "info", "message": f"Story '{story_title}' does not exist"}

    # Remove story
    current_stories = [s for s in current_stories if s["title"] != story_title and s["story_id"] != story_id]

    try:
        update_response = (
            supabase.table("users")
            .update({"stories": current_stories})
            .eq("user_id", user_id)
            .execute()
        )
        if update_response.data is None:
            raise HTTPException(status_code=500, detail="❌ Error updating stories")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error updating stories: {e}")

    return {"status": "success", "message": f"Story '{story_title}' deleted", "stories": current_stories}


# def get_user_stories(user_id: str):
#     result = supabase.table("users").select("stories").eq("user_id", user_id).execute()
#     if not result.data:
#         raise HTTPException(status_code=404, detail="❌ User not found")

#     story_dicts = result.data[0].get("stories", [])
#     print(story_dicts)

#     if not story_dicts:
#         return {"status": "info", "message": "No stories found"}

#     stories = []
#     for story in story_dicts:
#         title = story["title"]
#         story_id = story["story_id"]

#         progress = get_progress(user_id, story_id)

#         story_data = {
#             "title": title,
#             "story_id": story_id,
#             "latest_chapter_id": 0,
#             "continue_scene_id": 0,
#             "word_count": 0,
#         }

#         if progress:
#             story_data["latest_chapter_id"] = progress.get("latest_chapter_id", 0)
#             story_data["continue_scene_id"] = progress.get("continue_scene_id", 0)
#             story_data["word_count"] = progress.get("word_count", 0)

#         stories.append(story_data)

#     return {"status": "success", "stories": stories}

# def get_user_profile_data(user_id):
#     """
#     Fetches age, nickname, and user_tag for a given user_id.
#     Returns None if no user found.
#     """
#     try:
#         response = supabase.table("users") \
#             .select("age, nickname, user_tag") \
#             .eq("user_id", user_id) \
#             .single() \
#             .execute()

#         if response.data:
#             return {"status": "success", "data": response.data}
#         else:
#             return {"status": "success", "data": None}
#     except Exception as e:
#         #print(f"⚠️ Error fetching user info: {e}")
#         return {"status": "failed"}


def get_user_profile_with_stories(user_id: str):
    """
    Fetch user profile (age, nickname, user_tag) and their stories with progress.
    """
    try:
        # Fetch profile + stories in one query
        response = (
            supabase.table("users")
            .select("age, nickname, user_tag, stories")
            .eq("user_id", user_id)
            .single()
            .execute()
        )

        if not response.data:
            raise HTTPException(status_code=404, detail="❌ User not found")

        user_data = {
            "age": response.data.get("age"),
            "nickname": response.data.get("nickname"),
            "user_tag": response.data.get("user_tag"),
        }

        # Process stories
        story_dicts = response.data.get("stories", [])
        stories = []

        for story in story_dicts:
            title = story.get("title")
            story_id = story.get("story_id")

            progress = get_progress(user_id, story_id)

            story_data = {
                "title": title,
                "story_id": story_id,
                "latest_chapter_id": 0,
                "continue_scene_id": 0,
                "word_count": 0,
            }

            if progress:
                story_data["latest_chapter_id"] = progress.get("latest_chapter_id", 0)
                story_data["continue_scene_id"] = progress.get("continue_scene_id", 0)
                story_data["word_count"] = progress.get("word_count", 0)

            stories.append(story_data)

        return {
            "status": "success",
            "profile": user_data,
            "stories": stories if stories else [],
        }

    except Exception as e:
        print(f"⚠️ Error fetching user profile with stories: {e}")
        return {"status": "failed", "message": str(e)}


# res = get_user_profile_data("sad")
# print(res)