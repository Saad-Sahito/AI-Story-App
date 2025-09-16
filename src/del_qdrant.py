
# import streamlit as st
# from agents.director_agent import DirectorGraph
# from memory.short_term_memory import ShortTermMemory
# from memory.long_term_memory import LongTermMemory

# # Initialize components only once
# if "director" not in st.session_state:
#     if "chapter_id" not in st.session_state:
#         st.session_state.chapter_id = 1

#     st.session_state.director = DirectorGraph()
#     st.session_state.short_term_memory = ShortTermMemory()
#     st.session_state.long_term_memory = LongTermMemory()
#     st.session_state.is_generating = False   # flag for streaming

# st.set_page_config(page_title="AI Story Engine", layout="centered")

# st.title("🎬 AI Story Engine")
# st.write("Craft your story premise, and let the Director bring it to life.")

# with st.form("story_form"):
#     title = st.text_input("Story Title", placeholder="The Forgotten Kingdom")
#     setting = st.text_area("Setting", placeholder="Ancient forest with magical ruins...")
#     characters = st.text_area("Main Characters", placeholder="A young mage, a rogue knight...")
#     genre = st.selectbox("Genre", ["Fantasy", "Sci-Fi", "Mystery", "Romance", "Thriller"])
#     tone = st.selectbox("Tone", ["Lighthearted", "Dark", "Epic", "Comedic", "Suspenseful"])
#     submitted = st.form_submit_button("🎥 Start Story Generation")

# if submitted:
#     # Combine premise details into a single user message
#     premise = f"""
#     Title: {title}
#     Setting: {setting}
#     Characters: {characters}
#     Genre: {genre}
#     Tone: {tone}
#     """

#     with st.spinner("🎬 Directing your story..."):
#         if 'story' not in st.session_state:
#             st.session_state.story_synopsis = ""
#             st.session_state.story_text = ""

#         # First synopsis (premise-based)
#         synopsis_text = st.session_state.director.set_story_premise(premise=premise)
#         st.session_state.story_synopsis = synopsis_text + "\n\n---\n\n"

#         st.text_area("Story Synopsis Output", value=st.session_state.story_synopsis, height=400)

# # Only show "Generate Next Chapter" button when not generating
# if "story_synopsis" in st.session_state:
#     if not st.session_state.is_generating:
#         if st.button("Generate Next Chapter"):
#             st.session_state.is_generating = True
#             placeholder = st.empty()

#             initial_state = {
#                 "messages": [],
#                 "current_chapter_id": st.session_state.chapter_id,
#             }

#             final_state = st.session_state.director.run(initial_state)
#             #st.session_state.story_text += final_state["scene_memory"].get("scene_so_far", "")

#             placeholder.text_area("Story Output", value=st.session_state.story_text, height=400)

#             st.session_state.is_generating = False
#             st.session_state.chapter_id += 1



# st.markdown("---")
# st.subheader("📚 Long-Term Memory Scenes")
# scenes = st.session_state.long_term_memory.get_all()
# if scenes:
#     for idx, scene in enumerate(scenes, start=1):
#         st.markdown(f"**Scene {idx}:** {scene}")
# else:
#     st.write("No scenes stored yet.")

# # --- Sidebar UI ---
# st.sidebar.header("User Choices")

# st.session_state.user_input = ""
# st.session_state.user_prompt = ""
# st.sidebar.text_area("Choice Prompt", value=st.session_state.user_prompt or "")
# user_input = st.sidebar.text_input("Choice ...", placeholder="Summon a dragon...")
# if st.sidebar.button("Send Choice"):
#     st.sidebar.write(f"Choice sent: {user_input}")
#     st.session_state.user_input = user_input


# import requests
# import os

# api_key = os.environ.get("GROQ_API_KEY")
# url = "https://api.groq.com/openai/v1/models"

# headers = {
#     "Authorization": f"Bearer {api_key}",
#     "Content-Type": "application/json"
# }

# response = requests.get(url, headers=headers)

# print(response.json())


import collections
from qdrant_client import QdrantClient

host: str = "localhost"
port: int = 6333

client = QdrantClient(host=host, port=port)
"""Delete all collections in the Qdrant instance."""
# Get all existing collections
collections = client.get_collections().collections

# Delete each one
for c in collections:
    client.delete_collection(c.name)
    print(f"Deleted collection: {c.name}")
