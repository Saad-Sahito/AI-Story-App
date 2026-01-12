# src/plot_engine/story_metadata.py

import json
import asyncio
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_helpers import StoryHelpers

class MetaDataForm(BaseModel):
    story_title: str
    genre: List[str]
    sub_genre: List[str]
    themes: str
    tone: str
    story_structure: str
    total_acts: Optional[int]
    target_length: Optional[int]
    target_audience_age: Optional[int]

class MetaDataGenerator:
    def __init__(self):
        pass

    async def generate_metadata(self, context: str, past_metadata: dict) -> Tuple[MetaDataForm, dict, dict]:
        system_prompt = f"""Modify the Metadata if needed looking at the new story context and the current metadata.
If you think if some or all values should remain the same or dont have enough context to change or add a value then output as is, without modifying, or an empty string '', if already empty.

For Genre:
Choose maximum of three from: 'Fantasy', 'Mystery','Comedy','Sci-Fi', 'Romance', 'Adventure', 'Horror', 'Thriller/Suspense', 'Crime', 'Drama', 'Tragedy'.
For Sub-Genre:
Choose between these respective of selected genres.
"Thriller/Suspense": ["Psychological Thriller", "Action Thriller", "Legal Thriller", "Domestic Thriller", "Conspiracy Thriller", "Spy/Espionage", "Medical Thriller"],
  "Mystery": ["Cozy Mystery", "Whodunit", "Hard-Boiled", "Police Procedural", "Private Investigator", "Locked-Room", "Noir Mystery", "Caper/Heist Mystery"],
  "Horror": ["Supernatural", "Psychological Horror", "Slasher", "Body Horror", "Folk Horror", "Cosmic/Lovecraftian", "Gothic Horror", "Zombie"],
  "Romance": ["Contemporary Romance", "Historical Romance", "Paranormal Romance", "Romantic Suspense", "Romantic Comedy", "Sports Romance", "Dark Romance", "Fantasy Romance"],
  "Comedy": ["Romantic Comedy", "Dark Comedy", "Satire", "Slapstick", "Buddy Comedy", "Screwball Comedy", "Parody"],
  "Drama": ["Family Drama", "Coming-of-Age", "Psychological Drama", "Social Issue Drama", "Melodrama", "Historical Drama"],
  "Tragedy": ["Classical Tragedy", "Revenge Tragedy", "Domestic Tragedy", "Modern Tragedy", "Shakespearean Tragedy"],
  "Adventure": ["Swashbuckling", "Pulp Adventure", "Survival Adventure", "Jungle/Exploration", "High-Seas/Pirate"],
  "Crime": ["Heist/Caper", "Police Procedural", "Noir", "Gangster/Mafia", "True Crime-Inspired", "Organized Crime"],
  "Fantasy": ["High Fantasy", "Urban Fantasy", "Dark Fantasy", "Portal/Isekai", "Mythic/Fairy-Tale", "Grimdark", "Sword & Sorcery"],
  "Sci-Fi": ["Space Opera", "Hard Sci-Fi", "Cyberpunk", "Dystopian", "Post-Apocalyptic", "Military Sci-Fi", "Time Travel", "First Contact"]

Do not generate any other genre or sub-genre that is not present above, make sure the string value for these are exactly as defined above.
Json Output:
{MetaDataForm.model_json_schema()}
"""

        human_prompt = f"""
Current Metadata: {json.dumps(past_metadata, indent=2)}
Story context So Far: {json.dumps(context)}
"""

        # --- LLM CALL ---
        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
        )


        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        # --- Parse JSON output ---
        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=MetaDataForm
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]

        print(f"→ Generated MetaData...")
        return json_data, author_tokens, utility_tokens
