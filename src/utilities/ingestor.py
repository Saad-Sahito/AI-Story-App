# src/utilities/ingestor.py
import json
import asyncio
import gc
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.utilities.story_helpers import StoryHelpers
from src.llm_client.llm_client import ingestor_client


class EntityDetails(BaseModel):
    old_name: str = Field(description="Previous or current name of the character or world element")
    new_name: str = Field(description="New name if changed, otherwise same as old_name")
    details: str = Field(description="Scene-relevant summary details")


class SceneBundle(BaseModel):
    story_summary: str = Field(description="Detailed summary of the scene")
    character_details: Dict[str, EntityDetails] = Field(
        description="Dictionary: {character_key: {old_name, new_name, details}}"
    )
    world_details: Dict[str, EntityDetails] = Field(
        description="Dictionary: {world_key: {old_name, new_name, details}}"
    )

scene_parser = PydanticOutputParser(pydantic_object=SceneBundle)


class CharacterMemory(BaseModel):
    name: str
    progression: List[str] = Field(default_factory=list, description="Detailed progression history entries, formatted as 'Act {act} Chapter {chapter} Scene {scene}: {detailed change description, including what changed, why, and current state}'")
    current_summary: str = Field(description="Current inline summary of character's actions/motivations/changes")
    current_traits: List[str] = Field(default_factory=list)
    current_relationships: Dict[str, str] = Field(default_factory=dict)
    current_emotional_state: Optional[str] = None
    current_goals: Optional[str] = None
    current_status: Optional[str] = None


class WorldElementMemory(BaseModel):
    name: str
    progression: List[str] = Field(default_factory=list, description="Detailed progression history entries, formatted as 'Act {act} Chapter {chapter} Scene {scene}: {detailed change description, including what changed, why, and current state}'")
    current_summary: str = Field(description="Current summary of how this world element appeared/changed")
    current_atmosphere: Optional[str] = None
    current_culture: Optional[str] = None
    current_events: Optional[str] = None
    current_connections: Union[Dict[str, str], str] = Field(default_factory=dict)


class ChapterBundle(BaseModel):
    summary: str = Field(description="Detailed summary of the entire chapter")
    character_summary: Dict[str, CharacterMemory]
    world_summary: Dict[str, WorldElementMemory]

chapter_parser = PydanticOutputParser(pydantic_object=ChapterBundle)


class ActSummary(BaseModel):
    act_summary: str = Field(description="Detailed summary of the entire Act")
    character_progressions: Dict[str, CharacterMemory] = Field(description="Cumulative character progressions up to this act")
    world_progressions: Dict[str, WorldElementMemory] = Field(description="Cumulative world element progressions up to this act")

act_ingestor_parser = PydanticOutputParser(pydantic_object=ActSummary)

# ============================================================================
# INGESTOR CLASS
# ============================================================================
class Ingestor:
    @staticmethod
    async def ingest_scene(
        act_id: int,
        chapter_id: int,
        scene_id: int,
        scene_text: str,
        chars: List[str],
        worlds: List[str],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        system_prompt = f"""You are the Scene Breakdown Agent. Extract structured information from the scene.
Always include:

* act_id, chapter_id and scene_id in each entity's details.

* Both old_name and new_name fields for every character and world element:

  * If no rename occurred, keep both the same.

  * If renamed, set old_name to the previous name and new_name to the new one.

Only include relevant character and world names that appear or are mentioned. If a name change occurs, clearly indicate it through old_name/new_name instead of creating extra keys.
Respond ONLY in JSON matching this schema: {scene_parser.get_format_instructions()}"""

        human_prompt = f"""Current Act: {act_id}, Current Chapter: {chapter_id}, Current Scene: {scene_id}
Scene: {scene_text}
Character Names: {chars} World Names: {worlds}"""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                clean_resp, SceneBundle, scene_parser
            )
            if success:
                return result

            try:
                fixed_resp = await StoryHelpers.load_json_with_retry(clean_resp, scene_parser)

                del clean_resp
                gc.collect()

                if isinstance(fixed_resp, dict):
                    fixed_clean = json.dumps(fixed_resp)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    fixed_clean, SceneBundle, scene_parser
                )
                del fixed_clean
                if success:
                    return result
            except Exception as inner_e:
                pass

        return None

    @staticmethod
    async def ingest_chapter(
        act_id: int,
        chapter_id: int,
        current_chap_summary: str,
        char_details: Dict[str, Any],
        world_details: Dict[str, Any],
        max_retries: int = 3
    ) -> Optional[Dict[str, Any]]:
        system_prompt = f"""You are the Chapter Breakdown Agent.
Analyze the entire chapter text, keeping indicated references to user decisions wherever indicated and generate a structured breakdown.
OUTPUT MUST BE VALID JSON ONLY. NO MARKDOWN. NO EXPLANATION.
MANDATORY FIELDS:

1. "summary": Detailed chapter summary (string)

2. "character_summary": Dict of CharacterMemory objects (use exact names from story)

3. "world_summary": Dict of WorldElementMemory objects (MUST INCLUDE ALL WORLD ELEMENTS FROM STORY + MEMORY)

CRITICAL:

* "world_summary" is REQUIRED. If no new world elements, include known ones from memory.

* Use EXACT keys from provided Character Details and World Details.

* For each CharacterMemory and WorldElementMemory, update the progression list by appending new entries based on changes in this chapter, formatted as 'Act {act_id} Chapter {chapter_id}: [detailed description of change, including what changed, why, previous state, and new current state]'. If no change, append a note confirming stability.

* Update current_ fields to reflect the state at the end of this chapter.

* Build upon provided previous details to create cumulative progression history.

* For missing fields in memory, set to empty string "" or null as specified.

* NEVER omit "world_summary" — it will crash the system.

EXAMPLE STRUCTURE: {{ "summary": "Chapter summary here...", "character_summary": {{ "Alex Rivera": {{ "name": "Alex Rivera", "progression": ["Act 1 Chapter 1: Initial introduction as hero...", "Act 1 Chapter 2: Faced betrayal, now distrustful..."], "current_summary": "...", ... }} }}, "world_summary": {{ "Neo-Tokyo": {{ "name": "Neo-Tokyo", "progression": [...], ... }}, "Grid": {{ ... }} }} }}
Provided Character Details (use these keys and build progression): {json.dumps(char_details, indent=2)}
Provided World Details (use these keys and build progression): {json.dumps(world_details, indent=2)}
{chapter_parser.get_format_instructions()}"""

        human_prompt = f"""Current Act: {act_id} Current Chapter: {chapter_id} Chapter Text: {current_chap_summary}
Character Details: {char_details}
World Details: {world_details}"""

        def safe_validate(json_str: str):
            if not json_str or not json_str.strip():
                return False, None, "Empty input"
            try:
                parsed = chapter_parser.parse(json_str)
                return True, parsed.model_dump() if hasattr(parsed, 'model_dump') else parsed.dict(), None
            except Exception as e1:
                try:
                    validated = ChapterBundle.model_validate_json(json_str)
                    return True, validated.model_dump() if hasattr(validated, 'model_dump') else validated.dict(), None
                except Exception as e2:
                    return False, None, f"Parse error: {e1}; Validate error: {e2}"

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            if not raw_text:
                clean_resp = "{}"
            else:
                clean_resp = StoryHelpers._strip_code_fences(raw_text)

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            elif not isinstance(clean_resp, str):
                clean_resp = str(clean_resp)
            if not clean_resp.strip():
                clean_resp = "{}"

            del raw_text, resp
            gc.collect()

            success, result, exc = safe_validate(clean_resp)
            if success:
                return result

            # JSON Fixer Fallback
            try:
                fixed_resp = await StoryHelpers.load_json_with_retry(clean_resp, chapter_parser)
                if not fixed_resp:
                    continue
                if isinstance(fixed_resp, dict):
                    fixed_clean = json.dumps(fixed_resp)
                elif not isinstance(fixed_clean, str):
                    fixed_clean = str(fixed_clean)
                if not fixed_clean.strip():
                    fixed_clean = "{}"

                success, result, exc = safe_validate(fixed_clean)
                if success:
                    return result
            except Exception as e:
                pass

        return None
    
    @staticmethod
    async def ingest_act(act_id: int, chapters_text: str, char_details: Dict[str, Any], world_details: Dict[str, Any], max_retries: int = 3):
        """
        Ingests entire act using act chapter summaries, and builds cumulative progressions for characters and worlds.
        """
        system_prompt = f"""
You are the Act Breakdown Agent. Extract relevant and absolutely important info from the Act.
Always include chapter number reference alongside relevant text.
For character_progressions and world_progressions, build cumulative histories by aggregating and appending act-level changes based on the provided chapter summaries and previous details.
Append to progression lists with entries like 'Act {act_id}: [detailed act-level change summary, integrating chapter changes, current state]'.
Update current_ fields to reflect the state at the end of this act.

Respond ONLY in JSON with this schema:
{act_ingestor_parser.get_format_instructions()}
"""
        # print("COMBINED TEXT FOR ACT INGESTION: ",combined_text)
        human_prompt = f"""
Current Act: {act_id}

Act Chapters Summaries:
{chapters_text}

Previous Character Details (build upon for progressions): {json.dumps(char_details, indent=2)}
Previous World Details (build upon for progressions): {json.dumps(world_details, indent=2)}
"""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, ActSummary, act_ingestor_parser)
            
            if success:
                return result

            else:
                print(f"[Attempt {attempt}] Act ingestion validation failed:", exc)
            
            try:
                fixed_resp = await StoryHelpers.load_json_with_retry(clean_resp, act_ingestor_parser)
                
                del clean_resp
                gc.collect()
                
                if isinstance(fixed_resp, dict):
                    fixed_clean = json.dumps(fixed_resp)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, ActSummary, act_ingestor_parser)
                del fixed_clean
                
                if success:
                    return result
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised:", inner_e)
            
            if attempt < max_retries:
                print(f"Retrying ingest_act... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted; returning minimal structure.")
                del system_prompt, human_prompt
                return "failed"