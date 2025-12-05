"""
Highly Flexible Ingestor with adaptive schemas
- progression lists are strictly preserved and appended
- all other fields are dynamic/optional — LLM decides what's relevant
- OPTIMIZED: World elements are now hierarchical and selective
"""

import json
import asyncio
import gc
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.utilities.story_helpers import StoryHelpers
from src.llm_client.llm_client import ingestor_client


# ============================================================================
# FLEXIBLE & MINIMAL PYDANTIC MODELS
# ============================================================================

class EntityDetails(BaseModel):
    old_name: str = Field(description="Previous or current name of the character or world element")
    new_name: str = Field(description="New name if changed, otherwise same as old_name")
    details: str = Field(description="Scene-relevant summary details")
    significance: Optional[str] = Field(
        default=None,
        description="Why this location matters (for world elements only)"
    )


class SceneBundle(BaseModel):
    story_summary: str = Field(description="Detailed summary of the scene")
    character_details: Dict[str, EntityDetails] = Field(
        default_factory=dict,
        description="Dictionary: {character_key: {old_name, new_name, details}} — only include entities that appear or are referenced"
    )
    world_details: Dict[str, EntityDetails] = Field(
        default_factory=dict,
        description="""Dictionary: {world_key: {old_name, new_name, details, significance}} 
        IMPORTANT: Extract ONLY significant locations following these rules:
        1. Use BROAD geographic areas (e.g., 'northern_district', 'harbor_quarter') NOT specific buildings
        2. Extract specific locations ONLY if they are:
           - Recurring across multiple scenes
           - Central to plot events
           - Symbolically/thematically important
        3. Consolidate similar locations (e.g., multiple taverns → 'tavern_district' or just the main tavern)
        4. Prefer 3-7 world elements per scene maximum
        5. When in doubt, go broader rather than more specific"""
    )

# scene_parser = PydanticOutputParser(pydantic_object=SceneBundle)


# === FLEXIBLE CHARACTER MEMORY (only progression is enforced) ===
class CharacterMemory(BaseModel):
    name: str = Field(description="Exact name of the character")
    progression: List[str] = Field(
        default_factory=list,
        description="""MUST follow format: 'Act {act} Chapter {chapter}: {detailed change description}'.
        This list MUST be cumulative — include all previous entries + new one."""
    )
    # Everything else is optional and free-form
    current_summary: Optional[str] = None
    current_traits: Optional[List[str]] = None
    current_relationships: Optional[Dict[str, str]] = None
    current_emotional_state: Optional[str] = None
    current_goals: Optional[str] = None
    current_status: Optional[str] = None
    # Allow any additional dynamic fields
    model_config = {"extra": "allow"}


# === FLEXIBLE WORLD MEMORY (only progression enforced) ===
class WorldElementMemory(BaseModel):
    name: str = Field(description="Exact name of the world element")
    progression: List[str] = Field(
        default_factory=list,
        description="""MUST follow format: 'Act {act} Chapter {chapter}: {detailed change description}'.
        Cumulative — include all prior entries + new one."""
    )
    hierarchy_level: Optional[str] = Field(
        default=None,
        description="'broad' (regions/districts) or 'specific' (individual important locations)"
    )
    frequency: Optional[int] = Field(
        default=None,
        description="Number of times referenced across the story"
    )
    current_summary: Optional[str] = None
    current_atmosphere: Optional[str] = None
    current_culture: Optional[str] = None
    current_events: Optional[str] = None
    current_connections: Optional[Union[Dict[str, str], str]] = None
    # Allow any additional dynamic fields
    model_config = {"extra": "allow"}


# === FLEXIBLE CHAPTER BUNDLE ===
class ChapterBundle(BaseModel):
    summary: str = Field(description="Detailed summary of the entire chapter")
    character_summary: Dict[str, CharacterMemory] = Field(default_factory=dict)
    world_summary: Dict[str, WorldElementMemory] = Field(default_factory=dict)
    # Allow any extra top-level fields (e.g. themes, foreshadowing, etc.)
    model_config = {"extra": "allow"}


# chapter_parser = PydanticOutputParser(pydantic_object=ChapterBundle)


# === FLEXIBLE ACT SUMMARY ===
class ActSummary(BaseModel):
    act_summary: str = Field(description="High-level summary of the entire Act")
    character_progressions: Dict[str, CharacterMemory] = Field(default_factory=dict)
    world_progressions: Dict[str, WorldElementMemory] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


# act_ingestor_parser = PydanticOutputParser(pydantic_object=ActSummary)


class Ingestor:

    @staticmethod
    async def ingest_scene(
        act_id: int,
        chapter_id: int,
        scene_id: int,
        scene_text: str,
        chars: List[str],
        worlds: List[str],
    ) -> Optional[Dict[str, Any]]:
        """Enhanced scene ingestion with smarter world element extraction"""
        
        # Build context of existing world elements with their hierarchy
        world_context = ""
        if worlds:
            world_context = "\n".join([f"  - {world} (already tracked)" for world in worlds])

        system_prompt = f"""You are a precise Scene Breakdown Agent with SMART location extraction.

**CRITICAL WORLD ELEMENT RULES:**
1. Extract BROAD locations by default (districts, regions, general areas)
2. Extract SPECIFIC locations ONLY if they meet AT LEAST ONE of these criteria:
   - Appears in multiple scenes (recurring)
   - Central to a major plot event
   - Has strong thematic/symbolic significance
   - A named, story-critical landmark

3. CONSOLIDATION RULES:
   - Multiple small locations in same area → Use the broader area name
   - Generic places (random tavern, unnamed alley) → Skip or use district name
   - "The marketplace in the eastern quarter" → Just "eastern_quarter"

4. TARGET: 3-7 world elements per scene maximum (fewer is better)

5. Use hierarchical naming:
   - Broad: "harbor_district", "northern_territories", "capital_city"
   - Specific: "ivory_tower", "blacksmith_forge_of_theron" (only if truly important)

Existing tracked worlds:
{world_context if world_context else 'None yet'}

Output valid JSON only. No markdown. No explanations.

{SceneBundle.model_json_schema()}"""

        human_prompt = f"""Act {act_id} Chapter {chapter_id} Scene {scene_id}

Known Characters: {', '.join(chars) if chars else 'None predefined'}

Scene Text:
{scene_text}

Extract structured data. Remember: BE SELECTIVE with world elements. Quality over quantity.
Focus on significant, recurring, or broadly-defined locations."""

        
        resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            fixed_resp = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=SceneBundle)
            return fixed_resp.model_dump()
        except BaseException as e:
            raise f"[Scene Ingestor] Fixer failed: {e}"


        

    @staticmethod
    async def ingest_chapter(
        act_id: int,
        chapter_id: int,
        current_chap_summary: str,
        char_details: Dict[str, Any],
        world_details: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Chapter ingestion with world element consolidation"""
        
        char_context = json.dumps(char_details, indent=2) if char_details else "None"
        world_context = json.dumps(world_details, indent=2) if world_details else "None"

        system_prompt = f"""You are a Chapter Progression Tracker with INTELLIGENT location consolidation.

**MANDATORY RULES:**
1. Every character and world element MUST have a "progression" list that includes ALL previous entries + one new entry:
   "Act {act_id} Chapter {chapter_id}: [description of what changed or remained stable]"

2. **WORLD ELEMENT CONSOLIDATION:**
   - Merge similar/adjacent locations into broader areas when appropriate
   - Track frequency: increment count for recurring locations
   - Mark hierarchy_level: "broad" or "specific"
   - Drop world elements that appeared once and had no significant impact
   - Aim for 5-12 world elements per chapter (consolidate more if needed)

3. Everything else is flexible: Include only relevant fields.

Previous state (build upon this exactly for progression lists):
Characters:
{char_context}

World Elements:
{world_context}

Output valid JSON only. No markdown.

{ChapterBundle.model_json_schema()}"""

        human_prompt = f"""Act {act_id} - Chapter {chapter_id}

Full Chapter Text:
{current_chap_summary}

Generate cumulative chapter breakdown.
**CONSOLIDATE world elements:** Merge minor locations into broader areas. Keep only significant ones.
Preserve and extend all progression lists.
Update frequency counts for recurring locations."""

        resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            fixed_resp = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=ChapterBundle)
            return fixed_resp.model_dump()
        except BaseException as e:
            raise f"[Chapter Ingestor] Fixer failed: {e}"




    @staticmethod
    async def ingest_act(
        act_id: int,
        chapters_text: str,
        char_details: Dict[str, Any],
        world_details: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Act ingestor with final world element pruning"""
        
        char_context = json.dumps(char_details, indent=2) if char_details else "None"
        world_context = json.dumps(world_details, indent=2) if world_details else "None"

        system_prompt = f"""You are an Act-Level Story Architect with STRATEGIC location curation.

**Critical Rules:**
1. Every character and world element MUST have a "progression" list with:
   - All prior act-level entries
   - ONE new entry: "Act {act_id}: [major arc transformation across this act, referencing key chapters]"

2. **WORLD ELEMENT PRUNING:**
   - Keep ONLY locations that:
     a) Appear in 3+ chapters, OR
     b) Are central to act-level plot developments, OR
     c) Have strong thematic significance
   - Remove one-off or minor locations
   - Prefer broad geographic areas over specific buildings
   - Target: 8-15 world elements per act maximum

3. All other fields optional and adaptive.

Previous Progressions (preserve and extend):
Characters:
{char_context}

World (prune and consolidate):
{world_context}

Output valid JSON only.

{ActSummary.model_json_schema()}"""

        human_prompt = f"""Act {act_id} - All Chapter Summaries:
{chapters_text}

Synthesize into act-level summary with cumulative progressions.
**PRUNE world elements:** Keep only truly significant locations.
Extend progression lists with one new Act {act_id} entry per entity."""

        resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            fixed_resp = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=ActSummary)
            
            return fixed_resp.model_dump()
            
        except BaseException as e:
            raise f"[Act Ingestor] Fixer failed: {e}"