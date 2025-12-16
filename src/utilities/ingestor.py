"""
Balanced & Highly Flexible Story Ingestor v2
- Equal weight on characters AND world elements
- All progression entries are tagged with Act/Chapter(/Scene)
- Smart consolidation & pruning for both entities and locations
- Minimal enforced structure, maximum adaptability
"""

import json
import gc
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from src.utilities.story_helpers import StoryHelpers
from src.llm_client.llm_client import ingestor_client


# ============================================================================
# FLEXIBLE & BALANCED PYDANTIC MODELS
# ============================================================================

class EntityDetails(BaseModel):
    name: str = Field(description="Current name of the character or world element")
    previous_names: List[str] = Field(default=[], description="Former names if renamed")
    details: str = Field(description="Concise, scene-relevant description or change summary")
    significance: Optional[str] = Field(default=None, description="Why this entity/location is important in this scene (if applicable)")


class SceneBundle(BaseModel):
    act_id: int
    chapter_id: int
    scene_id: int
    story_summary: str = Field(description="Rich, detailed summary of the scene")

    # Only include entities that actually appear or are meaningfully referenced
    characters: Dict[str, EntityDetails] = Field(
        default_factory=dict,
        description="Key: character identifier. Only significant appearances."
    )
    locations: Dict[str, EntityDetails] = Field(
        default_factory=dict,
        description="""Key: location identifier. STRICT rules:
        - Prefer BROAD areas (e.g., 'harbor_district', 'capital_city')
        - Specific places ONLY if recurring, plot-central, or thematically vital
        - Consolidate similar locations
        - Max 5–8 locations per scene (fewer = better)"""
    )


class CharacterMemory(BaseModel):
    name: str = Field(description="Canonical name of the character")
    progression: List[str] = Field(
        default_factory=list,
        description="""Cumulative list. Every entry MUST be formatted as:
        'Act {act} Chapter {chap} Scene {scene}: {what changed or was revealed}' 
        or 'Act {act} Chapter {chap}: {chapter-level change}' for chapter/act summaries.
        NEVER break cumulative order."""
    )
    frequency: Optional[int] = Field(default=0, description="Total scenes this character has appeared/referenced in")
    current_role: Optional[str] = None
    current_motivation: Optional[str] = None
    current_relationships: Optional[Dict[str, str]] = None
    current_status: Optional[str] = None
    current_location: Optional[str] = None
    # Fully dynamic beyond this
    model_config = {"extra": "allow"}


class LocationMemory(BaseModel):
    name: str = Field(description="Canonical name of the location")
    progression: List[str] = Field(
        default_factory=list,
        description="""Cumulative. Format:
        'Act {act} Chapter {chap} Scene {scene}: {what happened or changed here}'
        or broader for chapter/act level."""
    )
    hierarchy: Optional[str] = Field(default=None, description="'broad' (region/district) or 'specific' (named landmark)")
    frequency: Optional[int] = Field(default=0, description="Total scenes this location has been mentioned in")
    current_state: Optional[str] = None
    current_atmosphere: Optional[str] = None
    connected_to: Optional[List[str]] = None
    thematic_role: Optional[str] = None
    # Fully dynamic
    model_config = {"extra": "allow"}


class ChapterBundle(BaseModel):
    act_id: int
    chapter_id: int
    summary: str = Field(description="Comprehensive chapter summary")
    characters: Dict[str, CharacterMemory] = Field(default_factory=dict)
    locations: Dict[str, LocationMemory] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class ActSummary(BaseModel):
    act_id: int
    act_summary: str = Field(description="High-level summary of the entire act")
    characters: Dict[str, CharacterMemory] = Field(default_factory=dict)
    locations: Dict[str, LocationMemory] = Field(default_factory=dict)
    model_config = {"extra": "allow"}


class Ingestor:

    # ============================================================================
    # SCENE INGESTION – Balanced character & location focus
    # ============================================================================
    @staticmethod
    async def ingest_scene(
        act_id: int,
        chapter_id: int,
        scene_id: int,
        scene_text: str,
        known_characters: List[str],
        known_locations: List[str],
    ) -> Optional[Dict[str, Any]]:

        known_chars = ", ".join(known_characters) if known_characters else "None"
        known_locs = "\n".join([f"  - {loc}" for loc in known_locations]) if known_locations else "None yet"

        system_prompt = f"""You are a precision Scene Analyst. Your job is to extract ONLY meaningful characters and locations from the scene with equal rigor.

CHARACTER RULES (as strict as location rules):
- Only include characters who speak, act, are described in detail, or are meaningfully referenced
- Minor unnamed guards, passersby, crowds → ignore unless plot-relevant
- Consolidate: e.g "the three assassins" → treat as one entity if they function as a unit
- Describe exactly why and when the character is significant or when describing it in 'details', dont just mention 'in this scene', but rather precise context, as the scene would be lost afterwards. MUST also mention Act/Chapter/Scene number, with every detail/significance.


LOCATION RULES:
- Prefer broad geographic areas (districts, forests, cities)
- Specific buildings/landmarks only if recurring, plot-critical, or symbolically loaded
- Consolidate similar places
- Describe exactly why and when the location is significant or when describing it in 'details', dont just mention 'in this scene', but rather precise context, as the scene would be lost afterwards. MUST also mention Act/Chapter/Scene number, with every detail/significance.


Existing tracked characters: {known_chars}
Existing tracked locations:
{known_locs}

Every progression entry must include Act/Chapter/Scene numbers.

Output VALID JSON only. No markdown.

{SceneBundle.model_json_schema()}"""

        human_prompt = f"""Act {act_id} Chapter {chapter_id} Scene {scene_id}

Scene text:
{scene_text}

Extract structured scene data. Be highly selective with both characters and locations.
Tag all changes with full Act/Chapter/Scene context."""

        resp, tokens = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            parsed, utility_tokens = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=SceneBundle)
            if isinstance(parsed, tuple):
                parsed = parsed[0]
            result = parsed.model_dump()
            # print("INGEST SCENE: ", result)
            result["act_id"] = act_id
            result["chapter_id"] = chapter_id
            result["scene_id"] = scene_id
            return result, tokens, utility_tokens
        except Exception as e:
            raise RuntimeError(f"[Scene Ingestor] JSON fixing failed: {e}")


    # ============================================================================
    # CHAPTER INGESTION – Cumulative + balanced consolidation
    # ============================================================================
    @staticmethod
    async def ingest_chapter(
        act_id: int,
        chapter_id: int,
        chapter_text: str,
        current_characters: Dict[str, Any],
        current_locations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        char_ctx = json.dumps(current_characters, indent=2) if current_characters else "None"
        loc_ctx = json.dumps(current_locations, indent=2) if current_locations else "None"

        system_prompt = f"""You are a Chapter Memory Consolidator.

MANDATORY:
- Every character and location MUST have a "progression" list that includes ALL prior entries + ONE new entry tagged:
  "Act {act_id} Chapter {chapter_id}: [summary of change/development in this chapter]"

SMART CONSOLIDATION:
Characters:
- Merge one-off nameless characters into archetypes if needed
- Drop characters who appeared once and had zero impact
- Increment frequency counters

Locations:
- Merge minor locations into broader areas
- Drop one-off insignificant places
- Prefer broad regions; keep specific ones only if recurring or pivotal
- Target: 8–18 total locations by end of chapter

Previous state (extend progression lists exactly):
Characters:
{char_ctx}

Locations:
{loc_ctx}

Output valid JSON only.

{ChapterBundle.model_json_schema()}"""

        human_prompt = f"""Act {act_id} - Chapter {chapter_id}

Full chapter summary:
{chapter_text}

Update cumulative memory. Extend every progression list with a new "Act {act_id} Chapter {chapter_id}: ..." entry.
Consolidate ruthlessly but intelligently — keep only what matters long-term."""

        resp, tokens = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            parsed, utility_tokens = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=ChapterBundle)
            if isinstance(parsed, tuple):
                parsed = parsed[0]
            return parsed.model_dump(), tokens, utility_tokens
        except Exception as e:
            raise RuntimeError(f"[Chapter Ingestor] JSON fixing failed: {e}")


    # ============================================================================
    # ACT INGESTION – Final pruning, equal character/location treatment
    # ============================================================================
    @staticmethod
    async def ingest_act(
        act_id: int,
        all_chapter_summaries: str,
        final_characters: Dict[str, Any],
        final_locations: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:

        char_ctx = json.dumps(final_characters, indent=2) if final_characters else "None"
        loc_ctx = json.dumps(final_locations, indent=2) if final_locations else "None"

        system_prompt = f"""You are an Act-Level Story Architect.

RULES:
1. Extend every progression list with exactly one new entry:
   "Act {act_id}: [major arc transformation or stabilization across the entire act]"

2. FINAL PRUNING (applied equally to characters and locations):
   - Keep characters who appeared in 3+ chapters OR drive major plot/theming
   - Keep locations that appeared in 3+ chapters OR are act-defining
   - Remove one-off or peripheral entities
   - Target: 12–25 characters and 10–20 locations max per act

Previous state (preserve and extend):
Characters:
{char_ctx}

Locations:
{loc_ctx}

Output valid JSON only.

{ActSummary.model_json_schema()}"""

        human_prompt = f"""Act {act_id} - Complete Act Content:
{all_chapter_summaries}

Synthesize act-level summary.
Add one final progression entry per entity tagged with "Act {act_id}: ...".
Prune aggressively but fairly — only enduring characters and locations survive."""

        resp, tokens = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        del raw_text, resp
        gc.collect()

        try:
            parsed, utility_tokens = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=ActSummary)
            if isinstance(parsed, tuple):
                parsed = parsed[0]
            result = parsed.model_dump()
            result["act_id"] = act_id
            return result, tokens, utility_tokens
        except Exception as e:
            raise RuntimeError(f"[Act Ingestor] JSON fixing failed: {e}")