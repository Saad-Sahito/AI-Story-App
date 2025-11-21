# src/utilities/improved_ingestor.py

"""
Improved Ingestor with better prompts based on entity extraction best practices
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
# PYDANTIC MODELS (same as before)
# ============================================================================

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
    progression: List[str] = Field(
        default_factory=list, 
        description="Detailed progression history entries, formatted as 'Act {act} Chapter {chapter}: {detailed change description}'"
    )
    current_summary: str = Field(description="Current inline summary of character's actions/motivations/changes")
    current_traits: List[str] = Field(default_factory=list)
    current_relationships: Dict[str, str] = Field(default_factory=dict)
    current_emotional_state: Optional[str] = None
    current_goals: Optional[str] = None
    current_status: Optional[str] = None


class WorldElementMemory(BaseModel):
    name: str
    progression: List[str] = Field(
        default_factory=list, 
        description="Detailed progression history entries, formatted as 'Act {act} Chapter {chapter}: {detailed change description}'"
    )
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
    character_progressions: Dict[str, CharacterMemory] = Field(
        description="Cumulative character progressions up to this act"
    )
    world_progressions: Dict[str, WorldElementMemory] = Field(
        description="Cumulative world element progressions up to this act"
    )

act_ingestor_parser = PydanticOutputParser(pydantic_object=ActSummary)



class Ingestor:
    """
    Ingestor with enhanced prompts following entity extraction best practices:
    - Explicit entity definitions
    - Clear output format specifications
    - Few-shot examples where helpful
    - Constraint-based focusing
    - Step-by-step extraction guidance
    """
    
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
        """Enhanced scene ingestion with better entity extraction"""
        
        # Build explicit entity definitions
        char_definitions = "\n".join([f"  - {char}: A character in the story" for char in chars])
        world_definitions = "\n".join([f"  - {world}: A world element/location in the story" for world in worlds])
        
        system_prompt = f"""You are a Scene Breakdown Agent specializing in structured entity extraction from narrative text.

=== YOUR TASK ===
Extract and structure information about characters and world elements that appear in the scene.

=== ENTITY DEFINITIONS ===

CHARACTERS (extract if they appear, are mentioned, or are referenced):
{char_definitions if chars else "  - No predefined characters (extract any mentioned)"}

WORLD ELEMENTS (extract if they appear, are mentioned, or influence the scene):
{world_definitions if worlds else "  - No predefined world elements (extract any mentioned)"}

=== EXTRACTION RULES ===

1. SCOPE: Only extract entities that are:
   - Directly present in the scene
   - Mentioned by other characters
   - Referenced in narration
   - Influencing events in the scene

2. NAME CONSISTENCY:
   - Use "old_name" and "new_name" fields for ALL entities
   - If no name change: set both to the SAME name
   - If renamed: old_name = previous name, new_name = new name
   - NEVER create separate entries for the same entity

3. DETAILS FIELD: Include for each entity:
   - What they DID in this scene (actions)
   - What they SAID or thought (if applicable)
   - How they CHANGED (emotions, knowledge, relationships)
   - Their CONDITION at scene end (emotional/physical state)

4. CONTEXT REFERENCES:
   - Always include: Act {act_id}, Chapter {chapter_id}, Scene {scene_id}
   - Format: "In Act {act_id} Chapter {chapter_id} Scene {scene_id}, [entity] did X..."

=== OUTPUT FORMAT ===
{scene_parser.get_format_instructions()}

=== EXAMPLE OUTPUT ===
{{
  "story_summary": "In Act 1 Chapter 2 Scene 3, Alice confronted Marcus in the warehouse, discovering he had been lying about the artifact's location. Their heated argument revealed Alice's growing distrust and Marcus's desperation.",
  "character_details": {{
    "Alice": {{
      "old_name": "Alice",
      "new_name": "Alice",
      "details": "In Act 1 Chapter 2 Scene 3, Alice confronted Marcus with evidence of his lies. She displayed anger and betrayal, her trust completely shattered. By scene end, she resolved to work alone, emotional state: furious and determined."
    }},
    "Marcus": {{
      "old_name": "Marcus",
      "new_name": "Marcus",
      "details": "In Act 1 Chapter 2 Scene 3, Marcus attempted to justify his deception but became increasingly defensive. He appeared desperate, revealing fear of consequences. Emotional state: anxious and cornered."
    }}
  }},
  "world_details": {{
    "Warehouse District": {{
      "old_name": "Warehouse District",
      "new_name": "Warehouse District",
      "details": "In Act 1 Chapter 2 Scene 3, the abandoned warehouse served as a tense confrontation space. Atmosphere: dark, echoing, isolated. The setting amplified the characters' paranoia and secrecy."
    }}
  }}
}}

=== CRITICAL REMINDERS ===
- Output ONLY valid JSON matching the schema
- NO markdown code fences
- NO explanatory text outside the JSON
- Include ALL entities that appear or are mentioned
- Use exact Act/Chapter/Scene numbers in details"""

        human_prompt = f"""=== SCENE TO ANALYZE ===

Location Context: Act {act_id}, Chapter {chapter_id}, Scene {scene_id}

Known Characters: {', '.join(chars) if chars else 'None predefined - extract any mentioned'}
Known World Elements: {', '.join(worlds) if worlds else 'None predefined - extract any mentioned'}

Scene Text:
{scene_text}

=== EXTRACT ENTITIES ===
Follow the rules above and output structured JSON."""

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
        """Enhanced chapter ingestion with progression tracking"""
        
        # Format previous details for clarity
        char_context = json.dumps(char_details, indent=2) if char_details else "No previous character data"
        world_context = json.dumps(world_details, indent=2) if world_details else "No previous world data"
        
        system_prompt = f"""You are a Chapter Breakdown Agent specializing in cumulative story progression tracking.

=== YOUR TASK ===
Analyze the chapter text and generate a structured breakdown that BUILDS UPON previous character and world states.

=== CRITICAL PROGRESSION RULES ===

1. APPEND TO PROGRESSION LISTS (DO NOT REPLACE):
   - Each character's "progression" list MUST include ALL previous entries PLUS new entry
   - Each world element's "progression" list MUST include ALL previous entries PLUS new entry
   - New entry format: "Act {act_id} Chapter {chapter_id}: [What changed: previous state → new state, why it changed, significance]"

2. PROGRESSION ENTRY REQUIREMENTS:
   For Characters:
   - Emotional changes: "was [emotion] → now [emotion] because [reason]"
   - Knowledge gained: "learned that [information], now knows [new understanding]"
   - Goal shifts: "was seeking [old goal] → now pursuing [new goal]"
   - Relationship changes: "relationship with [other] changed from [old] to [new]"
   - Physical changes: "condition changed from [old] to [new]"
   
   For World Elements:
   - Atmospheric shifts: "atmosphere was [old] → now [new] due to [events]"
   - Physical changes: "location changed from [old state] to [new state]"
   - Political/social changes: "power structure shifted from [old] to [new]"
   - Plot relevance: "significance changed from [old] to [new]"

3. CURRENT STATE UPDATES:
   - Update ALL "current_" fields to reflect END of this chapter
   - current_summary: Most recent state (1-2 sentences)
   - current_emotional_state: Single emotion word or short phrase
   - current_goals: What they're pursuing NOW
   - current_status: Physical/social/plot status NOW
   - current_relationships: Dict of {{character_name: relationship_description}}

4. STABILITY HANDLING:
   If a character/world had NO significant change:
   - Still append: "Act {act_id} Chapter {chapter_id}: No significant change, maintained [state]"
   - Keep previous "current_" values or note stability

5. NEW ENTITIES:
   If a character/world appears for FIRST time:
   - progression: ["Act {act_id} Chapter {chapter_id}: First appearance - [initial state and role]"]
   - Fill all current_ fields based on this chapter

=== DATA TO BUILD UPON ===

Previous Character States:
{char_context}

Previous World States:
{world_context}

=== OUTPUT FORMAT ===
{chapter_parser.get_format_instructions()}

=== EXAMPLE CORRECT OUTPUT ===
{{
  "summary": "Chapter summary here covering all major events...",
  "character_summary": {{
    "Alice": {{
      "name": "Alice",
      "progression": [
        "Act 1 Chapter 1: Introduced as naive protagonist seeking truth, emotional state: curious and hopeful",
        "Act 1 Chapter 2: Discovered Marcus's betrayal, trust shattered → now suspicious and angry, learned that allies can deceive, goal shifted from finding truth to exposing lies"
      ],
      "current_summary": "Alice is now a hardened investigator driven by anger and distrust, working alone to expose Marcus's deception.",
      "current_traits": ["determined", "distrustful", "resourceful", "angry"],
      "current_relationships": {{
        "Marcus": "former ally, now enemy due to his betrayal",
        "Detective Chen": "reluctant ally, growing trust"
      }},
      "current_emotional_state": "angry with underlying hurt",
      "current_goals": "expose Marcus's lies and recover the artifact alone",
      "current_status": "physically exhausted, socially isolated, but mentally sharp and driven"
    }},
    "Marcus": {{
      "name": "Marcus",
      "progression": [
        "Act 1 Chapter 1: Introduced as Alice's trusted mentor, emotional state: calm and guiding",
        "Act 1 Chapter 2: Revealed as liar, desperation exposed → now cornered and fearful, goal shifted from mentoring to self-preservation"
      ],
      "current_summary": "Marcus is a desperate man whose lies have been exposed, now trying to salvage his situation.",
      "current_traits": ["desperate", "manipulative", "fearful", "cornered"],
      "current_relationships": {{
        "Alice": "former protégé, now adversary hunting him"
      }},
      "current_emotional_state": "fearful and desperate",
      "current_goals": "escape Alice's pursuit and hide evidence",
      "current_status": "on the run, reputation destroyed, allies abandoning him"
    }}
  }},
  "world_summary": {{
    "Warehouse District": {{
      "name": "Warehouse District",
      "progression": [
        "Act 1 Chapter 1: Introduced as abandoned industrial area, atmosphere: eerie and forgotten",
        "Act 1 Chapter 2: Became site of confrontation, atmosphere shifted from merely eerie → actively hostile and tense, now associated with betrayal"
      ],
      "current_summary": "The Warehouse District is now tainted by the confrontation, representing broken trust.",
      "current_atmosphere": "hostile, tense, claustrophobic",
      "current_culture": "abandoned by law, haven for secrets",
      "current_events": "Alice and Marcus's confrontation exposed lies",
      "current_connections": {{"City Center": "far from civilized areas, deliberately isolated"}}
    }}
  }}
}}

=== CRITICAL VALIDATION CHECKLIST ===
Before outputting, verify:
☐ Every character has progression list with ALL previous entries + new entry
☐ Every world element has progression list with ALL previous entries + new entry  
☐ New progression entries follow format: "Act X Chapter Y: [change details]"
☐ All current_ fields updated to reflect END of chapter
☐ world_summary is NOT empty (required field)
☐ Output is valid JSON with NO markdown, NO explanations outside JSON
☐ All entity keys match EXACTLY the names from provided previous details"""

        human_prompt = f"""=== CHAPTER TO ANALYZE ===

Act: {act_id}
Chapter: {chapter_id}

Chapter Text (all scenes combined):
{current_chap_summary}

Previous Character Details (MUST build upon these):
{char_context}

Previous World Details (MUST build upon these):
{world_context}

=== GENERATE STRUCTURED BREAKDOWN ===
Remember: APPEND to progression lists, UPDATE current_ fields, NEVER omit world_summary."""

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
                elif isinstance(fixed_resp, str):
                    fixed_clean = fixed_resp
                else:
                    fixed_clean = str(fixed_resp)
                    
                if not fixed_clean.strip():
                    fixed_clean = "{}"

                success, result, exc = safe_validate(fixed_clean)
                if success:
                    return result
            except Exception as e:
                pass

        return None
    
    @staticmethod
    async def ingest_act(
        act_id: int, 
        chapters_text: str, 
        char_details: Dict[str, Any], 
        world_details: Dict[str, Any], 
        max_retries: int = 3
    ):
        """Enhanced act ingestion with cumulative progression"""
        
        char_context = json.dumps(char_details, indent=2) if char_details else "No previous character data"
        world_context = json.dumps(world_details, indent=2) if world_details else "No previous world data"
        
        system_prompt = f"""You are an Act Breakdown Agent creating high-level summaries with cumulative entity tracking.

=== YOUR TASK ===
Synthesize the act's chapter summaries into a coherent act-level narrative while tracking character and world progressions.

=== ACT-LEVEL PROGRESSION RULES ===

1. AGGREGATE CHAPTER CHANGES:
   - Review all chapter-level progression entries
   - Synthesize into act-level understanding
   - Append ONE act-level entry per entity: "Act {act_id}: [major arc changes across all chapters]"

2. ACT ENTRY FORMAT:
   "Act {act_id}: [Entity's] major transformation - started as [initial state in act], through [key events in chapters], ended as [final state in act]. Key turning point: [chapter X event]. Overall significance: [impact on story]"

3. HIGHLIGHT KEY CHAPTERS:
   Reference specific chapters where major changes occurred
   Example: "Act 2: Alice's trust completely shattered (Chapter 3), leading to isolation (Chapter 5)"

4. CUMULATIVE PROGRESSION:
   - Include ALL previous act entries from char_details/world_details
   - Add new act entry
   - Update current_ fields to reflect END of this act

=== OUTPUT FORMAT ===
{act_ingestor_parser.get_format_instructions()}

=== EXAMPLE OUTPUT ===
{{
  "act_summary": "Act 2 covered the protagonist's descent into paranoia across 5 chapters. Key events included the betrayal revelation (Ch 2), the failed alliance (Ch 4), and the desperate gambit (Ch 5). The act concludes with the protagonist isolated but determined.",
  "character_progressions": {{
    "Alice": {{
      "name": "Alice",
      "progression": [
        "Act 1: Introduced as naive truth-seeker, discovered the conspiracy, ended with determination to investigate",
        "Act 2: Trust completely shattered (Chapter 3 betrayal), descended into paranoia and isolation (Chapters 4-5), ended as hardened lone wolf. Transformation from hopeful investigator → suspicious vigilante. Key turning point: Marcus's betrayal in Chapter 3."
      ],
      "current_summary": "Alice is now a hardened, isolated vigilante operating outside all alliances, driven by vengeance and distrust.",
      "current_traits": ["paranoid", "ruthless", "isolated", "skilled"],
      "current_relationships": {{"Everyone": "distrusts all, works alone"}},
      "current_emotional_state": "cold determination with underlying trauma",
      "current_goals": "expose the conspiracy alone, regardless of cost",
      "current_status": "physically exhausted, emotionally damaged, socially isolated, but highly dangerous"
    }}
  }},
  "world_progressions": {{
    "City": {{
      "name": "City",
      "progression": [
        "Act 1: Introduced as seemingly normal urban setting, hints of corruption beneath surface",
        "Act 2: Conspiracy revealed to reach highest levels (Chapter 2), atmosphere shifted from normal → oppressive and surveillance-heavy (Chapters 3-5). Public unaware of rot beneath. Became character in its own right as hostile environment."
      ],
      "current_summary": "The City is now revealed as thoroughly corrupt system actively hunting the protagonist.",
      "current_atmosphere": "oppressive, paranoid, hostile to truth-seekers",
      "current_culture": "facade of normalcy hiding systemic corruption",
      "current_events": "manhunt for Alice, cover-up operations ongoing",
      "current_connections": {{"Underground": "only safe zones outside official control"}}
    }}
  }}
}}

Previous Character Progressions:
{char_context}

Previous World Progressions:
{world_context}

=== VALIDATION ===
☐ act_summary covers major events across all chapters
☐ Each entity has ONE new act-level progression entry
☐ Progression entries reference specific chapters for major turning points
☐ All current_ fields updated to end-of-act state
☐ Valid JSON, no markdown"""

        human_prompt = f"""Act {act_id} - Chapter Summaries:
{chapters_text}

Previous Character Progressions:
{char_context}

Previous World Progressions:
{world_context}

Generate act-level breakdown with cumulative progressions."""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                clean_resp, ActSummary, act_ingestor_parser
            )
            
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

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    fixed_clean, ActSummary, act_ingestor_parser
                )
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
                return None