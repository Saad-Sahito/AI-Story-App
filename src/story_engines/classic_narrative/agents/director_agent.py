# src/story_engines/classic_adventure/agents/director_agent.py

import json
import asyncio
import gc
import traceback
from typing import Any, Dict, List, Literal, Optional, Union
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
from .shared_scene_planner import UserSceneContext
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import director_client, ingestor_client
from config_vars import tier_1_monthly_words_limit, tier_2_monthly_words_limit


# ============================================================================
# STATE AND MODELS
# ============================================================================

@dataclass
class StoryState:
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    story_word_count: int = 0
    current_chapter_word_count: int = 0
    current_act_id: int = 1
    chapter_plan: Optional[Dict[str, Any]] = field(default_factory=dict)
    next_action: str = ""
    error_message: Optional[str] = None


class ScenePlan(BaseModel):
    scene_id: int = Field(..., description="Unique identifier for the scene, sequential starting from 1")
    chapter_number: int = Field(..., description="Sequential chapter index in the story")
    chapter_title: str = Field(..., description="Title of the chapter this scene belongs to")
    narrative_purpose: str = Field(..., description="What this chapter accomplishes")
    emotional_arc: str = Field(..., description="Emotional progression of this chapter")
    chapter_closure_condition: str = Field(..., description="Condition for chapter resolution")
    scene_goal: str = Field(..., description="Immediate purpose of this scene")
    main_characters: List[str] = Field(..., description="Characters appearing in this scene")
    location: str = Field(..., description="Where the scene takes place")
    time_context: str = Field(..., description="When the scene occurs")
    key_events: List[str] = Field(..., description="Key narrative events that must occur")
    emotional_beats: List[str] = Field(..., description="Target emotional beats")
    thematic_notes: List[str] = Field(..., description="How this scene reinforces story themes")
    word_count: int = Field(..., description="Target word count for the scene")
    style_guide: str = Field(..., description="prose_style, pov, tense, narrative_voice that the writer must follow")

class DirectorOutput(BaseModel):
    scenes: List[ScenePlan] = Field(..., description="List of detailed scene plans for Scene Writer")
    action: Literal["generate_and_ingest", "END"] = Field(description="'generate_and_ingest' to continue, 'END' if story complete")

director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)


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
    chapter_id: str
    summary: str = Field(description="Inline summary of character's actions/motivations/changes")
    traits: List[str] = Field(default_factory=list)
    relationships: Dict[str, str] = Field(default_factory=dict)
    emotional_state: Optional[str] = None
    goals: Optional[str] = None
    status_changes: Optional[str] = None

class WorldElementMemory(BaseModel):
    name: str
    chapter_id: str
    summary: str = Field(description="Summary of how this world element appeared/changed")
    atmosphere: Optional[str] = None
    culture: Optional[str] = None
    events: Optional[str] = None
    connections: Union[Dict[str, str], str] = Field(default_factory=dict)

class ChapterBundle(BaseModel):
    summary: str = Field(description="Detailed summary of the entire chapter")
    character_summary: Dict[str, CharacterMemory]
    world_summary: Dict[str, WorldElementMemory]

chapter_parser = PydanticOutputParser(pydantic_object=ChapterBundle)


# ============================================================================
# INGESTOR CLASS
# ============================================================================

class Ingestor:
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system

    async def ingest_scene(self, state: StoryState, scene_text: str, writer_token_usage: dict, max_retries: int = 3) -> Dict[str, str]:
        chars = await self.memory.get_long_term_characters_names()
        worlds = await self.memory.get_long_term_worlds_names()
        
        system_prompt = f"""
You are the Scene Breakdown Agent. Extract structured information from the scene.

Always include:
- `chapter_id` and `scene_id` in each entity's details.
- Both `old_name` and `new_name` fields for every character and world element:
    - If no rename occurred, keep both the same.
    - If renamed, set `old_name` to the previous name and `new_name` to the new one.

Only include relevant character and world names that appear or are mentioned.
If a name change occurs, clearly indicate it through old_name/new_name instead of creating extra keys.

Respond ONLY in JSON matching this schema:
{scene_parser.get_format_instructions()}
"""
        
        human_prompt = f"""
Current Chapter: {state.current_chapter_id}, Current Scene: {state.scene_id}

Scene:
{scene_text}

Character Names: {chars}
World Names: {worlds}
"""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, SceneBundle, scene_parser)
            
            if success:
                new_word_count = StoryHelpers._count_words_split(scene_text)
                state.current_chapter_word_count += new_word_count
                state.story_word_count += new_word_count
                
                await self.memory.update_story_progress(metadata={
                    "latest_chapter_id": state.current_chapter_id, 
                    "continue_scene_id": state.scene_id + 1, 
                    "story_word_count": state.story_word_count,
                    "chapter_word_count": state.current_chapter_word_count,
                    #"director_token_usage": director_token_usage,
                    "writer_token_usage": writer_token_usage
                })
                del system_prompt, human_prompt, new_word_count
                return result
            else:
                print(f"[Attempt {attempt}] Scene ingestion validation failed:", exc)

            try:
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, SceneBundle, scene_parser)
                del fixed_clean
                
                if success:
                    new_word_count = StoryHelpers._count_words_split(scene_text)
                    state.current_chapter_word_count += new_word_count
                    state.story_word_count += new_word_count
                    
                    await self.memory.update_story_progress(metadata={
                        "latest_chapter_id": state.current_chapter_id, 
                        "continue_scene_id": state.scene_id + 1, 
                        "story_word_count": state.story_word_count,
                        "chapter_word_count": state.current_chapter_word_count,
                        #"director_token_usage": director_token_usage,
                        "writer_token_usage": writer_token_usage
                    })
                    #await self.memory.update_user_monthly_word_count(word_count=new_word_count)
                    del system_prompt, human_prompt, new_word_count
                    return result

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised:", inner_e)

            if attempt < max_retries:
                print(f"Retrying ingest_scene... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted; returning minimal structure.")
                del system_prompt, human_prompt
                return {"story_summary": "", "character_details": {}, "world_details": {}}

    async def ingest_chapter(self, state: StoryState, current_chap_summary: str, max_retries: int = 3) -> Dict[str, Any]:
        print("Ingesting chapter...")

        world_details = await self.memory.get_long_term_recent_worlds(state.current_chapter_id)
        char_details = await self.memory.get_long_term_recent_characters(state.current_chapter_id)

        system_prompt = f"""
You are the Chapter Breakdown Agent.

Analyze the entire chapter text and generate a structured breakdown.

Output JSON with:
1. summary - Detailed chapter summary
2. character_summary - Dict of CharacterMemory objects
3. world_summary - Dict of WorldElementMemory objects

Preserve exact names and only use information provided.
No markdown, explanations, or text outside JSON.

{chapter_parser.get_format_instructions()}
"""

        human_prompt = f"""
Current Chapter: {state.current_chapter_id}
Chapter Text:
{current_chap_summary}

Character Details:
{char_details}

World Details:
{world_details}
"""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            print(clean_resp)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, ChapterBundle, chapter_parser)
            
            if success:
                await self.memory.add_post_chapter_bundle(
                    parts=result, 
                    metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title,"act_id": state.current_act_id}
                )
                
                await self.memory.increment_chapter(word_count_delta=0, scene_id=1)
                
                state.current_chapter_id += 1
                state.scene_id = 1
                
                print("✅ Chapter Complete!")
                result.update({
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    
                })
                del system_prompt, human_prompt
                return result

            print(f"[Attempt {attempt}] Chapter ingestion validation failed:", exc)

            try:
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, ChapterBundle, chapter_parser)
                del fixed_clean
                
                if success:
                    await self.memory.add_post_chapter_bundle(
                        parts=result, 
                        metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title,"act_id": state.current_act_id}
                    )
                    
                    await self.memory.increment_chapter(word_count_delta=0, scene_id=1)
                    
                    state.current_chapter_id += 1
                    state.scene_id = 1
                    
                    print("✅ Chapter Complete!")
                    result.update({
                        "current_chapter_id": state.current_chapter_id,
                        "scene_id": state.scene_id,
                        
                    })
                    del system_prompt, human_prompt
                    return result

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised:", inner_e)

            if attempt < max_retries:
                print(f"Retrying ingest_chapter... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted; returning minimal structure.")
                del system_prompt, human_prompt
                return {
                    "summary": "",
                    "character_summary": {},
                    "world_summary": {},
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    
                }


# ============================================================================
# DIRECTOR GRAPH
# ============================================================================

class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("director_node")
        self.graph.add_node("director_node", self.director_node)
        self.graph.add_node("generate_and_ingest", self.generate_and_ingest_node)
        self.graph.add_node("ingest_chapter", self.ingest_chapter)
        self.graph.add_node("story_complete", self.story_complete)
        self.graph.add_node("error_termination", self.error_termination)

        self.graph.add_conditional_edges(
            "director_node",
            lambda state: state.next_action,
            {
                "generate_and_ingest": "generate_and_ingest",
                "END": "story_complete",
                "ERROR": "error_termination",
            },
        )
        
        def route_after_generate(state: StoryState):
            if state.next_action == "END":
                return "END"
            if state.next_action == "ERROR":
                return "error_termination"
            return "ingest_chapter"
        
        self.graph.add_conditional_edges(
            "generate_and_ingest",
            route_after_generate,
            {
                "ingest_chapter": "ingest_chapter",
                "END": END,
                "error_termination": "error_termination",
            }
        )
        
        #self.graph.add_edge("generate_and_ingest", END)
        self.graph.add_edge("ingest_chapter", END)
        self.graph.add_edge("story_complete", END)
        self.graph.add_edge("error_termination", END)
        self.compiled = self.graph.compile()
        
        self.act_title = ""
        self.current_chap_summary = ""
        self.llm_temp = 0.7
        self.model = ""
        

    async def story_complete(self, state: StoryState):
        print("✅ Marking story as complete")
        await self.memory.mark_story_complete()
        status_payload = {"type": "status", "message": "story complete"}
        try:
            self.scene_chunk_callback(status_payload)
        except Exception as e:
            print(f"⚠️ scene_chunk_callback raised: {e}")
        return state

    async def error_termination(self, state: StoryState):
        print(f"❌ Terminating due to error: {state.error_message or 'Unknown error'}")
        status_payload = {"type": "status", "message": f"Error: {state.error_message or 'Unknown error'}"}
        try:
            self.scene_chunk_callback(status_payload)
        except Exception as e:
            print(f"⚠️ scene_chunk_callback raised: {e}")
        return state

    async def _get_current_chapter_outline(self, state: StoryState) -> Optional[Dict[str, Any]]:
        try:
            act_plan_json = await self.memory.get_long_term_document(
                metadata={
                    'type': 'act_plan',
                    'act_id': state.current_act_id,
                    
                    'story_title': state.story_title
                }
            )
            
            if not act_plan_json:
                print(f"⚠️ No act plan found for act {state.current_act_id}")
                return None
            
            act_plan = json.loads(act_plan_json)
            chapter_outlines = act_plan.get('chapter_outlines', [])
            self.act_title = act_plan.get('act_title', 'Unknown')
            self.scene_chunk_callback({"type":"act_title", "message": self.act_title})
            for chapter_outline in chapter_outlines:
                if chapter_outline.get('chapter_number') == state.current_chapter_id:
                    print(f"✅ Found chapter outline for chapter {state.current_chapter_id}")
                    return chapter_outline
            
            print(f"⚠️ Chapter {state.current_chapter_id} not found in act {state.current_act_id} plan")
            return None
            
        except Exception as e:
            print(f"❌ Error loading chapter outline: {e}")
            traceback.print_exc()
            return None

    async def _chapter_outline_to_scene_plans(
        self, 
        chapter_outline: Dict[str, Any], 
        state: StoryState,
        director_context: str
    ) -> List[Dict[str, Any]]:
        story_seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': state.story_title}
        )
        story_seed = json.loads(story_seed_json) if story_seed_json else {}
        
        # Extract chapter target word count and expected scenes from chapter outline
        chapter_word_target = chapter_outline.get('target_word_count', 3000)
        expected_scenes = len(chapter_outline.get('key_scenes', [])) or 3
        
        # Calculate target word count per scene
        target_words_per_scene = int(chapter_word_target / expected_scenes)
        
        print(f"📊 Chapter Word Count Planning:")
        print(f"   Chapter target: {chapter_word_target} words")
        print(f"   Expected scenes: {expected_scenes}")
        print(f"   → Target per scene: {target_words_per_scene} words")
        
        system_prompt = f"""
You are the Scene Planner working under the Director.

You have received a chapter outline from the Act Plan. Your job is to break it into {expected_scenes} detailed scene plans that the Scene Writer will execute.

Chapter Outline:
{json.dumps(chapter_outline, indent=2)}

Story Style Guidelines (to be passed to scene writer):
- POV: {story_seed.get('style_guide', {}).get('pov', 'Third-person')}
- Prose Style: {story_seed.get('style_guide', {}).get('prose_style', '')}
- Tense: {story_seed.get('style_guide', {}).get('tense', '')}
- Narrative Voice: {story_seed.get('style_guide', {}).get('narrative_voice', '')}
- Tone: {story_seed.get('tone', 'Balanced')}
- Genre: {story_seed.get('genre', 'Fiction')}

╔═══════════════════════════════════════════════════════════╗
║ WORD COUNT REQUIREMENTS (CRITICAL)                         ║
╠═══════════════════════════════════════════════════════════╣
║ Chapter Target:     {chapter_word_target:>5} words         ║
║ Expected Scenes:    {expected_scenes:>5}                   ║
║ Target Per Scene:   {target_words_per_scene:>5} words      ║
╚═══════════════════════════════════════════════════════════╝

Your Output:
Create {expected_scenes} detailed scene plans that:
1. Follow the chapter's key_scenes list
2. Progress toward the chapter's closure condition: "{chapter_outline.get('ends_when', 'Chapter completes')}"
3. Match the emotional beats: {chapter_outline.get('emotional_beats', 'N/A')}
4. Distribute the chapter's word count ({chapter_word_target} words) across scenes

Each scene plan must include:
- scene_goal: What this scene accomplishes
- main_characters: Who appears
- location: Where it takes place
- time_context: When it occurs
- key_events: What must happen
- emotional_beats: Emotional progression
- thematic_notes: How it reinforces themes
- word_count: Target words for this scene (approximately {target_words_per_scene} words each)

CRITICAL WORD COUNT RULES:
- Each scene should target approximately {target_words_per_scene} words
- Total across all scenes must equal {chapter_word_target} words
- Distribute evenly unless specific scenes require more/less
- Scene Writer will enforce these limits strictly

Output ONLY valid JSON matching this schema:
{director_parser.get_format_instructions()}

Set action to "generate_and_ingest" unless the story is complete.
"""

        human_prompt = f"""
Chapter Number: {state.current_chapter_id}
Chapter Title: {chapter_outline.get('chapter_title', 'Untitled Chapter')}
Chapter Goal: {chapter_outline.get('chapter_goal', 'Continue story')}
Current Story Word Count: {state.story_word_count}

Story Context (Recent):
{director_context}

Create {expected_scenes} scene plans that bring this chapter to life, with each scene targeting approximately {target_words_per_scene} words.
"""

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                resp, director_tokens = await director_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=self.llm_temp,
                    model=self.model
                )
                print(director_tokens)
                if resp is None:
                    print(f"Retrying _chapter_outline_to_scene_plans... (attempt {attempt+1})")
                    continue
                raw_text = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(raw_text)
                
                self.director_token_usage["prompt_tokens"] += director_tokens["prompt_tokens"]
                self.director_token_usage["completion_tokens"] += director_tokens["completion_tokens"]
                self.director_token_usage["total_tokens"] += director_tokens["total_tokens"]
                #print("_chapter_outline_to_scene_plans response: ", clean_resp)
                del raw_text, resp, director_tokens
                gc.collect()
                
                if isinstance(clean_resp, dict):
                    clean_resp = json.dumps(clean_resp)
                
                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    clean_resp, DirectorOutput, director_parser
                )
                
                if success:
                    scenes = result.get("scenes", [])
                    
                    # Validate and adjust word counts
                    total_words = sum(s.get('word_count', target_words_per_scene) for s in scenes)
                    if abs(total_words - chapter_word_target) > chapter_word_target * 0.2:
                        print(f"⚠️ Adjusting scene word counts: planned {total_words} vs target {chapter_word_target}")
                        # Redistribute evenly
                        for scene in scenes:
                            scene['word_count'] = target_words_per_scene
                    
                    print(f"✅ Generated {len(scenes)} scene plans for chapter {state.current_chapter_id}")
                    for i, s in enumerate(scenes, 1):
                        print(f"   Scene {i}: {s.get('word_count', 0)} words - {s.get('scene_goal', 'N/A')}")
                    
                    return scenes

                print(f"[Attempt {attempt}] Scene planning validation failed:", exc)

                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    fixed_clean, DirectorOutput, director_parser
                )
                del fixed_clean
                
                if success:
                    scenes = result.get("scenes", [])
                    
                    # Validate and adjust word counts
                    total_words = sum(s.get('word_count', target_words_per_scene) for s in scenes)
                    if abs(total_words - chapter_word_target) > chapter_word_target * 0.2:
                        print(f"⚠️ Adjusting scene word counts: planned {total_words} vs target {chapter_word_target}")
                        for scene in scenes:
                            scene['word_count'] = target_words_per_scene
                    
                    print(f"✅ Generated {len(scenes)} scene plans (after json_fixer)")
                    for i, s in enumerate(scenes, 1):
                        print(f"   Scene {i}: {s.get('word_count', 0)} words - {s.get('scene_goal', 'N/A')}")
                    
                    return scenes

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)

            except Exception as e:
                print(f"[Attempt {attempt}] Error in scene planning: {e}")

            if attempt < max_retries:
                print(f"Retrying scene planning... (attempt {attempt+1})")
                continue
            else:
                print("⚠️ All retries exhausted; returning empty scene list")
                return []

        return []

    async def director_node(self, state: StoryState) -> Dict:
        print(f"\n🎬 Director Node: Chapter {state.current_chapter_id}, Act {state.current_act_id}")
        # Check if story is complete
        try:
            len_acts_chapters = 0
            # To get total chapter outlines length
            try:
                act_plan_json = await self.memory.get_long_term_document(
                    metadata={'type': 'act_plan', 'act_id': state.current_act_id, 'story_title': state.story_title}
                )
                act_plan = json.loads(act_plan_json)
                chapter_outlines = act_plan.get('chapter_outlines', [])
                len_acts_chapters += len(chapter_outlines)
            except:
                state.error_message = f"No act plan found for act {state.current_act_id}"
                state.next_action = "ERROR"
                return state.__dict__
            
            if state.current_act_id > 1:
                for i in range(1, state.current_act_id):
                    try:
                        act_plan_json = await self.memory.get_long_term_document(
                            metadata={'type': 'act_plan', 'act_id': i, 'story_title': state.story_title}
                        )
                        act_plan = json.loads(act_plan_json)
                        chapter_outlines = act_plan.get('chapter_outlines', [])
                        len_acts_chapters += len(chapter_outlines)
                    except:
                        state.error_message = f"No act plan found for act {state.current_act_id}"
                        state.next_action = "ERROR"
                        return state.__dict__

            print("len_acts_chapters: ",len_acts_chapters)
            if not chapter_outlines or state.current_chapter_id > len_acts_chapters:
                # Check if there's a next act
                len_acts_chapters = 0
                # next_act_plan = await self.memory.get_long_term_document(
                #     metadata={
                #         'type': 'act_plan',
                #         'act_id': state.current_act_id + 1,
                #         'story_title': state.story_title
                #     }
                # )
                # if not next_act_plan:
                print("✅ No more chapters or acts; story is complete")
                state.next_action = "END"
                return state.__dict__
                # else:
                #     state.current_act_id += 1
                #     state.current_chapter_id = 1
                #     print(f"🎬 Advancing to Act {state.current_act_id}, Chapter 1")

        except Exception as e:
            state.error_message = f"Error checking story completion: {str(e)}"
            state.next_action = "ERROR"
            return state.__dict__

        # Check for existing chapter plan
        existing_plan = await self.memory.get_long_term_document(
            metadata={
                "type": "chapter_plan",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        
        if existing_plan:
            print(f"📋 Found existing chapter plan for chapter {state.current_chapter_id}")
            try:
                if isinstance(existing_plan, str):
                    plan_dict = json.loads(existing_plan)
                else:
                    plan_dict = existing_plan
                
                chapter_plan_scenes = plan_dict.get("chapter_plan", [])
                
                if chapter_plan_scenes and len(chapter_plan_scenes) > 0:
                    max_scene_id = max(int(s.get("scene_id", 0)) for s in chapter_plan_scenes)
                    
                    print(f"🎬 Current scene_id: {state.scene_id}, Max scene in plan: {max_scene_id}")
                    
                    if state.scene_id > max_scene_id:
                        print(f"✅ All scenes completed for chapter {state.current_chapter_id}")
                        state.next_action = "ingest_chapter"
                        state.chapter_plan = chapter_plan_scenes
                        return state.__dict__
                    else:
                        print(f"▶️ Continuing with scene {state.scene_id}")
                        state.next_action = "generate_and_ingest"
                        state.chapter_plan = chapter_plan_scenes
                        return state.__dict__
                        
            except Exception as e:
                state.error_message = f"Error parsing existing chapter plan: {str(e)}"
                state.next_action = "ERROR"
                return state.__dict__
        
        # Generate new chapter plan
        print(f"🎭 Generating NEW scene plans for chapter {state.current_chapter_id}")
        
        chapter_outline = await self._get_current_chapter_outline(state)
        
        if not chapter_outline:
            state.error_message = f"Could not find chapter outline for chapter {state.current_chapter_id} in act {state.current_act_id}"
            state.next_action = "ERROR"
            return state.__dict__
        
        self.current_chap_summary = await self.memory.search_single_episodic_story(
            act_number=state.current_act_id,
            chapter_number=state.current_chapter_id - 1,
            summary_type="chapter summary"
        )
        
        director_context = await self.memory.get_director_context(
            current_act_number=state.current_act_id,
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=5
        )
        
        scene_plans = await self._chapter_outline_to_scene_plans(
            chapter_outline=chapter_outline,
            state=state,
            director_context=director_context
        )
        
        if not scene_plans:
            state.error_message = f"Failed to generate scene plans for chapter {state.current_chapter_id}"
            state.next_action = "ERROR"
            return state.__dict__
        
        chapter_plan_doc = {"chapter_plan": scene_plans}
        await self.memory.add_long_term_document(
            text=json.dumps(chapter_plan_doc),
            metadata={
                "chapter_id": state.current_chapter_id,
                "act_id": state.current_act_id,
                "type": "chapter_plan",
                "story_title": state.story_title
            }
        )
        await self.memory.update_story_progress({"director_token_usage": self.director_token_usage})
        print(f"✅ Saved {len(scene_plans)} scene plans for chapter {state.current_chapter_id}")
        
        del director_context
        gc.collect()
        
        state.next_action = "generate_and_ingest"
        state.chapter_plan = scene_plans
        return state.__dict__

    async def generate_and_ingest_node(self, state: StoryState):
        print("🎞️ Called Generate and Ingest Node!")

        if scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE is None:
            state.error_message = "CLASSIC_SCENE_PLANNER_SERVICE is None"
            state.next_action = "ERROR"
            return state.__dict__

        scenes_list = []
        if isinstance(state.chapter_plan, list):
            scenes_list = state.chapter_plan
        elif isinstance(state.chapter_plan, dict):
            scenes_list = state.chapter_plan.get("scenes", [])
        
        if not scenes_list:
            state.error_message = "No valid chapter plan or scenes found"
            state.next_action = "ERROR"
            return state.__dict__

        redis_client = None
        try:
            from setup.shared_redis_pool import get_redis_client
            redis_client = await get_redis_client()
        except Exception as e:
            print(f"⚠️ Could not initialize Redis client: {e}")

        for scene_dict in scenes_list:
            # Ensure user monthly word count compatibility
            monthly_wc_data = await self.memory.get_monthly_word_count()
            if monthly_wc_data.get('tier') == 1:
                if monthly_wc_data.get('monthly_word_count')  >= tier_1_monthly_words_limit:
                    status="User monthly word count limit reached for tier free"
                    state.error_message = f"Scene {scene.scene_id} failed: {status}"
                    state.next_action = "ERROR"
                    return state.__dict__
            elif monthly_wc_data.get('tier') == 2:
                if monthly_wc_data.get('monthly_word_count')  >= tier_2_monthly_words_limit:
                    status="User monthly word count limit reached for tier scribe"
                    state.error_message = f"Scene {scene.scene_id} failed: {status}"
                    state.next_action = "ERROR"
                    return state.__dict__
            del monthly_wc_data

            if isinstance(scene_dict, dict):
                scene = ScenePlan(**scene_dict)
            else:
                scene = scene_dict
                
            if scene.scene_id > state.scene_id:
                break
            if scene.scene_id != state.scene_id:
                print(f"⏭️ Skipping scene {scene.scene_id}, already completed.")
                continue

            director_instructions = json.dumps(scene.model_dump(), indent=2)
            state.scene_id = scene.scene_id
            scene_target_word_count = scene.word_count
            
            print(f"\n🎬 Running Scene {scene.scene_id} | Goal: {scene.scene_goal}")
            print(f"📊 Target: {scene_target_word_count} words")

            user_context = UserSceneContext.create_for_user(
                user_id=self.memory.user_id,
                story_id=self.memory.story_id,
                director_instructions=director_instructions,
                scene_chunk_callback=self.scene_chunk_callback,
                llm_temp=self.llm_temp,
                model=self.model,
                token_usage=self.writer_token_usage,
                target_length=self.target_length,
                scene_target_length=scene_target_word_count
            )

            scene_text, scene_cluster, status, writer_tokens = await scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE.run_scene(
                user_context=user_context,
                stop_event=self.stop_event
            )
                        
            if status != "SUCCESS":
                del user_context
                gc.collect()
                print(f"🛑 Scene {scene.scene_id} failed with status: {status}")
                status_payload = {"type": "status", "ERROR": f"Scene {scene.scene_id} failed: {status}"}
                try:
                    self.scene_chunk_callback(status_payload)
                except Exception as e:
                    print(f"⚠️ scene_chunk_callback raised: {e}")
                
                state.error_message = f"Scene {scene.scene_id} failed: {status}"
                state.next_action = "ERROR"
                return state.__dict__

            print(f"✅ Scene {scene.scene_id} completed successfully")
            new_word_count = StoryHelpers._count_words_split(scene_text)
            await self.memory.update_user_monthly_word_count(word_count=new_word_count)
            del new_word_count
            if redis_client:
                try:
                    queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"
                    resume_payload = {"type": "save"}
                    try:
                        self.scene_chunk_callback(resume_payload)
                    except Exception as e:
                        print(f"⚠️ scene_chunk_callback raised: {e}")

                    user_choice = False
                    for _ in range(3600):
                        choice = await redis_client.lpop(queue_key)
                        if choice in (b"1", "1", 1):
                            user_choice = True
                            print("✅ User chose to continue")
                            break
                        elif choice is not None:
                            print(f"🛑 User chose not to continue ({choice})")
                            break
                        await asyncio.sleep(1.0)

                    if not user_choice:
                        state.next_action = "END"
                        return state.__dict__

                except Exception as e:
                    print(f"⚠️ Redis handling error: {e}")

            self.writer_token_usage = writer_tokens
            del writer_tokens
            ingestor = Ingestor(self.memory)
            self.scene_chunk_callback({"type":"status", "message": "saving story"})
            scene_bundle = await ingestor.ingest_scene(
                state=state,
                scene_text=scene_text,
                #director_token_usage=self.director_token_usage,
                writer_token_usage=self.writer_token_usage
            )
            del ingestor
            gc.collect()

            await self.memory.add_story_scene_cluster(
                text=scene_cluster,
                metadata={
                    "act_id": state.current_act_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "scene_id": state.scene_id,
                    "chapter_word_count": state.current_chapter_word_count,
                    "act_title": self.act_title
                }
            )

            await self.memory.add_post_scene_bundle(
                scene_bundle=scene_bundle,
                metadata={
                    "act_id": state.current_act_id,
                    "scene_id": state.scene_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "type": "scene summary"
                }
            )

            del scene_bundle, scene_cluster, scene_text
            gc.collect()

            state.scene_id += 1

        print(f"✅ All scenes for chapter {state.current_chapter_id} completed!")
        state.next_action = "ingest_chapter"
        return state.__dict__

    async def ingest_chapter(self, state: StoryState):
        print(f"📚 Ingesting chapter {state.current_chapter_id}...")
        
        ingestor = Ingestor(self.memory)
        self.scene_chunk_callback({"type":"status", "message": "saving chapter"})
        await ingestor.ingest_chapter(
            state=state,
            current_chap_summary=self.current_chap_summary
        )
        try:
            self.scene_chunk_callback({"chapter_complete": True})
            await self.memory.update_story_progress({"chapter_word_count": 0})
        except Exception as e:
            print(f"❌ ERROR: scene_chunk_callback raised: {e}")
            import traceback
            traceback.print_exc()
        
        self.current_chap_summary = ""
        del ingestor
        gc.collect()
        
        return state

    async def run(self, scene_chunk_callback, stop_event: asyncio.Event | None = None):
        print("🎬 Running act-based director agent...")
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()

        try:
            story_progress = await self.memory.get_story_progress()
            story_word_count = story_progress.get("story_word_count", 0)
            current_chapter_word_count = story_progress.get("chapter_word_count", 0)
            if not story_progress:
                print("❌ No story progress found")
                return "No story progress found"
            
            self.llm_temp = story_progress.get("tone_temp", 0.7)
            self.model = story_progress.get("model", "None")
            def parse_token_usage(value):
                default = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if not value:
                    return default
                try:
                    if isinstance(value, str):
                        parsed = json.loads(value)
                        if not isinstance(parsed, dict) or not all(
                            key in parsed for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
                        ):
                            return default
                        return parsed
                    elif isinstance(value, dict):
                        if not all(key in value for key in ["prompt_tokens", "completion_tokens", "total_tokens"]):
                            return default
                        return value
                    return default
                except json.JSONDecodeError:
                    return default
            
            # Initialize token usage fields
            self.director_token_usage = parse_token_usage(story_progress.get("director_token_usage"))
            self.writer_token_usage = parse_token_usage(story_progress.get("writer_token_usage"))
            self.target_length = story_progress.get("target_length", 50000)
            
            complete = story_progress.get("complete", False)
            if complete:
                print("✅ Story already complete")
                return "Story already complete."
            
            initialized_state = StoryState(
                current_chapter_id=story_progress.get("latest_chapter_id", 1),
                scene_id=story_progress.get("continue_scene_id", 1),
                story_title=story_progress.get("story_title", "None"),
                story_word_count=story_word_count,
                current_chapter_word_count=current_chapter_word_count,
                current_act_id=story_progress.get("current_act_id", 1),
                next_action=""
            )
            
            print(f"📖 Story State: Chapter {initialized_state.current_chapter_id}, Scene {initialized_state.scene_id}, Act {initialized_state.current_act_id}")
            print(f"📊 Current Chapter Word Count: {initialized_state.current_chapter_word_count}")

            task = asyncio.create_task(
                self.compiled.ainvoke(
                    initialized_state,
                    {"recursion_limit": 50, "stop_event": self.stop_event}
                )
            )

            while not task.done():
                if stop_event.is_set():
                    print("🛑 Stop event received – cancelling DirectorGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ DirectorGraph task cancelled cleanly")
                    return None
                await asyncio.sleep(0.2)

            result = await task
            print("✅ Director Node Finished:", result)
            return result

        except asyncio.CancelledError:
            print("🛑 DirectorGraph CancelledError caught")
            raise
        except Exception as e:
            print(f"❌ Error in DirectorGraph.run: {e}")
            traceback.print_exc()
            raise
        finally:
            print("🎬 DirectorGraph stopped gracefully")

