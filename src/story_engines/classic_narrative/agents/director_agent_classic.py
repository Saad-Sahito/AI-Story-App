# src/story_engines/classic_adventure/agents/director_agent.py

import json
import asyncio
import gc
import traceback
from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from .shared_scene_planner import UserSceneContext
from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import director_client
from config_vars import tier_1_monthly_words_limit, tier_2_monthly_words_limit
from src.utilities.ingestor import Ingestor
from setup.shared_redis_pool import get_redis_client


# ============================================================================
# STATE AND MODELS
# ============================================================================
class ChapterOutline(BaseModel):
    chapter_number: int
    chapter_title: str
    chapter_goal: str
    target_word_count: int
    key_scenes: List[str]
    emotional_beats: List[str]
    ends_when: str
    thematic_elements: List[str]
    agent_focus: Dict[str, str]

@dataclass
class StoryState:
    current_act_id: int = 1
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    story_word_count: int = 0
    current_chapter_word_count: int = 0
    seed: Optional[Dict[str, Any]] = field(default_factory=dict)
    chapter_plan: Optional[ChapterOutline] = field(default_factory=dict)
    act_plan: Optional[Dict[str, Any]] = field(default_factory=dict)
    next_action: str = ""
    error_message: Optional[str] = None

# class ScenePlan(BaseModel):
#     scene_id: int = Field(..., description="Unique identifier for the scene, sequential starting from 1")
#     chapter_number: int
#     chapter_title: str
#     narrative_purpose: str = Field(alias="scene_goal")
#     main_characters: List[str]
#     location: str
#     time_context: str
#     key_events: List[str]
#     emotional_beats: List[str]
#     thematic_notes: List[str]
#     symbolic_elements: List[str] = field(default_factory=list)
#     word_count: int
#     style_guide: str

class DetailedSceneDirective(BaseModel):
    """Director's specific instructions to Scene Writer"""
    scene_id: int
    
    # From Author's beat
    author_intent: str = Field(description="What Author planned")
    
    # Director's creative additions
    opening_image: str = Field(description="Specific sensory detail to open with")
    scene_blocking: str = Field(
        description="Physical staging: who is where, doing what, spatial relationships"
    )
    pacing_direction: str = Field(
        description="'Quick cuts for tension' or 'Linger on reaction' or 'Slow reveal'"
    )
    
    dialogue_guidance: str = Field(
        description="Tone of exchanges, power dynamics, subtext to convey"
    )
    
    sensory_palette: List[str] = Field(
        description="Which senses to emphasize: ['smell of rain', 'texture of rough stone', 'distant bells']"
    )
    
    emotional_camera: str = Field(
        description="Whose POV emotional filter: 'Through protagonist's paranoia' or 'Detached observation'"
    )
    
    scene_rhythm: str = Field(
        description="'Long flowing sentences → sharp short ones at revelation' or 'Staccato throughout'"
    )
    
    key_details_to_include: List[str] = Field(
        description="Specific objects, gestures, or moments that MUST appear"
    )
    
    closing_image: str = Field(
        description="Last sensory detail or action before scene ends"
    )
    
    transition_to_next: str = Field(
        description="How this scene should flow into next: 'Hard cut' or 'Zoom out to...' or 'Echo the opening'"
    )
    
    # Tactical adjustments
    word_count: int
    style_adjustments: str = Field(
        description="'More internal monologue' or 'Action-heavy' or 'Dialogue-driven' based on chapter flow"
    )


class ChapterDirectorBrief(BaseModel):
    """Director's overview before planning scenes"""
    chapter_rhythm: str = Field(description="Overall pacing strategy for this chapter")
    visual_motif: str = Field(description="Recurring image or color for this chapter")
    tonal_consistency: str = Field(description="How to maintain tone across scenes")
    variety_strategy: str = Field(description="How scenes differ to prevent monotony")

class DirectorOutput(BaseModel):
    scenes: List[DetailedSceneDirective] = Field(..., description="List of detailed scene plans for Scene Writer")
    action: Literal["generate_and_ingest", "END"] = Field(
        description="'generate_and_ingest' to continue, 'END' if story complete"
    )

chapter_director_brief_parser = PydanticOutputParser(pydantic_object=ChapterDirectorBrief)
chapter_outline_parser = PydanticOutputParser(pydantic_object=ChapterOutline)
director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)


# ============================================================================
# DIRECTOR GRAPH
# ============================================================================
class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("load_act_node")
        self.graph.add_node("load_act_node", self.load_act_node)
        self.graph.add_node("chapter_planner_node", self.chapter_planner_node)
        self.graph.add_node("scene_planner_node", self.scene_planner_node)
        self.graph.add_node("generate_scenes_node", self.generate_scenes_node)
        self.graph.add_node("quality_validate_node", self.quality_validate_node)
        self.graph.add_node("ingest_chapter_node", self.ingest_chapter_node)
        self.graph.add_node("story_complete_node", self.story_complete_node)
        self.graph.add_node("error_node", self.error_node)

        # Edges
        self.graph.add_edge("load_act_node", "chapter_planner_node")
        self.graph.add_edge("chapter_planner_node", "scene_planner_node")
        self.graph.add_conditional_edges(
            "scene_planner_node",
            lambda state: state.next_action,
            {
                "generate_scenes": "generate_scenes_node",
                "END": "story_complete_node",
                "ERROR": "error_node"
            }
        )
        self.graph.add_edge("generate_scenes_node", "quality_validate_node")
        self.graph.add_conditional_edges(
            "quality_validate_node",
            lambda state: state.next_action,
            {
                "ingest_chapter": "ingest_chapter_node",
                "retry_scenes": "scene_planner_node",
                "ERROR": "error_node"
            }
        )
        self.graph.add_edge("ingest_chapter_node", END)
        self.graph.add_edge("story_complete_node", END)
        self.graph.add_edge("error_node", END)

        self.compiled = self.graph.compile()

        self.current_chap_summary = ""
        self.llm_temp = 0.7
        self.model = ""
        self.act_title = ""
        self.director_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.writer_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.target_length = 50000

    # ============================================================================
    # NODES
    # ============================================================================

    async def load_act_node(self, state: StoryState) -> Dict:
        print(f"\nLoad Act Node: Act {state.current_act_id}")
        act_plan_json = await self.memory.get_long_term_document(
            metadata={'type': 'act_plan', 'act_id': state.current_act_id, 'story_title': state.story_title}
        )
        if not act_plan_json:
            state.error_message = f"No act plan for act {state.current_act_id}"
            state.next_action = "ERROR"
            return state.__dict__

        state.act_plan = json.loads(act_plan_json)
        self.act_title = state.act_plan.get('act_title', 'Unknown Act')
        self.scene_chunk_callback({"type": "act_title", "message": self.act_title})

        # Load story seed for style guide
        story_seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': state.story_title}
        )
        state.seed = json.loads(story_seed_json) if story_seed_json else {}

        return state.__dict__

    async def chapter_planner_node(self, state: StoryState) -> Dict:
        print(f"Chapter Planner: Chapter {state.current_chapter_id}")

        # Check if chapter plan already exists
        existing_plan = await self.memory.get_long_term_document(
            metadata={
                "type": "chapter_plan",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        existing_plan = ChapterOutline(**existing_plan)

        if existing_plan:
            #plan_dict = json.loads(existing_plan) if isinstance(existing_plan, str) else existing_plan
            state.chapter_plan = existing_plan
            print(f"Using existing chapter plan for chapter {state.current_chapter_id}")
            state.next_action = "scene_planner"
            return state.__dict__
        plot_outline = await self.memory.get_long_term_document(
            metadata={
                "type": "expanded_plot_outline",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        # Otherwise, generate new chapter outline
        chapter_outline = state.act_plan['chapter_seeds'][state.current_chapter_id - 1]  # 0-indexed
        if not chapter_outline:
            state.error_message = f"Chapter {state.current_chapter_id} outline not found"
            state.next_action = "ERROR"
            return state.__dict__

        # Convert to structured ChapterOutline
        target_word_count =  plot_outline.get('target_length', 3000) // len(plot_outline.get('act_summaries')) // len(state.act_plan['chapter_seeds'])  # Even dist
        #expected_scenes = len(chapter_outline.get('key_scenes', [])) or 3

        system_prompt = f"""Convert this chapter seed into a structured plan.

Story Style: POV = {state.seed.get('style_guide', {}).get('pov', 'Third-person')}, 
Prose = {state.seed.get('prose_style', '')}, 
Tense = past, 
Tone = {state.seed.get('tone', 'Balanced')}

Chapter Target Word Count: {target_word_count}

{chapter_outline_parser.get_format_instructions()}"""

        human_prompt = f"Chapter {state.current_chapter_id}: {chapter_outline.get('chapter_title', 'Untitled')} Chapter Seed: {json.dumps(chapter_outline, indent=2)}"

        resp, tokens = await director_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=self.llm_temp,
            model=self.model
        )
        self.director_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
        self.director_token_usage["completion_tokens"] += tokens["completion_tokens"]
        self.director_token_usage["total_tokens"] += tokens["total_tokens"]

        clean_resp = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        result = await StoryHelpers.load_json_with_retry(clean_resp, chapter_outline_parser)

        chapter_plan: ChapterOutline = result
        chapter_plan.target_word_count = target_word_count

        # Save chapter plan
        await self.memory.add_long_term_document(
            text=json.dumps({"chapter_plan": chapter_plan}),
            metadata={
                "type": "chapter_plan",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )

        state.chapter_plan = chapter_plan
        state.next_action = "scene_planner"
        return state.__dict__

    
    async def scene_planner_node(self, state: StoryState) -> Dict:
        """Director interprets Author's scene beats and creates detailed directives"""
        print(f"Scene Planner: Chapter {state.current_chapter_id}")
        
        act_plan = state.act_plan
        
        # Get Author's scene beats
        author_scene_beats = act_plan.get('chapter_scene_beats', {}).get(int(state.current_chapter_id), [])
        
        if not author_scene_beats:
            state.error_message = "No scene beats from Author"
            state.next_action = "ERROR"
            return state.__dict__
        
        # Load connected agents and integrated world (fix your bug)
        agents_json = await self.memory.get_long_term_document(
            metadata={'type': 'connected_agents', 'story_title': state.story_title}
        )
        connected_agents = json.loads(agents_json)
        
        world_json = await self.memory.get_long_term_document(
            metadata={'type': 'integrated_world', 'story_title': state.story_title}
        )
        integrated_world = json.loads(world_json)
        
        # Get context from previous chapter
        prev_chapter_summary = await self.memory.search_single_episodic_story(
            act_number=state.current_act_id,
            chapter_number=state.current_chapter_id - 1,
            summary_type="chapter summary"
        )
        
        director_context = await self.memory.get_director_context(
            current_act_number=state.current_act_id,
            current_chapter_number=state.current_chapter_id,
            query=prev_chapter_summary or "",
            k=5
        )
        
        # DIRECTOR'S CREATIVE WORK: First, create chapter-level strategy
        chapter_brief_prompt = f"""You are the Director. The Author has given you scene beats. 
    Before directing individual scenes, create your VISION for this chapter.

    Consider:
    - How should this chapter FEEL different from previous ones?
    - What visual or sensory motif unifies it?
    - What pacing rhythm best serves these events?
    - How do scenes build on each other?

    Author's Chapter Plan:
    {state.chapter_plan.model_dump_json(indent=2)}

    Author's Scene Beats:
    {json.dumps(author_scene_beats, indent=2)}

    Story Context So Far:
    {director_context}

    Previous Chapter: {prev_chapter_summary}

    {chapter_director_brief_parser.get_format_instructions()}"""

        chapter_brief_resp, tokens1 = await director_client(
            system_prompt="You are a film director creating your vision for a chapter.",
            human_prompt=chapter_brief_prompt,
            llm_temp=0.8,
            model=self.model
        )
        
        clean_brief = StoryHelpers._extract_content(chapter_brief_resp)
        clean_brief = StoryHelpers._strip_code_fences(clean_brief)
        brief_json = await StoryHelpers.load_json_with_retry(clean_brief, chapter_director_brief_parser)
        chapter_brief = chapter_director_brief_parser.parse(brief_json)
        
        self.director_token_usage["prompt_tokens"] += tokens1["prompt_tokens"]
        self.director_token_usage["completion_tokens"] += tokens1["completion_tokens"]
        self.director_token_usage["total_tokens"] += tokens1["total_tokens"]
        
        # DIRECTOR'S CREATIVE WORK: Now create detailed scene directives
        system_prompt = f"""You are the Director translating Author's scene beats into detailed directives for the Scene Writer.

    The Author says WHAT happens. You specify HOW it happens.

    For each scene beat, provide:
    1. Opening image (specific sensory detail)
    2. Scene blocking (spatial staging, who where)
    3. Pacing direction (how fast/slow, when to linger)
    4. Dialogue guidance (tone, subtext, power dynamics)
    5. Sensory palette (which senses to engage)
    6. Emotional camera (whose perspective filters emotion)
    7. Scene rhythm (sentence structure guidance)
    8. Key details that MUST appear
    9. Closing image
    10. Transition to next scene

    Your Chapter Vision:
    {chapter_brief.model_dump_json(indent=2)}

    Agents Available:
    {json.dumps([{
        'name': a['name'], 
        'type': a['type'],
        'distinctive_voice': a.get('distinctive_voice', ''),
        'plot_function': a.get('plot_function', '')
    } for a in connected_agents], indent=2)}

    World Context:
    Physical locations: {integrated_world.get('plot_relevant_locations', '')}
    World rules: {integrated_world.get('world_plot_interactions', '')}

    Story Style:
    POV: {state.seed.get('pov', 'Third-person')}
    Prose: {state.seed.get('prose_style', '')}
    Tone: {state.seed.get('tone', '')}

    {director_parser.get_format_instructions()}

    Think like a film director blocking a scene, but for prose."""

        human_prompt = f"""Create detailed scene directives for the Scene Writer.

    Author's Scene Beats (WHAT happens):
    {json.dumps(author_scene_beats, indent=2)}

    Recent Story Context:
    {director_context}

    Transform Author's beats into vivid, executable directives that specify HOW to write each scene.
    Make each scene feel distinct within your chapter vision.
    Total scenes: {len(author_scene_beats)}"""

        resp, tokens2 = await director_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=self.llm_temp,
            model=self.model
        )
        
        self.director_token_usage["prompt_tokens"] += tokens2["prompt_tokens"]
        self.director_token_usage["completion_tokens"] += tokens2["completion_tokens"]
        self.director_token_usage["total_tokens"] += tokens2["total_tokens"]
        
        clean_resp = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        result_json = await StoryHelpers.load_json_with_retry(clean_resp, director_parser)
        result: DirectorOutput = director_parser.parse(result_json)
        
        scenes = [s.model_dump() for s in result.scenes]
        
        # Store Director's directives
        await self.memory.add_long_term_document(
            text=json.dumps(scenes),
            metadata={
                "type": "director_scene_directives",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        
        state.next_action = "generate_scenes"
        return state.__dict__


    async def generate_scenes_node(self, state: StoryState) -> Dict:
        """Execute Director's directives through Scene Writer"""
        print(f"Generate Scenes: Chapter {state.current_chapter_id}, Scene {state.scene_id}")
        
        # Load DIRECTOR's directives (not Author's beats)
        scenes_json = await self.memory.get_long_term_document(
            metadata={
                "type": "director_scene_directives",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        director_directives = json.loads(scenes_json)
        
        redis_client = None
        try:
            redis_client = await get_redis_client()
        except Exception as e:
            print(f"Redis init failed: {e}")
        
        for directive_dict in director_directives:
            directive = DetailedSceneDirective(**directive_dict)
            
            if directive.scene_id > state.scene_id:
                break
            if directive.scene_id != state.scene_id:
                print(f"Skipping completed scene {directive.scene_id}")
                continue

            # Word count limit check
            monthly = await self.memory.get_monthly_word_count()
            if monthly.get('tier') == 1 and monthly.get('monthly_word_count', 0) >= tier_1_monthly_words_limit:
                state.error_message = "Free tier word limit reached"
                state.next_action = "ERROR"
                return state.__dict__
            if monthly.get('tier') == 2 and monthly.get('monthly_word_count', 0) >= tier_2_monthly_words_limit:
                state.error_message = "Scribe tier word limit reached"
                state.next_action = "ERROR"
                return state.__dict__

            # Pass Director's FULL directive to Scene Writer
            director_instructions = directive.model_dump_json(indent=2)
            state.scene_id = directive.scene_id
            
            user_context = UserSceneContext.create_for_user(
                user_id=self.memory.user_id,
                story_id=self.memory.story_id,
                director_instructions=director_instructions,  # Full directive, not just beats
                scene_chunk_callback=self.scene_chunk_callback,
                llm_temp=self.llm_temp,
                model=self.model,
                token_usage=self.writer_token_usage,
                target_length=self.target_length,
                scene_target_length=directive.word_count,
                user_age=self.min_age
            )

            scene_text, scene_cluster, status, writer_tokens = await scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE.run_scene(
                user_context=user_context,
                stop_event=self.stop_event
            )

            if status != "SUCCESS":
                state.error_message = f"Scene {directive.scene_id} failed: {status}"
                state.next_action = "ERROR"
                return state.__dict__

            word_count = StoryHelpers._count_words_split(scene_text)
            await self.memory.update_user_monthly_word_count(word_count=word_count)
            self.writer_token_usage = writer_tokens

            # Redis pause
            if redis_client:
                queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"
                self.scene_chunk_callback({"type": "save"})
                user_choice = False
                for _ in range(3600):
                    choice = await redis_client.lpop(queue_key)
                    if choice in (b"1", "1", 1):
                        user_choice = True
                        break
                    elif choice is not None:
                        break
                    await asyncio.sleep(1.0)
                if not user_choice:
                    state.next_action = "END"
                    return state.__dict__

            # Ingest scene
            chars = await self.memory.get_long_term_characters_names()
            worlds = await self.memory.get_long_term_worlds_names()
            self.scene_chunk_callback({"type": "status", "message": "saving story"})

            scene_bundle = await Ingestor.ingest_scene(
                chapter_id=state.current_chapter_id,
                scene_id=state.scene_id,
                scene_text=scene_text,
                chars=chars,
                worlds=worlds
            )

            state.current_chapter_word_count += word_count
            state.story_word_count += word_count

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
                parts=scene_bundle or {},
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "act_id": state.current_act_id
                }
            )

            await self.memory.update_story_progress({
                "latest_chapter_id": state.current_chapter_id,
                "continue_scene_id": state.scene_id + 1,
                "story_word_count": state.story_word_count,
                "chapter_word_count": state.current_chapter_word_count,
                "writer_token_usage": self.writer_token_usage
            })

            state.scene_id += 1
            gc.collect()

        state.next_action = "ingest_chapter"
        return state.__dict__

    async def quality_validate_node(self, state: StoryState) -> Dict:
        state.next_action = "ingest_chapter"  # Placeholder
        return state.__dict__

    async def ingest_chapter_node(self, state: StoryState) -> Dict:
        print(f"Ingesting chapter {state.current_chapter_id}...")
        world_details = await self.memory.get_long_term_recent_worlds(state.current_chapter_id)
        char_details = await self.memory.get_long_term_recent_characters(state.current_chapter_id)
        self.scene_chunk_callback({"type": "status", "message": "saving chapter"})

        chapter_bundle = await Ingestor.ingest_chapter(
            chapter_id=state.current_chapter_id,
            current_chap_summary=self.current_chap_summary,
            char_details=char_details,
            world_details=world_details
        )

        if chapter_bundle:
            await self.memory.add_post_chapter_bundle(
                parts=chapter_bundle,
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "act_id": state.current_act_id
                }
            )
            await self.memory.increment_chapter(word_count_delta=0, scene_id=1)
            state.current_chapter_id += 1
            state.scene_id = 1
            self.scene_chunk_callback({"chapter_complete": True})
            await self.memory.update_story_progress({"chapter_word_count": 0})
        self.current_chap_summary = ""
        gc.collect()
        return state.__dict__

    async def story_complete_node(self, state: StoryState):
        await self.memory.mark_story_complete()
        self.scene_chunk_callback({"type": "status", "message": "story complete"})
        return state.__dict__

    async def error_node(self, state: StoryState):
        self.scene_chunk_callback({
            "type": "status",
            "message": f"Error: {state.error_message or 'Unknown error'}"
        })
        return state.__dict__

    # ============================================================================
    # RUN
    # ============================================================================
    async def run(self, scene_chunk_callback, stop_event: asyncio.Event | None = None):
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()

        try:
            progress = await self.memory.get_story_progress()
            if not progress:
                return "No story progress"

            self.llm_temp = progress.get("tone_temp", 0.7)
            self.model = progress.get("model", "")
            self.min_age = progress.get("min_age", 13)
            self.target_length = progress.get("target_length", 50000)
            self.director_token_usage = self._parse_tokens(progress.get("director_token_usage"))
            self.writer_token_usage = self._parse_tokens(progress.get("writer_token_usage"))

            if progress.get("complete"):
                return "Story already complete"

            state = StoryState(
                current_act_id=progress.get("current_act_id", 1),
                current_chapter_id=progress.get("latest_chapter_id", 1),
                scene_id=progress.get("continue_scene_id", 1),
                story_title=progress.get("story_title", "None"),
                story_word_count=progress.get("story_word_count", 0),
                current_chapter_word_count=progress.get("chapter_word_count", 0)
            )

            task = asyncio.create_task(self.compiled.ainvoke(state, {"recursion_limit": 50, "stop_event": self.stop_event}))
            while not task.done():
                if self.stop_event.is_set():
                    task.cancel()
                    await task
                    return None
                await asyncio.sleep(0.2)
            return await task

        except asyncio.CancelledError:
            raise
        except Exception as e:
            traceback.print_exc()
            raise
        finally:
            print("DirectorGraph stopped")

    def _parse_tokens(self, value):
        default = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not value:
            return default
        try:
            parsed = json.loads(value) if isinstance(value, str) else value
            if isinstance(parsed, dict) and all(k in parsed for k in default):
                return parsed
        except:
            pass
        return default