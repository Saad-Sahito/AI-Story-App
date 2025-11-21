# src/story_engines/classic_adventure/agents/director_agent.py

import json
import asyncio
import gc
import traceback
import re
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



# ============================================================================
# STATE AND MODELS
# ============================================================================


class SceneWriterDirective(BaseModel):
    """
    COMPLETE scene instructions for Scene Writer.
    This is the ONLY context Scene Writer gets — must be comprehensive.
    """
    scene_id: int
    chapter_id: int
    act_id: int
    
    # =========================================================================
    # STORY EVENTS - THE SPINE
    # =========================================================================
    
    story_events: List[str] = Field(
        description="3-5 CONCRETE actions/discoveries that MUST happen (order matters)"
    )
    
    scene_purpose: str = Field(
        description="advance_plot | develop_character | reveal_world | create_emotion | setup_payoff"
    )
    
    # =========================================================================
    # AGENT DETAILS - WHO & THEIR OBJECTIVES
    # =========================================================================
    
    agents_in_scene: Dict[str, Dict[str, Any]] = Field(
        description="""
        agent_name -> {
            'role': str (their story role),
            'objective': str (what they want THIS scene),
            'current_status': str (physical/emotional state they enter with),
            'distinctive_voice': str (speech patterns, word choice, cadence),
            'recent_arc': str (what's happened to them recently),
            'relationship_to_others': dict (how they relate to each agent in scene)
        }
        """
    )
    
    # =========================================================================
    # WORLD & SETTING - THE STAGE
    # =========================================================================
    
    setting: Dict[str, str] = Field(
        description="""
        {
            'location': str (specific place this scene occurs),
            'atmosphere': str (sensory details: light, sound, smell, temperature),
            'active_world_elements': list (magic system active, tech present, dangers, etc),
            'time_of_day': str,
            'weather_or_environmental': str
        }
        """
    )
    
    # =========================================================================
    # STATE CHANGES - WHAT SHIFTS AFTER THIS SCENE
    # =========================================================================
    
    character_state_changes: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="""
        agent_name -> {
            'learns': list (new facts they discover),
            'believes_shift': dict ('old_belief' -> 'new_belief'),
            'emotional_state_after': str,
            'status_after': str (physically or socially changed?),
            'relationship_shifts': dict (how their view of others changes)
        }
        """
    )
    
    world_state_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="What physically/metaphysically changes (corrupted location, broken object, revealed secret)"
    )
    
    # =========================================================================
    # SETUP & PAYOFF TRACKING
    # =========================================================================
    
    introduces_for_later: List[str] = Field(
        default_factory=list,
        description="New questions, objects, relationships, tensions introduced here (must payoff later)"
    )
    
    resolves_from_earlier: List[str] = Field(
        default_factory=list,
        description="What this scene answers/resolves from earlier setup"
    )
    
    # =========================================================================
    # RHYTHM & PACING
    # =========================================================================
    
    emotional_arc: str = Field(
        description="How emotion evolves: 'tense opening → intimate confession → explosive confrontation → tentative resolution'"
    )
    
    pacing: str = Field(
        description="fast (short sentences, quick beats) | medium (balanced) | slow (introspection, lingering moments)"
    )
    
    # =========================================================================
    # CONSTRAINTS & TARGETS
    # =========================================================================
    
    target_word_count: int = Field(description="Target scene length")
    
    expected_paragraph_structure: str = Field(
        description="Opening action beat | Development | Climax | Resolution (rough structure)"
    )
    
    # =========================================================================
    # CRITICAL CONTEXT FOR SCENE WRITER
    # =========================================================================
    
    immediate_context: str = Field(
        description="What happened in the last scene(s)? What's the reader's mindset entering this?"
    )
    
    thematic_echo: str = Field(
        description="How does this scene echo or advance one of the story's core themes?"
    )
    
    conflict_context: str = Field(
        description="Which story conflicts are active in this scene?"
    )
    
    # =========================================================================
    # OPTIONAL: SPECIFIC DIALOGUE OR ACTION REQUIREMENTS
    # =========================================================================
    
    must_include_moments: List[str] = Field(
        default_factory=list,
        description="If there are specific beats that MUST happen (handed down from Author), list here"
    )
    
    forbidden_elements: List[str] = Field(
        default_factory=list,
        description="What should NOT happen (contradicts earlier setup, too mature for age, etc)"
    )


# class ChapterOutline(BaseModel):
#     chapter_number: int
#     chapter_title: str
#     chapter_goal: str
#     target_word_count: int
#     key_scenes: List[str]
#     emotional_beats: List[str]
#     ends_when: str
#     thematic_elements: List[str]
#     agent_focus: Dict[str, str]

@dataclass
class StoryState:
    current_act_id: int = 1
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    story_word_count: int = 0
    current_chapter_word_count: int = 0
    seed: Optional[Dict[str, Any]] = field(default_factory=dict)
    act_plan: Optional[Dict[str, Any]] = field(default_factory=dict)
    next_action: str = ""
    error_message: Optional[str] = None

# class ChapterDirectorBrief(BaseModel):
#     """Director's overview before planning scenes"""
#     chapter_rhythm: str = Field(description="Overall pacing strategy for this chapter")
#     visual_motif: str = Field(description="Recurring image or color for this chapter")
#     tonal_consistency: str = Field(description="How to maintain tone across scenes")
#     variety_strategy: str = Field(description="How scenes differ to prevent monotony")

class DirectorOutput(BaseModel):
    scenes: List[SceneWriterDirective] = Field(..., description="List of concrete scene directives for Scene Writer")
    action: Literal["generate_and_ingest", "END"] = Field(
        description="'generate_and_ingest' to continue, 'END' if story complete"
    )

# chapter_director_brief_parser = PydanticOutputParser(pydantic_object=ChapterDirectorBrief)
# chapter_outline_parser = PydanticOutputParser(pydantic_object=ChapterOutline)
director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)

# ============================================================================
# GENERIC CONTEXT FORMATTER (replaces old DirectorContextBuilder)
# ============================================================================

class ContextFormatter:
    @staticmethod
    def format_any(obj: Any) -> str:
        if hasattr(obj, '__dict__'):
            data = obj.__dict__
        elif isinstance(obj, dict):
            data = obj
        else:
            data = {k: v for k, v in obj.__dict__.items()} if hasattr(obj, '__dict__') else vars(obj)

        parts = [
            "╔═══════════════════════════════════════════════════════════╗",
            "║ SCENE DIRECTIVE FOR WRITER",
            "╚═══════════════════════════════════════════════════════════╝",
        ]
        for key, value in data.items():
            if key == "previous_scenes_in_chapter":
                continue
            title = key.replace("_", " ").upper()
            parts.append(ContextFormatter.format_section(title, value))
        return "\n".join(parts)
    
    @staticmethod
    def _format_value(value: Any, depth: int = 0) -> str:
        indent = "  " * depth
        if isinstance(value, dict):
            lines = []
            for k, v in value.items():
                if isinstance(v, (dict, list)) and v:
                    lines.append(f"{indent}{k}:")
                    lines.append(ContextFormatter._format_value(v, depth + 1))
                else:
                    lines.append(f"{indent}{k}: {v}")
            return "\n".join(lines)
        elif isinstance(value, list):
            if not value:
                return "(none)"
            lines = []
            for i, item in enumerate(value[:10], 1):  # limit explosion
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}{i}. ")
                    lines.append(ContextFormatter._format_value(item, depth + 1))
                else:
                    lines.append(f"{indent}- {item}")
            if len(value) > 10:
                lines.append(f"{indent}... ({len(value)-10} more)")
            return "\n".join(lines)
        else:
            return str(value)

    @staticmethod
    def format_section(title: str, data: Any) -> str:
        if data is None or (isinstance(data, (dict, list)) and not data):
            return f"╠ {title}\n║ (none)\n"
        content = ContextFormatter._format_value(data, depth=1)
        return f"╠ {title}\n{content}\n"

    @staticmethod
    def full_context(snapshot: 'StoryContextSnapshot') -> str:
        parts = [
            "╔═══════════════════════════════════════════════════════════╗",
            "║ DIRECTOR CONTEXT SNAPSHOT",
            "╚═══════════════════════════════════════════════════════════╝",
        ]
        for key, value in snapshot.__dict__.items():
            if key == "previous_scenes_in_chapter":
                continue
            title = key.replace("_", " ").upper()
            parts.append(ContextFormatter.format_section(title, value))
        return "\n".join(parts)


@dataclass
class StoryContextSnapshot:
    story_seed: Dict
    final_plot: Dict
    act_plan: Dict
    chapter_word_count_target: int
    act_setup_payload: List[str]
    act_payoff_payload: List[str]
    connected_agents: List[Dict]
    integrated_world: Dict
    conflict_matrix: Dict
    character_progressions: List[Dict]
    world_progressions: List[Dict]
    recent_chapters_summary: str
    current_chapter_pivots: List[Dict] = field(default_factory=list)

# ============================================================================
# DIRECTOR GRAPH
# ============================================================================
class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        self.story_context: Optional[StoryContextSnapshot] = None
        self.current_chapter_outline: Optional[Dict[str, Any]] = None
        self.memory = memory_system
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("load_act_node")
        self.graph.add_node("load_act_node", self.load_act_node)
        self.graph.add_node("scene_planner_node", self.scene_planner_node)
        self.graph.add_node("generate_scenes_node", self.generate_scenes_node)
        self.graph.add_node("quality_validate_node", self.quality_validate_node)
        self.graph.add_node("ingest_chapter_node", self.ingest_chapter_node)
        self.graph.add_node("story_complete_node", self.story_complete_node)
        self.graph.add_node("error_node", self.error_node)

        # Edges
        self.graph.add_edge("load_act_node", "scene_planner_node")

        self.graph.add_conditional_edges(
            "scene_planner_node",
            lambda state: state.next_action,
            {
                "generate_scenes": "generate_scenes_node",
                "END": "story_complete_node",
                "ERROR": "error_node"
            }
        )
        self.graph.add_conditional_edges(
            "generate_scenes_node",
            lambda state: state.next_action,
            {
                "END": END,
                "ERROR": "error_node",
                "quality_validate_node": "quality_validate_node"
            }
        )
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

        # self.current_chap_summary = ""
        self.llm_temp = 0.7
        #self.model = ""
        self.act_title = ""
        self.min_age = 13
        self.pov = "third person"
        self.tense = "past"
        self.voice = "standard narrative"
        self.tone = "light"
        self.director_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.writer_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.target_length = 50000
    
    # ============================================================================
    # CONTEXT BUILDING
    # ============================================================================

    async def _build_context(self, story_title: str, act_id: int, chapter_id: int) -> StoryContextSnapshot:
        seed = json.loads(await self.memory.get_long_term_document(metadata={'type': 'story_seed', 'story_title': story_title}))
        self.pov = seed['pov']
        self.voice = seed['prose_style']
        self.tone = seed['tone']
        plot = json.loads(await self.memory.get_long_term_document(metadata={'type': 'expanded_plot_outline', 'story_title': story_title}))
        act_plan = json.loads(await self.memory.get_long_term_document(metadata={'type': 'act_plan', 'act_id': act_id, 'story_title': story_title}))
        agents = json.loads(await self.memory.get_long_term_document(metadata={'type': 'connected_agents', 'story_title': story_title}))
        world = json.loads(await self.memory.get_long_term_document(metadata={'type': 'integrated_world', 'story_title': story_title}))
        conflict = json.loads(await self.memory.get_long_term_document(metadata={'type': 'conflict_matrix', 'story_title': story_title}))


        # Extract the high-level chapter outline (still stored inside act_plan for backward compat)
        rich_chapter = next(
            (c for c in act_plan.get("chapter_outlines", []) if isinstance(c, dict) and c.get("chapter_number") == chapter_id),
            {}
        )
        target_word_count = rich_chapter.get("target_word_count")
        char_prog = await self.memory.get_long_term_recent_characters(chapter_id)
        world_prog = await self.memory.get_long_term_recent_worlds(chapter_id)
        recent_summary = await self.memory.search_single_episodic_story(act_number=act_id, chapter_number=chapter_id, summary_type="scene summary")
        #print(rich_chapter.get("pivot_points"))
        return StoryContextSnapshot(
            story_seed=seed,
            final_plot=plot,
            act_plan=act_plan,
            chapter_word_count_target=target_word_count,
            act_setup_payload=act_plan.get("setup_this_act", []),
            current_chapter_pivots=rich_chapter.get("pivot_points"),
            act_payoff_payload=act_plan.get("payoff_this_act", []),
            connected_agents=agents,
            integrated_world=world,
            conflict_matrix=conflict,
            character_progressions=char_prog,
            world_progressions=world_prog,
            recent_chapters_summary=recent_summary or ""
        )
    
    # ============================================================================
    # NODES
    # ============================================================================

    async def load_act_node(self, state: StoryState) -> Dict:
        print(f"\n📖 Load Act Node: Act {state.current_act_id}")
        
        act_plan_json = await self.memory.get_long_term_document(
            metadata={'type': 'act_plan', 'act_id': state.current_act_id, 'story_title': state.story_title}
        )
        if not act_plan_json:
            state.error_message = f"No act plan for act {state.current_act_id}"
            state.next_action = "ERROR"
            return state.__dict__
        
        state.act_plan = json.loads(act_plan_json)
        self.act_title = state.act_plan.get('act_title', 'Unknown Act')
        
        # NEW: Build comprehensive context snapshot
        print("📚 Building full story context snapshot...")
        self.story_context = await self._build_context(state.story_title, state.current_act_id, state.current_chapter_id)
        
        self.scene_chunk_callback({"type": "act_title", "message": self.act_title})
        print(f"✓ Act loaded: {self.act_title}")
        print(f"✓ Context snapshot: {len(self.story_context.connected_agents)} agents.")
        
        return state.__dict__
    
    
    
    async def scene_planner_node(self, state: StoryState) -> Dict:
        print(f"Scene Planner: Chapter {state.current_chapter_id}")

        # ─── CACHED DIRECTIVES? ─────────────────────────────────────────────────────
        cached = await self.memory.get_long_term_document(metadata={
            "type": "director_scene_directives",
            "act_id": state.current_act_id,
            "chapter_id": state.current_chapter_id,
            "story_title": state.story_title
        })
        if cached:
            print("Using cached scene directives")
            state.next_action = "generate_scenes"
            return state.__dict__

        # ─── NO PIVOTS → SKIP ───────────────────────────────────────────────────────
        pivot_points = getattr(self.story_context, "current_chapter_pivots", [])
        if not pivot_points:
            print("No pivot points found → skipping to generate_scenes")
            state.next_action = "generate_scenes"
            return state.__dict__

        total_chapter_words: int = self.story_context.chapter_word_count_target
        # ─── PARSER FOR FULL RICH DIRECTIVES ────────────────────────────────────────
        # We'll trick the parser: tell it to output DirectorOutput, but with scenes as SceneWriterDirective
        # This works because DirectorOutput.scenes is List[ConcreteSceneDirective], but we override instructions
        #rich_director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)

        system_prompt = f"""You are the Director. Your only job is to turn high-level story pivots into 100% executable, concrete SceneWriterDirective objects that can never be misinterpreted.

CRITICAL RULES (never break these):

1. Every story_events list (3–5 per scene) MUST consist exclusively of concrete, filmable, speakable actions.  
   Allowed:  
   - "Lirael tells Kael she will defect during the Calm Storm"  
   - "Kael pulls out his half-finished signal jammer and demonstrates it fizzles"  
   - "Voren enters and touches his staff to Lirael's loyalty vine tattoo"  
   Forbidden forever:  
   - "Lirael wrestles with doubt"  
   - "Tension rises"  
   - "Kael shows mixed loyalty"  
   - "The vines react to dishonesty"

2. If a pivot says "Lirael confesses her plan", you MUST break it into at least two concrete events (she says the plan out loud + the other character reacts in a specific, visible way).

3. Magic/system rules in this scene MUST be consistent with every previous scene in the full story context. If the vines glowed brighter with honesty last chapter, they cannot suddenly wilt with honesty this chapter.

4. Every single field in SceneWriterDirective must be filled, but story_events is the spine — everything else (objectives, state changes, emotional arc) must flow logically from those concrete events.

5. Target word count per scene: calculate exactly (total_chapter_words // total_scenes_this_chapter). Never round vaguely.

6. Output ONLY valid JSON matching DirectorOutput schema. No explanations, no markdown, no extra text.

Total chapter word target: {total_chapter_words}
Full story context (use it religiously for continuity):
{ContextFormatter.full_context(self.story_context)}
"""
        all_rich_directives = []
        scene_id_counter = state.scene_id
        
        total_pivots = len(pivot_points)

        for idx, pivot in enumerate(pivot_points, 1):
            print(f"  Planning pivot {idx}/{total_pivots}: {pivot.get('pivot_type', 'unknown')}")

            scenes_per_pivot: int = 2 if total_pivots <= 4 else 1
            words_per_scene_temp = int(total_chapter_words/(scenes_per_pivot * total_pivots))
            words_per_scene = max(300, words_per_scene_temp)
            print("Director words per scene: ", words_per_scene)
            human_prompt = f"""Generate exactly {scenes_per_pivot} complete, production-ready SceneWriterDirective object(s) for this pivot.

PIVOT TO COVER:
{json.dumps(pivot, indent=2)}

HARD REQUIREMENT:
- Each scene target: {words_per_scene} words (±15%)

EVERY story_events list MUST be 3–5 concrete, observable actions or spoken lines.  
Examples of correct events for a confession scene:
  • Lirael sits beside Kael and says, “I’m leaving during the Calm Storm.”
  • Kael’s hand tightens on his gadget; he whispers, “The vines will force me to report you.”
  • Kael slides a brass cylinder across the table and says, “This jammer isn’t finished yet.”
  • Voren’s staff taps twice in the doorway; he asks, “Why is your vine dim?”

Do NOT write vague or internal events. If you cannot think of concrete actions, create an extra micro-beat (a look, a touch, an object) to make it visible.

All other fields (agents_in_scene, character_state_changes, setting, emotional_arc, etc.) must derive directly from these concrete events — no freelancing.

Output ONLY the pure JSON DirectorOutput object with "action": "generate_and_ingest" and a "scenes" array of {scenes_per_pivot} complete SceneWriterDirective objects.
"""

            try:
                resp, tokens = await director_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=0.35
                )

                self.director_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                self.director_token_usage["completion_tokens"] += tokens["completion_tokens"]
                self.director_token_usage["total_tokens"] += tokens["total_tokens"]

                clean_resp = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(clean_resp)

                director_output: DirectorOutput = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=director_parser
                )

                rich_scenes: List[dict] = director_output.scenes

                for scene_dict in rich_scenes:
                    scene_dict = scene_dict.model_dump()
                    scene_dict.update({
                        "scene_id": scene_id_counter,
                        "chapter_id": state.current_chapter_id,
                        "act_id": state.current_act_id,
                        "target_word_count": words_per_scene
                    })

                    all_rich_directives.append(scene_dict)
                    scene_id_counter += 1

                print(f"    → Generated {len(rich_scenes)} rich32 rich scenes")

            except Exception as e:
                print(f"    Pivot {idx} failed: {e}")
                traceback.print_exc()
                state.error_message = f"Rich scene planning failed on pivot {idx}: {e}"
                state.next_action = "ERROR"
                return state.__dict__

        # ─── PERSIST FINAL RICH DIRECTIVES ──────────────────────────────────────────
        await self.memory.update_story_progress({
                "director_token_usage": self.director_token_usage
            })
        await self.memory.add_long_term_document(
            text=json.dumps(all_rich_directives, indent=2),
            metadata={
                "type": "director_scene_directives",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )

        print(f"Chapter {state.current_chapter_id}: {len(all_rich_directives)} FULL rich scene directives ready")
        state.next_action = "generate_scenes"
        return state.__dict__
        
    async def generate_scenes_node(self, state: StoryState) -> Dict:
        print(f"✍️ Generate Scenes: Chapter {state.current_chapter_id}, Scene {state.scene_id}")
        
        # Load scene directives
        scenes_json = await self.memory.get_long_term_document(
            metadata={
                "type": "director_scene_directives",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        
        director_directives = json.loads(scenes_json)
        
        for directive_dict in director_directives:
            directive = SceneWriterDirective(**directive_dict)
            directive_text = ContextFormatter.format_any(directive)
            if directive.scene_id != state.scene_id:
                continue
            # later:
            scene_target_length: int = int(directive.target_word_count)
            
            # NEW: Format complete directive for Scene Writer
            # directive_for_writer = format_scene_directive_for_scene_writer(directive)
            
            # print(f"  Scene {state.scene_id}: {directive.scene_purpose}")
            # print(f"  Events: {', '.join(directive.story_events[:2])}")
            
            # Pass FORMATTED directive (not raw JSON)
            user_context = UserSceneContext.create_for_user(
                user_id=self.memory.user_id,
                story_id=self.memory.story_id,
                director_instructions=directive_text,
                scene_chunk_callback=self.scene_chunk_callback,
                token_usage=self.writer_token_usage,
                llm_temp=self.llm_temp,
                scene_target_length=scene_target_length,
                user_age=self.min_age,
                pov=self.pov,
                voice=self.voice,
                tone=self.tone,
                tense=self.tense
            )
            
            scene_text, scene_cluster, status, writer_tokens = await scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE.run_scene(
                ctx=user_context,
                stop_event=self.stop_event
            )

            if status != "SUCCESS":
                state.error_message = f"Scene {directive_text.scene_id} failed: {status}"
                state.next_action = "ERROR"
                return state.__dict__

            word_count = StoryHelpers._count_words_split(scene_text)
            await self.memory.update_user_monthly_word_count(word_count=word_count)
            self.writer_token_usage = writer_tokens

            from setup.shared_redis_pool import get_redis_client 
            redis_client = await get_redis_client()   
            # Redis pause
            if redis_client:
                queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"
                self.scene_chunk_callback({"type": "save"})
                user_choice = False
                for _ in range(3600):
                    choice = await redis_client.lpop(queue_key)
                    if choice in (b"1", "1", 1):
                        user_choice = True
                        print("User chose to continue")
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
                act_id=state.current_act_id,
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
                scene_bundle=scene_bundle or {},
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
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

        state.next_action = "quality_validate_node"
        return state.__dict__

    async def quality_validate_node(self, state: StoryState) -> Dict:
        state.next_action = "ingest_chapter"  # Placeholder
        return state.__dict__

    async def ingest_chapter_node(self, state: StoryState) -> Dict:
        print(f"Ingesting chapter {state.current_chapter_id}...")
        world_details = await self.memory.get_long_term_recent_worlds(state.current_chapter_id)
        char_details = await self.memory.get_long_term_recent_characters(state.current_chapter_id)
        self.scene_chunk_callback({"type": "status", "message": "saving chapter"})
        current_chap_summary = await self.memory.search_single_episodic_story(
            act_number=state.current_act_id,
            chapter_number=state.current_chapter_id,
            summary_type="scene summary"
        )
        chapter_bundle = await Ingestor.ingest_chapter(
            act_id=state.current_act_id,
            chapter_id=state.current_chapter_id,
            current_chap_summary=current_chap_summary,
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
        # self.current_chap_summary = ""
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
            #self.model = progress.get("model", "")
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