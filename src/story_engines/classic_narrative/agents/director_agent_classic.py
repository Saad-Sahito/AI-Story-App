# src/story_engines/classic_adventure/agents/director_agent.py

import json
import asyncio
import gc
import traceback
import re
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
# from langchain_core.output_parsers import StringOutputParser
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
    
    story_events: List[str] = Field(
        description="3-5 CONCRETE actions/discoveries that MUST happen (order matters)"
    )
    
    scene_purpose: str = Field(
        description="Primary purpose: advance_plot | develop_character | reveal_world | create_emotion | setup_payoff | other"
    )
    
    agents_in_scene: Dict[str, Dict[str, Any]] = Field(
        description="""agent_name: { 'role': str, 'gender': str, 'objective': str, 'current_status': str, 'distinctive_voice': str, 'relationships': dict, ... }"""
    )
    
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible container for location, atmosphere, pacing, emotional_arc, tone, etc."
    )
    
    state_changes: Dict[str, Any] = Field(
        default_factory=dict,
        description="What changes as result of this scene (characters, world, plot)"
    )
    
    targets: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="paragraph_count, dialogue_ratio, pov, etc."
    )
    
    notes: str = Field(default="", description="Freeform notes from Director")
    target_word_count: int = Field(description="Target scene length")


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


class DirectorOutput(BaseModel):
    scenes: List[SceneWriterDirective] = Field(..., description="List of concrete scene directives for Scene Writer")
    action: Literal["generate_and_ingest", "END"] = Field(
        description="'generate_and_ingest' to continue, 'END' if story complete"
    )


# ============================================================================
# BEST-IN-CLASS JSON SCHEMA INSTRUCTION (2025 STANDARD)
# ============================================================================
# def best_json_instructions(model: type[BaseModel]) -> str:
#     schema = model.model_json_schema()
    
#     def format_field(name: str, info: dict, *, required: bool = False, depth: int = 0):
#         indent = "  " * depth
#         req = " [REQUIRED]" if required else ""
#         desc = f" — {info.get('description', '').strip()}" if info.get('description') else ""
#         type_ = info.get('type', 'any')
#         if '$ref' in info:
#             type_ = info['$ref'].split('/')[-1]
#         elif 'anyOf' in info:
#             refs = [r['$ref'].split('/')[-1] for r in info['anyOf'] if '$ref' in r]
#             type_ = " | ".join(refs) if refs else 'union'
        
#         default = ""
#         if 'default' in info and info['default'] is not None:
#             default = f" (default: {json.dumps(info['default'], ensure_ascii=False)})"
        
#         return f"{indent}- {name}: {type_}{default}{req}{desc}"

#     lines = [
#         "YOU MUST OUTPUT EXACTLY ONE VALID JSON OBJECT WITH THESE FIELDS:",
#         "═" * 78,
#     ]
    
#     required = set(schema.get("required", []))
#     for field_name, field_info in schema["properties"].items():
#         is_req = field_name in required
#         lines.append(format_field(field_name, field_info, required=is_req, depth=0))
        
#         if field_info.get("type") == "array" and "items" in field_info:
#             items = field_info["items"]
#             if "$ref" in items:
#                 ref_name = items["$ref"].split("/")[-1]
#                 lines.append(f"  → Each item is a {ref_name} object")
#             elif "properties" in items:
#                 lines.append(f"  → Each item contains:")
#                 for sub_name, sub_info in items["properties"].items():
#                     sub_req = sub_name in items.get("required", [])
#                     lines.append(format_field(sub_name, sub_info, required=sub_req, depth=2))
        
#         if "properties" in field_info:
#             lines.append(f"  → Contains:")
#             for sub_name, sub_info in field_info["properties"].items():
#                 sub_req = sub_name in field_info.get("required", [])
#                 lines.append(format_field(sub_name, sub_info, required=sub_req, depth=2))

#     lines.extend([
#         "",
#         "CRITICAL RULES:",
#         "- Output ONLY the JSON. First character must be {, last must be }",
#         "- No markdown, no code fences, no explanations, no extra text",
#         "- If you cannot comply, output: {\"error\": \"failed to generate valid JSON\"}",
#         "",
#         "Begin JSON now:"
#     ])
    
#     return "\n".join(lines)


# Pre-compute once — this is your new "get_format_instructions()"
DIRECTOR_JSON_INSTRUCTIONS = DirectorOutput.model_json_schema()


# ============================================================================
# GENERIC CONTEXT FORMATTER
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
            for i, item in enumerate(value[:100], 1):
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
    # final_plot: Dict
    act_plan: Dict
    chapter_word_count_target: int
    act_setup_payload: List[str]
    act_payoff_payload: List[str]
    connected_agents: List[Dict]
    integrated_world: Dict
    character_progressions: List[Dict]
    # conflict_matrix: Dict
    world_progressions: List[Dict]
    story_so_far_context: Dict
    entire_story_tracker: Dict
    current_chapter_anchors: List[Dict] = field(default_factory=list)


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

        # Edges (unchanged)
        self.graph.add_edge("load_act_node", "scene_planner_node")
        self.graph.add_conditional_edges("scene_planner_node", lambda s: s.next_action, {
            "generate_scenes": "generate_scenes_node",
            "END": "story_complete_node",
            "ERROR": "error_node"
        })
        self.graph.add_conditional_edges("generate_scenes_node", lambda s: s.next_action, {
            "END": END,
            "ERROR": "error_node",
            "quality_validate_node": "quality_validate_node"
        })
        self.graph.add_conditional_edges("quality_validate_node", lambda s: s.next_action, {
            "ingest_chapter": "ingest_chapter_node",
            "retry_scenes": "scene_planner_node",
            "ERROR": "error_node"
        })
        self.graph.add_edge("ingest_chapter_node", END)
        self.graph.add_edge("story_complete_node", END)
        self.graph.add_edge("error_node", END)

        self.compiled = self.graph.compile()

        self.user_id = None
        self.llm_temp = 0.7
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
        self.pov = seed['pov'] if seed['pov'] else self.pov
        self.voice = seed['prose_style'] if seed['prose_style'] else self.voice
        self.tone = seed['tone'] if seed['tone'] else self.tone
        # plot = json.loads(await self.memory.get_long_term_document(metadata={'type': 'expanded_plot_outline', 'story_title': story_title}))
        act_plan = json.loads(await self.memory.get_long_term_document(metadata={'type': 'act_plan', 'act_id': act_id, 'story_title': story_title}))
        agents = json.loads(await self.memory.get_long_term_document(metadata={'type': 'connected_agents', 'story_title': story_title}))
        world = json.loads(await self.memory.get_long_term_document(metadata={'type': 'integrated_world', 'story_title': story_title}))
        # conflict = json.loads(await self.memory.get_long_term_document(metadata={'type': 'conflict_matrix', 'story_title': story_title}))
        tracker = json.loads(await self.memory.get_long_term_document(metadata={'type': 'story_tracker', 'story_title': story_title}))

        director_context = await self.memory.get_director_context(current_act_number=act_id, current_chapter_number=chapter_id)

        # Extract the high-level chapter outline (still stored inside act_plan for backward compat)
        rich_chapter = next(
            (c for c in act_plan.get("chapter_outlines", []) if isinstance(c, dict) and c.get("chapter_number") == chapter_id),
            {}
        )
        target_word_count = rich_chapter.get("target_word_count")
        char_prog = await self.memory.get_long_term_recent_characters(chapter_id)
        world_prog = await self.memory.get_long_term_recent_worlds(chapter_id)
        # recent_summary = await self.memory.search_single_episodic_story(act_number=act_id, chapter_number=chapter_id, summary_type="scene summary")
        #print(rich_chapter.get("anchor_points"))
        return StoryContextSnapshot(
            story_seed=seed,
            # final_plot=plot,
            act_plan=act_plan,
            chapter_word_count_target=target_word_count,
            act_setup_payload=act_plan.get("setup_this_act", []),
            current_chapter_anchors=rich_chapter.get("anchor_points"),
            act_payoff_payload=act_plan.get("payoff_this_act", []),
            connected_agents=agents,
            integrated_world=world,
            # conflict_matrix=conflict,
            character_progressions=char_prog,
            world_progressions=world_prog,
            story_so_far_context=director_context,
            entire_story_tracker=tracker['act_tracking']
            # recent_chapters_summary=recent_summary or ""
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

        anchor_points = getattr(self.story_context, "current_chapter_anchors", [])
        if not anchor_points:
            print("No anchor points → skipping to generate_scenes")
            state.next_action = "generate_scenes"
            return state.__dict__

        total_chapter_words: int = self.story_context.chapter_word_count_target

        system_prompt = f"""You are the Director — a merciless cinematic surgeon who turns story anchors into tight, filmable, irreversible scenes, no matter the genre or tone.

YOUR NON-NEGOTIABLE LAWS (apply to every genre, every time)
1. ONE DRAMATIC FUNCTION PER SCENE — FULL STOP  
   If a previous scene in this chapter already did “prove the magic key works”, “trade an object for access”, “disable surveillance”, or “receive a new McGuffin”, you are FORBIDDEN to do it again.  
   Invent a new function or merge with an existing scene. Repetition = instant death.

2. MINIMAL INTERNAL STATES IN NARRATION
   Never name emotions, thoughts, or realisations. Never use physical clichés as shorthand for feeling.
   Emotion and thought ONLY exist when they erupt into observable behaviour, dialogue, or world reaction.

3. THE PROTAGONIST (or current POV character) MUST HAVE AGENCY
   In every scene containing the focal character, they MUST perform at least one deliberate, concrete action that creates a new fact in the world. No passive witnessing.

4. EVERY SCENE MUST CREATE AT LEAST ONE IRREVERSIBLE CHANGE  
   Something must break, burn, die, get pregnant, get broadcast, get tattooed, get sold, get publicly confessed — anything that cannot be undone by the end of the book, let alone the chapter.

5. NO REPETITION OF DRAMATIC FUNCTION
   Never create two scenes in the same chapter that serve the same purpose 
   (“someone delivers bad news”, “someone tempts the hero”, “someone explains lore”, etc.).

6. GENRE DICTATES FLAVOUR, NOT RULES
   - In comedy: irreversible change can be a pie in the face that starts a food fight that gets someone fired.
   - In romance: it can be the first kiss that immediately gets interrupted by an ex walking in.
   - In cozy mystery: it can be knocking over a teacup that reveals a hidden letter.
   - In tragedy: it can be blood on the floor.
   The laws stay identical; only the texture changes.

7. DIALOGUE MUST HAVE A WINNER AND A LOSER  
   Every line is an attempt to dominate, seduce, blackmail, beg, or lie. Never neutral exposition.

POSITIVE COMMANDMENTS (always)

- Use specific, tactile objects as turning points.
- Make the environment react to choices in real time (weather, lights, animals, technology, magic, social media notifications — whatever fits the genre).
- Dialogue must sound like people trying to win, seduce, survive, or lie — never like exposition.
- Heighten pressure appropriate to tone (a ticking bomb, an awkward silence, a phone about to vibrate, a cat about to knock over a vase).

OUTPUT RULES (unchanged schema)

- story_events: 4–6 numbered, filmable beats of action or exact dialogue
- emotional_arc: always phrased as observable shift (“Fidgeting with ring → Ring thrown into the fire”)
- state_changes: must contain at least one bullet starting with “IRREVERSIBLE:” (funny or devastating, doesn’t matter)
- must inlude gender with each agent mentioned under 'agents_in_scene'

Total chapter word target: {total_chapter_words}

Full story context (never contradict):
{ContextFormatter.full_context(self.story_context)}

{DIRECTOR_JSON_INSTRUCTIONS}
"""

        all_rich_directives = []
        scene_id_counter = state.scene_id
        total_anchors = len(anchor_points)

        for idx, anchor in enumerate(anchor_points, 1):
            print(f"  Planning anchor {idx}/{total_anchors}: {anchor.get('anchor_type', 'unknown')}")

            scenes_per_anchor = 1
            # Only allow 2+ if the anchor explicitly says "estimated_scenes: 2+" in the act plan
            if anchor.get("estimated_scenes", 1) > 1:
                scenes_per_anchor = anchor["estimated_scenes"]
            words_per_scene = int(total_chapter_words / (scenes_per_anchor * total_anchors))
            # words_per_scene = max(300, int(total_chapter_words / (scenes_per_anchor * total_anchors)))
            print(f"  → Generating {scenes_per_anchor} scene(s) @ ~{words_per_scene} words each")
# - Location: {anchor.get('location', 'Not specified')}
# - Time: {anchor.get('time_context', 'Not specified')}
# - Character Positions: {json.dumps(anchor.get('character_positions', {}), indent=2)}
# - Transition from previous: {anchor.get('transition_from_previous', 'N/A')}
# - Required character states: {json.dumps(anchor.get('required_character_states', {}), indent=2)}
            human_prompt = f"""Convert this anchor into {scenes_per_anchor} concrete, executable scene(s) for a novel.
ENTIRE CHAPTER ANCHOR POINTS (For Context Reference):
{json.dumps(anchor_points, indent=2)}

════════════════════════════════════════════════════════════════════════════════
ANCHOR TO DRAMATIZE
════════════════════════════════════════════════════════════════════════════════

{json.dumps(anchor, indent=2)}

════════════════════════════════════════════════════════════════════════════════
SCENE SPECIFICATIONS
════════════════════════════════════════════════════════════════════════════════

Target word count per scene: {words_per_scene} words (±15% acceptable)

Story events requirements:
- 3-5 concrete, observable actions or lines of dialogue
- Each must be specific enough that a writer knows EXACTLY what to show
- If the anchor suggests emotion or thought, convert to: body language, objects, environmental reactions, or dialogue
- Order events for maximum dramatic impact (not necessarily chronological with anchor)

Context fields to populate:
- Include only fields relevant to THIS scene's genre/needs
- Prioritize: location, atmosphere, pacing, emotional_arc
- Add genre-specific fields as appropriate (clues, magic_effects, combat_details, romantic_tension, etc.)

Agent profiles:
- All five core fields are MANDATORY: role, objective, current_status, distinctive_voice, relationships
- Add optional fields that create dramatic specificity (hidden_agenda, physical_tells, secrets_kept, etc.)

State changes:
- What do characters LEARN (concrete facts, not feelings)?
- How do relationships SHIFT (observable behavioral changes)?
- What physical/world changes occur?

════════════════════════════════════════════════════════════════════════════════
CREATIVE EXPECTATIONS
════════════════════════════════════════════════════════════════════════════════

This is your chance to elevate the material:
- If the anchor is flat, add environmental pressure or obstacles
- If it's generic, find the specific detail that makes it memorable
- If it's rushed, break the moment into beats that breathe
- If it's safe, find the surprising-yet-logical choice

Ask yourself: "Could I write this scene with the details I've provided?"
If not, add more concrete specificity.

This is a text-based novel not a movie or screenplay.

No markdown. No code fences. No explanations. Just pure JSON.
"""


            try:
                resp, tokens = await director_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=self.llm_temp
                )

                self.director_token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                self.director_token_usage["completion_tokens"] += tokens["completion_tokens"]
                self.director_token_usage["total_tokens"] += tokens["total_tokens"]

                clean_resp = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(clean_resp).strip()

                # Just pass the MODEL, not the parser
                clean_resp = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=DirectorOutput  # or SceneOutput — just the class!
                )

                

                rich_scenes: List[dict] = [s.model_dump() for s in clean_resp.scenes]

                for scene_dict in rich_scenes:
                    scene_dict.update({
                        "scene_id": scene_id_counter,
                        "chapter_id": state.current_chapter_id,
                        "act_id": state.current_act_id,
                        "target_word_count": words_per_scene
                    })
                    all_rich_directives.append(scene_dict)
                    scene_id_counter += 1

                print(f"    → Generated {len(rich_scenes)} rich scenes")

            except Exception as e:
                print(f"    anchor {idx} failed: {e}")
                traceback.print_exc()
                state.error_message = f"Rich scene planning failed on anchor {idx}: {e}"
                state.next_action = "ERROR"
                return state.__dict__

        # Persist
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
            
            if directive.scene_id != state.scene_id:
                continue
            # Check monthly word count
            monthly_wc_data = await self.memory.get_monthly_word_count()
            if monthly_wc_data.get('tier') == 1:
                if monthly_wc_data.get('monthly_word_count') >= tier_1_monthly_words_limit:
                    self.scene_chunk_callback({"type": "error", "message": "User monthly word count limit reached for tier 'free'"})
                    raise
            elif monthly_wc_data.get('tier') == 2:
                if monthly_wc_data.get('monthly_word_count') >= tier_2_monthly_words_limit:
                    self.scene_chunk_callback({"type": "error", "message": "User monthly word count limit reached for tier 'scribe'"})
                    raise
            del monthly_wc_data
        
            scene_target_length: int = int(directive.target_word_count)
            directive_text = ContextFormatter.format_any(directive)

            prev_scene = await self.memory.get_story_cluster(chapter_id=state.current_chapter_id) if state.scene_id > 1 else ""
            if prev_scene != "":
                prev_scene = prev_scene['text'][-1]
                if prev_scene['type'] == 'text':
                    prev_scene = prev_scene['scene_text']
            user_context = UserSceneContext.create_for_user(
                user_id=self.memory.user_id,
                story_id=self.memory.story_id,
                prev_scene=prev_scene,
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
                state.error_message = f"Scene {state.scene_id} failed: {status}"
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
                known_characters=chars,
                known_locations=worlds
            )

            state.current_chapter_word_count += word_count
            state.story_word_count += word_count
            await self.memory.add_post_scene_bundle(
                    scene_bundle=scene_bundle or {},
                    metadata={
                        "chapter_id": state.current_chapter_id,
                        "scene_id": state.scene_id,
                        "story_title": state.story_title,
                        "act_id": state.current_act_id
                    }
                )

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
            
           
            await self.memory.update_story_progress({
                "latest_chapter_id": state.current_chapter_id,
                "continue_scene_id": state.scene_id + 1,
                "story_word_count": state.story_word_count,
                "chapter_word_count": state.current_chapter_word_count,
                "writer_token_usage": self.writer_token_usage
            })

            state.scene_id += 1
            self.scene_chunk_callback({"type": "saved", "message": "story saved"})

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
            chapter_text=current_chap_summary,
            current_characters=char_details,
            current_locations=world_details
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

            chapter_outlines = [
                c for c in self.story_context.act_plan.get("chapter_outlines", []) 
            ]
            highest_chapter = 0
            if chapter_outlines:
                highest_chapter_dict = max(chapter_outlines, key=lambda c: c.get("chapter_number", 0))
                highest_chapter = highest_chapter_dict.get("chapter_number")
                # print(f"The highest chapter number found is: {highest_chapter}")
            else:
                highest_chapter = 0
                print("The chapter_outlines list is empty or could not be found.")
            
            if state.current_chapter_id == highest_chapter:
                self.scene_chunk_callback({"type": "act_complete", "message": "complete"})

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
    async def run(self, scene_chunk_callback, user_id, stop_event: asyncio.Event | None = None):
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()
        self.user_id = user_id
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
