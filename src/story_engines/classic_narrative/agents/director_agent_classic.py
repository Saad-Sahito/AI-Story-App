# src/story_engines/classic_adventure/agents/director_agent.py

import json
import asyncio
import gc
import traceback
from typing import Any, Dict, List, Optional, Literal
from langgraph.graph import StateGraph, END
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from .shared_scene_planner import UserSceneContext
from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import director_client
from config_vars import tier_1_monthly_words_limit, tier_2_monthly_words_limit
from src.utilities.ingestor import Ingestor
from .story_author import ChapterOutline, ActPlan
# ============================================================================
# STATE AND MODELS
# ============================================================================
class SceneWriterDirective(BaseModel):
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
        description="""agent_name: { 
            'role': str, 
            'gender': str, 
            'objective': str,  
            'current_status': str,  
            'distinctive_voice': str,
            'core_motivation': str,
            'relationships': dict,
            
            # Contextual Depth (populated if available)
            'active_subplot': str, # If this scene advances a subplot
            'backstory_trigger': str, # If a specific past event is relevant
            ... 
        }"""
    )
    
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="""location, atmosphere, pacing, emotional_arc, tone, thematic_beat, world_rules_active, mechanics_to_clarify
        """
    )
    
    state_changes: Dict[str, Any] = Field(
        description="""characters, relationships, world changes, and:
        
        'emotional_beats': [
            {
                'type': 'betrayal' | 'sacrifice' | 'confession' | 'loss' | 'triumph',
                'character': 'Name',
                'requires_setup': ['trust built in prev scene', 'hinted conflicting loyalty'],
                'payoff_if_earned': 'Reader feels gutted, not confused'
            }
        ]
        """
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
            try:
                data = {k: v for k, v in obj.__dict__.items()} if hasattr(obj, '__dict__') else vars(obj)
            except:
                data = str(obj)

        parts = [
            "╔═══════════════════════════════════════════════════════════╗",
            "║ SCENE DIRECTIVE FOR WRITER",
            "╚═══════════════════════════════════════════════════════════╝",
        ]
        if isinstance(data, dict):
            for key, value in data.items():
                if key == "previous_scenes_in_chapter":
                    continue
                title = key.replace("_", " ").upper()
                parts.append(ContextFormatter.format_section(title, value))
        else:
            parts.append(str(data))
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
            "╚═══════════════════════════════════════════════════════════",
        ]
        
        if snapshot.central_conflict:
            parts.append(f"\n🎯 CENTRAL CONFLICT\n{snapshot.central_conflict}\n")
        
        if snapshot.character_core_drives:
            parts.append("\n👥 CHARACTER MOTIVATIONS")
            for name, drive in list(snapshot.character_core_drives.items())[:8]:
                parts.append(f"  • {name}: {drive}")
            parts.append("")
        
        if snapshot.relationship_web:
            parts.append("\n🔗 KEY RELATIONSHIPS")
            for name, relationships in list(snapshot.relationship_web.items())[:6]:
                for other, tension in list(relationships.items())[:2]:
                    parts.append(f"  • {name} ↔ {other}: {tension}")
            parts.append("")
        
        if snapshot.world_rules:
            parts.append("\n🌍 WORLD CONSTRAINTS")
            for rule in snapshot.world_rules[:5]:
                parts.append(f"  • {rule}")
            parts.append("")
        
        # Add Chapter Context specifically
        if snapshot.chapter_context:
             parts.append(ContextFormatter.format_section("CHAPTER CONTEXT", snapshot.chapter_context))

        return "\n".join(parts)

@dataclass
class StoryContextSnapshot:
    story_seed: Dict
    act_plan: Dict
    chapter_word_count_target: int
    act_setup_payload: List[str]
    act_payoff_payload: List[str]
    connected_agents: List[Dict]
    integrated_world: Dict
    character_progressions: List[Dict]
    world_progressions: List[Dict]
    story_so_far_context: Dict
    entire_story_tracker: Dict
    chapter_context: Dict
    current_chapter_anchors: List[Dict] = field(default_factory=list)
    
    # NEW: Critical missing context
    central_conflict: str = ""  
    character_core_drives: Dict[str, str] = field(default_factory=dict)
    relationship_web: Dict[str, Dict[str, str]] = field(default_factory=dict) 
    world_rules: List[str] = field(default_factory=list)  
    active_themes: Dict[str, str] = field(default_factory=dict) 

    # Epic Mode Enhancements
    backstory_highlights: List[Dict] = field(default_factory=list)
    key_locations: List[Dict] = field(default_factory=list)
    npc_pool: List[Dict] = field(default_factory=list)
    active_subplots: List[Dict] = field(default_factory=list)
    story_mode: str = "standard"


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
            "quality_validate_node": "quality_validate_node",
            "END": "story_complete_node",
            "ERROR": "error_node"
        })
        self.graph.add_conditional_edges("quality_validate_node", lambda s: s.next_action, {
            "generate_scenes": "generate_scenes_node",
            # "retry_scenes": "scene_planner_node",
            "ERROR": "error_node"
        })
        self.graph.add_conditional_edges("generate_scenes_node", lambda s: s.next_action, {
            "END": END,
            "ERROR": "error_node",
            "ingest_chapter": "ingest_chapter_node"
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
        self.genre_list = []
        self.director_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.writer_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.ingestor_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.utility_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.target_length = 50000

    # ============================================================================
    # CONTEXT BUILDING
    # ============================================================================

    async def _build_context(self, story_title: str, act_id: int, chapter_id: int):
        seed = json.loads(await self.memory.get_long_term_document(
             metadata={'type': 'story_seed', 'story_title': story_title}
        ))
        
        # DETERMINE MODE
        target_length = seed.get('target_length', 50000)
        story_mode = "compact" if target_length < 40000 else "epic" if target_length >= 80000 else "standard"
        
        # Initialize Epic containers
        backstories = []
        world_guide = {}
        subplot_arch = {}

        # BUILD MODE-APPROPRIATE CONTEXT
        if story_mode == "epic":
            try:
                # Load epic enhancements (assuming they are stored as JSON strings of the objects)
                backstories_json = await self.memory.get_long_term_document(
                    metadata={'type': 'character_backstories', 'story_title': story_title}
                )
                if backstories_json:
                    backstories = json.loads(backstories_json)

                world_guide_json = await self.memory.get_long_term_document(
                    metadata={'type': 'world_guide', 'story_title': story_title}
                )
                if world_guide_json:
                    world_guide = json.loads(world_guide_json)

                subplot_json = await self.memory.get_long_term_document(
                    metadata={'type': 'subplot_architecture', 'story_title': story_title}
                )
                if subplot_json:
                    subplot_arch = json.loads(subplot_json)
            except Exception as e:
                print(f"Warning: Failed to load some epic context: {e}")

        self.pov = seed.get('pov', self.pov)
        self.voice = seed.get('prose_style', self.voice)
        self.tone = seed.get('tone', self.tone)
        self.genre_list = seed.get('genre', [])
        
        act_plan = json.loads(await self.memory.get_long_term_document(metadata={'type': 'act_plan', 'act_id': act_id, 'story_title': story_title}))
        agents = json.loads(await self.memory.get_long_term_document(metadata={'type': 'connected_agents', 'story_title': story_title}))
        world = json.loads(await self.memory.get_long_term_document(metadata={'type': 'integrated_world', 'story_title': story_title}))
        conflict_matrix = json.loads(await self.memory.get_long_term_document(metadata={'type': 'conflict_matrix', 'story_title': story_title}))
        
        tracker_json = await self.memory.get_long_term_document(metadata={'type': 'story_tracker', 'story_title': story_title})
        tracker = json.loads(tracker_json) if tracker_json else {'act_tracking': {}}

        director_context = await self.memory.get_director_context(current_act_number=act_id, current_chapter_number=chapter_id)

        rich_chapter = next(
            (c for c in act_plan.get("chapter_outlines", []) if isinstance(c, dict) and c.get("chapter_number") == chapter_id),
            {}
        )
        
        # --- ROBUST EXTRACTION ---
        
        # Extract character motivations and relationships
        char_drives = {}
        relationship_web = {}
        for agent in agents:
            name = agent.get('name', 'Unknown')
            # Fallback chain for plot function
            char_drives[name] = agent.get('plot_function') or agent.get('essence') or agent.get('role', '')
            
            # Robust relationship extraction (checks multiple likely keys)
            rels = agent.get('relationships') or agent.get('key_relationships') or agent.get('relationship_web', {})
            if rels:
                relationship_web[name] = rels
        
        # Robust World Rules extraction
        world_rules = []
        # 1. Check explicit "world_rules" list (added in patched Author)
        if 'world_rules' in world and isinstance(world['world_rules'], list):
            world_rules.extend(world['world_rules'])
        
        # 2. Check for dynamic keys (Legacy/Flexible fallback)
        for key, val in world.items():
            k_lower = key.lower()
            if any(term in k_lower for term in ['magic', 'tech', 'law', 'rule', 'system', 'constraint']):
                if isinstance(val, str):
                    world_rules.append(f"{key.capitalize()}: {val[:100]}...") # truncate for tokens
        
        # Extract active themes
        active_themes = {theme: f"Explored through {seed.get('genre', ['story'])[0]} lens" 
                     for theme in seed.get('themes', [])}
        
        target_word_count = rich_chapter.get("target_word_count", 2000)
        char_prog = await self.memory.get_long_term_recent_characters(chapter_id)
        world_prog = await self.memory.get_long_term_recent_worlds(chapter_id)

        # Build targeted context
        chapter_context = self.build_chapter_planner_context(
            story_data={
                "seed": seed,
                "connected_agents": agents,
                "integrated_world": world,
                "conflict_matrix": conflict_matrix,
                "story_tracker": tracker.get('act_tracking', {}),
                "character_backstories": backstories,
                "world_guide": world_guide,
                "subplot_architecture": subplot_arch
            },
            act_plan=ActPlan(**act_plan), # Using the imported pydantic model for type hinting helper
            chapter_outline=ChapterOutline(**rich_chapter) if rich_chapter else None,
            story_mode=story_mode,
            recent_chapters_summary=director_context
        )

        return StoryContextSnapshot(
            story_seed=seed,
            act_plan=act_plan,
            chapter_word_count_target=target_word_count,
            act_setup_payload=act_plan.get("setup_this_act", []),
            current_chapter_anchors=rich_chapter.get("anchor_points"),
            act_payoff_payload=act_plan.get("payoff_this_act", []),
            connected_agents=agents,
            integrated_world=world,
            character_progressions=char_prog,
            world_progressions=world_prog,
            story_so_far_context=director_context,
            entire_story_tracker=tracker.get('act_tracking', {}),
            central_conflict=conflict_matrix.get('central_conflict', ''),
            character_core_drives=char_drives,
            relationship_web=relationship_web,
            world_rules=world_rules,
            active_themes=active_themes,
            chapter_context=chapter_context,
            backstory_highlights=backstories,
            key_locations=world_guide.get('location_dossiers', []) if world_guide else [],
            npc_pool=world_guide.get('minor_character_pool', []) if world_guide else [],
            active_subplots=subplot_arch.get('subplots', []) if subplot_arch else [],
            story_mode=story_mode
        )
    
    def _infer_subplot_status(self, subplot: Dict, current_act: int) -> str:
        """Infer where we are in the subplot arc"""
        # Handle dict access for subplot
        act_integration = subplot.get('act_integration', {})
        # Convert string keys to int if necessary
        act_keys = []
        for k in act_integration.keys():
            try:
                act_keys.append(int(k))
            except:
                pass
        
        total_acts_in_subplot = len(act_keys)
        acts_completed = len([a for a in act_keys if a < current_act])
        
        if acts_completed == 0:
            return "Setup phase"
        elif acts_completed < total_acts_in_subplot - 1:
            return "Development phase"
        else:
            return "Resolution phase"
        
    def build_chapter_planner_context(
        self,
        story_data: Dict[str, Any],
        act_plan: ActPlan,
        chapter_outline: ChapterOutline,
        story_mode: str,
        recent_chapters_summary: str
    ) -> Dict[str, Any]:
        """
        Build highly targeted context for chapter/scene planner.
        Only include what's relevant to THIS chapter.
        """
        if not chapter_outline:
            return {}

        # Extract agents present in this chapter
        agents_in_chapter = set()
        locations_in_chapter = set()
        
        for anchor in chapter_outline.anchor_points:
            # Handle list of strings or list of objects
            if hasattr(anchor, 'agents_present'):
                agents_in_chapter.update(anchor.agents_present)
            
            if hasattr(anchor, 'location'):
                locations_in_chapter.add(anchor.location)
        
        # Filter connected agents to only those in this chapter
        all_agents = story_data["connected_agents"]
        relevant_agents = []
        for a in all_agents:
            # Handle Pydantic object or Dict
            name = a.get('name') if isinstance(a, dict) else a.name
            if name in agents_in_chapter:
                relevant_agents.append(a)

        base_context = {
            "chapter_outline": chapter_outline.model_dump() if hasattr(chapter_outline, 'model_dump') else chapter_outline,
            "relevant_agents": relevant_agents,
            "story_so_far": recent_chapters_summary,
        }
        
        if story_mode == "compact":
            # STILL INCLUDE ESSENTIALS - just more condensed
            base_context["world_snippet"] = {
                "foundation": story_data["integrated_world"].get("foundation", ""),
                "key_rules": story_data["integrated_world"].get("world_rules", [])[:3],  # Top 3 rules
            }
            base_context["conflict_core"] = {
                "central": story_data["conflict_matrix"].get("central_conflict", ""),
                "escalation": story_data["conflict_matrix"].get("escalation_path", "")[:200] + "..."  # Truncate
            }
            return base_context
        elif story_mode == "standard":
            # Moderate world + conflict context
            base_context.update({
                "world_context": {
                    "foundation": story_data["integrated_world"].get("foundation", ""),
                    "plot_integration": story_data["integrated_world"].get("plot_integration", ""),
                },
                "conflict_context": {
                    "central_conflict": story_data["conflict_matrix"].get("central_conflict", ""),
                    "escalation_path": story_data["conflict_matrix"].get("escalation_path", ""),
                },
            })
            return base_context
        
        elif story_mode == "epic":
            # Rich, filtered context
            backstories = story_data.get("character_backstories", [])
            world_guide = story_data.get("world_guide", {})
            subplot_arch = story_data.get("subplot_architecture", {})
            
            # Enrich agents with relevant backstory snippets
            enriched_agents = []
            # Normalize backstories list of dicts
            backstory_dict = {bs.get('name'): bs for bs in backstories}
            
            for agent in relevant_agents:
                agent_dict = agent if isinstance(agent, dict) else agent.model_dump()
                agent_name = agent_dict.get("name")
                
                if agent_name in backstory_dict:
                    bs = backstory_dict[agent_name]
                    # Inject snippet
                    events = bs.get('formative_events', '')
                    agent_dict["backstory_snippet"] = events[:400] + "..." if events else ""
                    agent_dict["active_secrets"] = bs.get('secrets', [])[:2]
                    agent_dict["voice_profile"] = bs.get('voice_profile', '')
                    agent_dict["psychological_notes"] = bs.get('psychological_profile', '')[:300] + "..."
                
                enriched_agents.append(agent_dict)
            
            # Filter locations
            location_details = []
            if world_guide and 'location_dossiers' in world_guide:
                for loc in world_guide['location_dossiers']:
                    if loc.get("name") in locations_in_chapter:
                        location_details.append(loc)
            
            # Filter NPCs (only from relevant locations)
            available_npcs = []
            if world_guide and 'minor_character_pool' in world_guide:
                for npc in world_guide['minor_character_pool']:
                    # Heuristic: if npc location matches
                    if npc.get("location") in locations_in_chapter:
                        available_npcs.append(npc)
            
            # Filter subplots (only those active in this act)
            active_subplots = []
            if subplot_arch and 'subplots' in subplot_arch:
                current_act = act_plan.act_number
                for sp in subplot_arch['subplots']:
                    # Normalize act_integration keys
                    act_integ = sp.get('act_integration', {})
                    # Check if this subplot has a beat in current act (using string or int keys)
                    beat = act_integ.get(current_act) or act_integ.get(str(current_act))
                    
                    if beat:
                        active_subplots.append({
                            "title": sp.get('title'),
                            "premise": sp.get('premise'),
                            "character_owner": sp.get('character_owner'),
                            "beat_this_act": beat,
                            "current_status": self._infer_subplot_status(sp, current_act)
                        })
            
            base_context.update({
                "relevant_agents": enriched_agents,  # Overwrite with enriched version
                "location_details": location_details,
                "available_npcs": available_npcs,
                "active_subplots": active_subplots,
                "world_systems": {
                    "magic_tech": world_guide.get('system_documentation', {}) if world_guide else {},
                },
            })
            
            return base_context
        
        return base_context
    
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
            state.next_action = "quality_validate_node"
            return state.__dict__

        anchor_points = getattr(self.story_context, "current_chapter_anchors", [])
        if not anchor_points:
            print("No anchor points → skipping to generate_scenes")
            state.next_action = "generate_scenes"
            return state.__dict__

        total_chapter_words: int = self.story_context.chapter_word_count_target
        story_mode = getattr(self.story_context, "story_mode", "standard")
        
        all_rich_directives = []
        scene_id_counter = state.scene_id

        # ====================================================================
        # STRATEGY 1: BATCH PROCESSING (Compact & Standard)
        # Faster, cheaper, better flow for normal chapters.
        # ====================================================================
#         if story_mode in ["compact", "standard"]:
#             print(f"⚡ Using Batch Planning Strategy ({story_mode} mode)")
            
#             system_prompt = f"""You are the Director. Plan the ENTIRE CHAPTER based on these anchors.

# CORE CONTEXT:
# Conflict: {self.story_context.central_conflict}
# Themes: {', '.join(self.story_context.active_themes.keys())}
# World Rules: {' | '.join(self.story_context.world_rules)}

# CHARACTER DRIVES:
# {chr(10).join(f"• {name}: {drive}" for name, drive in list(self.story_context.character_core_drives.items())[:8])}

# RELATIONSHIPS:
# {chr(10).join(f"• {name} vs {other}: {tension}" 
#               for name, rels in list(self.story_context.relationship_web.items())[:5]
#               for other, tension in list(rels.items())[:2])}

# RULES:
# 1. Output ALL scenes for the chapter in one list.
# 2. Distribute the total word count ({total_chapter_words}) appropriately across scenes.
# 3. Ensure scenes flow logically from one to the next (cause & effect).
# 4. Do not drop any anchors - every anchor must be represented by at least one scene.
# 5. Create as many scenes as needed to pacing (usually 1-2 per anchor).

# {DIRECTOR_JSON_INSTRUCTIONS}
# """

#             human_prompt = f"""Plan the full chapter sequence.

# TOTAL CHAPTER TARGET: {total_chapter_words} words.

# ANCHOR POINTS (The Roadmap):
# {json.dumps(anchor_points, indent=2)}

# Generate concrete scene directives that bridge these anchors into a seamless narrative."""

#             try:
#                 resp, tokens = await director_client(
#                     system_prompt=system_prompt,
#                     human_prompt=human_prompt,
#                     llm_temp=self.llm_temp
#                 )
#                 self.director_token_usage = StoryHelpers._add_tokens_to_total(self.director_token_usage, tokens)

#                 clean_resp = StoryHelpers._extract_content(resp)
#                 clean_resp = StoryHelpers._strip_code_fences(clean_resp).strip()

#                 batch_output, utility_tokens = await StoryHelpers.load_json_with_retry(
#                     text=clean_resp,
#                     parser=DirectorOutput
#                 )
#                 if isinstance(batch_output, tuple):
#                     batch_output = batch_output[0]
#                 self.utility_token_usage = StoryHelpers._add_tokens_to_total(self.utility_token_usage, utility_tokens)
                
#                 rich_scenes: List[dict] = [s.model_dump() for s in batch_output.scenes]
                
#                 # Assign IDs and metadata
#                 for scene_dict in rich_scenes:
#                     scene_dict.update({
#                         "scene_id": scene_id_counter,
#                         "chapter_id": state.current_chapter_id,
#                         "act_id": state.current_act_id
#                         # target_word_count is presumably set by LLM in batch mode, 
#                         # or we can enforce a split here if needed, but letting LLM decide is better for flow.
#                     })
#                     all_rich_directives.append(scene_dict)
#                     scene_id_counter += 1
                    
#                 print(f"    → Batch generated {len(rich_scenes)} scenes")

#             except Exception as e:
#                 print(f"    Batch planning failed: {e}")
#                 traceback.print_exc()
#                 state.error_message = f"Batch planning failed: {e}"
#                 state.next_action = "ERROR"
#                 return state.__dict__

        # ====================================================================
        # STRATEGY 2: LOOP PROCESSING (Epic & Fallback)
        # Granular control for complex/long chapters.
        # ====================================================================
        # else:
        print(f"🐢 Using Iterative Planning Strategy ({story_mode} mode)")
        
        total_anchors = len(anchor_points)
        for idx, anchor in enumerate(anchor_points, 1):
            print(f"  Planning anchor {idx}/{total_anchors}: {anchor.get('anchor_type', 'unknown')}")

            # Initial scenes per anchor logic...
            scenes_per_anchor = anchor.get("estimated_scenes", 1)
            words_per_scene = int(total_chapter_words / (scenes_per_anchor * total_anchors))
            while words_per_scene > 800:
                scenes_per_anchor += 1
                words_per_scene = int(total_chapter_words / (scenes_per_anchor * total_anchors))
            
            system_prompt = f"""You are the Director converting story anchors into executable scene directives.

CORE CONTEXT:
Conflict: {self.story_context.central_conflict}
Themes: {', '.join(self.story_context.active_themes.keys())}
World Rules: {' | '.join(self.story_context.world_rules)}

CHARACTER DRIVES:
{chr(10).join(f"• {name}: {drive}" for name, drive in self.story_context.character_core_drives.items())}

RELATIONSHIP TENSIONS:
{chr(10).join(f"• {name} vs {other}: {tension}" 
              for name, rels in self.story_context.relationship_web.items()
              for other, tension in rels.items())}

RULES:
1. One dramatic function per scene (no repetition in chapter)
2. Show internal states via action/dialogue only
3. Protagonist must take deliberate, change-creating actions
4. Every scene has irreversible consequences
5. Varied dramatic functions across scenes
6. Tone fits genres: {', '.join(self.genre_list)}
7. Dialogue is always a power struggle

Scene directives must include:
- agents_in_scene: with objective (current want) + current_status (emotional state)
- story_events: concrete, motivation-honoring actions
- state_changes: psychology/relationship/world shifts
- context: include thematic_beat (theme explored)

Target chapter words: {total_chapter_words}
Full context: {ContextFormatter.full_context(self.story_context)}

{DIRECTOR_JSON_INSTRUCTIONS}
"""


            human_prompt = f"""Convert this anchor into {scenes_per_anchor} concrete, executable scene(s).

CHAPTER ANCHOR POINTS (context only):
{json.dumps(anchor_points, indent=2)}

════════════════════════════════
ANCHOR TO DRAMATIZE
════════════════════════════════
{json.dumps(anchor, indent=2)}

════════════════════════════════
SPECIFICATIONS
════════════════════════════════

Target per scene: {words_per_scene} words (±15%)
Adjust content based on this target.

Story events:
- 3–5 specific, observable actions/dialogue lines
- Convert thoughts/emotions to body language, objects, environment, or dialogue
- Order for maximum dramatic impact

Context fields:
- Mandatory: location, atmosphere, pacing, emotional_arc, thematic_beat, world_rules_active
- Add genre-relevant fields (e.g., clues, magic_effects, combat_details, romantic_tension)

Agents_in_scene (all mandatory):
- role, gender, objective (now), current_status, distinctive_voice, core_motivation, relationships (tensions in scene)
- Optional: hidden_agenda, physical_tells, secrets_kept, etc.

State changes:
- Concrete facts learned
- Observable relationship shifts
- Physical/world changes

Weave in backstories/subplots from anchor context if relevant and characters are present.

Elevate the material:
- Add pressure/obstacles if flat
- Add memorable specific details if generic
- Build breathing room if rushed
- Choose surprising-yet-logical actions if safe

Ensure a writer could execute the scene exactly from your details.

Text-based novel (no screenplay format).

Output: Pure JSON only. No markdown, explanations, or fences.
"""
            try:
                resp, tokens = await director_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=self.llm_temp
                )
                self.director_token_usage = StoryHelpers._add_tokens_to_total(self.director_token_usage, tokens)

                clean_resp = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(clean_resp).strip()

                single_output, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=DirectorOutput
                )
                if isinstance(single_output, tuple):
                    single_output = single_output[0]
                self.utility_token_usage = StoryHelpers._add_tokens_to_total(self.utility_token_usage, utility_tokens)

                rich_scenes = [s.model_dump() for s in single_output.scenes]
                for scene_dict in rich_scenes:
                    scene_dict.update({
                        "scene_id": scene_id_counter,
                        "chapter_id": state.current_chapter_id,
                        "act_id": state.current_act_id,
                        "target_word_count": words_per_scene
                    })
                    all_rich_directives.append(scene_dict)
                    scene_id_counter += 1

            except Exception as e:
                print(f"    Anchor {idx} failed: {e}")
                state.error_message = f"Iterative planning failed on anchor {idx}: {e}"
                state.next_action = "ERROR"
                return state.__dict__

        # ====================================================================
        # FINALIZE
        # ====================================================================
        
        # Persist results
        await self.memory.update_story_progress({
            "director_token_usage": self.director_token_usage,
            "utility_token_usage": self.utility_token_usage
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

        print(f"Chapter {state.current_chapter_id}: {len(all_rich_directives)} directives ready")
        state.next_action = "quality_validate_node"
        return state.__dict__
        
    async def quality_validate_node(self, state: StoryState) -> Dict:
        """
        Validate entire chapter scene plan against act plan and story tracker.
        Only runs once per chapter (scene_id == 1).
        Checks for contradictions, genre fidelity, and act plan alignment.
        """
        
        # Only validate at chapter start
        if state.scene_id != 1:
            print(f"Scene {state.scene_id} - skipping validation (already done for chapter)")
            state.next_action = "generate_scenes"
            return state.__dict__
        
        print(f"Quality Validation: Chapter {state.current_chapter_id}, Act {state.current_act_id}")
        print("=" * 60)
        
        # Load scene directives
        scenes_json = await self.memory.get_long_term_document(
            metadata={
                "type": "director_scene_directives",
                "act_id": state.current_act_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        
        if not scenes_json:
            print("No scene directives found - skipping validation")
            state.next_action = "ERROR"
            return state.__dict__
        
        director_directives = json.loads(scenes_json)
        
        # Load act plan
        act_plan_json = await self.memory.get_long_term_document(
            metadata={
                'type': 'act_plan',
                'act_id': state.current_act_id,
                'story_title': state.story_title
            }
        )
        act_plan = json.loads(act_plan_json)
        
        # Load story tracker
        tracker_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_tracker', 'story_title': state.story_title}
        )
        tracker = json.loads(tracker_json) if tracker_json else {'act_tracking': {}}
        
        # Load seed for genre/tone
        seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': state.story_title}
        )
        seed = json.loads(seed_json)
        
        # Load world rules
        world_json = await self.memory.get_long_term_document(
            metadata={'type': 'integrated_world', 'story_title': state.story_title}
        )
        world = json.loads(world_json)
        
        # Load conflict matrix
        conflict_json = await self.memory.get_long_term_document(
            metadata={'type': 'conflict_matrix', 'story_title': state.story_title}
        )
        conflict_matrix = json.loads(conflict_json)
        
        # Get current chapter outline from act plan
        current_chapter_outline = None
        for chapter in act_plan.get("chapter_outlines", []):
            if chapter.get("chapter_number") == state.current_chapter_id:
                current_chapter_outline = chapter
                break
        
        if not current_chapter_outline:
            print("Chapter outline not found in act plan - skipping validation")
            state.next_action = "ingest_chapter"
            return state.__dict__
        
        # Build validation context
        print("Building validation context...")
        
        # Get story so far for continuity check
        story_so_far = await self.memory.get_director_context(
            current_act_number=state.current_act_id,
            current_chapter_number=state.current_chapter_id
        )
        
        # Extract current act tracking
        current_act_tracking = tracker.get('act_tracking', {}).get(str(state.current_act_id), {})
        
        system_prompt = f"""You are a Story Consistency Validator checking chapter scene plans for contradictions and fidelity.

YOUR TASK: Validate the Director's scene plan against the Act Plan and Story Tracker.

VALIDATION CRITERIA:

1. **ACT PLAN FIDELITY**
   - Does the scene plan honor ALL anchor points from the chapter outline?
   - Are anchor story beats present in the scene events?
   - Do scenes follow the chapter's creative seed/purpose?

2. **STORY TRACKER CONSISTENCY**
   - Setup elements: Are previously introduced elements respected?
   - Payoff elements: If this act should resolve something, do scenes address it?
   - Character moments: Do key agent moments from tracker appear?

3. **GENRE & TONE ADHERENCE**
   - Genre: {', '.join(seed.get('genre', []))}
   - Tone: {seed.get('tone', 'Balanced')}
   - Are scenes consistent with genre conventions?
   - Does emotional tone match seed specifications?

4. **WORLD RULES CONSISTENCY**
   - Do scene events violate established world rules?
   - Are magic/tech systems used correctly?
   - Do locations match previous descriptions?

5. **CHARACTER CONSISTENCY**
   - Do character actions align with their established motivations?
   - Are relationship dynamics consistent with previous chapters?
   - Do character states (emotional/physical) make logical sense?

6. **CONFLICT CONSISTENCY**
   - Does the scene plan advance the central conflict?
   - Are conflicts escalating appropriately for Act {state.current_act_id}?
   - Do obstacles feel organic to established story?

7. **CONTINUITY**
   - Do scenes logically follow from previous chapter events?
   - Are there unexplained jumps in time/location/character state?
   - Do events have proper cause-effect relationships?


If validation passes:
- Return directives EXACTLY as provided (no changes)

If issues found:
- Fix ONLY the problematic elements
- Preserve as much of original plan as possible


CRITICAL RULES:
- Return COMPLETE list of ALL scenes (not just changed ones)
- Maintain original scene IDs, chapter IDs, act IDs
- Keep target_word_count unchanged unless it violates chapter budget
- Only modify story_events, context, state_changes, or notes fields if needed
- Be surgical - minimal changes only - But output complete plan regardless.

Age-appropriate for: {seed.get('target_audience_age', 13)}+ audience


Output clean JSON array with no markdown fences.
FORMAT:
{DIRECTOR_JSON_INSTRUCTIONS}"""

        human_prompt = f"""Validate this chapter scenes plan.

═══════════════════════════════════════════════════════════════════════════════
CHAPTER OUTLINE (from Act Plan)
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(current_chapter_outline, indent=2)}

═══════════════════════════════════════════════════════════════════════════════
ACT TRACKING (Setup/Payoff Requirements)
═══════════════════════════════════════════════════════════════════════════════

Setup this act (must introduce):
{json.dumps(current_act_tracking.get('setup_elements', []), indent=2)}

Payoff this act (must resolve):
{json.dumps(current_act_tracking.get('payoff_elements', []), indent=2)}

Key agent moments this act:
{json.dumps(current_act_tracking.get('key_agent_moments', {}), indent=2)}

═══════════════════════════════════════════════════════════════════════════════
WORLD RULES (Cannot be violated)
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(world.get('world_rules', []), indent=2)}

Foundation: {world.get('foundation', 'N/A')}

═══════════════════════════════════════════════════════════════════════════════
CENTRAL CONFLICT
═══════════════════════════════════════════════════════════════════════════════

{conflict_matrix.get('central_conflict', 'N/A')}

Escalation path: {conflict_matrix.get('escalation_path', 'N/A')}

═══════════════════════════════════════════════════════════════════════════════
STORY SO FAR (for continuity)
═══════════════════════════════════════════════════════════════════════════════

{story_so_far[:1000] + '...' if story_so_far and len(story_so_far) > 1000 else story_so_far or 'Start of story'}

═══════════════════════════════════════════════════════════════════════════════
DIRECTOR'S SCENE PLAN (TO VALIDATE)
═══════════════════════════════════════════════════════════════════════════════

{json.dumps(director_directives, indent=2)}

═══════════════════════════════════════════════════════════════════════════════

Validate this scene plan and return the complete corrected version (or original if no issues)."""

        try:
            print("Running validation LLM check...")
        
            response, tokens = await director_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.6
            )
        
            self.director_token_usage = StoryHelpers._add_tokens_to_total(
                self.director_token_usage, tokens
            )
        
            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp).strip()
        
            # Parse into Pydantic model
            parsed_output, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=DirectorOutput
            )
        
            # Handle case where load_json_with_retry returns (model, tokens) tuple
            if isinstance(parsed_output, tuple):
                validated_output = parsed_output[0]
            else:
                validated_output = parsed_output
        
            self.utility_token_usage = StoryHelpers._add_tokens_to_total(
                self.utility_token_usage, utility_tokens
            )
        
            # CRITICAL: Convert to list of plain dicts for JSON comparison and storage
            validated_directives_list = [scene.model_dump() for scene in validated_output.scenes]
        
            # Now compare using JSON strings (both sides are list[dict])
            original_directives_list = director_directives  # This is already list[dict] from json.loads()
        
            changes_made = (
                json.dumps(validated_directives_list, sort_keys=True) !=
                json.dumps(original_directives_list, sort_keys=True)
            )
        
            if len(validated_directives_list) != len(original_directives_list):
                print(f"Scene count mismatch: {len(validated_directives_list)} vs {len(original_directives_list)}")
                print("Falling back to original directives to prevent data loss")
                validated_directives_list = original_directives_list
                changes_made = False
        
            if changes_made:
                print("Validation made changes — saving updated directives")
                # Save the corrected version (as list of dicts)
                await self.memory.add_long_term_document(
                    text=json.dumps(validated_directives_list, indent=2),
                    metadata={
                        "type": "director_scene_directives",
                        "act_id": state.current_act_id,
                        "chapter_id": state.current_chapter_id,
                        "story_title": state.story_title
                    }
                )
            else:
                print("No issues found — proceeding with original directives")
        
        except Exception as e:
            print(f"Validation failed with error: {e}")
            print("Continuing with original directives")
            traceback.print_exc()
            # On any error, fall back safely
            validated_directives_list = director_directives
        
        print("=" * 60)
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
            # print(directive)
            directive_text = ContextFormatter.format_any(directive)
            # print("DIRECTIVE TEXT: ",  directive_text)

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
                writer_token_usage=self.writer_token_usage,
                utility_token_usage=self.utility_token_usage,
                llm_temp=self.llm_temp,
                scene_target_length=scene_target_length,
                user_age=self.min_age,
                pov=self.pov,
                voice=self.voice,
                tone=self.tone,
                tense=self.tense,
                genre=self.genre_list
            )
            
            scene_text, scene_cluster, status, writer_tokens, utility_tokens = await scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE.run_scene(
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
            self.utility_token_usage = utility_tokens

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
            scene_bundle, ingestor_tokens, utility_tokens = await Ingestor.ingest_scene(
                act_id=state.current_act_id,
                chapter_id=state.current_chapter_id,
                scene_id=state.scene_id,
                scene_text=scene_text,
                known_characters=chars,
                known_locations=worlds
            )

            self.ingestor_token_usage = StoryHelpers._add_tokens_to_total(self.ingestor_token_usage, ingestor_tokens)
            self.utility_token_usage = StoryHelpers._add_tokens_to_total(self.utility_token_usage, utility_tokens)

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
                "writer_token_usage": self.writer_token_usage,
                "utility_token_usage": self.utility_token_usage,
                "ingestor_token_usage": self.ingestor_token_usage
            })

            state.scene_id += 1
            self.scene_chunk_callback({"type": "saved", "message": "story saved"})

            gc.collect()

        state.next_action = "ingest_chapter"
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
        chapter_bundle, ingestor_tokens, utility_tokens = await Ingestor.ingest_chapter(
            act_id=state.current_act_id,
            chapter_id=state.current_chapter_id,
            chapter_text=current_chap_summary,
            current_characters=char_details,
            current_locations=world_details
        )

        self.ingestor_token_usage = StoryHelpers._add_tokens_to_total(self.ingestor_token_usage, ingestor_tokens)
        self.utility_token_usage = StoryHelpers._add_tokens_to_total(self.utility_token_usage, utility_tokens)

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
            await self.memory.update_story_progress({
                "chapter_word_count": 0, 
                "ingestor_token_usage": self.ingestor_token_usage, 
                "utility_token_usage": self.utility_token_usage
                })
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
            self.min_age = progress.get("min_age", 13)
            self.target_length = progress.get("target_length", 50000)
            self.director_token_usage = StoryHelpers._parse_tokens(progress.get("director_token_usage"))
            self.writer_token_usage = StoryHelpers._parse_tokens(progress.get("writer_token_usage"))
            self.ingestor_token_usage = StoryHelpers._parse_tokens(progress.get("ingestor_token_usage"))
            self.utility_token_usage = StoryHelpers._parse_tokens(progress.get("utility_token_usage"))

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

