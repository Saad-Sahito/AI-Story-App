
import json
from typing import Any, Dict, List, Optional, Tuple, Literal
# from openai import max_retries
from pydantic import BaseModel, Field, ConfigDict
# from langchain_core.output_parsers import PydanticOutputParser
from src.memory.memory_system import StoryMemorySystem
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_structure_decider import find_best_structure, find_best_structure_name
from src.utilities.story_helpers import StoryHelpers
from config_vars import author_story_rules_negative
import asyncio
# import random
import hashlib


# ============================================================================
# FLEXIBLE BASE MODEL - Allows arbitrary additional fields
# ============================================================================

class FlexibleBase(BaseModel):
    """Base class allowing LLM to add arbitrary creative fields"""
    model_config = ConfigDict(extra="allow")


# ============================================================================
# MINIMAL PYDANTIC MODELS - Essential fields only
# ============================================================================

class MinimalStorySeed(FlexibleBase):
    """Minimal story foundation - LLM can extend based on genre needs"""
    title: Optional[str] = None
    pov: str
    tone: str
    genre: List[str]
    sub_genre: List[str]
    setting: Optional[str] = None
    prose_style: str
    themes: List[str]
    target_audience_age: int
    target_length: int
    protagonist_specs: Dict[str, Any] = Field(default_factory=dict)
    act_count: int
    story_structure: str


class WorldFoundation(FlexibleBase):
    """Minimal world building - LLM adds genre-appropriate details"""
    core_concept: str = Field(
        description="1-3 sentence essence of this world"
    )
    # LLM adds dynamically: magic_system, tech_level, social_structure, 
    # geography, culture, threats, etc. based on genre


class NarrativeAgent(FlexibleBase):
    """Minimal agent definition - LLM expands based on role importance"""
    name: str
    role: str = Field(description="protagonist, antagonist, mentor, ally, etc.")
    essence: str = Field(description="1-2 sentence core identity, gender etc")
    # LLM adds: background, psychology, relationships, quirks, appearance, etc.


class NarrativeAgentList(FlexibleBase):
    """Wrapper for agent list"""
    agents: List[NarrativeAgent]


class ConflictMatrix(FlexibleBase):
    """Minimal conflict structure - LLM adds layers as genre demands"""
    central_conflict: str = Field(description="Primary plot-driving tension")
    escalation_path: str = Field(description="How conflicts intensify")
    # LLM adds: character_conflicts, systemic_conflicts, thematic_conflicts,
    # moral_dilemmas, twist_opportunities, etc.


class MinimalPlotOutline(FlexibleBase):
    """Initial plot conception - flexible structure"""
    title: str
    premise: str = Field(description="1-2 sentence story hook")
    central_question: str = Field(description="What question drives this story?")
    target_length: int
    narrative_arc: str = Field(description="High-level story progression")
    required_character_roles: List[str] = Field(
        description="Roles needed: protagonist, antagonist, mentor, etc. MUST include ALL roles."
    )



class ExpandedPlotOutline(FlexibleBase):
    """Detailed plot - LLM adds richness appropriate to length/genre"""
    title: str
    premise: str
    act_summaries: List[str] = Field(description="Rich prose summaries per act FOR EACH ACT NOW")
    narrative_flow: str = Field(description="Complete plot in flowing prose")
    # LLM adds: subplot_threads, relationship_arcs, symbolic_layers,
    # thematic_development, structure_beats, etc.

class AgeAppropriatenessReport(BaseModel):
    age_rating: Literal["A", "T", "M"] = Field(
        description="'A' for ages 8+, 'T' for ages 13+, 'M' for ages 18+"
    )
    plot_outline_pass: bool = Field(description="Does the plot pass (True) or fail (False) for age appropriateness")



class ConnectedNarrativeAgent(FlexibleBase):
    """Agent after plot integration - flexible depth"""
    name: str
    role: str
    essence: str
    plot_function: str = Field(description="How this agent drives the plot")
    agent_arc: str = Field(description="How this agent changes through story")
    # LLM adds: relationships, distinctive_voice, key_moments, etc.


class ConnectedNarrativeAgentList(FlexibleBase):
    agents: List[ConnectedNarrativeAgent]


class IntegratedWorld(FlexibleBase):
    """World after plot integration"""
    foundation: str = Field(description="Core world elements")
    plot_integration: str = Field(
        description="How world rules enable/complicate conflicts"
    )
    # LLM adds: key_locations, thematic_resonance, world_arc, etc.


class QualityReport(FlexibleBase):
    """Story quality validation"""
    strengths: List[str]
    concerns: List[str]
    recommendations: List[str]
    overall_assessment: str


class ActTracking(FlexibleBase):
    """What gets introduced and resolved each act"""
    act_number: int
    setup_elements: List[str] = Field(
        description="Things introduced this act for later use"
    )
    payoff_elements: List[str] = Field(
        description="Things resolved from earlier acts"
    )
    key_agent_moments: Dict[str, str] = Field(
        description="Critical moments for each agent this act"
    )


class StoryTracker(FlexibleBase):
    """Lightweight story-level consistency tracking"""
    act_tracking: Dict[int, ActTracking]

# ============================================================================
# KEEP ACT PLAN STRUCTURE INTACT (used by Director)
# ============================================================================

class ChapterAnchorPoints(FlexibleBase):
    """Major story moments - FLEXIBLE BEAT STRUCTURE for creative flow"""
    anchor_number: int
    anchor_type: str = Field(
        description="Flexible type: revelation | confrontation | decision | loss | discovery | betrayal | transformation | twist | alliance | sacrifice | epiphany | or custom type for unique beats"
    )
    
    what_happens: str = Field(
        description="Vivid 2-4 sentence description capturing the essence, emotion, and key actions of this story moment - allow for nuance and subtext"
    )
    
    location: str = Field(
        description="Evocative location description, including atmosphere and sensory details to inspire immersion"
    )
    
    time_context: Optional[str] = Field(
        default="",
        description="Temporal placement relative to previous anchor, including mood of time passage (e.g., 'dawn after a sleepless night' or 'weeks later in mounting tension')"
    )
    
    character_positions: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Initial emotional/physical states and positions of key characters - focus on motivations and internal conflicts"
    )
    
    transition_from_previous: Optional[str] = Field(
        default="",
        description="Narrative bridge from last anchor, emphasizing character growth, foreshadowing, or thematic links"
    )
    
    story_function: Optional[str] = Field(
        default="",
        description="Narrative purpose: how this beat advances plot, deepens characters, explores themes, or builds world - encourage layered impacts"
    )
    
    agents_present: List[str] = Field(description="Characters or entities central to this beat, including potential for unexpected participants")
    
    emotional_target: Optional[str] = Field(
        default="",
        description="Intended emotional resonance for readers - aim for complex, multifaceted feelings (e.g., bittersweet triumph, creeping dread)"
    )
    
    estimated_scenes: int = Field(
        default=1,
        description="Flexible scene count to dramatize this anchor (1-5, allowing for epic sequences or intimate moments)"
    )
    
    scene_breakdown: List[str] = Field(
        default_factory=list,
        description="If estimated_scenes > 1, evocative summaries of each scene's focus, tone, and key turning points"
    )
    
    introduces: Optional[List[str]] = Field(
        default_factory=list,
        description="New elements: tensions, mysteries, relationships, world lore, or thematic questions to spark intrigue"
    )
    
    resolves: Optional[List[str]] = Field(
        default_factory=list,
        description="Payoffs: resolved arcs, answered questions, or evolved dynamics from prior setup"
    )
    
    required_character_states: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Pre-anchor character mindsets, relationships, or conditions - keep flexible for organic development"
    )
    
    resulting_character_states: Optional[Dict[str, str]] = Field(
        default_factory=dict,
        description="Post-anchor evolutions: growth, setbacks, revelations, or shifts in alliances and worldview"
    )


class ChapterOutline(FlexibleBase):
    chapter_number: int
    anchor_points: List[ChapterAnchorPoints]
    target_word_count: int
    creative_seed: str = Field(
        description="2-4 sentence inspirational hook for the chapter, blending themes, tone, and unique flair"
    )


class ActPlan(FlexibleBase):
    """Complete act plan - FLEXIBLE STRUCTURE for Director to adapt creatively"""
    act_number: int
    act_title: str
    act_purpose: str
    
    chapter_outlines: List[ChapterOutline]
    
    emotional_progression: Optional[str] = Field(
        default=""
    )
    structure_alignment: Optional[str] = Field( 
        default=""
    )
    act_climax: str
    
    setup_this_act: List[str] = Field(default_factory=list)
    payoff_this_act: List[str] = Field(default_factory=list)
    agent_state_changes: Dict[str, str] = Field(default_factory=dict)
    thematic_weave: str = Field(
        default="",
        description="How themes interlace through the act for deeper resonance"
    )
    world_building_elements: List[str] = Field(
        default_factory=list,
        description="Opportunities for immersive world expansion or lore reveals"
    )

# ============================================================================
# COMPONENT CLASSES
# ============================================================================

class WorldBuilder:
    """Generates independent world foundations with creative flexibility"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_world_foundation(
        self,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[WorldFoundation, dict]:
        """Generate world with genre-appropriate flexibility"""
        
        system_prompt = f"""You are a master world-builder creating immersive, internally consistent worlds for a novel.

Build a world for: {', '.join(seed.genre)}
Tone: {seed.tone}
Setting hint: {seed.setting or 'Create something fitting'}
Themes: {', '.join(seed.themes)}

// ADD FIELDS BASED ON GENRE NEEDS:
// High Fantasy: "magic_system", "ancient_history", "mystical_creatures", "world_threats"
// Sci-Fi: "tech_level", "alien_species", "space_politics", "scientific_laws"
// Historical: "time_period", "historical_tensions", "social_hierarchy", "cultural_details"
// Mystery/Thriller: "urban_environment", "power_structures", "hidden_dangers", "investigative_landscape"
// Romance: "social_world", "meeting_places", "relationship_obstacles", "community_dynamics"
// Horror: "atmospheric_dread", "supernatural_rules", "isolation_factors", "threat_nature"
// Literary: "symbolic_geography", "emotional_landscape", "thematic_spaces"

// Add sensory details, cultural elements, systemic rules, tensions, unique features
// Be creative! Include whatever makes THIS world vivid and functional for THIS story

OUTPUT FORMAT (JSON):
{WorldFoundation.model_json_schema()}
Age-appropriate for: {seed.target_audience_age} year old audience

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Create a rich world foundation for this story.

Seed details:
{seed.model_dump_json(indent=2)}

Build a world that could support many stories, with inherent conflicts and memorable details.
Add genre-appropriate fields that bring this world to life."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=WorldFoundation
        )

        return json_data, tokens
    
    async def integrate_with_conflict(
        self,
        world_foundation: WorldFoundation,
        conflict_matrix: ConflictMatrix,
        agents: List[ConnectedNarrativeAgent],
        model: str = "None"
    ) -> Tuple[IntegratedWorld, dict]:
        """Show how world intersects with story conflicts"""
        
        system_prompt = f"""You have an established world. Now show how it intersects with specific story conflicts.

// ADD RELEVANT FIELDS:
// "key_locations": specific places where events occur
// "thematic_resonance": how world mirrors story themes  
// "world_arc": how world changes due to story events
// "symbolic_elements": world features that carry meaning
// "obstacle_sources": how world creates challenges

// Add whatever fields show rich world-plot connection

OUTPUT FORMAT (JSON):
{IntegratedWorld.model_json_schema()}

Don't change the world - find organic connections with the story.
Output clean JSON with no markdown fences."""
        
        human_prompt = f"""World Foundation:
{world_foundation.model_dump_json(indent=2)}

Story Conflicts:
{conflict_matrix.model_dump_json(indent=2)}

Agents:
{json.dumps([a.model_dump() for a in agents], indent=2, ensure_ascii=False)}

Show how this world naturally enables and complicates these conflicts."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=IntegratedWorld
        )

        return json_data, tokens


class AgentGenesis:
    """Generates independent narrative agent foundations with flexible depth"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def generate_narrative_agents(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        required_roles: List[str],
        model: str = "None"
    ) -> Tuple[List[NarrativeAgent], dict]:
        """Generate agents with genre-appropriate detail"""
        
        agent_count = len(required_roles)
        
        system_prompt = f"""You are designing narrative agents for a {'/'.join(seed.genre)} novel.

YOU MUST OUTPUT EXACTLY {agent_count} AGENTS IN A SINGLE RESPONSE.
DO NOT STOP AFTER A FEW - OUTPUT ALL {agent_count} AGENTS.

{author_story_rules_negative}

OUTPUT FORMAT (JSON):
{NarrativeAgentList.model_json_schema()}

// ADD FIELDS BASED ON ROLE IMPORTANCE:
// Major characters: "background", "psychology", "desires", "fears", 
//                   "relationships", "arc_potential", "distinctive_traits",
//                   "voice_style", "contradictions", "secrets"
// Minor characters: "function", "key_trait", "relationship_to_protagonist"
// Forces/concepts: "manifestation", "influence", "symbolic_meaning"

// Romance genre: "emotional_baggage", "love_language", "intimacy_fears"
// Thriller genre: "hidden_agenda", "pressure_points", "betrayal_capacity"
// Fantasy genre: "magical_ability", "destiny_connection", "ancient_ties"

// Be creative! Add what makes each agent vivid and functional

Create one agent per required role. MUST Include their gender. Most should be individual characters unless the role explicitly describes a group/system.
Use protagonist specs if provided.
Age-appropriate for: {seed.target_audience_age} year old audience.

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Generate {agent_count} narrative agents for these roles:
{json.dumps(required_roles, indent=2)}

Seed:
{seed.model_dump_json(indent=2)}

World Context:
{world_foundation.model_dump_json(indent=2)}

Protagonist specs (if any):
{json.dumps(seed.protagonist_specs, indent=2)}

Create agents with independent desires and fears shaped by their world.
Add detail appropriate to each role's importance."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        agents_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=NarrativeAgentList
        )

        if hasattr(agents_json, "agents"):
            agents = agents_json.agents
        elif isinstance(agents_json, dict):
            if "agents" in agents_json:
                raw_list = agents_json["agents"]
            else:
                raw_list = next((v for v in agents_json.values() if isinstance(v, list)), None)
                if raw_list is None:
                    raise ValueError("Could not find agent list in response")
            
            agents = [NarrativeAgent(**item) if isinstance(item, dict) else item for item in raw_list]
        else:
            raise ValueError("Unexpected parsed type from LLM")

        if not agents or not all(isinstance(a, NarrativeAgent) for a in agents):
            raise ValueError("Failed to produce valid NarrativeAgent instances")

        print(f"Successfully created {len(agents)} narrative agents")
        return agents, tokens
    
    async def connect_agents_to_plot(
        self,
        narrative_agents: List[NarrativeAgent],
        conflict_matrix: ConflictMatrix,
        plot_outline: MinimalPlotOutline,
        prose_style: str,
        model: str = "None"
    ) -> Tuple[List[ConnectedNarrativeAgent], dict]:
        """Connect agents to plot with flexible integration"""
        
        system_prompt = f"""You're connecting fully-formed agents to a specific plot for a novel.

OUTPUT FORMAT (JSON):
{ConnectedNarrativeAgentList.model_json_schema()}
      
// ADD AS NEEDED:
// "key_relationships": dynamics with other agents
// "distinctive_voice": speech patterns if applicable
// "turning_points": critical moments in their arc
// "internal_conflict": personal struggles
// "thematic_role": what they represent

// Add fields that show how THIS agent intersects with THIS plot


Find ORGANIC connections between agents' existing psychology and the plot.
Don't change their core nature - show how their desires naturally engage with conflicts.
MUST Include their gender. 

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Connect these agents to the story:

Agents:
{json.dumps([a.model_dump() for a in narrative_agents], indent=2)}

Plot:
{plot_outline.model_dump_json(indent=2)}

Conflicts:
{conflict_matrix.model_dump_json(indent=2)}

Show how each agent's existing desires/fears naturally intersect with the plot."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        agents_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ConnectedNarrativeAgentList
        )

        if hasattr(agents_json, "agents"):
            connected = agents_json.agents
        elif isinstance(agents_json, dict):
            raw = agents_json.get("agents") or next((v for v in agents_json.values() if isinstance(v, list)), None)
            connected = [ConnectedNarrativeAgent(**item) if isinstance(item, dict) else item for item in raw]
        else:
            raise ValueError("Failed to parse connected agents")

        return connected, tokens


class ConflictArchitect:
    """Generates multi-dimensional conflict with genre-appropriate complexity"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _scale_conflict_complexity(self, target_length: int, base_conflicts: ConflictMatrix) -> ConflictMatrix:
        """Simplify conflict layers for shorter stories"""
        if target_length >= 40000:
            return base_conflicts
        
        # For short stories, the LLM will naturally add fewer fields
        # No need to artificially simplify since we're using FlexibleBase
        return base_conflicts
    
    async def generate_conflict_layers(
        self,
        plot_outline: MinimalPlotOutline,
        seed: MinimalStorySeed,
        narrative_agents: List[NarrativeAgent],
        world_foundation: WorldFoundation,
        model: str = "None"
    ) -> Tuple[ConflictMatrix, dict]:
        """Generate conflict analysis with flexible depth"""
        
        complexity_guidance = ""
        if seed.target_length < 15000:
            complexity_guidance = """
SHORT STORY MODE: Focus primarily on central conflict.
Keep layers minimal - one clear tension is enough."""
        elif seed.target_length < 40000:
            complexity_guidance = """
NOVELLA MODE: Central conflict + 1-2 character/thematic layers."""
        else:
            complexity_guidance = """
NOVEL MODE: Multiple interwoven conflict layers encouraged."""
        
        system_prompt = f"""You are analyzing conflicts for a {'/'.join(seed.genre)} story (novel).

{complexity_guidance}

OUTPUT FORMAT (JSON):
{ConflictMatrix.model_json_schema()}
  
// ADD LAYERS APPROPRIATE TO LENGTH/GENRE:
// "character_conflicts": personal desires clashing
// "systemic_conflicts": world structures creating obstacles  
// "thematic_conflicts": philosophical/moral questions
// "relationship_tensions": interpersonal dynamics
// "internal_struggles": protagonist's psychology
// "moral_dilemma": core ideological tension (both sides valid)
// "twist_opportunities": where conflicts could reverse/surprise
// "subplot_conflicts": secondary tension threads

// Mystery: "suspect_conflicts", "clue_tensions"
// Romance: "emotional_barriers", "misunderstanding_sources"
// Thriller: "escalating_threats", "trust_breakdowns"

// Add fields that create rich, layered conflict for THIS story


Make conflicts feel inevitable yet surprising.
Age-appropriate for: {seed.target_audience_age} years old.

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Build conflict matrix for this story:

Themes: {', '.join(seed.themes)}
Tone: {seed.tone}

Plot Foundation:
{plot_outline.model_dump_json(indent=2)}

Agents:
{json.dumps([a.model_dump() for a in narrative_agents], indent=2)}

World:
{world_foundation.model_dump_json(indent=2)}

Create conflicts that emerge naturally from agents and world."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ConflictMatrix
        )
        
        scaled_matrix = self._scale_conflict_complexity(
            target_length=seed.target_length,
            base_conflicts=json_data
        )

        return scaled_matrix, tokens


class QualityController:
    """Validates story quality with focus on specificity"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def validate_story_elements(
        self,
        seed: MinimalStorySeed,
        plot: ExpandedPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        model: str = "None"
    ) -> Tuple[QualityReport, dict]:
        """Comprehensive quality assessment"""
        
        system_prompt = f"""You are a story development editor evaluating narrative quality for a novel.

CRITICAL: Check for VAGUENESS and LACK OF SPECIFICITY.

OUTPUT FORMAT (JSON):
{{
  "strengths": ["What's working well"],
  "concerns": ["Issues that need attention - BE SPECIFIC"],
  "recommendations": ["ACTIONABLE improvements with examples"],
  "overall_assessment": "Holistic view"
}}

ANALYZE FOR:

1. SPECIFICITY (MOST IMPORTANT)
   ✓ Character names used (not "the hero")
   ✓ Locations specific (not "somewhere", "the village")
   ✓ Events concrete (not "faces challenges")
   ✓ Act summaries match the total act count of {seed.act_count}, if less suggest spreading content appropriately.
   
2. PLOT COHERENCE
   ✓ Logical flow, clear cause/effect
   
3. AGENT AGENCY
   ✓ Characters drive plot with clear motivations
   
4. CONFLICT ESCALATION
   ✓ Stakes rise appropriately
   
5. THEMATIC INTEGRATION
   ✓ Themes manifest in concrete events
   
6. WORLD CONSISTENCY
   ✓ World follows its own rules
   
7. EMOTIONAL VARIETY
   ✓ Balance of tones and beats
   
8. ORIGINALITY
   ✓ Avoids clichés, has unique details

9. COMPLETENESS
    ✓ If the plot feels incomplete or is cutoff abruptly, note that specifically. Suggest ways to fill gaps.


Be brutally honest. Recommendations must be ACTIONABLE.

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Evaluate this story's quality:

Seed: {seed.model_dump_json(indent=2)}
Plot: {plot.model_dump_json(indent=2)}
Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
World: {world.model_dump_json(indent=2)}
Conflicts: {conflict_matrix.model_dump_json(indent=2)}

Focus on: Is the plot SPECIFIC enough for execution?"""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.7,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=QualityReport
        )
        
        return json_data, tokens


class StoryTrackerManager:
    """Manages incremental story tracking"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def update_tracker_after_act(
        self,
        story_title: str,
        act_number: int,
        act_plan: ActPlan,
        author_context: str,
        model: str = "None"
    ) -> Tuple[StoryTracker, dict]:
        """Update story tracker after act is planned"""
        
        system_prompt = f"""Update story continuity tracker for Act {act_number}.

Extract what was INTRODUCED (setup) and RESOLVED (payoff).

OUTPUT FORMAT (JSON):
{{
  "act_number": {act_number},
  "setup_elements": [
    "Specific new tensions/questions introduced",
    "Objects that will matter later",
    "Relationship changes needing resolution"
  ],
  "payoff_elements": [
    "Questions answered from earlier acts",
    "Conflicts resolved from setup",
    "Mysteries solved"
  ],
  "key_agent_moments": {{
    "Agent Name": "Their critical transformation this act",
    "Another Agent": "Their pivotal moment"
  }}
}}

Be concrete and specific. Only track elements that MATTER to ongoing story.
Only include agents with MAJOR moments.

Output clean JSON with no markdown fences."""

        human_prompt = f"""Extract tracking for Act {act_number}.

Full Story Context:
{author_context}

Act {act_number} Plan:
{act_plan.model_dump_json(indent=2)}

Extract setup/payoff tracking."""

        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        # act_tracking_parser = PydanticOutputParser(pydantic_object=ActTracking)
        act_tracking_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ActTracking
        )
        
        if isinstance(act_tracking_json, dict):
            act_tracking = ActTracking(**act_tracking_json)
        else:
            act_tracking = act_tracking_json
        
        return act_tracking, tokens
    
    async def get_or_create_tracker(
        self,
        memory_system: StoryMemorySystem,
        story_title: str
    ) -> Dict[int, ActTracking]:
        """Load existing tracker or create empty one"""
        
        try:
            tracker_json = await memory_system.get_long_term_document(
                metadata={'type': 'story_tracker', 'story_title': story_title}
            )
            
            if tracker_json:
                tracker_data = json.loads(tracker_json)
                
                tracker = {}
                for act_num_str, act_data in tracker_data.get('act_tracking', {}).items():
                    act_num = int(act_num_str)
                    if isinstance(act_data, dict):
                        tracker[act_num] = ActTracking(**act_data)
                    else:
                        tracker[act_num] = act_data
                
                return tracker
            
        except Exception as e:
            print(f"⚠️ Could not load existing tracker: {e}")
        
        return {}
    
    async def save_tracker(
        self,
        memory_system: StoryMemorySystem,
        story_title: str,
        tracker: Dict[int, ActTracking]
    ):
        """Save updated tracker to memory"""
        
        tracker_data = {
            "act_tracking": {
                str(act_num): (
                    act_tracking.model_dump() 
                    if hasattr(act_tracking, 'model_dump') 
                    else act_tracking
                )
                for act_num, act_tracking in tracker.items()
            }
        }
        
        await memory_system.add_long_term_document(
            text=json.dumps(tracker_data, indent=2),
            metadata={"type": "story_tracker", "story_title": story_title}
        )
        
        print(f"✅ Story tracker updated with Act {max(tracker.keys())} tracking")


class ActContextBuilder:
    """Builds rich, structured context for act planning"""
    
    @staticmethod
    def build_act_context(
        act_number: int,
        final_plot: dict,
        seed: dict,
        connected_agents: list,
        integrated_world: dict,
        conflict_matrix: dict,
        story_tracker: dict,
        previous_acts: dict,
        story_so_far: str,
        word_guidance: tuple
    ) -> str:
        """Build comprehensive act context"""
        
        word_percentage, act_guidance = word_guidance
        suggested_words = int(seed.get('target_length', 50000) * word_percentage)
        suggested_chapters = max(2, suggested_words // 2500)
        
        context_parts = []
        
        context_parts.append(f"""
╔═══════════════════════════════════════════════════════════════════════════════
║ ACT {act_number} CONTEXT & GUIDANCE
╚═══════════════════════════════════════════════════════════════════════════════

ACT PURPOSE: {act_guidance}
TARGET: ~{suggested_words:,} words across {suggested_chapters} chapters
TONE: {seed.get('tone', 'Balanced')}
PROSE STYLE: {seed.get('prose_style', 'Standard narrative')}
""")
        
        act_summaries = final_plot.get('act_summaries', [])
        if act_number <= len(act_summaries):
            context_parts.append(f"""
┌─ ACT SUMMARY ────────────────────────────────────────────────────────────────
{act_summaries[act_number - 1]}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        outstanding_setup = []
        for prev_act_num in range(1, act_number):
            prev_tracking = story_tracker.get(str(prev_act_num), {})
            if isinstance(prev_tracking, dict):
                setup = prev_tracking.get('setup_elements', [])
                outstanding_setup.extend([f"(Act {prev_act_num}) {item}" for item in setup])
        
        current_tracking = story_tracker.get(str(act_number), {})
        if isinstance(current_tracking, dict):
            expected_payoffs = current_tracking.get('payoff_elements', [])
            expected_setups = current_tracking.get('setup_elements', [])
        else:
            expected_payoffs = []
            expected_setups = []
        
        context_parts.append(f"""
┌─ STORY CONTINUITY ───────────────────────────────────────────────────────────

OUTSTANDING SETUP (needs payoff in this or future acts):
{ActContextBuilder._format_list(outstanding_setup[:10], indent=2) if outstanding_setup else "  • (none - this is Act 1)"}

EXPECTED PAYOFFS THIS ACT:
{ActContextBuilder._format_list(expected_payoffs, indent=2)}

EXPECTED SETUP THIS ACT (for future payoff):
{ActContextBuilder._format_list(expected_setups, indent=2)}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        context_parts.append("\n┌─ AGENT ARCS THIS ACT ────────────────────────────────────────────────────────")
        
        tracker_agents = {}
        if isinstance(current_tracking, dict):
            tracker_agents = current_tracking.get('key_agent_moments', {})
        
        for agent in connected_agents:
            agent_name = agent.get('name', 'Unknown')
            agent_role = agent.get('role', '')
            agent_arc = agent.get('agent_arc', '')
            
            moment = tracker_agents.get(agent_name, "")
            
            context_parts.append(f"""
{agent_name} ({agent_role})
  Arc: {agent_arc}
  This Act: {moment if moment else 'Key player in unfolding conflicts'}
""")
        
        context_parts.append("└──────────────────────────────────────────────────────────────────────────────\n")
        
        context_parts.append(f"""
┌─ CONFLICT ESCALATION ────────────────────────────────────────────────────────
Central: {conflict_matrix.get('central_conflict', '')}

Escalation: {conflict_matrix.get('escalation_path', '')}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        world_summary = integrated_world.get('foundation', integrated_world.get('core_concept', ''))
        context_parts.append(f"""
┌─ WORLD & SETTING ────────────────────────────────────────────────────────────
{world_summary}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        if previous_acts and act_number > 1:
            context_parts.append(f"""
┌─ CONSEQUENCES FROM ACT {act_number - 1} ─────────────────────────────────────
{ActContextBuilder._summarize_previous_act(previous_acts.get(act_number - 1, {}))}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        context_parts.append(f"""
┌─ STORY SO FAR ───────────────────────────────────────────────────────────────
{story_so_far}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        themes = seed.get('themes', [])
        if themes:
            narrative_flow = final_plot.get('narrative_flow', '')
            context_parts.append(f"""
┌─ THEMATIC FOCUS ─────────────────────────────────────────────────────────────
Core Themes: {', '.join(themes)}

Development:
{narrative_flow[:500] + '...' if len(narrative_flow) > 500 else narrative_flow}
└──────────────────────────────────────────────────────────────────────────────
""")
        
        return "\n".join(context_parts)
    
    @staticmethod
    def _format_list(items: list, indent: int = 0) -> str:
        """Format list items with indentation"""
        if not items:
            return " " * indent + "(none specified)"
        
        formatted = []
        for item in items:
            if isinstance(item, str):
                clean = item.strip()
                if clean:
                    formatted.append(" " * indent + f"• {clean}")
        
        return "\n".join(formatted) if formatted else " " * indent + "(none specified)"
    
    @staticmethod
    def _summarize_previous_act(prev_act: dict) -> str:
        """Extract key consequences from previous act"""
        climax = prev_act.get('act_climax', '')
        payoffs = prev_act.get('payoff_this_act', [])
        changes = prev_act.get('agent_state_changes', {})
        
        summary = f"Climax: {climax}\n"
        if payoffs:
            summary += f"Payoffs: {', '.join(payoffs[:3])}\n"
        
        if changes:
            summary += "Character Changes: " + ", ".join(
                [f"{agent} ({status})" for agent, status in list(changes.items())[:3]]
            )
        
        return summary


# ============================================================================
# MAIN STORY AUTHOR CLASS
# ============================================================================

class StoryAuthor:
    """
    World-class story planning system using snowflake expansion method.
    Now with flexible creative output while maintaining structural integrity.
    """
    
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        
        # Initialize component systems
        self.world_builder = WorldBuilder(better_author_client)
        self.agent_genesis = AgentGenesis(author_fast_client)
        self.conflict_architect = ConflictArchitect(author_client)
        self.quality_controller = QualityController(author_fast_client)
        self.tracker_manager = StoryTrackerManager(author_fast_client)
        
        # LLM cache
        self.llm_cache = {}
    
    # async def _cached_author_llm_call(
    #     self,
    #     system_prompt: str,
    #     human_prompt: str,
    #     llm_temp: float,
    #     model: str = "None"
    # ):
    #     """Cached LLM wrapper"""
    #     key = hashlib.md5(
    #         (system_prompt + human_prompt + str(llm_temp) + model).encode()
    #     ).hexdigest()
        
    #     if key in self.llm_cache:
    #         return self.llm_cache[key], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
    #     response, tokens = await author_client(
    #         system_prompt=system_prompt,
    #         human_prompt=human_prompt,
    #         llm_temp=llm_temp,
    #         model=model
    #     )
        
    #     self.llm_cache[key] = response
    #     return response, tokens
    
    def validate_act_count(self, act_count: int, structure_name: str) -> int:
        if structure_name == "Three Act Structure":
            return 3
        elif structure_name == "Fichtean Curve":
            return 3
        elif structure_name == "Save the Cat Beat Sheet":
            return 3
        # elif structure_name == "Freytag's Pyramid":
        #     return 5
        elif act_count < 3:
            return 3
        elif act_count > 5:
            return 5
        else:
            return act_count
        
    def create_minimal_seed(self, user_context: Dict[str, Any]) -> MinimalStorySeed:
        """Extract minimal story seed from user context"""
        
        genres = user_context.get('Genre', ['Fiction'])
        sub_genre = user_context.get('Sub-Genre', [])
        themes = user_context.get('Additional Themes', [])
        
        protagonist_specs = {
            k: v for k, v in {
                'name': user_context.get('protagonist_name'),
                'age': user_context.get('protagonist_age'),
                'gender': user_context.get('protagonist_gender'),
                'archetype': user_context.get('protagonist_archetype'),
                'core_trait': user_context.get('protagonist_core_trait'),
                'background': user_context.get('protagonist_background'),
                'desire': user_context.get('protagonist_desire'),
                'fear': user_context.get('protagonist_fear'),
                'relationships': user_context.get('protagonist_relationships'),
                'physical_description': user_context.get('protagonist_physical_description')
            }.items() if v is not None
        }
        
        word_count_map = {
            "Short Long Story (7,500 - 15,000 words)": 12000,
            "Novelette (15,000 - 25,000 words)": 20000,
            "Novella (25,000 - 40,000 words)": 35000,
            "Novel Chapter (40,000 - 60,000 words)": 50000,
            "Full Novel (60,000 - 90,000 words)": 80000,
            "Epic / Series (90,000 - 150,000+ words)": 135000
        }
        
        target_length_str = user_context.get('Length', 'Novelette (15,000 - 25,000 words)')
        target_length = word_count_map.get(target_length_str, 20000)
        
        excluded_titles = {None, "None", "", " ", "Untitled Story", "Unitled story", 
                          "untitled story", "Null", "NULL", "Nill", "NILL", "null", "nill"}
        title = user_context.get('Title')
        title = None if title in excluded_titles else title
        
        target_audience_age = user_context.get('target_audience_age', 13)
        target_audience_age = max(8, min(target_audience_age, 18))

        if target_length <= 75000:
            act_count = 3
        else:
            act_count = 5
        if genres == []:
            genres = ["Fiction"]
        
        story_structure = find_best_structure_name(genres)
        act_count_validated = self.validate_act_count(act_count=act_count, structure_name=story_structure)
        act_count = act_count_validated
        return MinimalStorySeed(
            title=title,
            pov=user_context.get('POV', 'Third-person'),
            tone=user_context.get('Tone', 'Balanced'),
            genre=genres,
            sub_genre=sub_genre,
            setting=user_context.get('Setting'),
            prose_style=user_context.get('Guide Prose', ['Standard narrative'])[0],
            themes=themes,
            target_audience_age=target_audience_age,
            target_length=target_length,
            protagonist_specs=protagonist_specs,
            act_count=act_count,
            story_structure=story_structure
        )

    async def generate_minimal_plot_outline(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        model: str = "None",
        max_retries: int = 4
    ) -> Tuple[MinimalPlotOutline, dict]:
        """Generate initial plot with flexible character roles and auto-retries."""

        # --- 1. Compute dynamic role thresholds based on target length ---
        target_len = seed.target_length  # 12,000 to 120,000 expected

        # Clamp target length in case out of expected range
        min_len, max_len = 12000, 120000
        norm = max(0.0, min(1.0, (target_len - min_len) / (max_len - min_len)))

        # Linear interpolate between 4 and 15 roles
        dynamic_min_roles = int(4 + norm * (15 - 5))
        # dynamic_max_roles = 15  # still a hard max

        print(f"→ Target length: {target_len}, requiring at least {dynamic_min_roles} roles.")

        structure_template = find_best_structure(seed.story_structure)

        system_prompt = f"""You are a master plot architect creating the SOUL of a story (novel).

This is where magic begins - you're not filling a form, you're birthing a narrative that will haunt readers.

═══════════════════════════════════════════════════════════════════════════════
THE STORY'S HEARTBEAT
═══════════════════════════════════════════════════════════════════════════════

Genre Fusion: {'/'.join(seed.genre)} {f"+ {'/'.join(seed.sub_genre)}" if seed.sub_genre else ""}
Emotional Core: {seed.tone}
World Whisper: {world_foundation.core_concept}
Thematic DNA: {', '.join(seed.themes)}

Structure Inspiration (don't slavishly follow - let it guide, not bind):
{structure_template}

═══════════════════════════════════════════════════════════════════════════════
YOUR CREATIVE MANDATE
═══════════════════════════════════════════════════════════════════════════════

  title: A title that SINGS - evocative, memorable, thematically resonant,
  
  premise:  THE HOOK - Not a dry summary, but the irresistible question.
             One breathless sentence that makes someone say 'I NEED to read this.'
             Lead with character desire colliding with impossible stakes.
             Example: 'A grief-stricken cartographer must choose between mapping 
             her father's killer and losing the only person who sees past her scars.',
  
  central_question: The BURNING question that will torment readers.
                       Not 'what happens?' but 'what will they CHOOSE?'
                       Frame as moral/emotional dilemma, not plot mechanics.
                       Example: 'Can she forgive herself for choosing ambition over love?',
  
  narrative_arc:  Tell me the story in FLOWING PROSE, not bullet points.
                    Paint the emotional journey: where they start broken, how the world 
                    tests them, what they lose, what they become. Make me FEEL the arc.
                    Use sensory details, metaphor, rhythm. This should read like the 
                    back-cover copy of a bestseller - compelling, evocative, alive.
                    
                    Instead of: 'Protagonist goes on journey, faces challenges, wins'
                    Try: 'She enters the mist with maps and certainty, but the jungle 
                    strips her bare—each betrayal a vine tightening, each loss a root 
                    pulling her deeper into the question: what if the treasure she seeks 
                    is the very thing destroying her?',

    required_character_roles:
    Think FUNCTION not formula. What narrative forces does THIS story need?
    
    CORE ROLES (adapt to story):
    - Protagonist (but WHO specifically? 'Haunted explorer'? 'Reluctant heir'?)
    - Antagonist (external force, internal shadow, or both?)
    
    STORY-SPECIFIC ROLES (be creative and specific):
    Instead of: 'mentor' → 'guilt-ridden former expedition leader'
    Instead of: 'ally' → 'indigenous guide with prophecy burden'
    Instead of: 'love interest' → 'rival cartographer hiding family secret'
    
    SCALE TO STORY:
    - Intimate character study: 4-7 roles (each deeply developed)
    - Epic multi-threaded: 12-18 roles (varied importance)
    
    GENRE EXAMPLES:
    Mystery: 'detective with addiction', 'unreliable witness', 'victim's vengeful sibling'
    Romance: 'best friend saboteur', 'ex who catalyzes growth', 'mentor championing love'
    Fantasy: 'mage bound by taboo', 'shapeshifter testing loyalty', 'oracle refusing destiny'
    Thriller: 'whistleblower with family hostage', 'handler with hidden agenda'
    
    Think: What roles create MAXIMUM emotional/thematic collision?

    // MUST contain at least {dynamic_min_roles} roles.
    // MUST NOT be empty.
    // MUST NOT collapse to generic labels.
    Example of good role list (format, not content):

    [
    "haunted apprentice cartographer seeking identity",
    "exiled queen manipulating the protagonist from afar",
    "mentor who lost their faith after a failed rebellion",
    "rival explorer who becomes an uneasy ally",
    "oracle child who speaks only in dreams",
    "soldier carrying a forbidden truth",
    "villain’s enforcer who doubts their orders"
    ]

  // OPTIONAL - ADD WHATEVER SPARKS YOUR VISION:
  // "opening_image": vivid first scene idea
  // "thematic_question": philosophical core
  // "genre_twist": how you'll subvert expectations
  // "emotional_palette": the feelings this story explores
  // "narrative_voice": who's telling this and why
  // Don't ask permission - just add what makes THIS story unforgettable


Base Output Schema:
{MinimalPlotOutline.model_json_schema()}

═══════════════════════════════════════════════════════════════════════════════
CREATIVE COMMANDMENTS
═══════════════════════════════════════════════════════════════════════════════

1. **CHARACTER OVER PLOT**: Roles should be people first, functions second
   Bad: "mentor, ally, antagonist"
   Good: "disgraced father-figure seeking redemption through the protagonist's quest"

2. **EMOTIONAL SPECIFICITY**: What does this story FEEL like?
   Bad: "Protagonist faces challenges and grows"
   Good: "She mistakes ambition for identity until betrayal carves her hollow"

3. **SUBVERT CLICHÉS**: If your first instinct is "chosen one" or "love triangle," 
   spin it. What's the version no one's seen? The angle that surprises?

4. **THEMATIC RESONANCE**: Every role, every beat should echo {', '.join(seed.themes)}
   Don't just mention themes - EMBODY them in character dynamics and stakes

5. **VOICE**: Write like you're pitching this to someone at a bar who's had two drinks
   and is leaning in, captivated. Conversational, vivid, irresistible.

6. **AGE-APPROPRIATE COMPLEXITY**: For {seed.target_audience_age} year olds
   - Under 13: Clear heroes/villains, hopeful endings, moral lessons
   - 13-15: Moral ambiguity, consequences, bittersweet growth
   - 16-18: Philosophical depth, tragedy, no easy answers

═══════════════════════════════════════════════════════════════════════════════
THE TEST
═══════════════════════════════════════════════════════════════════════════════

Before submitting, ask yourself:
- Would *I* read this based on the premise alone?
- Do the character roles make me curious about their dynamics?
- Does the narrative arc have emotional progression, not just plot progression?
- Can I picture the story's unique flavor from these bones?

If not - dig deeper, get weirder, find the heart.

Output clean JSON with no markdown fences."""


        human_prompt = f"""Create the foundational DNA of an unforgettable {'/'.join(seed.genre)} story.

═══════════════════════════════════════════════════════════════════════════════
STORY SEED
═══════════════════════════════════════════════════════════════════════════════

{seed.model_dump_json(indent=2)}

═══════════════════════════════════════════════════════════════════════════════
WORLD FOUNDATION
═══════════════════════════════════════════════════════════════════════════════

{world_foundation.model_dump_json(indent=2)}

═══════════════════════════════════════════════════════════════════════════════
YOUR MISSION
═══════════════════════════════════════════════════════════════════════════════

Don't just answer the questions - create a premise so compelling it demands to be told.

Make the roles SPECIFIC to this world and premise. Not "the mentor" but 
"the guilt-haunted admiral whose stolen artifact holds the key to redemption."

Make the arc EMOTIONAL. Not "they go on a journey" but "ambition devours her 
until she's forced to choose between the map and her soul."

Write with URGENCY and VOICE. This is the moment the story crystalizes from 
possibility into inevitability.

Be bold. Be specific. Make me care."""

        # --- 2. Retry loop ---
        attempt = 0
        last_tokens = {}

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 LLM plot generation attempt {attempt}/{max_retries + 1}")

            # --- 3. Call LLM ---
            response, tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.9,
                model=model
            )
            last_tokens = tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                outline: MinimalPlotOutline = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=MinimalPlotOutline
                )
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            role_count = len(outline.required_character_roles)
            print(f"→ Produced {role_count} roles: {outline.required_character_roles}")

            # --- 4. Check role thresholds ---
            if role_count >= dynamic_min_roles:
                print(f"✓ Accepted. Role count meets threshold ({dynamic_min_roles}).")
                return outline, tokens
            else:
                print(f"⚠️ Insufficient roles ({role_count} < {dynamic_min_roles}). Retrying...")
                await asyncio.sleep(0.8 * attempt)

        # --- 5. If max retries exhausted ---
        print("🚨 Max retries exhausted. Returning last valid attempt (even if insufficient roles).")
        return outline, last_tokens

    async def expand_plot_outline(
        self,
        minimal_plot: MinimalPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        seed: MinimalStorySeed,
        model: str = "None",
        max_retries: int = 4
    ) -> Tuple[ExpandedPlotOutline, dict]:
        """Expand minimal plot with creative flexibility and enforce act count matching."""

        target_act_count = seed.act_count
        print(f"→ Target act count: {target_act_count}")

        structure_template = find_best_structure(seed.story_structure)

        system_prompt = f"""
You are a disciplined narrative generator.
Expand this minimal plot outline into a FULL story blueprint (for a story novel).
You have full creative freedom within these boundaries, make it unique and compelling.
Do not add content outside the requested JSON. 
Use only information provided through variables.

════════ STORY INPUT ════════
Premise: {minimal_plot.premise}
Central Question: {minimal_plot.central_question}
Structure Inspiration: {structure_template}
Emotional Core: {seed.tone}
Prose Voice: {seed.prose_style}
Target Length: {seed.target_length} words across {target_act_count} acts
Title: {minimal_plot.title}
Genre: {'/'.join(seed.genre)}
Audience Age: {seed.target_audience_age}

════════ Creative Mandate ════════

  title: {minimal_plot.title},
  premise: {minimal_plot.premise},

  act_summaries: 
    Write each act in 200–300 words of continuous, cinematic narrative prose.

    Requirements:
    - Begin with sensory grounding (sight, sound, smell, texture).
    - Use character names frequently. No generic labels like 'the protagonist'.
    - Always reference specific places, not generic locations.
    - Include at least 5 named characters per act.
    - Include at least 3 clearly named locations per act.
    - Include concrete events with consequences.
    - Show internal emotional movement.
    - Use imagery, metaphor, varied pacing.
    - End each act with a strong hook or revelation.
    - No lists. No outlining language. No meta commentary. Story only.
  ,

  narrative_flow: Write 500–800 words of flowing, cinematic prose describing the emotional and narrative arc of the full story. 
  Structure:
  - Opening mood, wound, world texture
  - Rising complications and deepening desires
  - Midpoint revelation that shifts meaning
  - Climax: transformation or impossible choice
  - Resolution: new equilibrium with visible emotional change
  Use sensory details, emotional precision, and clear scene-based language.,

  // Extra Creative Fields:
  // Add ONLY fields that meaningfully expand this specific story.
  // Do not copy samples blindly. Invent fields appropriate to { '/'.join(seed.genre) }.

    // Length of Act Summaries List must be {target_act_count}

OUTPUT MUST validate against:
{ExpandedPlotOutline.model_json_schema()}

════════ CREATIVE GUIDELINES ════════
Follow these principles when writing:

1. Show character through actions and spoken voice, not labels or summaries.
2. Dramatize events as mini-scenes. Avoid vague summaries.
3. Use precise emotional language.
4. Let themes emerge through conflict and choices, not exposition.
5. Anchor key beats in physical sensory detail.
6. Vary sentence rhythm for effect.
7. Avoid generic placeholders—always choose concrete details.
8. Match complexity and tone to the target audience.

════════ QUALITY CHECK (BEFORE OUTPUT) ════════
Ensure:
- Scenes are clear and visualizable.
- Characters and places are specific and repeated meaningfully.
- Narrative_flow feels like a continuous story, not a list.
- No contradictions with provided inputs.
- No meta commentary or instructions in output.
- Only valid JSON is returned.
- Make sure all act summaries are present. This is the final production version.

Generate the clean JSON only.
"""

        
        human_prompt = f"""Weave the full tapestry of this {'/'.join(seed.genre)} story.

═══════════════════════════════════════════════════════════════════════════════
THE RAW MATERIALS
═══════════════════════════════════════════════════════════════════════════════

MINIMAL PLOT SKELETON:
{minimal_plot.model_dump_json(indent=2)}

THE CAST (breathe life into these):
{json.dumps([a.model_dump() for a in agents], indent=2)}

THE WORLD (make it pulse):
{world.model_dump_json(indent=2)}

THE CONFLICTS (make them bleed):
{conflict_matrix.model_dump_json(indent=2)}

═══════════════════════════════════════════════════════════════════════════════
YOUR MANDATE
═══════════════════════════════════════════════════════════════════════════════

Transform these building blocks into a story that LIVES.

ACT SUMMARIES: Write {target_act_count} acts as FLOWING NARRATIVE
- Each 200-300 words of immersive storytelling
- Names, locations, emotions, sensory details
- Make me see it, feel it, fear for them

NARRATIVE FLOW: The complete emotional arc in 500-800 words
- Opening wound to final transformation
- Cinematic, sensory, rhythmic
- This is the story's SOUL—make it unforgettable

ADDITIONAL FIELDS: Add whatever makes THIS story sing
- Relationship arcs if intimacy matters
- Symbolic layers if themes demand it
- Suspense architecture if tension is king
- INVENT fields I haven't thought of

Generate the plot keeping the desired story length in mind. Make sure it is compatible with the target length.
Target Length: {seed.target_length} words across {target_act_count} acts

HONOR THEMES: {', '.join(seed.themes)}
EARN EMOTION: Every beat must RESONATE

Don't report—ENCHANT. Don't list—BEWITCH. 

This is the story that will make people be in awe. Make it worthy."""

        attempt = 0
        last_tokens = {}

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 LLM outline expansion attempt {attempt}/{max_retries + 1}")

            # --- LLM CALL ---
            response, tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.9,
                model=model
            )
            last_tokens = tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON output ---
            try:
                json_data: ExpandedPlotOutline = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            # --- VALIDATION: ACT COUNT ---
            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {target_act_count})")

            if generated_count == target_act_count:
                print("✓ Accepted. Act summary count matches the target.")
                return json_data, tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {target_act_count}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        # --- MAX RETRIES EXHAUSTED ---
        print("🚨 Max retries exhausted. Returning last generated outline (mismatch unresolved).")
        return json_data, last_tokens

    
    async def _enforce_age_appropriateness(
        self,
        final_plot: ExpandedPlotOutline,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[AgeAppropriatenessReport, dict]:
        """Age-appropriateness safety filter"""
        target_age = seed.target_audience_age

        system_prompt = f"""You are an age-appropriateness pass agent.
Target audience age: {target_age} years old (8–18 range).

Scan the plot and rate its maturity. If any content is too mature for age {target_age},
Output False.
If the plot passes the age filter then output True

Thresholds:
- Age < 13 → max mild violence, no sexual content, clean language
- Age 13–15 → moderate violence ok, implied romance only, mild language  
- Age 16–18 → intense violence ok, moderate/implied sexual content ok, moderate language ok

Rate the plot for target Audience: 'A' for ages 8+, 'T' for ages 13+, 'M' for ages 18+, regardless of given target audience age.

Output Format:
{AgeAppropriatenessReport.model_json_schema()}"""

        human_prompt = f"""Target age: {target_age}

Plot to check:
{final_plot.model_dump_json(indent=2)}

Review and rate."""

        response, tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,
            model=model
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=AgeAppropriatenessReport
        )

        return json_data, tokens
    
    async def refine_plot_with_quality_feedback(
        self,
        plot: ExpandedPlotOutline,
        quality_report: QualityReport,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        act_count_target: int,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict]:
        """Refine plot based on quality assessment"""
        
        if not quality_report.concerns:
            return plot, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        vagueness_concerns = [c for c in quality_report.concerns if any(
            word in c.lower() for word in ['vague', 'generic', 'specific', 'abstract', 'concrete', 'detail']
        )]
        
        specificity_focus = ""
        if vagueness_concerns:
            specificity_focus = """
🚨 CRITICAL PRIORITY: ADDRESS VAGUENESS

The current plot is too abstract. Make it CONCRETE:

REPLACE:
- "the hero" → character name (e.g., "Kael")
- "faces challenges" → specific obstacles
- "learns something" → specific revelation
- "goes somewhere" → named location

"""
        
        system_prompt = f"""You are refining a plot based on editorial feedback.

{specificity_focus}

Address specific concerns while maintaining story integrity.
Only fix what's flagged, but be SPECIFIC and CONCRETE where you make changes.

OUTPUT: Same JSON structure as input.
Output clean JSON with no markdown fences."""
    
        human_prompt = f"""Current Plot:
{plot.model_dump_json(indent=2)}

Quality Concerns (ADDRESS ALL):
{json.dumps(quality_report.concerns, indent=2)}

The story must have EXACTLY {act_count_target} acts in act_summaries. Adjust accordingly if act_summaries length is different.
If suggestions include changes length of act_summaries or narrative flow, adjust accordingly. You may add or delete act_summaries from its list as needed.
You have creative freedom to enhance the plot. If you feel its missing elements not mentioned in the report, improve them too.

Recommendations:
{json.dumps(quality_report.recommendations, indent=2)}

Available Resources:
Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
World: {world.model_dump_json(indent=2)}
Conflicts: {conflict_matrix.model_dump_json(indent=2)}

Refine the plot with specific names, locations, and concrete events."""
        max_retries = 4
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 LLM Refined outline expansion attempt {attempt}/{max_retries + 1}")

            # --- LLM CALL ---
            response, tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.75,
                model=model
            )
            last_tokens = tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON output ---
            try:
                json_data: ExpandedPlotOutline = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            # --- VALIDATION: ACT COUNT ---
            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {act_count_target})")

            if generated_count == act_count_target:
                print("✓ Accepted. Act summary count matches the target.")
                return json_data, tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {act_count_target}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        # --- MAX RETRIES EXHAUSTED ---
        print("🚨 Max retries exhausted. Returning last generated outline (mismatch unresolved).")
        return json_data, last_tokens
    
    async def _create_story_tracker(
        self,
        act_count: int,
        final_plot: ExpandedPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        conflict_matrix: ConflictMatrix,
        model: str
    ) -> Tuple[StoryTracker, dict]:
        """Lightweight story-level tracking"""
        
        system_prompt = f"""Create a simple tracking document for story consistency.

For each of the {act_count} acts, identify:
1. What major elements get INTRODUCED (characters, conflicts, questions, items, relationships)
2. What gets RESOLVED from earlier

Keep it simple and genre-agnostic. 
For a mystery: introduce clues, resolve whodunit
For romance: introduce attraction, resolve will-they-won't-they  
For action: introduce threat, resolve confrontation
For comedy: introduce misunderstanding, resolve reveal

DO NOT produce more or less acts than given!

{StoryTracker.model_json_schema()}"""

        human_prompt = f"""Story outline:
{final_plot.model_dump_json(indent=2)}

Agents:
{json.dumps([a.model_dump() for a in connected_agents], indent=2)}

Conflicts:
{conflict_matrix.model_dump_json(indent=2)}

Create act-by-act tracking of what gets introduced and resolved."""

        response, tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=StoryTracker)
        
        return json_data, tokens
    
    async def orchestrate_story_planning(
        self,
        user_context: Dict[str, Any],
        model: str = "None"
    ) -> Dict[str, Any]:
        """
        Main orchestration: Full story planning workflow using snowflake method
        Returns complete story foundation ready for act/chapter planning
        """
        
        print("🎬 ORCHESTRATING WORLD-CLASS STORY PLANNING (SNOWFLAKE METHOD)")
        print("=" * 60)
        
        # PHASE 1: Minimal Seed
        print("\n📋 Phase 1: Creating story seed...")
        seed = self.create_minimal_seed(user_context)
        structure_template = find_best_structure(seed.story_structure)
        print(f"✓ Seed created | Genres: {seed.genre} | Target: {seed.target_length} words")
        
        # PHASE 2: Independent Foundations (Snowflake Layer 1)
        print("\n🌍 Phase 2: Building independent world foundation...")
        world_foundation, world_tokens = await self.world_builder.generate_world_foundation(
            seed, model
        )
        print(f"✓ World foundation created")
        
        # PHASE 3: Minimal Plot (Snowflake Layer 2)
        print("\n📖 Phase 3: Generating minimal plot outline...")
        minimal_plot, plot_tokens = await self.generate_minimal_plot_outline(
            seed, world_foundation, model
        )
        print(f"✓ Minimal plot: '{minimal_plot.title}' ({seed.act_count} acts)")
        print(f"✓ Requires {len(minimal_plot.required_character_roles)} roles")
        
        # PHASE 4: Generate Agents (Snowflake Layer 3)
        print("\n👥 Phase 4: Generating narrative agents for required roles...")
        narrative_agents, agent_tokens = await self.agent_genesis.generate_narrative_agents(
            seed, world_foundation, minimal_plot.required_character_roles, model
        )
        print(f"✓ {len(narrative_agents)} narrative agents created")
        
        # PHASE 5: Conflict Matrix (Snowflake Layer 4)
        print("\n⚔️ Phase 5: Analyzing conflicts...")
        conflict_matrix, conflict_tokens = await self.conflict_architect.generate_conflict_layers(
            minimal_plot, seed, narrative_agents, world_foundation, model
        )
        print(f"✓ Multi-dimensional conflict matrix created")
        
        # PHASE 6: Connect Elements (Snowflake Layer 5)
        print("\n🔗 Phase 6: Connecting elements to plot...")
        prose_style = seed.prose_style
        connected_agents, connect_tokens = await self.agent_genesis.connect_agents_to_plot(
            narrative_agents, conflict_matrix, minimal_plot, prose_style, model
        )

        integrated_world, integrate_tokens = await self.world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents, model
        )
        
        print(f"✓ Agents connected to plot")
        print(f"✓ World integrated with conflicts")
        
        # PHASE 7: Expand Plot (Snowflake Layer 6)
        print("\n📈 Phase 7: Expanding plot outline...")
        expanded_plot, expand_tokens = await self.expand_plot_outline(
            minimal_plot, connected_agents, integrated_world,
            conflict_matrix, seed, model
        )
        print(f"✓ Plot expanded with full detail")
        
        # PHASE 8: Age-appropriateness filter
        print("\n🔞 Phase 8: Age-appropriateness filter...")
        age_report, age_filter_tokens = await self._enforce_age_appropriateness(
            final_plot=expanded_plot,
            seed=seed,
            model=model
        )
        print(age_report)
        if not age_report.plot_outline_pass:
            raise "Age Filter pass failed"
        
        if age_report.age_rating == 'A':
            seed.target_audience_age = 8
        elif age_report.age_rating == 'T':
            seed.target_audience_age = 13
        elif age_report.age_rating == 'M':
            seed.target_audience_age = 18
        
        print(f"✓ Content validated for age {seed.target_audience_age}+")
        
        # PHASE 9: Quality Validation
        print("\n🔍 Phase 9: Quality validation...")
        quality_report, quality_tokens = await self.quality_controller.validate_story_elements(
            seed, expanded_plot, connected_agents,
            integrated_world, conflict_matrix, model
        )
        print(f"✓ Quality assessment complete")
        print(f"  Strengths: {len(quality_report.strengths)}")
        print(f"  Concerns: {len(quality_report.concerns)}")

        # PHASE 10: Refinement if needed
        final_plot = expanded_plot
        refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if quality_report.concerns:
            critical_issues = [c for c in quality_report.concerns if 'vague' in c.lower() or 'specific' in c.lower() or 'generic' in c.lower()]
            
            if critical_issues:
                print("\n⚠️  CRITICAL: Plot too vague, running refinement...")
                print(f"   Issues: {len(critical_issues)}")
            else:
                print(f"\n🔄 Phase 10: Refining {len(quality_report.concerns)} concerns...")
            
            final_plot, refine_tokens = await self.refine_plot_with_quality_feedback(
                plot=expanded_plot, quality_report=quality_report, agents=connected_agents,
                world=integrated_world, conflict_matrix=conflict_matrix, act_count_target=seed.act_count, model=model
            )
            
            # if critical_issues:
            #     print("   Re-validating after refinement...")
            #     recheck_report, recheck_tokens = await self.quality_controller.validate_story_elements(
            #         seed, final_plot, connected_agents,
            #         integrated_world, conflict_matrix, model
            #     )
            #     refine_tokens["prompt_tokens"] += recheck_tokens.get("prompt_tokens", 0)
            #     refine_tokens["completion_tokens"] += recheck_tokens.get("completion_tokens", 0)
                
            #     remaining_critical = [c for c in recheck_report.concerns if 'vague' in c.lower() or 'specific' in c.lower()]
            #     if remaining_critical:
            #         print(f"⚠️  Still has {len(remaining_critical)} vagueness issues (proceeding anyway)")
            #     else:
            #         print("✅ Vagueness issues resolved")
            
            print(f"✓ Plot refined")
        else:
            print("\n✓ Phase 10: No refinement needed")

        # PHASE 11: Story Tracker
        print("\n📊 Phase 11: Creating story tracker...")
        story_tracker, tracker_tokens = await self._create_story_tracker(
            act_count=seed.act_count,
            final_plot=final_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix,
            model=model
        )
        print(f"✓ Story tracker initialized")
        
        # Store in memory
        await self.memory.add_long_term_document(
            text=seed.model_dump_json(indent=2),
            metadata={"type": "story_seed", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=minimal_plot.model_dump_json(indent=2),
            metadata={"type": "minimal_plot_outline", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=final_plot.model_dump_json(indent=2),
            metadata={"type": "expanded_plot_outline", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=world_foundation.model_dump_json(indent=2),
            metadata={"type": "world_foundation", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=json.dumps([a.model_dump() for a in narrative_agents], indent=2),
            metadata={"type": "narrative_agents", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=conflict_matrix.model_dump_json(indent=2),
            metadata={"type": "conflict_matrix", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=json.dumps([a.model_dump() for a in connected_agents], indent=2),
            metadata={"type": "connected_agents", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=integrated_world.model_dump_json(indent=2),
            metadata={"type": "integrated_world", "story_title": final_plot.title}
        )
        await self.memory.add_long_term_document(
            text=quality_report.model_dump_json(indent=2),
            metadata={"type": "quality_report", "story_title": final_plot.title}
        )
        
        # Calculate total tokens
        total_tokens = {
            "prompt_tokens": sum([
                world_tokens.get("prompt_tokens", 0),
                agent_tokens.get("prompt_tokens", 0),
                plot_tokens.get("prompt_tokens", 0),
                conflict_tokens.get("prompt_tokens", 0),
                connect_tokens.get("prompt_tokens", 0),
                integrate_tokens.get("prompt_tokens", 0),
                expand_tokens.get("prompt_tokens", 0),
                quality_tokens.get("prompt_tokens", 0),
                refine_tokens.get("prompt_tokens", 0),
                age_filter_tokens.get("prompt_tokens", 0),
                tracker_tokens.get("prompt_tokens", 0)
            ]),
            "completion_tokens": sum([
                world_tokens.get("completion_tokens", 0),
                agent_tokens.get("completion_tokens", 0),
                plot_tokens.get("completion_tokens", 0),
                conflict_tokens.get("completion_tokens", 0),
                connect_tokens.get("completion_tokens", 0),
                integrate_tokens.get("completion_tokens", 0),
                expand_tokens.get("completion_tokens", 0),
                quality_tokens.get("completion_tokens", 0),
                refine_tokens.get("completion_tokens", 0),
                age_filter_tokens.get("completion_tokens", 0),
                tracker_tokens.get("completion_tokens", 0)
            ]),
            "total_tokens": 0
        }
        total_tokens["total_tokens"] = (
            total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]
        )
        
        print("\n" + "=" * 60)
        print("✅ STORY PLANNING COMPLETE")
        print(f"📊 Total tokens used: {total_tokens['total_tokens']:,}")
        print("=" * 60)
        
        return {
            "seed": seed,
            "world_foundation": world_foundation,
            "narrative_agents": narrative_agents,
            "conflict_matrix": conflict_matrix,
            "connected_agents": connected_agents,
            "integrated_world": integrated_world,
            "minimal_plot": minimal_plot,
            "final_plot": final_plot,
            "quality_report": quality_report,
            "story_tracker": story_tracker,
            "structure_template": structure_template,
            "tokens": total_tokens
        }

    def get_act_percentages_and_guidance(self, structure_name: str, total_acts: int) -> List[Tuple[float, str]]:
        """Returns act percentages and guidance based on structure"""
        if structure_name == "Three Act Structure":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.25, "SETUP: Establish world, agents, inciting incident, and stakes."),
                (0.50, "CONFRONTATION: Rising action, complications, midpoint reversal."),
                (0.25, "RESOLUTION: Climax, falling action, denouement.")
            ]
        
        elif structure_name == "Freytag's Pyramid":
            if total_acts != 5:
                if total_acts == 3:
                    return [
                        (0.20, "EXPOSITION AND RISING: Introduce setting/agents and build tension."),
                        (0.60, "CLIMAX AND FALLING: Peak conflict and consequences."),
                        (0.20, "DENOUEMENT: Resolution.")
                    ]
                elif total_acts == 4:
                    return [
                        (0.15, "EXPOSITION: Introduce agents and setting."),
                        (0.40, "RISING ACTION: Build tension."),
                        (0.25, "CLIMAX AND FALLING: Turning point and consequences."),
                        (0.20, "DENOUEMENT: Resolution.")
                    ]
                else:
                    raise ValueError(f"{structure_name} best fits 5 acts.")
            return [
                (0.15, "EXPOSITION: Introduce agents, setting, and initial conflict."),
                (0.35, "RISING ACTION: Escalate complications."),
                (0.15, "CLIMAX: Moment of highest tension."),
                (0.20, "FALLING ACTION: Unravel consequences."),
                (0.15, "DENOUEMENT: Final resolution.")
            ]
        
        elif structure_name == "The Hero's Journey":
            if total_acts == 3:
                return [
                    (0.25, "DEPARTURE: Ordinary world, call, threshold."),
                    (0.50, "INITIATION: Trials, allies/enemies, ordeal."),
                    (0.25, "RETURN: Reward, road back, resurrection.")
                ]
            elif total_acts == 4:
                return [
                    (0.20, "ORDINARY WORLD AND CALL"),
                    (0.30, "DEPARTURE AND TRIALS"),
                    (0.30, "ORDEAL AND REWARD"),
                    (0.20, "RETURN")
                ]
            elif total_acts == 5:
                return [
                    (0.15, "ORDINARY WORLD"),
                    (0.25, "DEPARTURE"),
                    (0.30, "INITIATION"),
                    (0.20, "ORDEAL"),
                    (0.10, "RETURN")
                ]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        elif structure_name == "Dan Harmon's Story Circle":
            base = [
                (0.20, "YOU/NEED: Establish comfort zone and unmet need."),
                (0.30, "GO/SEARCH/TAKE: Enter unfamiliar, pursue goal."),
                (0.30, "FIND/RETURN/PAY: Achieve goal but pay price."),
                (0.20, "CHANGE: Return transformed.")
            ]
            if total_acts == 4:
                return base
            elif total_acts == 3:
                return [
                    (0.25, "YOU/NEED/GO: Setup, need, enter unfamiliar."),
                    (0.50, "SEARCH/TAKE/FIND: Adapt, pursue, achieve."),
                    (0.25, "RETURN/CHANGE: Transform and resolve.")
                ]
            elif total_acts == 5:
                return [(0.20, base[0][1])] + [(0.20, base[1][1])] + [(0.20, base[2][1])] + [(0.20, base[3][1])] + [(0.20, "FINAL INTEGRATION")]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        elif structure_name == "Fichtean Curve":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.20, "INCITING INCIDENT: Jump into action."),
                (0.60, "RISING CRISES: Escalating challenges."),
                (0.20, "CLIMAX AND RESOLUTION: Peak confrontation.")
            ]
        
        elif structure_name == "Save the Cat Beat Sheet":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.25, "ACT 1: Setup, catalyst, debate."),
                (0.50, "ACT 2: B-story, midpoint, all is lost."),
                (0.25, "ACT 3: Finale, resolution.")
            ]
        
        elif structure_name == "Seven-Point Story Structure":
            base = [
                (0.20, "HOOK TO PLOT TURN 1"),
                (0.40, "PINCH 1 TO MIDPOINT"),
                (0.25, "PINCH 2 TO PLOT TURN 2"),
                (0.15, "RESOLUTION")
            ]
            if total_acts == 4:
                return base
            elif total_acts == 3:
                return [
                    (0.25, "HOOK TO MIDPOINT"),
                    (0.50, "PINCH 2 TO TURN 2"),
                    (0.25, "RESOLUTION")
                ]
            elif total_acts == 5:
                return [(0.15, "HOOK")] + base[:3] + [(0.10, "FINAL RESOLUTION")]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        even_perc = 1.0 / total_acts
        return [(even_perc, f"ACT {i+1}: Progress with rising tension.") for i in range(total_acts)]

    def get_enhanced_act_planning_system_prompt(
        self,
        act_number: int,
        total_acts: int,
        act_guidance: str,
        themes: list,
        tone: str,
        prose_style: str,
        target_act_word_count: int,
        word_count: int,
        min_age: int,
        structure_config: dict
    ) -> str:
        """Directive system prompt for act planning - EMPHASIZE CREATIVITY"""
        
        return f"""You are crafting ACT {act_number} of {total_acts} with world-class storytelling flair.

YOUR ROLE: Sculpt a vibrant story SPINE - dynamic turning points that pulse with creativity, emotion, and surprise.
EMBRACE: Rich subtext, layered motivations, thematic depth, and innovative beats. Avoid formulaic rigidity.
You are crafting ACT {act_number} of {total_acts} with world-class storytelling flair.

CRITICAL RULE — NEVER REPEAT YOURSELF
════════════════════════════════════════════════
• Never reuse the same action, phrase, prop, or atmospheric detail more than once per act
• Every anchor must feel distinctly new in mood, imagery, stakes, or location
• Surprise me at least once per chapter
• Earn every emotional beat — nothing comes easy
════════════════════════════════════════════════

YOUR ROLE: Sculpt a vibrant, unpredictable story spine that pulses with surprise, subtext, and emotional truth.
EMBRACE creative freedom. Break the template if the story demands it.
═══════════════════════════════════════════════════════════════════════════════
STRUCTURE & CONTEXT
═══════════════════════════════════════════════════════════════════════════════

Act Purpose: {act_guidance}
Tone: {tone} | Prose: {prose_style}
Themes: {', '.join(themes)}
Age Range: {min_age}+
Total Entire Story Length: ~{word_count} words

Flexible Structure Guidelines (Adapt as needed for creativity):
- Chapters: {structure_config.get('min_chapters', 2)}-{structure_config.get('max_chapters', 5)} (or more if epic scope demands)
- Anchors per chapter: {structure_config['anchors_per_chapter']} (vary for pacing variety)
- Scenes per anchor: {structure_config['scenes_per_anchor']} (expand for complexity)
- Words across all chapters: ~{target_act_word_count} (flexible for narrative flow)
- Words per chapter: Should be decent, not too short or too long, prefer chapter count over chapter length

═══════════════════════════════════════════════════════════════════════════════
ANCHOR POINT GUIDELINES (FLEXIBLE FOR CREATIVITY)
═══════════════════════════════════════════════════════════════════════════════

Each anchor point SHOULD inspire with:

1. **IMMERSIVE GROUNDING**
location: Vivid, atmospheric details
character_positions: Emotional/psychological starting points

2. **DYNAMIC TEMPORAL FLOW**
time_context: Evocative passage of time
transition_from_previous: Thematic or character-driven bridges

3. **CREATIVE SCENE CRAFTING**
estimated_scenes: 1-5 for flexible dramatization
scene_breakdown: Inspiring outlines if multi-scene

4. **DEEP CHARACTER ARCING**
required_character_states: Loose preconditions for organic entry
resulting_character_states: Transformative shifts with nuance

5. **NARRATIVE WEAVE**
introduces: Spark fresh intrigue and layers
resolves: Satisfying payoffs with emotional weight

Feel free to innovate on anchor_type or add custom fields if the story demands.
Surprise me at least once per chapter.
Earn every emotional beat — nothing is handed to the characters.

═══════════════════════════════════════════════════════════════════════════════
DENSITY GUIDANCE (ADAPT FOR RICHNESS)
═══════════════════════════════════════════════════════════════════════════════

{self._get_density_guidance(word_count)}

{ActPlan.model_json_schema()}

Output clean JSON with no markdown fences."""

    def _get_density_guidance(self, total_words: int) -> str:
        """Length-appropriate anchor density guidance - PRIORITIZE RICHNESS"""
        if total_words < 15000:
            return """SHORT STORY MODE (< 15K):
- 1-3 anchors per chapter for focused intensity
- Weave deep emotion into core conflict
- Layer subtext without subplots"""
        
        elif total_words < 40000:
            return """NOVELLA MODE (15-40K):
- 2-4 anchors per chapter for balanced exploration
- Intertwine plot, character, and theme creatively
- 1-2 subplots to add texture
- Each anchor = 800-1500 words of nuanced prose"""
        
        else:
            return """NOVEL MODE (40K+):
- 3-5+ anchors per chapter for epic layering
- Multiple interwoven threads and subplots
- Subplots enhance themes and world-building
- Vary anchor scale for dynamic pacing"""

    def get_enhanced_act_planning_human_prompt(
        self,
        chapter_number: int,
        context_str: str,
        act_number: int,
        total_acts: int,
        suggested_words: int
    ) -> str:
        """Well-organized human prompt - INSPIRE CREATIVITY"""
        if chapter_number == 0:
            chapter_number = 1
        
        return f"""{context_str}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Craft Act {act_number} of {total_acts} with appropriate number of chapters, starting from chapter {chapter_number} (~{suggested_words} words total combined chapters, flexible for flow).

For EACH chapter:
1. Creative seed: 3-5 sentences igniting imagination, blending themes, tone, and unique twists
2. List 2-5+ anchor points in CHRONOLOGICAL ORDER, each bursting with potential
3. Infuse anchors with vivid fields: location, time_context, character_positions, transition_from_previous - but innovate freely

═══════════════════════════════════════════════════════════════════════════════
VALIDATION CHECKLIST (GUIDE, NOT CONSTRAINT)
═══════════════════════════════════════════════════════════════════════════════

For each anchor, aim to:
☐ LOCATION evokes senses and mood
☐ TIME_CONTEXT builds atmospheric continuity
☐ TRANSITION_FROM_PREVIOUS weaves narrative magic
☐ REQUIRED_CHARACTER_STATES allows character-driven entry
☐ RESULTING_CHARACTER_STATES sparks profound change
☐ ESTIMATED_SCENES fits the beat's epic or intimate scale
☐ If multi-scene, SCENE_BREAKDOWN inspires dramatic arcs

Generate the Act keeping the desired act word length in mind. Make sure it is compatible with the target length.
Target Length: {suggested_words} words

═══════════════════════════════════════════════════════════════════════════════
CREATIVE EXPECTATIONS (UNLEASH IMAGINATION)
═══════════════════════════════════════════════════════════════════════════════

Dream big:
- Pay off prior acts with surprising depth
- Seed future acts with tantalizing hooks  
- Balance introduction and resolution with creative flair
- Pace emotions: from quiet introspection to explosive catharsis
- Dive into character psyches: motivations, flaws, growth
- Infuse world-building, subplots, and thematic echoes for richness"""
    
    @staticmethod
    def calculate_chapter_structure(total_story_length: int, structure_name: str) -> dict:
        """
        Smart chapter structure:
        - Base ranges from total word count (original logic)
        - Adjusts min/max chapters based on narrative structure complexity
        - Output format stays EXACTLY the same as before
        """

        # 1. BASE CHAPTER TIERS (original behaviour)
        if total_story_length < 15_000:
            base = {
                "anchors_per_chapter": "1-2",
                "scenes_per_anchor": 1,
                "min_chapters": 2,
                "max_chapters": 4
            }
        elif total_story_length < 40_000:
            base = {
                "anchors_per_chapter": "2-3",
                "scenes_per_anchor": 1,
                "min_chapters": 4,
                "max_chapters": 6
            }
        elif total_story_length < 75_000:
            base = {
                "anchors_per_chapter": "3-4",
                "scenes_per_anchor": 1,
                "min_chapters": 6,
                "max_chapters": 10
            }
        else:
            base = {
                "anchors_per_chapter": "4-5",
                "scenes_per_anchor": 2,
                "min_chapters": 8,
                "max_chapters": 16
            }

        # 2. STRUCTURE-BASED WEIGHTING
        # More complex structures get more chapters
        structure_complexity = {
            "Three Act Structure": 1.0,          # simple, balanced
            "Fichtean Curve": 1.0,               # still simple
            "Save the Cat Beat Sheet": 1.1,      # midpoint & beats = slightly more
            "Dan Harmon's Story Circle": 1.15,   # cycles & internal change = more
            "Seven-Point Story Structure": 1.2,  # more plot points, expand slightly
            "The Hero's Journey": 1.3,           # 12-step monomyth → needs space
            "Freytag's Pyramid": 1.3             # exposition → climax → fall → denouement
        }

        factor = structure_complexity.get(structure_name, 1.0)

        # 3. APPLY STRUCTURE WEIGHT TO CHAPTER RANGE
        min_ch = round(base["min_chapters"] * factor)
        max_ch = round(base["max_chapters"] * factor)

        # hard limits (keep logic sane)
        min_ch = max(1, min_ch)
        max_ch = max(min_ch + 1, max_ch)

        # 4. RETURN IN ORIGINAL FORMAT
        return {
            "anchors_per_chapter": base["anchors_per_chapter"],
            "scenes_per_anchor": base["scenes_per_anchor"],
            "min_chapters": min_ch,
            "max_chapters": max_ch
        }

        
    async def plan_act(
        self,
        story_title: str,
        act_number: int,
        model: str = "None"
    ) -> Tuple[ActPlan, dict]:
        """Plan complete act with rich chapter seeds"""
        
        print(f"\n📋 PLANNING ACT {act_number}")
        print("=" * 60)
        
        print("📚 Loading story elements...")
        
        final_plot_json = await self.memory.get_long_term_document(
            metadata={'type': 'expanded_plot_outline', 'story_title': story_title}
        )
        final_plot = json.loads(final_plot_json)
        
        agents_json = await self.memory.get_long_term_document(
            metadata={'type': 'connected_agents', 'story_title': story_title}
        )
        agents = json.loads(agents_json)
        
        world_json = await self.memory.get_long_term_document(
            metadata={'type': 'integrated_world', 'story_title': story_title}
        )
        world = json.loads(world_json)
        
        conflict_json = await self.memory.get_long_term_document(
            metadata={'type': 'conflict_matrix', 'story_title': story_title}
        )
        conflict_matrix = json.loads(conflict_json)

        tracker = await self.tracker_manager.get_or_create_tracker(
            self.memory, story_title
        )
        
        seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': story_title}
        )
        seed = json.loads(seed_json)
        
        progress = await self.memory.get_story_progress()
        current_word_count = progress.get('story_word_count', 0)
        target_total = final_plot.get('target_length', seed.get('target_length', 50000))
        remaining_words = target_total - current_word_count
        
        print(f"✓ Story progress: {current_word_count:,} / {target_total:,} words")
        
        previous_acts = {}
        for prev_act_num in range(1, act_number):
            try:
                act_doc = await self.memory.get_long_term_document(
                    metadata={'type': 'act_plan', 'act_id': prev_act_num, 'story_title': story_title}
                )
                if act_doc:
                    act_data = json.loads(act_doc)
                    act_plan_model = ActPlan(**act_data)
                    previous_acts[prev_act_num] = act_plan_model
            except Exception as e:
                print(f"Could not load Act {prev_act_num}: {e}")

        min_age = progress.get("min_age", 13)
        total_acts = len(final_plot.get('act_summaries', [])) or 3
        print("TOTAL ACTS ", total_acts)
        structure_name = seed['story_structure']
        percentages_and_guidances = self.get_act_percentages_and_guidance(structure_name, total_acts)
        word_percentage, act_guidance = percentages_and_guidances[act_number - 1]
        
        if act_number == total_acts:
            suggested_word_count = remaining_words
        else:
            suggested_word_count = int(remaining_words * word_percentage)

        structure_config = self.calculate_chapter_structure(
            total_story_length=target_total,
            structure_name=structure_name
        )

        # suggested_chapters = max(
        #     structure_config["min_chapters"],
        #     min(
        #         structure_config["max_chapters"],
        #         suggested_word_count
        #     )
        # )
        print(f"Act {act_number} guidance: {act_guidance}")
        print(f"Target: ~{suggested_word_count:,} ") #words across ~{suggested_chapters} chapters

        latest_chapter = progress.get('latest_chapter_id', 0)
        story_so_far = await self.memory.get_author_context(
            current_act_number=act_number
        )
        if not story_so_far:
            story_so_far = "Start of story."
        
        # Generate act plan
        act_context = ActContextBuilder.build_act_context(
            act_number=act_number,
            final_plot=final_plot,
            seed=seed,
            connected_agents=agents,
            integrated_world=world,
            conflict_matrix=conflict_matrix,
            story_tracker={str(k): v.model_dump() for k, v in tracker.items()},
            previous_acts={num: plan.model_dump() for num, plan in previous_acts.items()},       
            story_so_far=story_so_far,
            word_guidance=(word_percentage, act_guidance)
        )
        
        system_prompt = self.get_enhanced_act_planning_system_prompt(
            act_number=act_number,
            total_acts=total_acts,
            act_guidance=act_guidance,
            themes=seed.get('themes', []),
            tone=seed.get('tone', 'Balanced'),
            prose_style=seed.get('prose_style', 'Standard'),
            target_act_word_count=suggested_word_count,
            word_count=target_total,
            min_age=min_age,
            structure_config=structure_config
        )
        
        human_prompt = self.get_enhanced_act_planning_human_prompt(
            chapter_number=latest_chapter,
            context_str=act_context,
            act_number=act_number,
            total_acts=total_acts,
            suggested_words=suggested_word_count
        )

        # -------------------------------
        # LLM CALL + PARSE + VALIDATE LOOP
        # -------------------------------
        max_retries = 4
        attempt = 0
        # last_tokens = {}
        last_plan = None

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 Act Planning LLM Attempt {attempt}/{max_retries + 1}")

            response, tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.8,
                model=model
            )
            # last_tokens = tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON ---
            try:
                act_plan = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ActPlan
                )
            except Exception as e:
                print(f"⚠️ JSON parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            last_plan = act_plan

            # --- Validation #1: Chapter count ---
            chapter_count = len(act_plan.chapter_outlines)
            print(f"→ Generated {chapter_count} chapters")

            if chapter_count == 0:
                print(f"⚠️ Chapter count is zero. Retrying...")
                await asyncio.sleep(0.8 * attempt)
                continue

            # --- Validation #2: Non-zero word counts ---
            missing = [c for c in act_plan.chapter_outlines if not c.target_word_count]

            if missing:
                per_chapter = suggested_word_count / len(act_plan.chapter_outlines)
                for c in missing:
                    c.target_word_count = per_chapter

            # if zero_word_chapters:
            #     print(f"⚠️ Chapters with zero target_word_count: {zero_word_chapters}. Retrying...")
            #     await asyncio.sleep(0.8 * attempt)
            #     continue

            # --- Passed all checks ---
            print("✓ Act validated. Correct chapter count + valid word counts.")
            break

        # If retries exhausted but we still have something, return last valid plan
        if attempt > max_retries and last_plan:
            print("🚨 Max retries reached — returning last generated act plan (may be incomplete).")
            act_plan = last_plan


        # response, tokens = await better_author_client(
        #     system_prompt=system_prompt,
        #     human_prompt=human_prompt,
        #     llm_temp=0.8,
        #     model=model
        # )
        # clean_resp = StoryHelpers._extract_content(response)
        # clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        # Validate anchor completeness
        def validate_anchor_fields(act_plan: ActPlan) -> bool:
            """Check that all anchors have required spatial/temporal fields"""
            required_fields = ['location', 'time_context', 'character_positions', 'transition_from_previous']
            
            for chapter in act_plan.chapter_outlines:
                for anchor in chapter.anchor_points:
                    for field in required_fields:
                        if not getattr(anchor, field, None):
                            if field == 'transition_from_previous' and anchor.anchor_number == 1:
                                continue
                            print(f"⚠️  Warning: Anchor {anchor.anchor_number} missing '{field}'")
                            return False
            return True


        if not validate_anchor_fields(act_plan):
            print("⚠️  Some anchors missing required fields - may cause Director issues")

        print(f"📊 Updating story tracker for Act {act_number}...")
        
        act_tracking, tracker_tokens = await self.tracker_manager.update_tracker_after_act(
            story_title=story_title,
            act_number=act_number,
            act_plan=act_plan,
            author_context=story_so_far,
            model=model
        )
        tracker[act_number] = act_tracking     
        await self.tracker_manager.save_tracker(
            self.memory, story_title, tracker
        )
        total_tokens = {
            "prompt_tokens": tokens.get("prompt_tokens", 0) + tracker_tokens.get("prompt_tokens", 0),
            "completion_tokens": tokens.get("completion_tokens", 0) + tracker_tokens.get("completion_tokens", 0),
            "total_tokens": 0
        }
        total_tokens["total_tokens"] = total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]
        
        # Store in memory
        await self.memory.add_long_term_document(
            text=act_plan.model_dump_json(indent=2),
            metadata={"type": "act_plan", "act_id": act_number, "story_title": story_title}
        )
        
        print("=" * 60)
        print(f"✅ ACT {act_number} PLANNED: '{act_plan.act_title}'")
        print(f"📖 {len(act_plan.chapter_outlines)} chapters outlined")
        print(f"📊 Tracker updated:")
        print(f"   Setup: {len(act_tracking.setup_elements)} elements")
        print(f"   Payoff: {len(act_tracking.payoff_elements)} elements")
        print(f"   Agent moments: {len(act_tracking.key_agent_moments)}")
        print("=" * 60)
        
        return act_plan, total_tokens
    




# example user context
# user_context = {
#     'Genre': ['Fantasy', 'Adventure'],
#     'Sub-Genre': ['Epic Fantasy'],
#     'Tone': 'Hopeful yet dark',
#     'POV': 'Third-person limited',
#     'Length': 'Novella (25,000 - 40,000 words)',
#     'Setting': 'A world where magic is dying',
#     'Additional Themes': ['sacrifice', 'legacy', 'power corrupts'],
#     'target_audience_age': '16',
#     'Guide Prose': 'Lyrical with vivid imagery',
#     'protagonist_name': 'Kael',
#     'protagonist_age': '19',
#     'protagonist_archetype': 'Reluctant hero',
#     'protagonist_desire': 'To save his village',
#     'protagonist_fear': 'Becoming like his father'
# }

