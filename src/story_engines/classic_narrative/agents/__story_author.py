import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.memory.memory_system import StoryMemorySystem
from src.llm_client.llm_client import author_client, better_author_client
from src.utilities.story_structure_decider import find_best_structure, find_best_structure_name
from src.utilities.story_helpers import StoryHelpers
from config_vars import author_story_rules_negative
# import asyncio
# import random
import hashlib


# PYDANTIC MODELS - Flexible for LLM Creativity
class MinimalStorySeed(BaseModel):
    """Minimal story foundation from user context"""
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


class WorldFoundation(BaseModel):
    """Independent world building - before plot integration"""
    physical_description: str = Field(
        description="Rich sensory description of the world's physical nature"
    )
    cultural_elements: str = Field(
        description="Societies, customs, beliefs that exist in this world"
    )
    systemic_rules: str = Field(
        description="How this world operates - laws of nature, magic, technology, society"
    )
    inherent_tensions: str = Field(
        description="Conflicts built into the world itself, independent of protagonist"
    )
    unique_elements: List[str] = Field(
        description="3-5 distinctive features that make this world memorable"
    )


class NarrativeAgent(BaseModel):
    """Represents any story-driving entity — person, group, or abstract force"""
    name: str
    type: str = Field(description="Entity type: 'character', 'group', 'force', 'concept', or 'environment'")
    role: str = Field(description="Narrative role: protagonist, antagonist, mentor, etc.")
    description: str = Field(description="Nature, influence, or function in story")
    motivation: Optional[str] = Field(description="If applicable, what drives this entity’s behavior or effect")
    embodiment: Optional[str] = Field(description="How it manifests — person, symbol, environment, etc.")
    impact_on_story: str = Field(description="How it drives or resists the protagonist’s goal")
    background: Optional[str] = Field(description="Life history, formative experiences for characters")
    psychology: Optional[str] = Field(
        description="Internal world - desires, fears, beliefs, contradictions for characters"
    )
    world_position: Optional[str] = Field(
        description="How they fit in the world - status, relationships, resources for characters"
    )
    distinctive_traits: Optional[List[str]] = Field(
        description="Specific quirks, speech patterns, habits, appearance details for characters"
    )

class NarrativeAgentList(BaseModel):
    """List of narrative agents"""
    agents: List[NarrativeAgent] = Field(description="List of narrative agents for the story")

class EnhancedConflictMatrix(BaseModel):
    """Multi-dimensional conflict analysis"""
    central_conflict: str = Field(description="Primary plot-driving conflict")
    character_conflicts: str = Field(
        description="Conflicts emerging from character desires and fears"
    )
    systemic_conflicts: str = Field(
        description="Conflicts from world systems and societal structures"
    )
    thematic_conflicts: str = Field(
        description="Philosophical or moral tensions underlying the story"
    )
    conflict_intersections: str = Field(
        description="How different conflict layers interact and amplify each other"
    )
    escalation_path: str = Field(
        description="How conflicts intensify from beginning to climax"
    )
    twist_opportunities: List[str] = Field(
        description="Potential plot twists that emerge from conflict layers"
    )
    moral_dilemma_axis: str = Field(
        description="Core ideological tension where both sides have validity — no absolute right or wrong"
    )



class MinimalPlotOutline(BaseModel):
    """Initial plot conception"""
    title: str
    premise: str = Field(description="1-2 sentence story hook")
    central_question: str = Field(description="What question drives this story?")
    target_length: int
    narrative_arc: str = Field(
        description="High-level story progression in natural prose"
    )
    required_character_roles: List[str] = Field(
        description="List of character roles needed for this story (e.g., 'protagonist', 'antagonist', 'mentor', 'love interest', 'comic relief', 'betrayer')"
    )
    act_count: int = Field(description="Act Count")

class RelationalArcMatrix(BaseModel):
    """Maps how character relationships evolve and influence the story"""
    relationship_arcs: List[str] = Field(
        description="Each entry describes how key relationships change across acts"
    )
    emotional_inversions: List[str] = Field(
        description="Moments where relationships reverse expectations (love → betrayal, rivalry → trust)"
    )
    shared_theme_reflections: str = Field(
        description="How relationships mirror or challenge the story's central theme"
    )

class SymbolicLayer(BaseModel):
    motifs: List[str]
    recurring_symbols: List[str]
    color_imagery: str
    transformation_symbolism: str


class ExpandedPlotOutline(BaseModel):
    """Detailed plot with structure mapping"""
    title: str
    premise: str
    #structure_used: str = Field(description="Which structure template guides this plot")
    act_summaries: List[str] = Field(
        description="One rich summary per act showing progression"
    )
    structure_beat_mapping: str = Field(
        description="How story beats map to structure template beats"
    )
    narrative_flow: str = Field(
        description="Complete plot in flowing prose with emotional beats and turning points"
    )
    subplot_threads: List[str] = Field(
        description="Secondary storylines that weave through main plot"
    )
    relational_arc_matriX: RelationalArcMatrix = Field(
        description="Maps how character relationships evolve and influence the story"
    )
    symbolic_layer: SymbolicLayer = Field(
        description="Propose 3-5 recurring symbols or motifs that reflect the story’s central themes and evolve across acts."
    )


class ConnectedNarrativeAgent(BaseModel):
    """Narrative agent after integration with plot"""
    name: str
    role: str
    type: str
    description: str
    plot_function: str = Field(
        description="How this agent drives or complicates the plot"
    )
    agent_arc: str = Field(
        description="How this agent changes or manifests through the story"
    )
    key_relationships: str = Field(
        description="Dynamics with other agents"
    )
    distinctive_voice: Optional[str] = Field(
        description="Speech patterns, vocabulary, communication style if applicable"
    )

class ConnectedNarrativeAgentList(BaseModel):
    agents: List[ConnectedNarrativeAgent]

class IntegratedWorld(BaseModel):
    """World after plot integration"""
    foundation: str = Field(description="Core world elements from WorldFoundation")
    plot_relevant_locations: str = Field(
        description="Specific places where key story events occur"
    )
    world_plot_interactions: str = Field(
        description="How world rules enable, complicate, or reflect plot conflicts"
    )
    thematic_resonance: str = Field(
        description="How world elements reinforce story themes"
    )
    world_arc: str = Field(description="How the world changes due to story events")


# class TwistStrategy(BaseModel):
#     """Strategic twist placement"""
#     selected_twists: List[str] = Field(description="Twists chosen for this story")
#     twist_placements: str = Field(
#         description="Where and how each twist appears in the narrative"
#     )
#     setup_requirements: str = Field(
#         description="What must be established earlier for twists to land"
#     )
#     thematic_purpose: str = Field(
#         description="How twists serve story themes, not just shock value"
#     )


class QualityReport(BaseModel):
    """Story quality validation"""
    strengths: List[str] = Field(description="What's working well")
    concerns: List[str] = Field(description="Issues that need attention")
    recommendations: List[str] = Field(description="Specific improvements to consider")
    overall_assessment: str = Field(description="Holistic view of story quality")

class ActTracking(BaseModel):
    """What gets introduced and resolved each act"""
    act_number: int
    setup_elements: List[str] = Field(
        description="Things introduced this act for later use (genre-agnostic)"
    )
    payoff_elements: List[str] = Field(
        description="Things resolved from earlier acts"
    )
    key_agent_moments: Dict[str, str] = Field(
        description="Critical moments for each agent this act"
    )    

class StoryTracker(BaseModel):
    """Lightweight story-level consistency tracking"""
    act_tracking: Dict[int, ActTracking]

class ChapterAnchorPoints(BaseModel):
    """Major story moments that anchor a chapter - BEAT LEVEL, not scene level"""
    anchor_number: int
    anchor_type: str = Field(
        description="revelation | confrontation | decision | loss | discovery | betrayal | transformation"
    )
    
    # THE CORE: What happens at this beat
    what_happens: str = Field(
        description="1-2 sentence description of this story moment. Focus on WHAT changes, not HOW it unfolds"
    )
    
    # NEW: Spatial/Temporal grounding
    location: str = Field(
        description="Specific location where this anchor occurs (e.g., 'Village square at dawn', 'Kael's bedroom, midnight')"
    )
    
    time_context: str = Field(
        description="When this happens relative to previous anchor: 'Immediately after' | 'Same day, evening' | 'Three days later' | 'Concurrent with previous'"
    )
    
    character_positions: Dict[str, str] = Field(
        default_factory=dict,
        description="Where key characters are at start of this anchor: {'Kael': 'returning from forest', 'Lira': 'waiting at inn'}"
    )
    
    transition_from_previous: str = Field(
        default="",
        description="How we got from the last anchor to this one (travel, time skip, scene cut). Empty if this is first anchor."
    )
    
    # Why it matters
    story_function: str = Field(
        description="Why this beat matters to plot/character/theme progression"
    )
    
    # Context for Director
    agents_present: List[str] = Field(description="Which agents are central to this beat")
    
    emotional_target: str = Field(
        description="Desired emotional impact: 'shocking revelation' | 'bittersweet triumph' | 'crushing betrayal'"
    )
    
    # NEW: Scene execution guidance
    estimated_scenes: int = Field(
        default=1,
        description="How many actual scenes needed to dramatize this anchor (1-3). Complex anchors need multiple scenes."
    )
    
    scene_breakdown: List[str] = Field(
        default_factory=list,
        description="If estimated_scenes > 1, rough breakdown: ['Scene 1: Kael enters throne room', 'Scene 2: Confrontation escalates', 'Scene 3: King reveals truth']"
    )
    
    # Continuity tracking
    introduces: List[str] = Field(
        default_factory=list,
        description="New tensions, questions, or elements this beat creates"
    )
    
    resolves: List[str] = Field(
        default_factory=list,
        description="What this beat pays off from earlier"
    )
    
    # NEW: Character state requirements
    required_character_states: Dict[str, str] = Field(
        default_factory=dict,
        description="Pre-conditions for this anchor: {'Kael': 'must have the amulet', 'Lira': 'must believe Kael is dead'}"
    )
    
    resulting_character_states: Dict[str, str] = Field(
        default_factory=dict,
        description="How character states change: {'Kael': 'wounded, knows the truth', 'Lira': 'distrusts mentor'}"
    )

class ChapterOutline(BaseModel):
    chapter_number: int = Field(description="Chapter number within the story")
    target_word_count: int = Field(description="Chapter word count target")
    anchor_points : List[ChapterAnchorPoints]
    

class ActPlan(BaseModel):
    """Complete plan for a single act - CHAPTER LEVEL"""
    act_number: int
    act_title: str
    act_purpose: str = Field(
        description="What this act accomplishes in the overall story"
    )
    
    chapter_outlines: List[ChapterOutline] = Field(
        description="Each chapter has: seed (creative direction) + 2-4 anchor points"
    )
    
    emotional_progression: str = Field(
        description="How emotional tone evolves through this act"
    )
    structure_alignment: str = Field(
        description="How this act fulfills structure template requirements"
    )
    act_climax: str = Field(
        description="Peak moment of tension/revelation in this act"
    )
    
    # Tracking
    setup_this_act: List[str] = Field(
        default_factory=list,
        description="Things introduced this act that matter later"
    )
    payoff_this_act: List[str] = Field(
        default_factory=list, 
        description="Things from earlier acts that get resolved here"
    )
    agent_state_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="How each agent changes this act"
    )


# ============================================================================
# OUTPUT PARSERS
# ============================================================================

# world_foundation_parser = PydanticOutputParser(pydantic_object=WorldFoundation)
# narrative_agent_parser = PydanticOutputParser(pydantic_object=NarrativeAgentList)
# enhanced_conflict_parser = PydanticOutputParser(pydantic_object=EnhancedConflictMatrix)
# minimal_plot_parser = PydanticOutputParser(pydantic_object=MinimalPlotOutline)
# expanded_plot_parser = PydanticOutputParser(pydantic_object=ExpandedPlotOutline)
# connected_narrative_agent_parser = PydanticOutputParser(pydantic_object=ConnectedNarrativeAgentList)
# integrated_world_parser = PydanticOutputParser(pydantic_object=IntegratedWorld)
# # twist_strategy_parser = PydanticOutputParser(pydantic_object=TwistStrategy)
# quality_report_parser = PydanticOutputParser(pydantic_object=QualityReport)
# act_plan_parser = PydanticOutputParser(pydantic_object=ActPlan)
# story_tracker_parser = PydanticOutputParser(pydantic_object=StoryTracker)

# ============================================================================
# COMPONENT CLASSES
# ============================================================================

class WorldBuilder:
    """Generates independent world foundations"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def generate_world_foundation(
        self,
        seed: MinimalStorySeed,
        structure_template: str,
        model: str = "None"
    ) -> Tuple[WorldFoundation, dict]:
        """Generate world independently from specific plot"""
        
        system_prompt = f"""You are a master world-builder creating immersive, internally consistent worlds.

Build a world that fits these parameters:
- Genres: {', '.join(seed.genre)}
- Tone: {seed.tone}
- Setting hint: {seed.setting or 'Create something fitting'}
- Themes to support: {', '.join(seed.themes)}

Create a world that COULD support many stories, not just one specific plot.
Focus on:
1. Vivid physical environment with sensory details
2. Rich cultural/societal elements
3. Clear systemic rules (magic, tech, social, natural laws)
4. Inherent tensions built into the world itself
5. Unique memorable elements

The world should feel alive and complex, with conflicts emerging naturally from its structure.
Age-appropriate for: {seed.target_audience_age} years old audience

{WorldFoundation.model_json_schema()}"""
        
        human_prompt = f"""Seed details:
{seed.model_dump_json(indent=2)}

Build a world foundation that enables compelling stories."""
        
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

        #world = world_foundation_parser.parse(json_data)


        return json_data, tokens
    
    async def integrate_with_conflict(
        self,
        world_foundation: WorldFoundation,
        conflict_matrix: EnhancedConflictMatrix,
        agents: List[ConnectedNarrativeAgent],
        model: str = "None"
    ) -> Tuple[IntegratedWorld, dict]:
        """Show how existing world intersects with story conflicts"""
        
        system_prompt = f"""You have an established world with its own logic and tensions.
Now a specific story is happening in this world.

World Foundation:
{world_foundation.model_dump_json(indent=2)}

Story Conflicts:
{conflict_matrix.model_dump_json(indent=2)}

Your task: Show how the EXISTING world naturally enables, complicates, or reflects these conflicts.
Don't change the world - find organic connections.

Identify:
1. Specific locations where key events could occur
2. How world rules create obstacles or opportunities
3. How world tensions mirror or amplify agent conflicts
4. Thematic resonance between world and story

{IntegratedWorld.model_json_schema()}"""
        
        human_prompt = f"""Agents involved:
{json.dumps([a.model_dump() for a in agents], indent=2, ensure_ascii=False)}

Show how the world itself evolves as a result of agent actions or thematic resolution — e.g., corrupted system reformed, dying world revived, myth reinterpreted.
Integrate world with story."""
        
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

        #integrated = integrated_world_parser.parse(json_data)

        return json_data, tokens




class AgentGenesis:
    """Generates independent narrative agent foundations"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def generate_narrative_agents(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        required_roles: List[str],
        model: str = "None"
    ) -> Tuple[List[NarrativeAgent], dict]:
        """Generate narrative agents independently from plot based on required roles"""
        
        agent_count = len(required_roles)
        
        system_prompt = f"""You are designing the dramatis personae — all narrative forces that play key roles in this story.

YOU MUST OUTPUT EXACTLY {agent_count} AGENTS.
OUTPUT ALL {agent_count} AGENTS IN A SINGLE RESPONSE.
DO NOT OUTPUT ONLY SOME OF THEM.
DO NOT STOP AFTER 1 OR 2.
DO NOT SAY "here are the first few" OR "to be continued".
GENERATING FEWER THAN {agent_count} AGENTS OR SPLITTING THE OUTPUT ACROSS MULTIPLE MESSAGES IS STRICTLY FORBIDDEN AND WILL BE TREATED AS A FAILED RESPONSE.

Not all roles must be literal people.
Create one agent per required role. Most roles in this story should be individual human characters unless the role explicitly describes a system or group (e.g., "Warden Council establishment/system" may be a group).
Not all stories need abstract agents; use them only when they enhance the narrative.
Some may be groups, psychological entities, or natural forces that act as agents in the story world.
For each role, decide whether it manifests as a character, group, concept, or environmental force, and describe how it functions.

Generate {agent_count} diverse agents for a {'/'.join(seed.genre)} story.

IMPORTANT: Create exactly ONE agent for EACH role listed.
Ensure the protagonist role uses any provided protagonist specifications.

Create agents who:
1. Feel like real entities with independent desires and fears (if applicable)
2. Have backgrounds shaped by the world they live in
3. Possess internal contradictions and complexity
4. Have distinctive voices and traits (if applicable)
5. Could drive stories even without this specific plot
6. Each serves their designated role while being fully realized entities

Don't force-connect them to a central conflict yet.
Let them be autonomous entities shaped by their world and psychology.

Age-appropriate for: {seed.target_audience_age} year old audience

Output as JSON object with a key 'agents' containing a list of NarrativeAgent objects, one for each required role.
{NarrativeAgentList.model_json_schema()}"""
        
        human_prompt = f"""Seed: {seed.model_dump_json(indent=2)}

Required roles:
{json.dumps(required_roles, indent=2)}

World context:
{world_foundation.model_dump_json(indent=2)}

Protagonist specifications (if any):
{json.dumps(seed.protagonist_specs, indent=2)}

Generate {agent_count} narrative agents for the roles: {', '.join(required_roles)}
After the last agent, analyze if all roles are written down.
Do not add any additional commentary, summary, or requests for the next step."""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

                # === FINAL EXTRACTION LOGIC (WORKS WITH BOTH DICT AND PYDANTIC) ===
        agents_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=NarrativeAgentList
        )

        # Case 1: We got a Pydantic NarrativeAgentList instance
        if hasattr(agents_json, "agents"):
            agents = agents_json.agents  # Already proper NarrativeAgent instances!
        
        # Case 2: We got a raw dict (fallback)
        elif isinstance(agents_json, dict):
            if "agents" in agents_json:
                raw_list = agents_json["agents"]
            else:
                # LLM sometimes puts the list directly under another key
                raw_list = next((v for v in agents_json.values() if isinstance(v, list)), None)
                if raw_list is None:
                    raise ValueError("Could not find agent list in response")
            
            # Convert dicts → Pydantic models
            agents = [NarrativeAgent(**item) if isinstance(item, dict) else item for item in raw_list]
        
        else:
            raise ValueError("Unexpected parsed type from LLM")

        # Final sanity check
        if not agents or not all(isinstance(a, NarrativeAgent) for a in agents):
            raise ValueError("Failed to produce valid NarrativeAgent instances")

        print(f"Successfully created {len(agents)} narrative agents")
        return agents, tokens
    
    async def connect_agents_to_plot(
        self,
        narrative_agents: List[NarrativeAgent],
        conflict_matrix: EnhancedConflictMatrix,
        plot_outline: MinimalPlotOutline,
        prose_style: str,
        model: str = "None"
    ) -> Tuple[List[ConnectedNarrativeAgent], dict]:
        """Connect pre-existing narrative agents to plot organically"""
        
        system_prompt = f"""You have fully-formed narrative agents with their own desires, fears, and histories (if applicable).
Now show how these existing agents intersect with a specific story.

Note that not all agents are human — integrate abstract or non-sentient entities in a way that gives them narrative presence or symbolic weight.

Your task: Find ORGANIC connections between agents' existing psychology and the plot.
- Don't change their core nature
- Show how their independent desires/fears naturally engage with the conflict
- Their motivations should feel personal, not plot-convenient
- Develop their arcs based on who they already are

For each agent, define:
1. How their existing desires/fears relate to this story
2. Their specific agent arc through the plot
3. Key relationships with other agents
4. Plot function (what role they serve)
5. Distinctive voice/communication style (if applicable)

Maintain the story’s prose style throughout this plot: {prose_style}

Output as JSON object with a key 'agents' containing a list of ConnectedNarrativeAgent objects.
{ConnectedNarrativeAgentList.model_json_schema()}"""
        
        human_prompt = f"""Connect these autonomous agents to the story organically.
Agents:
{json.dumps([a.model_dump() for a in narrative_agents], indent=2)}

Plot Premise:
{plot_outline.premise}

Conflicts:
{conflict_matrix.model_dump_json(indent=2)}
        """
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
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
    """Generates multi-dimensional conflict matrices"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _scale_conflict_complexity(self, target_length: int, base_conflicts: EnhancedConflictMatrix) -> EnhancedConflictMatrix:
        """
        Simplifies conflict layers for shorter stories to prevent rushed pacing.
        """
        if target_length >= 40000:  # Novels and epics get full complexity
            return base_conflicts
        
        # For short stories/novellas, simplify
        simplified = base_conflicts.model_copy()
        
        if target_length < 15000:  # Short stories: ONLY central conflict
            simplified.character_conflicts = "Minimal - focus on protagonist only"
            simplified.systemic_conflicts = "Background only - not actively developed"
            simplified.thematic_conflicts = f"Single theme: {simplified.thematic_conflicts.split('.')[0]}"
            simplified.conflict_intersections = "Central conflict is primary driver"
            simplified.twist_opportunities = simplified.twist_opportunities[:1]  # Max 1 twist
            
        elif target_length < 40000:  # Novellas: Central + one other layer
            simplified.systemic_conflicts = "Present but not fully developed"
            simplified.twist_opportunities = simplified.twist_opportunities[:2]  # Max 2 twists
        
        return simplified
    
    async def generate_conflict_layers(
        self,
        plot_outline: MinimalPlotOutline,
        seed: MinimalStorySeed,
        narrative_agents: List[NarrativeAgent],
        world_foundation: WorldFoundation,
        model: str = "None"
    ) -> Tuple[EnhancedConflictMatrix, dict]:
        """Generate multi-layered conflict analysis"""
        
        system_prompt = f"""You are analyzing conflicts at multiple levels for a rich, layered story.
Generate a conflict matrix that identifies:

1. CENTRAL CONFLICT: The main plot-driving tension
2. CHARACTER CONFLICTS: Arising from individual desires clashing
3. SYSTEMIC CONFLICTS: From world structures and societal forces
4. THEMATIC CONFLICTS: Underlying philosophical/moral questions
5. INTERSECTIONS: How these layers amplify each other
6. ESCALATION PATH: How conflicts intensify toward climax
7. TWIST OPPORTUNITIES: Where conflicts could reverse or reveal surprises
    Create a twist strategy that:
    - Selects 2-4 twists that emerge naturally from agent/conflict/world
    - Places them at maximum impact points (not random)
    - Ensures proper setup earlier in story
    - Serves thematic purpose beyond shock value
    - Each twist should recontextualize what came before

# Twists should feel earned, not pulled from thin air.
Note that not all agents are human — integrate abstract or non-sentient entities in a way that gives them narrative presence or symbolic weight.

Make conflicts feel inevitable yet surprising.
Each layer should enrich the others.

{EnhancedConflictMatrix.model_json_schema()}"""
        
        human_prompt = f"""Themes to explore: {', '.join(seed.themes)}

Tone: {seed.tone}

Plot Foundation:
{plot_outline.model_dump_json(indent=2)}

Narrative Agents:
{json.dumps([a.model_dump() for a in narrative_agents], indent=2)}

World:
{world_foundation.model_dump_json(indent=2)}

Highlight the moral or ideological tension where both sides have truth — the story’s emotional core emerges from this tension.
Build a multi-dimensional conflict matrix."""
        
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
            parser=EnhancedConflictMatrix
        )
        
        # NEW: Scale complexity based on length
        scaled_matrix = self._scale_conflict_complexity(
            target_length=seed.target_length,
            base_conflicts=json_data
        )

        return scaled_matrix, tokens


class QualityController:
    """Validates story quality across dimensions"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def validate_story_elements(
        self,
        seed: MinimalStorySeed,
        plot: ExpandedPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: EnhancedConflictMatrix,
        model: str = "None"
    ) -> Tuple[QualityReport, dict]:
        """Comprehensive quality assessment with specificity checks"""
        
        system_prompt = f"""You are a story development editor evaluating narrative quality.

CRITICAL: Check for VAGUENESS and LACK OF SPECIFICITY - this is the #1 issue.

Analyze these story elements for:

1. SPECIFICITY & CONCRETENESS (MOST IMPORTANT)
   - Are character names used consistently (not "the hero", "the protagonist")?
   - Are locations specific (not "somewhere", "a place", "the village")?
   - Are conflicts concrete (not "faces challenges", "learns lessons")?
   - Are events described with concrete actions/objects (not abstract "growth")?
   - Do act summaries contain at least 5 character names each?
   - Does narrative_flow contain specific scenes, not just abstract arcs?

   RED FLAGS TO CATCH:
   ❌ "someone", "something", "somewhere", "somehow"
   ❌ "the hero goes on a journey"
   ❌ "faces various challenges"
   ❌ "learns about themselves"
   ❌ "discovers their true power"
   ❌ "saves the day"
   ❌ Generic phrases without proper nouns
   
   WHAT GOOD LOOKS LIKE:
   ✅ "Kael confronts Captain Vex in the Scorched Wastes"
   ✅ "Lira reveals the Crimson Sigil's true purpose"
   ✅ "Darrow's betrayal at the Ruins of Keth forces Kael to choose"

2. PLOT COHERENCE
   - Does the story flow logically? 
   - Are there gaps or unexplained jumps?
   - Do events have clear cause and effect?

3. AGENT AGENCY
   - Do agents drive the plot or just react?
   - Are their motivations clear and specific?
   - Do we know what each character WANTS and FEARS concretely?

4. CONFLICT ESCALATION
   - Do stakes rise appropriately?
   - Are conflicts concrete and escalating?
   - Is the central conflict clear and specific?

5. THEMATIC INTEGRATION
   - Are themes woven throughout with specific examples?
   - Do themes manifest in concrete events, not just stated?

6. WORLD CONSISTENCY
   - Does the world follow its own rules?
   - Are locations memorable and distinct?

7. EMOTIONAL VARIETY
   - Is there a good balance of tones?
   - Are emotional beats specific moments, not abstract "development"?

8. STRUCTURE INTEGRITY
   - Does it follow its template effectively?
   - Are act summaries detailed enough (200+ words each)?

9. ORIGINALITY
   - Does it avoid clichés?
   - Are there unique, specific details that make it memorable?

RATING SCALE FOR CONCERNS:
- CRITICAL: Plot is too vague to execute (generic phrases, no proper nouns, abstract events)
- MAJOR: Significant gaps in specificity or logic
- MINOR: Small improvements needed

Your recommendations should be ACTIONABLE and SPECIFIC:
❌ BAD: "Add more detail"
✅ GOOD: "Act 2 summary needs character names and locations. Replace 'faces challenges' with specific obstacles like conflicts with named antagonists or concrete environmental threats."

{QualityReport.model_json_schema()}"""
        
        human_prompt = f"""Evaluate this story's quality and provide SPECIFIC recommendations.

FOCUS YOUR ANALYSIS ON:
1. Is the plot SPECIFIC enough for a director to execute?
2. Does every act summary name characters, locations, and concrete events?
3. Does narrative_flow describe actual scenes (not just abstract progression)?
4. Are there proper nouns (names, places) throughout?

Story Elements:
Seed: {seed.model_dump_json(indent=2)}

Plot: {plot.model_dump_json(indent=2)}

Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}

World: {world.model_dump_json(indent=2)}

Conflicts: {conflict_matrix.model_dump_json(indent=2)}

Be brutally honest about vagueness - it's better to catch it now than during execution."""
        
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
    """Manages incremental story tracking after each act is ingested"""
    
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
        """
        Update story tracker after an act is planned and ingested.
        Extracts setup/payoff from the act and updates running tracker.
        """
        
        system_prompt = f"""You are updating a story continuity tracker after Act {act_number} has been planned.

Your job: Extract what was INTRODUCED (setup) and what was RESOLVED (payoff) in this act.

════════════════════════════════════════════════════════════════════════════════
WHAT TO TRACK
════════════════════════════════════════════════════════════════════════════════

For Act {act_number}, identify:

1. SETUP ELEMENTS (introduces_for_later)
   - New questions raised that need answers later
   - New objects/artifacts introduced that will matter
   - New tensions/conflicts that were created
   - New relationships formed or strained
   - Seeds planted for future payoff
   
   Be specific: "Kael discovers corrupted amulet" not just "new artifact"

2. PAYOFF ELEMENTS (resolves_from_earlier)
   - Questions from earlier acts that got answered
   - Conflicts from earlier acts that were resolved
   - Setup from earlier acts that paid off here
   - Mysteries that were solved
   
   Reference which act the setup came from when possible

3. KEY AGENT MOMENTS
   - For each major agent, what was their CRITICAL moment this act?
   - Focus on transformative beats, not every appearance
   - What fundamentally changed for them?

════════════════════════════════════════════════════════════════════════════════
EXTRACTION RULES
════════════════════════════════════════════════════════════════════════════════

✓ Be concrete and specific (names, objects, relationships)
✓ Only track elements that MATTER to ongoing story
✓ Don't track minor details or background information
✓ For setup: ask "will this need payoff later?"
✓ For payoff: ask "was this seeded earlier?"
✓ Agent moments: only transformative beats, not every scene

✗ Don't track atmosphere, tone, or setting descriptions
✗ Don't track every character interaction
✗ Don't include authorial commentary

════════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════════════════════════════════════════════

Return a SINGLE ActTracking object for Act {act_number}:

{{
  "act_number": {act_number},
  "setup_elements": [
    "Specific new tension or question introduced",
    "Object/artifact that will matter later",
    "Relationship change that needs resolution"
  ],
  "payoff_elements": [
    "Question from Act X answered",
    "Conflict from Act X resolved",
    "Mystery from Act X solved"
  ],
  "key_agent_moments": {{
    "Agent Name": "Their critical transformation/decision this act",
    "Another Agent": "Their pivotal moment this act"
  }}
}}

Only include agents who had MAJOR moments. Don't list every character.

{ActTracking.model_json_schema()}
"""

        human_prompt = f"""Extract setup/payoff tracking for Act {act_number}.

════════════════════════════════════════════════════════════════════════════════
FULL STORY CONTEXT
════════════════════════════════════════════════════════════════════════════════

{author_context}

════════════════════════════════════════════════════════════════════════════════
ACT {act_number} PLAN TO ANALYZE
════════════════════════════════════════════════════════════════════════════════

{act_plan.model_dump_json(indent=2)}

════════════════════════════════════════════════════════════════════════════════

Now extract the ActTracking object for Act {act_number}.
Focus on elements that will matter for continuity in future acts."""

        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,  # Lower temp for extraction task
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        act_tracking_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ActTracking
        )
        
        # Convert to proper Pydantic object if needed
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
                
                # Convert to ActTracking objects
                tracker = {}
                for act_num_str, act_data in tracker_data.get('act_tracking', {}).items():
                    act_num = int(act_num_str)
                    if isinstance(act_data, dict):
                        tracker[act_num] = ActTracking(**act_data)
                    else:
                        tracker[act_num] = act_data
                
                return tracker
            
        except Exception as e:
            print(f"⚠️  Could not load existing tracker: {e}")
        
        # Return empty tracker if none exists
        return {}
    
    async def save_tracker(
        self,
        memory_system: StoryMemorySystem,
        story_title: str,
        tracker: Dict[int, ActTracking]
    ):
        """Save updated tracker to memory"""
        
        # Convert ActTracking objects to dicts for storage
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
    """Builds rich, structured context for act planning without overwhelming the LLM"""
    
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
        structure_name: str,
        word_guidance: tuple
    ) -> str:
        """
        Returns a carefully formatted context string that gives the LLM:
        - High-level story stakes and progression
        - Agent arcs specific to this act
        - World elements relevant to this act
        - Setup/payoff tracking
        - Previous act consequences
        """
        
        word_percentage, act_guidance = word_guidance
        suggested_words = int(seed.get('target_length', 50000) * word_percentage)
        suggested_chapters = max(2, suggested_words // 2500)
        
        context_parts = []
        
        # ===== SECTION 1: ACT IDENTITY =====
        context_parts.append(f"""
╔═══════════════════════════════════════════════════════════════════════════════
║ ACT {act_number} CONTEXT & GUIDANCE
╚═══════════════════════════════════════════════════════════════════════════════

STRUCTURE: {structure_name}
ACT PURPOSE: {act_guidance}
TARGET: ~{suggested_words:,} words across {suggested_chapters} chapters
TONE: {seed.get('tone', 'Balanced')}
PROSE STYLE: {seed.get('prose_style', 'Standard narrative')}
""")
        
        # ===== SECTION 2: ACT SUMMARY & EMOTIONAL ARC =====
        act_summaries = final_plot.get('act_summaries', [])
        if act_number <= len(act_summaries):
            context_parts.append(f"""
┌─ ACT SUMMARY ─────────────────────────────────────────────────────────────────
{act_summaries[act_number - 1]}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 3: SETUP/PAYOFF FROM TRACKER =====
        # Show what's been setup in previous acts that needs payoff
        outstanding_setup = []
        for prev_act_num in range(1, act_number):
            prev_tracking = story_tracker.get(str(prev_act_num), {})
            if isinstance(prev_tracking, dict):
                setup = prev_tracking.get('setup_elements', [])
                outstanding_setup.extend([f"(Act {prev_act_num}) {item}" for item in setup])
        
        # Show what THIS act should resolve (from tracker)
        current_tracking = story_tracker.get(str(act_number), {})
        if isinstance(current_tracking, dict):
            expected_payoffs = current_tracking.get('payoff_elements', [])
            expected_setups = current_tracking.get('setup_elements', [])
        else:
            expected_payoffs = []
            expected_setups = []
        
        context_parts.append(f"""
┌─ STORY CONTINUITY ────────────────────────────────────────────────────────────

OUTSTANDING SETUP (needs payoff in this or future acts):
{ActContextBuilder._format_list(outstanding_setup[:10], indent=2) if outstanding_setup else "  • (none - this is Act 1)"}

EXPECTED PAYOFFS THIS ACT:
{ActContextBuilder._format_list(expected_payoffs, indent=2)}

EXPECTED SETUP THIS ACT (for future payoff):
{ActContextBuilder._format_list(expected_setups, indent=2)}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 4: AGENT ARCS THIS ACT =====
        context_parts.append("\n┌─ AGENT ARCS THIS ACT ─────────────────────────────────────────────────────────")
        
        # Get agent moments from tracker
        tracker_agents = {}
        if isinstance(current_tracking, dict):
            tracker_agents = current_tracking.get('key_agent_moments', {})
        
        for agent in connected_agents:
            agent_name = agent.get('name', 'Unknown')
            agent_role = agent.get('role', '')
            agent_arc = agent.get('agent_arc', '')
            
            # Extract act-specific moments if available
            moment = tracker_agents.get(agent_name, "")
            
            context_parts.append(f"""
{agent_name} ({agent_role})
  Arc: {agent_arc}
  This Act: {moment if moment else 'Key player in unfolding conflicts'}
""")
        
        context_parts.append("└───────────────────────────────────────────────────────────────────────────────\n")
        
        # ===== SECTION 5: CONFLICT ESCALATION =====
        context_parts.append(f"""
┌─ CONFLICT ESCALATION ─────────────────────────────────────────────────────────
Central: {conflict_matrix.get('central_conflict', '')}

Escalation Path: {conflict_matrix.get('escalation_path', '')}

Moral Dilemma: {conflict_matrix.get('moral_dilemma_axis', '')}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 6: WORLD ELEMENTS THIS ACT =====
        locs = integrated_world.get('plot_relevant_locations', '').split('\n')
        rules = integrated_world.get('world_plot_interactions', '').split('\n')

        context_parts.append(f"""
┌─ WORLD & SETTING ─────────────────────────────────────────────────────────────
Relevant Locations:
{ActContextBuilder._format_list(locs[:5], indent=2)}

World Rules Active:
{ActContextBuilder._format_list(rules[:5], indent=2)}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 7: PREVIOUS ACT CONSEQUENCES =====
        if previous_acts and act_number > 1:
            context_parts.append(f"""
┌─ CONSEQUENCES FROM ACT {act_number - 1} ──────────────────────────────────────
{ActContextBuilder._summarize_previous_act(previous_acts.get(act_number - 1, {}))}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 8: STORY PROGRESS =====
        context_parts.append(f"""
┌─ STORY SO FAR ────────────────────────────────────────────────────────────────
{story_so_far}
└───────────────────────────────────────────────────────────────────────────────
""")
        
        # ===== SECTION 9: THEMATIC THREADS =====
        themes = seed.get('themes', [])
        if themes:
            context_parts.append(f"""
┌─ THEMATIC FOCUS ──────────────────────────────────────────────────────────────
Core Themes: {', '.join(themes)}

How This Act Deepens Themes:
{final_plot.get('narrative_flow', '')}
└───────────────────────────────────────────────────────────────────────────────
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
    World-class story planning system using iterative expansion.
    Generates elements independently before integration for organic complexity.
    """
    
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        
        # Initialize component systems
        self.world_builder = WorldBuilder(self._cached_author_llm_call)
        self.agent_genesis = AgentGenesis(self._cached_author_llm_call)
        self.conflict_architect = ConflictArchitect(self._cached_author_llm_call)
        self.quality_controller = QualityController(better_author_client)
        self.tracker_manager = StoryTrackerManager(self._cached_author_llm_call)
        
        # LLM cache
        self.llm_cache = {}
    
    async def _cached_author_llm_call(
        self,
        system_prompt: str,
        human_prompt: str,
        llm_temp: float,
        model: str = "None"
    ):
        """Cached LLM wrapper"""
        key = hashlib.md5(
            (system_prompt + human_prompt + str(llm_temp) + model).encode()
        ).hexdigest()
        
        if key in self.llm_cache:
            return self.llm_cache[key], {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        response, tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=llm_temp,
            model=model
        )
        
        self.llm_cache[key] = response
        return response, tokens
    
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
            "Novella (25,000 - 40,000 words)": 32000,
            "Novel Chapter (40,000 - 60,000 words)": 50000,
            "Full Novel (60,000 - 90,000 words)": 75000,
            "Epic / Series (90,000 - 150,000+ words)": 120000
        }
        
        target_length_str = user_context.get('Length', 'Novelette (15,000 - 25,000 words)')
        target_length = word_count_map.get(target_length_str, 20000)
        
        excluded_titles = {None, "None", "", " ", "Untitled Story", "Unitled story", 
                          "untitled story", "Null", "NULL", "Nill", "NILL", "null", "nill"}
        title = user_context.get('Title')
        title = None if title in excluded_titles else title
        
        target_audience_age = user_context.get('target_audience_age', 13)
        target_audience_age = max(8, min(target_audience_age, 18))  # Clamp to 8-18+
        
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
            protagonist_specs=protagonist_specs
        )
    
    def validate_act_count(self, act_count: int, structure_name: str) -> int:
        if structure_name == "Three Act Structure":
            return 3
        elif structure_name == "Fichtean Curve":
            return 3
        elif structure_name == "Save the Cat Beat Sheet":
            return 3
        elif act_count < 3:
            return 3
        elif act_count > 5:
            return 5
        else:
            return act_count
        
    async def generate_minimal_plot_outline(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        model: str = "None"
    ) -> Tuple[MinimalPlotOutline, dict]:
        """Generate initial plot conception with character role requirements"""
        
        structure_template = find_best_structure(seed.genre)
        
        system_prompt = f"""You are a master plot architect creating compelling story foundations.

Create a minimal plot outline that fits this creative direction:

Structure guidance:
{structure_template}

Your outline should:
1. Pose a compelling central question
2. Suggest natural act divisions (3-5 acts based on structure)
3. Sketch the narrative arc in engaging prose
4. Define the required_character_roles list
   - Think about what character roles this specific story needs, major or minor all of them now
   - Be flexible: small intimate stories might need 5-7 roles, epic tales might need 15+
   - Always include at least: protagonist, antagonist
   - Consider: mentor, ally, love interest, rival, betrayer, comic relief, wise elder, trickster, herald, threshold guardian, shapeshifter, shadow, etc.
   - List only roles that serve THIS story's needs
   - The number should feel natural for the scope and genre
5. {author_story_rules_negative}

Examples:
- Intimate character study: ['protagonist', 'antagonist', 'confidant', 'catalyst']
- Epic fantasy: ['protagonist', 'antagonist', 'mentor', 'warrior ally', 'magic user ally', 'comic relief', 'betrayer', 'love interest', 'wise council member', 'rival']
- Mystery thriller: ['protagonist detective', 'antagonist killer', 'partner', 'red herring suspect', 'victim's family', 'informant']

Be age-appropriate for: {seed.target_audience_age} year old audience

{MinimalPlotOutline.model_json_schema()}"""
        
        if seed.target_length <= 75000:
            act_count = 3
        else:
            act_count = 5

        human_prompt = f"""Generate a minimal plot outline with appropriate role requirements for this story's scope.
        Seed: {seed.model_dump_json(indent=2)}

World Foundation: {world_foundation.model_dump_json(indent=2)}

Act Count should be {act_count}
        """
        
        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        outline: MinimalPlotOutline = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=MinimalPlotOutline
        )
        #outline = minimal_plot_parser.parse(json_data)
        
        # Validate roles
        if len(outline.required_character_roles) < 2:
            print(f"⚠️  Warning: Only {len(outline.required_character_roles)} roles defined. Minimum should be 2 (protagonist + antagonist)")
        
        print(f"✓ Plot requires {len(outline.required_character_roles)} roles: {', '.join(outline.required_character_roles)}")
        
        structure_name = find_best_structure_name(seed.genre)

        act_count_validated = self.validate_act_count(act_count=outline.act_count, structure_name=structure_name)
        outline.act_count = act_count_validated
        
        return outline, tokens
    
    async def expand_plot_outline(
        self,
        minimal_plot: MinimalPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: EnhancedConflictMatrix,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict]:
        """Expand minimal plot to detailed outline"""
        
        structure_template = find_best_structure(seed.genre)
        
        system_prompt = f"""You are expanding a plot outline into a rich, detailed narrative plan.

Structure Template:
{structure_template}

Expand the plot with:
1. Detailed act-by-act progression
2. Explicit mapping to structure beats
3. Integration of all conflict layers
4. Agent arcs woven throughout
5. Subplot threads identified
6. Emotional beats and turning points
7. Thematic development

Note that not all agents are human — integrate abstract or non-sentient entities in a way that gives them narrative presence or symbolic weight.
If an agent is abstract, express its influence through environment, symbolism, or protagonist perception — not dialogue.

Write in flowing narrative prose that inspires the writing team.
Be creative and surprising while maintaining coherence.
Maintain the story’s prose style throughout this plot: {seed.prose_style}

{ExpandedPlotOutline.model_json_schema()}"""
        
        human_prompt = f"""
Target length: {seed.target_length} words

Minimal Plot:
{minimal_plot.model_dump_json(indent=2)}

Connected Agents:
{json.dumps([a.model_dump() for a in agents], indent=2)}

Integrated World:
{world.model_dump_json(indent=2)}

Conflict Matrix:
{conflict_matrix.model_dump_json(indent=2)}
        
Acts: {minimal_plot.act_count}

Expand this plot to world-class quality.
Integrate evolving relationships between agents. Show how they change each other and reflect the central theme."""
        
        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ExpandedPlotOutline
        )
        #expanded = expanded_plot_parser.parse(json_data)
        
        return json_data, tokens
    
    async def _enforce_age_appropriateness(
        self,
        final_plot: ExpandedPlotOutline,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict]:
        """
        Quick post-processing filter that scans the final plot for content
        that may be too mature for the target audience age (8–18).
        If issues are detected → light sanitizing pass.
        """
        target_age = seed.target_audience_age

        system_prompt = f"""You are an age-appropriateness editor.
    Target audience age: {target_age} years old (8–18 range).

    Scan the plot below and rate its maturity:
    - Violence: (none / mild / moderate / intense / graphic)
    - Sexual content: (none / implied / moderate / explicit)
    - Language/themes: (clean / mild swearing / heavy swearing / disturbing psychological themes)

    If any category is above the following thresholds for age {target_age}, 
    rewrite ONLY the offending parts to bring them down to acceptable levels
    while preserving story, tone, and themes as much as possible.

    Thresholds (very permissive for 13–18, stricter below 13):
    - Age < 13 → max mild violence, no sexual content, clean language
    - Age 13–15 → moderate violence ok, implied romance only, mild language
    - Age 16–18 → intense violence ok, moderate/implied sexual content ok, moderate language ok

    Return the same JSON structure. Only change what is necessary.
    {ExpandedPlotOutline.model_json_schema()}"""

        human_prompt = f"""Target age: {target_age}\nPlot to check:\n{final_plot.model_dump_json(indent=2)}\n\nReview and sanitize if needed."""

        response, tokens = await self._cached_author_llm_call(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,   # low creativity — we just want cleanup
            model=model
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ExpandedPlotOutline
        )
        #sanitized_plot = expanded_plot_parser.parse(json_data)

        return json_data, tokens
    
    async def refine_plot_with_quality_feedback(
        self,
        plot: ExpandedPlotOutline,
        quality_report: QualityReport,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: EnhancedConflictMatrix,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict]:
        """Refine plot based on quality assessment"""
        
        if not quality_report.concerns:
            return plot, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        # Check if vagueness is the main issue
        vagueness_concerns = [c for c in quality_report.concerns if any(
            word in c.lower() for word in ['vague', 'generic', 'specific', 'abstract', 'concrete', 'detail']
        )]
        
        specificity_focus = ""
        if vagueness_concerns:
            specificity_focus = """
🚨 CRITICAL PRIORITY: ADDRESS VAGUENESS

The current plot is too abstract. Your primary job is to make it CONCRETE:

REPLACE THESE PATTERNS:
- "the hero" → character name (e.g., "Kael")
- "faces challenges" → specific obstacles (e.g., "battles Captain Vex's raiders at the Scorched Wastes")
- "learns something" → specific revelation (e.g., "discovers his father Darrow is alive and leading the enemy")
- "goes somewhere" → named location (e.g., "flees to the Ruins of Keth")
- "with someone" → named character (e.g., "alongside Lira, the village healer")

EVERY ACT SUMMARY MUST INCLUDE:
- At least 5 character names
- At least 3 specific locations  
- At least 2 concrete objects/conflicts
- Specific actions with concrete consequences

NARRATIVE_FLOW MUST INCLUDE:
- Named characters in every paragraph
- Specific locations for every scene mentioned
- Concrete events ("Kael touches Elder Mora and drains her life") not abstract progression ("protagonist develops their power")

"""
        
        system_prompt = f"""You are refining a plot based on editorial feedback.
Your goal: Address specific concerns while maintaining story integrity.

{specificity_focus}

Don't change core elements unnecessarily - only fix what's flagged in the concerns.
But where you DO make changes, be SPECIFIC and CONCRETE.

{ExpandedPlotOutline.model_json_schema()}"""
    
        human_prompt = f"""Current Plot:
{plot.model_dump_json(indent=2)}

Quality Concerns (ADDRESS ALL OF THESE):
{json.dumps(quality_report.concerns, indent=2)}

Recommendations:
{json.dumps(quality_report.recommendations, indent=2)}

Available to Use:
Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
World: {world.model_dump_json(indent=2)}
Conflicts: {conflict_matrix.model_dump_json(indent=2)}

Refine the plot to address these concerns. Be SPECIFIC - use character names, location names, and concrete events throughout."""
        
        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.75,
            model=model
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ExpandedPlotOutline
        )
        
        #refined = expanded_plot_parser.parse(json_data)

        return json_data, tokens
    
    async def reinforce_thematic_resonance(
            self, 
            plot: ExpandedPlotOutline, 
            agents: List[ConnectedNarrativeAgent], 
            world: IntegratedWorld, 
            themes: List[str], 
            model="None"
            ) -> Tuple[ExpandedPlotOutline, dict]:
        system_prompt = f"""You are a thematic development editor.
    Ensure the story’s major themes are reflected consistently in the plot, world, and agent arcs.
    Highlight recurring imagery, motifs, and emotional symbols.
    Don't change core elements unnecessarily.

    {ExpandedPlotOutline.model_json_schema()}"""
        
        human_prompt = f"""
    Themes: {', '.join(themes)}
    Plot: {plot.model_dump_json(indent=2)}
    Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
    World: {world.model_dump_json(indent=2)}
    """
        response, tokens = await better_author_client(system_prompt, human_prompt, 0.7, model)

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ExpandedPlotOutline
        )
        #refined_expanded = expanded_plot_parser.parse(json_data)
        
        return json_data, tokens
    
    async def _create_story_tracker(
        self,
        act_count: int,
        final_plot: ExpandedPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        conflict_matrix: EnhancedConflictMatrix,
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

        response, tokens = await self._cached_author_llm_call(
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
        Main orchestration: Full story planning workflow
        Returns complete story foundation ready for act/chapter planning
        """
        
        print("🎬 ORCHESTRATING WORLD-CLASS STORY PLANNING")
        print("=" * 60)
        
        # PHASE 1: Minimal Seed
        print("\n📋 Phase 1: Creating story seed...")
        seed = self.create_minimal_seed(user_context)
        structure_template = find_best_structure(seed.genre)
        print(f"✓ Seed created | Genres: {seed.genre} | Target: {seed.target_length} words")
        
        # PHASE 2: Independent Foundations (Parallel)
        print("\n🌍 Phase 2: Building independent foundations...")
        world_foundation, world_tokens = await self.world_builder.generate_world_foundation(
            seed, structure_template, model
        )
        print(f"✓ World foundation created")
        
        # PHASE 3: Minimal Plot (determines role requirements)
        print("\n📖 Phase 3: Generating minimal plot...")
        minimal_plot, plot_tokens = await self.generate_minimal_plot_outline(
            seed, world_foundation, model
        )
        print(f"✓ Minimal plot: '{minimal_plot.title}' ({minimal_plot.act_count} acts)")
        print(f"✓ Requires {len(minimal_plot.required_character_roles)} roles")
        
        # PHASE 4: Generate Agents Based on Plot Requirements
        print("\n👥 Phase 4: Generating narrative agents for required roles...")
        narrative_agents, agent_tokens = await self.agent_genesis.generate_narrative_agents(
            seed, world_foundation, minimal_plot.required_character_roles, model
        )
        print(f"✓ {len(narrative_agents)} narrative agents created")
        
        
        # PHASE 5: Conflict Matrix
        print("\n⚔️  Phase 5: Analyzing conflicts...")
        conflict_matrix, conflict_tokens = await self.conflict_architect.generate_conflict_layers(
            minimal_plot, seed, narrative_agents, world_foundation, model
        )
        print(f"✓ Multi-dimensional conflict matrix created")
        
        # PHASE 6: Connect Elements (Parallel)
        print("\n🔗 Phase 6: Connecting elements to plot...")
        prose_style = seed.prose_style
        # Run agent connection first
        connected_agents, connect_tokens = await self.agent_genesis.connect_agents_to_plot(
            narrative_agents, conflict_matrix, minimal_plot, prose_style, model
        )

        # Now run world integration using the actual list
        integrated_world, integrate_tokens = await self.world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents, model
        )
        
        print(f"✓ Agents connected to plot")

        print(f"✓ World integrated with conflicts")
        
        # PHASE 7: Expand Plot
        print("\n📈 Phase 7: Expanding plot outline...")
        expanded_plot, expand_tokens = await self.expand_plot_outline(
            minimal_plot, connected_agents, integrated_world,
            conflict_matrix, seed, model
        )
        print(f"✓ Plot expanded with full detail")
        # === Age-appropriateness safety pass ===
        print("\n🔞 Running age-appropriateness filter...")
        expanded_plot, age_filter_tokens = await self._enforce_age_appropriateness(
            final_plot=expanded_plot,
            seed=seed,
            model=model
        )
        # PHASE 8: Quality Validation
        print("\n🔍 Phase 8: Quality validation...")
        quality_report, quality_tokens = await self.quality_controller.validate_story_elements(
            seed, expanded_plot, connected_agents,
            integrated_world, conflict_matrix, model
        )
        print(f"✓ Quality assessment complete")
        print(f"  Strengths: {len(quality_report.strengths)}")
        print(f"  Concerns: {len(quality_report.concerns)}")

        # PHASE 9: Mandatory Refinement if Concerns Exist
        final_plot = expanded_plot
        refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        if quality_report.concerns:
            # Check for CRITICAL vagueness issues
            critical_issues = [c for c in quality_report.concerns if 'vague' in c.lower() or 'specific' in c.lower() or 'generic' in c.lower()]
            
            if critical_issues:
                print("\n⚠️  CRITICAL: Plot too vague, running refinement...")
                print(f"   Issues: {len(critical_issues)}")
            else:
                print(f"\n🔄 Phase 9: Refining {len(quality_report.concerns)} concerns...")
            
            final_plot, refine_tokens = await self.refine_plot_with_quality_feedback(
                expanded_plot, quality_report, connected_agents,
                integrated_world, conflict_matrix, model
            )
            
            # OPTIONAL: Re-validate if there were critical issues
            if critical_issues:
                print("   Re-validating after refinement...")
                recheck_report, recheck_tokens = await self.quality_controller.validate_story_elements(
                    seed, final_plot, connected_agents,
                    integrated_world, conflict_matrix, model
                )
                refine_tokens["prompt_tokens"] += recheck_tokens.get("prompt_tokens", 0)
                refine_tokens["completion_tokens"] += recheck_tokens.get("completion_tokens", 0)
                
                remaining_critical = [c for c in recheck_report.concerns if 'vague' in c.lower() or 'specific' in c.lower()]
                if remaining_critical:
                    print(f"⚠️  Still has {len(remaining_critical)} vagueness issues (proceeding anyway)")
                else:
                    print("✓ Vagueness issues resolved")
            
            print(f"✓ Plot refined")
        else:
            print("\n✓ Phase 9: No refinement needed")

        

        story_tracker, tracker_tokens = await self._create_story_tracker(
            act_count=minimal_plot.act_count,
            final_plot=final_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix,
            model=model
        )
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
        await self.memory.add_long_term_document(
            text=story_tracker.model_dump_json(indent=2),
            metadata={"type": "story_tracker", "story_title": final_plot.title}
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
        """
        Returns a list of (word_percentage: float, act_guidance: str) for each act, based on structure.
        Percentages sum to approximately 1.0. Raises ValueError if total_acts invalid for structure.
        """
        if structure_name == "Three Act Structure":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.25, "SETUP: Establish world, agents, inciting incident, and stakes."),
                (0.50, "CONFRONTATION: Rising action, complications, midpoint reversal, and deepening conflicts."),
                (0.25, "RESOLUTION: Climax, falling action, denouement, and thematic closure.")
            ]
        
        elif structure_name == "Freytag's Pyramid":
            if total_acts != 5:
                # Fallback: Adjust proportionally for 3-4 acts by merging phases
                if total_acts == 3:
                    return [
                        (0.20, "EXPOSITION AND RISING: Introduce setting/agents and build initial tension."),
                        (0.60, "CLIMAX AND FALLING: Peak conflict and unfolding consequences."),
                        (0.20, "DENOUEMENT: Resolution and new equilibrium.")
                    ]
                elif total_acts == 4:
                    return [
                        (0.15, "EXPOSITION: Introduce agents and setting."),
                        (0.40, "RISING ACTION: Build tension through complications."),
                        (0.25, "CLIMAX AND FALLING: Turning point and consequences."),
                        (0.20, "DENOUEMENT: Resolution and closure.")
                    ]
                else:
                    raise ValueError(f"{structure_name} best fits 5 acts; unsupported for {total_acts}.")
            return [
                (0.15, "EXPOSITION: Introduce agents, setting, and initial conflict."),
                (0.35, "RISING ACTION: Escalate complications and build toward peak."),
                (0.15, "CLIMAX: Moment of highest tension and turning point."),
                (0.20, "FALLING ACTION: Unravel consequences and resolve subplots."),
                (0.15, "DENOUEMENT: Final resolution and reflection.")
            ]
        
        elif structure_name == "The Hero's Journey":
            if total_acts == 3:
                return [
                    (0.25, "DEPARTURE: Ordinary world, call to adventure, refusal, mentor, crossing threshold."),
                    (0.50, "INITIATION: Trials, allies/enemies, approach inmost cave, ordeal."),
                    (0.25, "RETURN: Reward, road back, resurrection, return with elixir.")
                ]
            elif total_acts == 4:
                return [
                    (0.20, "ORDINARY WORLD AND CALL: Setup and inciting adventure."),
                    (0.30, "DEPARTURE AND TRIALS: Crossing threshold and initial challenges."),
                    (0.30, "ORDEAL AND REWARD: Deepest cave, climax, transformation."),
                    (0.20, "RETURN: Road back and final resolution.")
                ]
            elif total_acts == 5:
                return [
                    (0.15, "ORDINARY WORLD: Setup and call to adventure."),
                    (0.25, "DEPARTURE: Refusal, mentor, crossing threshold."),
                    (0.30, "INITIATION: Trials and approach to cave."),
                    (0.20, "ORDEAL: Climax and reward."),
                    (0.10, "RETURN: Road back, resurrection, elixir.")
                ]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        elif structure_name == "Dan Harmon's Story Circle":
            # Typically 8 steps; group into 3-5 acts
            base = [
                (0.20, "YOU/NEED: Establish agent in comfort zone and unmet need."),
                (0.30, "GO/SEARCH/TAKE: Enter unfamiliar situation, adapt, pursue goal."),
                (0.30, "FIND/RETURN/PAY: Achieve goal but pay price, begin change."),
                (0.20, "CHANGE: Return transformed, resolve circle.")
            ]
            if total_acts == 4:
                return base
            elif total_acts == 3:
                return [
                    (0.25, "YOU/NEED/GO: Setup, need, enter unfamiliar."),
                    (0.50, "SEARCH/TAKE/FIND: Adapt, pursue, achieve with price."),
                    (0.25, "RETURN/CHANGE: Transform and resolve.")
                ]
            elif total_acts == 5:
                return [(0.20, base[0][1])] + [(0.20, base[1][1])] + [(0.20, base[2][1])] + [(0.20, base[3][1])] + [(0.20, "FINAL INTEGRATION: Full circle closure.")]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        elif structure_name == "Fichtean Curve":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.20, "INCITING INCIDENT: Jump into action with immediate crisis."),
                (0.60, "RISING CRISES: Series of escalating challenges and complications."),
                (0.20, "CLIMAX AND RESOLUTION: Peak confrontation and rapid wrap-up.")
            ]
        
        elif structure_name == "Save the Cat Beat Sheet":
            if total_acts != 3:
                raise ValueError(f"{structure_name} requires exactly 3 acts.")
            return [
                (0.25, "ACT 1: Opening image, setup, theme, catalyst, debate, break into Act 2."),
                (0.50, "ACT 2: B-story, fun/games, midpoint, bad guys close in, all is lost, dark night."),
                (0.25, "ACT 3: Break into 3, finale (gather team/surprise/execute/dig deep/high tower), final image.")
            ]
        
        elif structure_name == "Seven-Point Story Structure":
            base = [
                (0.20, "HOOK TO PLOT TURN 1: Opening hook and first major plot shift."),
                (0.40, "PINCH 1 TO MIDPOINT: Building pressure, complications, midpoint revelation."),
                (0.25, "PINCH 2 TO PLOT TURN 2: Intensifying crises and second major shift."),
                (0.15, "RESOLUTION: Climax and final wrap-up.")
            ]
            if total_acts == 4:
                return base
            elif total_acts == 3:
                return [
                    (0.25, "HOOK TO MIDPOINT: Setup, turn 1, pinch 1, midpoint."),
                    (0.50, "PINCH 2 TO TURN 2: Escalation and major shift."),
                    (0.25, "RESOLUTION: Climax and ending.")
                ]
            elif total_acts == 5:
                return [(0.15, "HOOK: Opening setup.")] + base[:3] + [(0.10, "FINAL RESOLUTION: Denouement.")]
            else:
                raise ValueError(f"{structure_name} unsupported for {total_acts} acts.")
        
        # Fallback: Even distribution with generic guidance
        even_perc = 1.0 / total_acts
        return [(even_perc, f"ACT {i+1}: Progress the story with rising tension and agent development.") for i in range(total_acts)]

    def get_enhanced_act_planning_system_prompt(
        self,
        act_number: int,
        total_acts: int,
        structure_name: str,
        act_guidance: str,
        themes: list,
        tone: str,
        prose_style: str,
        word_count: int,
        min_age: int,
        structure_config: dict  # NEW PARAMETER
    ) -> str:
        """More directive system prompt that tells the LLM exactly what to prioritize"""
        
        return f"""You are planning ACT {act_number} of {total_acts}.

YOUR ROLE: Define the story SPINE — major turning points that drive this act.
NOT your job: Write scenes, dialogue, or detailed moment-to-moment action.

═══════════════════════════════════════════════════════════════════════════════
STRUCTURE & CONTEXT
═══════════════════════════════════════════════════════════════════════════════

Template: {structure_name}
Act Purpose: {act_guidance}
Tone: {tone} | Prose: {prose_style}
Themes: {', '.join(themes)}
Age Range: {min_age}+ (keep content appropriate)

Target Structure:
- Chapters: {structure_config.get('min_chapters', 2)}-{structure_config.get('max_chapters', 5)}
- Anchors per chapter: {structure_config['anchors_per_chapter']}
- Scenes per anchor: {structure_config['scenes_per_anchor']}
- Words per chapter: ~{structure_config['words_per_chapter']}

═══════════════════════════════════════════════════════════════════════════════
ANCHOR POINT REQUIREMENTS (CRITICAL - READ CAREFULLY)
═══════════════════════════════════════════════════════════════════════════════

Each anchor point MUST include:

1. **SPATIAL GROUNDING**
   location: Specific place (not "somewhere" or "the village" — use "Kael's cottage, main room")
   character_positions: Where each key character is at anchor start
   
2. **TEMPORAL GROUNDING**
   time_context: When this happens relative to previous anchor
   transition_from_previous: HOW we got from last anchor to this one
   
   Examples:
   ✓ GOOD: "Immediately after confrontation, Kael flees to stables"
   ✓ GOOD: "Two days later, having traveled through the forest"
   ✗ BAD: "Later" or "Next" (too vague)

3. **SCENE EXECUTION GUIDANCE**
   estimated_scenes: 1-3 (how many scenes to dramatize this anchor)
   scene_breakdown: If > 1 scene, list what each scene covers
   
   Examples:
   - Simple revelation = 1 scene
   - Complex confrontation with multiple stages = 2-3 scenes
   
4. **CHARACTER STATE TRACKING**
   required_character_states: What must be true BEFORE this anchor
   resulting_character_states: What changes BECAUSE of this anchor
   
   Examples:
   required: {{"Kael": "must have broken sword, must believe father is alive"}}
   resulting: {{"Kael": "wounded left arm, knows father is traitor, possesses map"}}

5. **CONTINUITY**
   introduces: New questions/tensions
   resolves: Earlier setup paid off
   
═══════════════════════════════════════════════════════════════════════════════
SPATIAL/TEMPORAL COHERENCE RULES
═══════════════════════════════════════════════════════════════════════════════

✓ Characters can only be in one place at a time
✓ Travel between locations takes time (acknowledge this in time_context)
✓ If an anchor requires a character to know something, previous anchor must have revealed it
✓ Physical states persist (injuries, exhaustion, possessions) unless explicitly changed
✓ Emotional states evolve logically (can't go from grief to joy without transition)

🚩 RED FLAGS:
- Character teleports between locations without transition
- Character forgets injury or important item
- Character knows something they shouldn't yet
- Two anchors in same chapter have contradictory times ("morning" then "dawn")

═══════════════════════════════════════════════════════════════════════════════
ANCHOR DENSITY GUIDANCE (BY STORY LENGTH)
═══════════════════════════════════════════════════════════════════════════════

{self._get_density_guidance(word_count)}

═══════════════════════════════════════════════════════════════════════════════
CRITICAL DISTINCTION: PLOT vs. CHARACTER
═══════════════════════════════════════════════════════════════════════════════

❌ WEAK: "Lira discovers the sigil on her palm"
   → This is plot. What does she *want*? What will she *sacrifice*?

✓ STRONG: "Lira discovers the sigil will erase her memories of her mother.
   She must choose between hiding (safety, certain loss) and seeking help
   (danger, possible salvation). She chooses to act."
   → This reveals character under pressure and explains her next move.

For each anchor, ask:
- What does the protagonist *want* in this moment?
- What are they willing to *risk* or *sacrifice*?
- What does this reveal about their values or blindspots?
- How does this anchor change what they believe is possible?

{ActPlan.model_json_schema()}"""

    def _get_density_guidance(self, total_words: int) -> str:
        """Returns length-appropriate guidance for anchor density"""
        if total_words < 15000:
            return """SHORT STORY MODE (< 15K words):
- Use 1-2 SUBSTANTIAL anchors per chapter
- Each anchor should support 1500-2000 words of dramatization
- Focus on ONE central conflict thread
- Avoid subplots that dilute focus
- Every anchor must advance the core story question

Example: 3-chapter short story = 5 total anchors maximum"""
    
        elif total_words < 40000:
            return """NOVELLA MODE (15-40K words):
- Use 2-3 anchors per chapter
- Balance plot advancement with character depth
- 1 main plot thread + 1 character/relationship subplot maximum
- Each anchor = 800-1200 words when dramatized

Example: 4-chapter novella = 10-12 total anchors"""
    
        else:
            return """NOVEL MODE (40K+ words):
- Use 3-4 anchors per chapter for novels, 4-5 for epics
- Multiple plot threads can interweave
- Subplots should tie to main themes
- Vary anchor complexity (some simple, some multi-scene)

Example: 6-chapter novel = 20-24 total anchors"""

    def get_enhanced_act_planning_human_prompt(
        self,
        chapter_number: int,
        context_str: str,
        act_number: int,
        total_acts: int,
        suggested_chapters: int,
        suggested_words: int
    ) -> str:
        """Well-organized human prompt that leverages the context builder"""
        if chapter_number == 0:
            chapter_number = 1
        
        return f"""{context_str}

═══════════════════════════════════════════════════════════════════════════════
YOUR TASK
═══════════════════════════════════════════════════════════════════════════════

Plan Act {act_number} of {total_acts} with {suggested_chapters} chapters, starting from chapter number {chapter_number}, (~{suggested_words} words total).

For EACH chapter:
1. Provide a brief creative seed (2-3 sentences)
2. List 2-4 anchor points in CHRONOLOGICAL ORDER
3. Each anchor includes ALL required fields (especially location, time_context, character_positions, transition_from_previous)

═══════════════════════════════════════════════════════════════════════════════
CONTINUITY VALIDATION CHECKLIST (Review before submitting)
═══════════════════════════════════════════════════════════════════════════════

For each anchor you create, verify:

☐ LOCATION is specific and matches world context
☐ TIME_CONTEXT shows how much time passed since last anchor
☐ TRANSITION_FROM_PREVIOUS explains how characters got here (travel? time skip? continuous?)
☐ REQUIRED_CHARACTER_STATES lists what must be true before this anchor
☐ RESULTING_CHARACTER_STATES shows what changes because of this anchor
☐ ESTIMATED_SCENES is realistic (1 scene for simple, 2-3 for complex)
☐ If estimated_scenes > 1, SCENE_BREAKDOWN lists what each scene covers

Example of GOOD anchor:
```json
{{
  "anchor_number": 2,
  "anchor_type": "confrontation",
  "location": "Throne room of Valdris Keep, late afternoon",
  "time_context": "Six hours after escaping dungeon",
  "character_positions": {{
    "Kael": "entering through servant's passage, still wounded",
    "King": "seated on throne, advisors flanking"
  }},
  "transition_from_previous": "Kael fled the dungeon, found the servant's passage Lira told him about, and made his way here",
  "what_happens": "Kael confronts the King about the betrayal, revealing evidence of the conspiracy",
  "required_character_states": {{
    "Kael": "possesses the sealed letter, wounded but mobile",
    "King": "unaware Kael has escaped"
  }},
  "resulting_character_states": {{
    "Kael": "branded as traitor, must flee the capital",
    "King": "knows Kael has evidence, dispatches hunters"
  }},
  "estimated_scenes": 2,
  "scene_breakdown": [
    "Scene 1: Kael's tense entry and initial accusation",
    "Scene 2: King's response and revelation of deeper conspiracy"
  ]
}}
```

═══════════════════════════════════════════════════════════════════════════════
CREATIVE EXPECTATIONS
═══════════════════════════════════════════════════════════════════════════════

Think about:
- How previous acts created setup that THIS act must pay off
- How THIS act sets up future acts
- When to introduce new complications vs. resolve existing ones
- Emotional pacing: where should readers catch breath? Where should tension peak?
- Character psychology: why would each character DO the things they do?

Create chapter outlines that give the DIRECTOR clear spatial/temporal continuity
so scenes flow naturally without characters teleporting or forgetting their state."""


    
    @staticmethod
    def calculate_chapter_structure(act_word_target: int, total_story_length: int) -> dict:
        """
        Returns optimal chapter/anchor structure based on story scope.
        Prevents over-planning short stories and under-planning epics.
        """
        if total_story_length < 15000:  # Short stories
            return {
                "words_per_chapter": 1800,
                "anchors_per_chapter": "1-2",
                "scenes_per_anchor": 1,
                "min_chapters": 2,
                "max_chapters": 4
            }
        elif total_story_length < 40000:  # Novellas
            return {
                "words_per_chapter": 2500,
                "anchors_per_chapter": "2-3",
                "scenes_per_anchor": 1,
                "min_chapters": 2,
                "max_chapters": 5
            }
        elif total_story_length < 75000:  # Novels
            return {
                "words_per_chapter": 3000,
                "anchors_per_chapter": "3-4",
                "scenes_per_anchor": 1,
                "min_chapters": 3,
                "max_chapters": 6
            }
        else:  # Epics
            return {
                "words_per_chapter": 3500,
                "anchors_per_chapter": "4-5",
                "scenes_per_anchor": 2,
                "min_chapters": 4,
                "max_chapters": 8
            }
        
    async def plan_act(
        self,
        story_title: str,
        act_number: int,
        model: str = "None"
    ) -> Tuple[ActPlan, dict]:
        """
        Plan a complete act with rich chapter seeds for chapter director.
        Uses structure-aware creativity.
        """
        
        print(f"\n📋 PLANNING ACT {act_number}")
        print("=" * 60)
        
        # Retrieve story elements from memory
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

        # NEW: Load story tracker
        tracker = await self.tracker_manager.get_or_create_tracker(
            self.memory, story_title
        )
        
        # twist_json = await self.memory.get_long_term_document(
        #     metadata={'type': 'twist_strategy', 'story_title': story_title}
        # )
        # twist_strategy = json.loads(twist_json)
        
        seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': story_title}
        )
        seed = json.loads(seed_json)
        
        # Get progress
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
                    # Convert to proper Pydantic model for consistency
                    act_plan_model = ActPlan(**act_data)
                    previous_acts[prev_act_num] = act_plan_model
            except Exception as e:
                print(f"Could not load Act {prev_act_num}: {e}")

        # Get progress & calculate word budget
        progress = await self.memory.get_story_progress()
        current_word_count = progress.get('story_word_count', 0)
        min_age = progress.get("min_age", 13)
        target_total = final_plot.get('target_length', seed['target_length'] if seed else 50000)
        # Calculate remaining words
        remaining_words = target_total - current_word_count
        total_acts = len(final_plot.get('act_summaries', [])) or 3
        structure_name = find_best_structure_name(seed['genre'] if seed else ['Fiction'])
        percentages_and_guidances = self.get_act_percentages_and_guidance(structure_name, total_acts)
        word_percentage, act_guidance = percentages_and_guidances[act_number - 1]
        # For final act, use ALL remaining words (no percentage)
        if act_number == total_acts:
            suggested_word_count = max(3000, remaining_words)  # Use all remaining
        else:
            suggested_word_count = max(4000, int(remaining_words * word_percentage))

        structure_config = self.calculate_chapter_structure(
            act_word_target=suggested_word_count,
            total_story_length=target_total
        )

        suggested_chapters = max(
            structure_config["min_chapters"],
            min(
                structure_config["max_chapters"],
                suggested_word_count // structure_config["words_per_chapter"]
            )
        )
        print(f"Act {act_number} guidance: {act_guidance}")
        print(f"Target: ~{suggested_word_count:,} words across ~{suggested_chapters} chapters")

        # Get rolling story summary
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
            structure_name=structure_name,
            word_guidance=(word_percentage, act_guidance)
        )
        
        system_prompt = self.get_enhanced_act_planning_system_prompt(
            act_number=act_number,
            total_acts=total_acts,
            structure_name=structure_name,
            act_guidance=act_guidance,
            themes=seed.get('themes', []),
            tone=seed.get('tone', 'Balanced'),
            prose_style=seed.get('prose_style', 'Standard'),
            word_count=suggested_word_count,
            min_age=min_age,
            structure_config=structure_config  # NEW
        )
        
        human_prompt = self.get_enhanced_act_planning_human_prompt(
            chapter_number=latest_chapter,
            context_str=act_context,
            act_number=act_number,
            total_acts=total_acts,
            suggested_chapters=suggested_chapters,
            suggested_words=suggested_word_count
        )

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.88,
            model=model
        )
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        # Validate anchor completeness
        def validate_anchor_fields(act_plan: ActPlan) -> bool:
            """Check that all anchors have required spatial/temporal fields"""
            required_fields = ['location', 'time_context', 'character_positions', 'transition_from_previous']
            
            for chapter in act_plan.chapter_outlines:
                for anchor in chapter.anchor_points:
                    for field in required_fields:
                        if not getattr(anchor, field, None):
                            if field == 'transition_from_previous' and anchor.anchor_number == 1:
                                continue  # First anchor in chapter doesn't need transition
                            print(f"⚠️  Warning: Anchor {anchor.anchor_number} missing '{field}'")
                            return False
            return True

        # In plan_act(), after parsing:
        act_plan = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ActPlan
        )

        # Validate
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



# # User context
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

# # Orchestrate story planning
# story_plan = await author.orchestrate_story_planning(
#     user_context=user_context,
#     model="claude-sonnet-4"
# )

# print("\n📖 Story Plan Complete!")
# print(f"Title: {story_plan['final_plot'].title}")
# print(f"Acts: {len(story_plan['final_plot'].act_summaries)}")
# print(f"Agents: {len(story_plan['connected_agents'])}")
# print(f"Quality Score: {story_plan['quality_report'].overall_assessment}")

# # Plan first act
# act_1 = await author.plan_act(
#     story_title=story_plan['final_plot'].title,
#     act_number=1,
#     model="claude-sonnet-4"
# )
