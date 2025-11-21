import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.memory.memory_system import StoryMemorySystem
from src.llm_client.llm_client import author_client, utility_client
from src.utilities.story_structure_decider import find_best_structure, find_best_structure_name
from src.utilities.story_helpers import StoryHelpers
from config_vars import author_story_rules_negative
import asyncio
import random
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
    act_count: int = Field(ge=3, le=5)
    target_length: int
    narrative_arc: str = Field(
        description="High-level story progression in natural prose"
    )
    required_character_roles: List[str] = Field(
        description="List of character roles needed for this story (e.g., 'protagonist', 'antagonist', 'mentor', 'love interest', 'comic relief', 'betrayer')"
    )

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
    relational_arc_matric: RelationalArcMatrix = Field(
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


class TwistStrategy(BaseModel):
    """Strategic twist placement"""
    selected_twists: List[str] = Field(description="Twists chosen for this story")
    twist_placements: str = Field(
        description="Where and how each twist appears in the narrative"
    )
    setup_requirements: str = Field(
        description="What must be established earlier for twists to land"
    )
    thematic_purpose: str = Field(
        description="How twists serve story themes, not just shock value"
    )


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

class ChapterPivotPoints(BaseModel):
    """Major story moments that anchor a chapter"""
    pivot_number: int = Field(description="Sequential pivot number within chapter of the act")
    pivot_type: str = Field(
        description="revelation | confrontation | decision | loss | discovery | betrayal | transformation | etc..."
    )
    what_changes: str = Field(
        description="What fundamentally shifts in the story at this moment"
    )
    agents_involved: List[str] = Field(description="Key agents present")
    story_function: str = Field(
        description="Why this pivot matters to overall plot/character/theme"
    )
    emotional_weight: str = Field(description="How this moment should feel")
    
    # Setup/Payoff tracking
    introduces_for_later: List[str] = Field(
        default_factory=list,
        description="New questions, objects, or tensions this pivot creates"
    )
    resolves_from_earlier: List[str] = Field(
        default_factory=list,
        description="What this pivot pays off from earlier in story"
    )
    
    # State changes
    character_knowledge_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="What characters learn/believe after this pivot"
    )
    relationship_changes: Dict[str, str] = Field(
        default_factory=dict,
        description="How relationships shift (e.g., 'protag-mentor: trust → suspicion')"
    )
    must_include_elements: List[str] = Field(
    default_factory=list,
    description="Immutable beats that MUST appear exactly (e.g., 'opening speakeasy heist with live jazz', 'lipstick message', 'heirloom ring glints')")

class ChapterOutline(BaseModel):
    chapter_number: int = Field(description="Chapter number within the story")
    target_word_count: int = Field(description="Chapter word count target")
    pivot_points : List[ChapterPivotPoints]
    

class ActPlan(BaseModel):
    """Complete plan for a single act - CHAPTER LEVEL"""
    act_number: int
    act_title: str
    act_purpose: str = Field(
        description="What this act accomplishes in the overall story"
    )
    
    chapter_outlines: List[ChapterOutline] = Field(
        description="Each chapter has: seed (creative direction) + 2-4 pivot points"
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

world_foundation_parser = PydanticOutputParser(pydantic_object=WorldFoundation)
narrative_agent_parser = PydanticOutputParser(pydantic_object=NarrativeAgentList)
enhanced_conflict_parser = PydanticOutputParser(pydantic_object=EnhancedConflictMatrix)
minimal_plot_parser = PydanticOutputParser(pydantic_object=MinimalPlotOutline)
expanded_plot_parser = PydanticOutputParser(pydantic_object=ExpandedPlotOutline)
connected_narrative_agent_parser = PydanticOutputParser(pydantic_object=ConnectedNarrativeAgentList)
integrated_world_parser = PydanticOutputParser(pydantic_object=IntegratedWorld)
twist_strategy_parser = PydanticOutputParser(pydantic_object=TwistStrategy)
quality_report_parser = PydanticOutputParser(pydantic_object=QualityReport)
act_plan_parser = PydanticOutputParser(pydantic_object=ActPlan)
story_tracker_parser = PydanticOutputParser(pydantic_object=StoryTracker)

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

{world_foundation_parser.get_format_instructions()}"""
        
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
            parser=world_foundation_parser
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

{integrated_world_parser.get_format_instructions()}"""
        
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
            parser=integrated_world_parser
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
Not all roles must be literal people.
You can choose to make roles literal people or abstract entities as fits the story best. Not all stories need abstract agents; use them only when they enhance the narrative.
Some may be groups, psychological entities, or natural forces that act as agents in the story world.
For each role, decide whether it manifests as a character, group, concept, or environmental force, and describe how it functions.

Generate {agent_count} diverse agents for a {'/'.join(seed.genre)} story.

IMPORTANT: Create exactly ONE agent for EACH role listed.
If multiple roles could apply to one agent (e.g., 'mentor' and 'ally'), choose the PRIMARY role.
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
{narrative_agent_parser.get_format_instructions()}"""
        
        human_prompt = f"""Seed: {seed.model_dump_json(indent=2)}

Required roles:
{json.dumps(required_roles, indent=2)}

World context:
{world_foundation.model_dump_json(indent=2)}

Protagonist specifications (if any):
{json.dumps(seed.protagonist_specs, indent=2)}

Generate {agent_count} narrative agents for the roles: {', '.join(required_roles)}"""
        
        response, tokens = await self.llm_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

                # === FINAL EXTRACTION LOGIC (WORKS WITH BOTH DICT AND PYDANTIC) ===
        agents_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=narrative_agent_parser
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
{connected_narrative_agent_parser.get_format_instructions()}"""
        
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
            llm_temp=0.85,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        agents_json = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=connected_narrative_agent_parser
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

{enhanced_conflict_parser.get_format_instructions()}"""
        
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
            parser=enhanced_conflict_parser
        )
       # matrix = enhanced_conflict_parser.parse(json_data)
        
        return json_data, tokens


# class TwistEvaluator:
#     """Strategic twist selection and placement"""
    
#     def __init__(self, llm_client):
#         self.llm_client = llm_client
    
#     async def create_twist_strategy(
#         self,
#         plot: ExpandedPlotOutline,
#         conflict_matrix: EnhancedConflictMatrix,
#         agents: List[ConnectedNarrativeAgent],
#         model: str = "None"
#     ) -> Tuple[TwistStrategy, dict]:
#         """Select and place twists strategically"""
        
#         system_prompt = f"""You are a master of plot twists that feel both surprising and inevitable.

# Available twist opportunities: {conflict_matrix.twist_opportunities}

# Create a twist strategy that:
# 1. Selects 2-4 twists that emerge naturally from agent/conflict/world
# 2. Places them at maximum impact points (not random)
# 3. Ensures proper setup earlier in story
# 4. Serves thematic purpose beyond shock value
# 5. Each twist should recontextualize what came before

# Twists should feel earned, not pulled from thin air.

# {twist_strategy_parser.get_format_instructions()}"""
        
#         human_prompt = f"""Develop a strategic twist placement plan.
#         Story Elements:
# Plot: {plot.model_dump_json(indent=2)}
# Conflicts: {conflict_matrix.model_dump_json(indent=2)}
# Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
#         """
        
#         response, tokens = await self.llm_client(
#             system_prompt=system_prompt,
#             human_prompt=human_prompt,
#             llm_temp=0.85,
#             model=model
#         )
        
#         strategy = twist_strategy_parser.parse(response.content.strip())
#         return strategy, tokens


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
        """Comprehensive quality assessment"""
        
        system_prompt = f"""You are a story development editor evaluating narrative quality.

Analyze these story elements for:

1. PLOT COHERENCE: Does the story flow logically? Are there gaps?
2. AGENT AGENCY: Do agents drive the plot or just react?
3. CONFLICT ESCALATION: Do stakes rise appropriately?
4. THEMATIC INTEGRATION: Are themes woven throughout?
5. WORLD CONSISTENCY: Does the world follow its own rules?
6. EMOTIONAL VARIETY: Is there a good balance of tones?
7. STRUCTURE INTEGRITY: Does it follow its template effectively?
8. ORIGINALITY: Does it avoid clichés?

Provide specific, actionable feedback.

{quality_report_parser.get_format_instructions()}"""
        
        human_prompt = f"""Evaluate this story's quality and provide recommendations.
        Story Elements:
Seed: {seed.model_dump_json(indent=2)}
Plot: {plot.model_dump_json(indent=2)}
Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
World: {world.model_dump_json(indent=2)}
Conflicts: {conflict_matrix.model_dump_json(indent=2)}
        """
        
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
            parser=quality_report_parser
        )
        #report = quality_report_parser.parse(json_data)
        
        return json_data, tokens

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
╔════════════════════════════════════════════════════════════════════════════╗
║ ACT {act_number} CONTEXT & GUIDANCE
╚════════════════════════════════════════════════════════════════════════════╝

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
┌─ ACT SUMMARY ─────────────────────────────────────────────────────────────┐
{act_summaries[act_number - 1]}
└───────────────────────────────────────────────────────────────────────────┘
""")
        
        # ===== SECTION 3: WHAT THIS ACT SETS UP & PAYS OFF =====
        tracker = story_tracker.get(str(act_number), {})
        setup_elements = tracker.get('setup_elements', [])
        payoff_elements = tracker.get('payoff_elements', [])
        
        context_parts.append(f"""
┌─ STORY CONTINUITY ────────────────────────────────────────────────────────┐
SETUP (introduce for later payoff):
{ActContextBuilder._format_list(setup_elements, indent=2)}

PAYOFF (resolve from earlier acts):
{ActContextBuilder._format_list(payoff_elements, indent=2)}
└───────────────────────────────────────────────────────────────────────────┘
""")
        
        # ===== SECTION 4: AGENT ARCS THIS ACT =====
        context_parts.append("\n┌─ AGENT ARCS THIS ACT ─────────────────────────────────────────────────┐")
        
        tracker_agents = tracker.get('key_agent_moments', {})
        
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
        
        context_parts.append("└───────────────────────────────────────────────────────────────────────┘\n")
        
        # ===== SECTION 5: CONFLICT ESCALATION =====
        context_parts.append(f"""
┌─ CONFLICT ESCALATION ─────────────────────────────────────────────────────┐
Central: {conflict_matrix.get('central_conflict', '')}

Escalation Path: {conflict_matrix.get('escalation_path', '')}

Moral Dilemma: {conflict_matrix.get('moral_dilemma_axis', '')}
└───────────────────────────────────────────────────────────────────────────┘
""")
        
        # ===== SECTION 6: WORLD ELEMENTS THIS ACT =====
        locs = integrated_world.get('plot_relevant_locations', '').split('\n')
        rules = integrated_world.get('world_plot_interactions', '').split('\n')

        context_parts.append(f"""
┌─ WORLD & SETTING ─────────────────────────────────────────────────────────┐
Relevant Locations:
{ActContextBuilder._format_list(locs, indent=2)}

World Rules Active:
{ActContextBuilder._format_list(rules, indent=2)}
└───────────────────────────────────────────────────────────────────────────┘
""")

        
        # ===== SECTION 7: PREVIOUS ACT CONSEQUENCES =====
        if previous_acts and act_number > 1:
            context_parts.append(f"""
┌─ CONSEQUENCES FROM ACT {act_number - 1} ───────────────────────────────────┐
{ActContextBuilder._summarize_previous_act(previous_acts.get(act_number - 1, {}))}
└───────────────────────────────────────────────────────────────────────────┘
""")
        
        # ===== SECTION 8: STORY PROGRESS =====
        context_parts.append(f"""
┌─ STORY SO FAR ────────────────────────────────────────────────────────────┐
{story_so_far}
└───────────────────────────────────────────────────────────────────────────┘
""")
        
        # ===== SECTION 9: THEMATIC THREADS =====
        themes = seed.get('themes', [])
        if themes:
            context_parts.append(f"""
┌─ THEMATIC FOCUS ──────────────────────────────────────────────────────────┐
Core Themes: {', '.join(themes)}

How This Act Deepens Themes:
{final_plot.get('narrative_flow', '')}
└───────────────────────────────────────────────────────────────────────────┘
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
            summary += f"Payoffs this act: {payoffs}\n"
        
        if changes:
            summary += "Character Changes: " + ", ".join(
                [f"{agent} ({status})" for agent, status in list(changes.items())]
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
        self.quality_controller = QualityController(self._cached_author_llm_call)
        
        # LLM cache
        self.llm_cache = {}
    
    async def _cached_author_llm_call(
        self,
        system_prompt: str,
        human_prompt: str,
        llm_temp: float,
        model: str
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
   - Think about what character roles this specific story NEEDS
   - Be flexible: small intimate stories might need 3-4 roles, epic tales might need 10-15
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

{minimal_plot_parser.get_format_instructions()}"""
        
        human_prompt = f"""Generate a minimal plot outline with appropriate role requirements for this story's scope.
        Seed: {seed.model_dump_json(indent=2)}

World Foundation: {world_foundation.model_dump_json(indent=2)}
        """
        
        response, tokens = await self._cached_author_llm_call(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        outline = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=minimal_plot_parser
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

{expanded_plot_parser.get_format_instructions()}"""
        
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
        
        response, tokens = await self._cached_author_llm_call(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=expanded_plot_parser
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
    {expanded_plot_parser.get_format_instructions()}"""

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
            parser=expanded_plot_parser
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
        
        system_prompt = f"""You are refining a plot based on editorial feedback.
Don't change core elements unnecessarily.

{expanded_plot_parser.get_format_instructions()}"""
        
        human_prompt = f"""Current Plot:
{plot.model_dump_json(indent=2)}

Quality Concerns:
{json.dumps(quality_report.concerns, indent=2)}

Recommendations:
{json.dumps(quality_report.recommendations, indent=2)}

Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
World: {world.model_dump_json(indent=2)}
Conflicts: {conflict_matrix.model_dump_json(indent=2)}

Refine the plot to address these concerns while maintaining what's working."""
        
        response, tokens = await self._cached_author_llm_call(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8,
            model=model
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=expanded_plot_parser
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

    {expanded_plot_parser.get_format_instructions()}"""
        human_prompt = f"""
    Themes: {', '.join(themes)}
    Plot: {plot.model_dump_json(indent=2)}
    Agents: {json.dumps([a.model_dump() for a in agents], indent=2)}
    World: {world.model_dump_json(indent=2)}
    """
        response, tokens = await self._cached_author_llm_call(system_prompt, human_prompt, 0.7, model)

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=expanded_plot_parser
        )
        #refined_expanded = expanded_plot_parser.parse(json_data)
        
        return json_data, tokens
    
    async def _create_story_tracker(
        self,
        final_plot: ExpandedPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        conflict_matrix: EnhancedConflictMatrix,
        model: str
    ) -> Tuple[StoryTracker, dict]:
        """Lightweight story-level tracking"""
        
        system_prompt = f"""Create a simple tracking document for story consistency.

For each act, identify:
1. What major elements get INTRODUCED (characters, conflicts, questions, items, relationships)
2. What gets RESOLVED from earlier

Keep it simple and genre-agnostic. 
For a mystery: introduce clues, resolve whodunit
For romance: introduce attraction, resolve will-they-won't-they  
For action: introduce threat, resolve confrontation
For comedy: introduce misunderstanding, resolve reveal

{story_tracker_parser.get_format_instructions()}"""

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
            llm_temp=0.7,
            model=model
        )
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data = await StoryHelpers.load_json_with_retry(clean_resp, story_tracker_parser)
        
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
        
        # PHASE 8: Quality Validation
        print("\n🔍 Phase 8: Quality validation...")
        quality_report, quality_tokens = await self.quality_controller.validate_story_elements(
            seed, expanded_plot, connected_agents,
            integrated_world, conflict_matrix, model
        )
        print(f"✓ Quality assessment complete")
        print(f"  Strengths: {len(quality_report.strengths)}")
        print(f"  Concerns: {len(quality_report.concerns)}")
        
        # PHASE 9: Refinement (if needed)
        final_plot = expanded_plot
        refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        refine_refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if quality_report.concerns:
            print("\n🔄 Phase 9: Refining based on feedback...")
            final_plot, refine_tokens = await self.refine_plot_with_quality_feedback(
                expanded_plot, quality_report, connected_agents,
                integrated_world, conflict_matrix, model
            )
            final_final_plot, refine_refine_tokens = await self.reinforce_thematic_resonance(plot=final_plot, agents=connected_agents, world=integrated_world, themes=seed.themes)
            print(f"✓ Plot refined")
        else:
            print("\n✓ Phase 9: No refinement needed")

        # === Age-appropriateness safety pass ===
        print("\n🔞 Running age-appropriateness filter...")
        final_plot, age_filter_tokens = await self._enforce_age_appropriateness(
            final_plot=final_final_plot,
            seed=seed,
            model=model
        )
        # PHASE 10: Twist Strategy
        # print("\n🎭 Phase 10: Planning twist strategy...")
        # twist_strategy, twist_tokens = await self.twist_evaluator.create_twist_strategy(
        #     final_final_plot, conflict_matrix, connected_agents, model
        # )
        # print(f"✓ {len(twist_strategy.selected_twists)} strategic twists planned")

        story_tracker, tracker_tokens = await self._create_story_tracker(
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
        # await self.memory.add_long_term_document(
        #     text=twist_strategy.model_dump_json(indent=2),
        #     metadata={"type": "twist_strategy", "story_title": final_final_plot.title}
        # )
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
                refine_refine_tokens.get("prompt_tokens", 0),
                #twist_tokens.get("prompt_tokens", 0)
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
                refine_refine_tokens.get("completion_tokens", 0),
                #twist_tokens.get("completion_tokens", 0)
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
            #"twist_strategy": twist_strategy,
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
        min_age: int
    ) -> str:
        """
        More directive system prompt that tells the LLM exactly what to prioritize
        """
        
        return f"""You are planning ACT {act_number} of {total_acts} for a world-class story.

    YOUR JOB:
    Create 2-4 CHAPTERS with rich PIVOT POINTS (major story moments).
    Do NOT write scenes. Do NOT write dialogue. Do NOT write description.
    Instead, map out the SPINE of each chapter — the critical turning points.

    ═══════════════════════════════════════════════════════════════════════════════
    STRUCTURE & PACING REQUIREMENTS
    ═══════════════════════════════════════════════════════════════════════════════

    Structure Template: {structure_name}
    Act Purpose: {act_guidance}
    Tone: {tone}
    Prose Style: {prose_style}
    Key Themes: {', '.join(themes)}

    Act {act_number}'s Role:
    - THIS is where we progress toward the {['setup', 'complication', 'climax'][min(act_number-1, 2)]}
    - Previous acts have established foundation; this act BUILDS on that
    - Next acts will depend on what YOU set up here
    - Pacing: balance revelations, confrontations, decisions, and consequences

    ═══════════════════════════════════════════════════════════════════════════════
    WHAT YOU MUST INCLUDE IN EACH CHAPTER
    ═══════════════════════════════════════════════════════════════════════════════

    For EACH chapter, provide:

    1. CHAPTER SEED (2-3 sentences)
    - The creative DNA of this chapter
    - What makes it FEEL different from other chapters
    - What drives the emotional tone
    Example: "Kael arrives at the abandoned tower where his father died. 
                The air is cold—but wrongly so, unnatural. As he searches for answers,
                he realizes his father's presence still lingers in corrupted magic."

    2. TARGET WORD COUNT
    - Roughly {word_count} words per chapter
    - No need to be exact; let Director adjust

    3. PIVOT POINTS (2-4 per chapter, ordered chronologically)
    Each pivot is ONE moment where something fundamentally shifts:
    - A revelation that reframes the narrative
    - A confrontation that shifts power or trust
    - A decision that locks in consequences
    - A loss that raises the stakes
    - A discovery that opens new paths forward
    - A betrayal that inverts a relationship
    - A transformation in a character or location

    ═══════════════════════════════════════════════════════════════════════════════
    PIVOT POINT STRUCTURE (REPEAT FOR EACH PIVOT)
    ═══════════════════════════════════════════════════════════════════════════════

    For each pivot, you MUST specify:

    pivot_type: (revelation | confrontation | decision | loss | discovery | betrayal | transformation)

    what_changes: Describe what fundamentally shifts.
    - Not "Kael learns magic"
    - But "Kael learns magic comes at cost of memory — forcing him to choose between power and remembering his sister"

    agents_involved: [List only agents who are CENTRAL to this pivot]
    - Don't list every character in the scene
    - List who DRIVES or is TRANSFORMED by this moment

    story_function: Why does this matter to the overall plot?
    - Connect to larger conflicts or character arcs
    - Make explicit the RIPPLE EFFECT

    emotional_weight: How should this FEEL?
    - "Shocking but inevitable" / "Bittersweet triumph" / "Crushing betrayal"
    - This guides the Director's scene construction

    introduces_for_later: What NEW tensions, objects, relationships, or questions does this pivot CREATE?
    - These MUST be payoff elsewhere in the story
    - These FEED into later acts
    - Be specific: "Question: Can Kael trust his mentor?" not just "introduces doubt"

    resolves_from_earlier: What from EARLIER in the story gets ANSWERED here?
    - Callbacks to setup
    - Payoff on seeded mysteries
    - Character relationship shifts that were building

    character_knowledge_changes: For each agent involved, what do they NOW KNOW/BELIEVE?
    - "Kael: learns his father was corrupted by dark magic"
    - "Mentor: realizes Kael is stronger than expected"
    - Track misconceptions being corrected

    relationship_changes: How do AGENT RELATIONSHIPS shift?
    - "Kael-Mentor: trust → suspicion"
    - "Kael-Love Interest: strangers → allies"
    - Be concise but specific

    ═══════════════════════════════════════════════════════════════════════════════
    CRITICAL CONSTRAINTS
    ═══════════════════════════════════════════════════════════════════════════════

    ✓ CONSISTENCY: Each pivot must align with agent arcs and established world rules
    ✓ ESCALATION: Pivots should build in intensity/stakes as chapter progresses
    ✓ SETUP/PAYOFF: Track what you introduce vs. what you resolve
    ✓ AGENT AUTHENTICITY: Agents act from their psychology, not plot convenience
    ✓ THEMATIC RESONANCE: Each pivot should echo at least one core theme
    ✓ AGE-APPROPRIATE: Content suitable for {min_age}-year-old audience
    ✓ REALISM: Even in fantasy/sci-fi, causes lead to effects

    ═══════════════════════════════════════════════════════════════════════════════
    FAILURE MODES TO AVOID
    ═══════════════════════════════════════════════════════════════════════════════

    ❌ Don't make pivots feel random or convenient
    ❌ Don't use pivots just for shock value
    ❌ Don't have agents acting against their established psychology
    ❌ Don't forget about setup/payoff tracking
    ❌ Don't introduce massive new elements without explanation
    ❌ Don't lose sight of the act's core purpose in the larger structure
    ❌ Don't create pivots that contradict world rules or earlier plot points

    ═══════════════════════════════════════════════════════════════════════════════

    Output ONLY valid JSON matching the ActPlan schema. Include ALL fields."""


    def get_enhanced_act_planning_human_prompt(
            self,
        chapter_number: int,
        context_str: str,
        act_number: int,
        total_acts: int,
        suggested_chapters: int,
        suggested_words: int
    ) -> str:
        """
        Well-organized human prompt that leverages the context builder
        """
        
        return f"""{context_str}

    ═══════════════════════════════════════════════════════════════════════════════
    YOUR TASK
    ═══════════════════════════════════════════════════════════════════════════════

    Plan Act {act_number} of {total_acts} with {suggested_chapters} chapters, startin from chapter number {chapter_number}, (~{suggested_words} words total).

    For EACH chapter:
    1. Provide a brief creative seed (2-3 sentences)
    2. List 2-4 pivot points in order
    3. Each pivot includes all required fields (type, what_changes, agents, etc.)

    Think about:
    • How previous acts created setup that THIS act must pay off
    • How THIS act sets up future acts
    • When to introduce new complications vs. resolve existing ones
    • Emotional pacing: where should readers catch breath? Where should tension peak?
    • Agent psychology: why would each agent DO the things they do?

    Create a chapter outline that the CHAPTER DIRECTOR can translate into actual scenes.
    Your job is the SPINE of the act. The Director handles the flesh."""

    
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
        story_tracker_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_tracker', 'story_title': story_title}
        )
        story_tracker = json.loads(story_tracker_json) if story_tracker_json else {}
        
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
        
        # === CRITICAL FIX: Properly parse ALL previous act plans ===
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
        remaining_words = max(5000, target_total - current_word_count)  # safety floor

        total_acts = len(final_plot.get('act_summaries', [])) or 3
        structure_name = find_best_structure_name(seed['genre'] if seed else ['Fiction'])

        percentages_and_guidances = self.get_act_percentages_and_guidance(structure_name, total_acts)
        word_percentage, act_guidance = percentages_and_guidances[act_number - 1]

        suggested_word_count = max(4000, int(remaining_words * word_percentage))
        suggested_chapters = max(2, min(6, suggested_word_count // 2200))  # ~2–6 chapters

        print(f"Act {act_number} guidance: {act_guidance}")
        print(f"Target: ~{suggested_word_count:,} words across ~{suggested_chapters} chapters")

        # Get rolling story summary
        latest_chapter = progress.get('latest_chapter_id', 0)
        story_so_far = await self.memory.get_director_context(
            current_act_number=act_number,
            current_chapter_number=latest_chapter + 1,
            query="",
            k=5
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
            story_tracker=story_tracker,
            previous_acts={num: plan.model_dump() for num, plan in previous_acts.items()},       
            story_so_far=story_so_far,
            structure_name=structure_name,
            word_guidance=(word_percentage, act_guidance)
        )
        
        # NEW: Use enhanced prompts
        system_prompt = self.get_enhanced_act_planning_system_prompt(
            act_number=act_number,
            total_acts=total_acts,
            structure_name=structure_name,
            act_guidance=act_guidance,
            themes=seed.get('themes', []),
            tone=seed.get('tone', 'Balanced'),
            prose_style=seed.get('prose_style', 'Standard'),
            word_count=suggested_word_count,
            min_age=min_age
        )
        
        human_prompt = self.get_enhanced_act_planning_human_prompt(
            chapter_number=latest_chapter,
            context_str=act_context,
            act_number=act_number,
            total_acts=total_acts,
            suggested_chapters=suggested_chapters,
            suggested_words=suggested_word_count
        )

        response, tokens = await self._cached_author_llm_call(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.88,
            model=model
        )
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        act_plan = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=act_plan_parser
        )
        #act_plan = act_plan_parser.parse(json_data)
        
        
        # Validate chapter count
        # if len(act_plan.chapter_seeds) < 2:
        #     print("⚠️  Warning: Act has fewer than 2 chapters")
        
        # Store in memory
        await self.memory.add_long_term_document(
            text=act_plan.model_dump_json(indent=2),
            metadata={"type": "act_plan", "act_id": act_number, "story_title": story_title}
        )
        
        print("=" * 60)
        print(f"✅ ACT {act_number} PLANNED: '{act_plan.act_title}'")
        print(f"📖 {len(act_plan.chapter_outlines)} chapters outlined")
        print("=" * 60)
        
        return act_plan, tokens



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
