# src/story_engines/classic_adventure/agents/story_author.py

import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.memory.memory_system import StoryMemorySystem
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_structure_decider import find_best_structure, find_best_structure_name
from src.utilities.story_helpers import StoryHelpers
from config_vars import author_story_rules_negative
import asyncio


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

class DetailedCharacterBackstory(FlexibleBase):
    """Deep character history for 80k+ novels"""
    name: str
    formative_events: str = Field(
        description="3-5 key life events that shaped this character (500-800 words)"
    )
    psychological_profile: str = Field(
        description="Internal landscape: fears, desires, contradictions, coping mechanisms"
    )
    relationship_history: Dict[str, str] = Field(
        default_factory=dict,
        description="Pre-story relationships with other characters"
    )
    secrets: List[str] = Field(
        default_factory=list,
        description="Hidden truths that could emerge during story"
    )
    voice_profile: str = Field(
        default="",
        description="Speech patterns, vocabulary, mannerisms"
    )
    subplot_seeds: List[str] = Field(
        default_factory=list,
        description="Personal story threads that could weave through main plot"
    )
    genre_specific_depth: Dict[str, Any] = Field(
        default_factory=dict,
        description="Genre-specific character elements"
    )


class EnhancedWorldGuide(FlexibleBase):
    """Comprehensive world documentation for 80k+ novels"""
    location_dossiers: List[Dict[str, Any]] = Field(
        description="5-10 key locations with history, culture, secrets"
    )
    system_documentation: Dict[str, str] = Field(
        default_factory=dict,
        description="Magic/tech/social systems with clear rules"
    )
    historical_timeline: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Key historical events informing current conflicts"
    )
    cultural_details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Customs, beliefs, taboos, celebrations"
    )
    minor_character_pool: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Pre-generated NPCs ready for deployment"
    )
    genre_specific_elements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Genre-specific world features"
    )


class SubplotThread(FlexibleBase):
    """Individual subplot architecture"""
    subplot_id: str
    title: str
    premise: str = Field(description="1-2 sentence subplot hook")
    character_owner: str = Field(description="Primary character driving this thread")
    supporting_characters: List[str] = Field(default_factory=list)
    thematic_connection: str = Field(
        description="How this subplot reinforces main themes"
    )
    act_integration: Dict[int, str] = Field(
        default_factory=dict,
        description="Key moments per act: {act_num: 'what happens'}"
    )
    resolution_type: str = Field(
        description="How this resolves: triumph, tragedy, transformation, etc."
    )


class SubplotArchitecture(FlexibleBase):
    """Complete subplot system for novel"""
    subplots: List[SubplotThread]
    integration_strategy: str = Field(
        description="How subplots weave with main plot"
    )

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

class ExpandedCompactPlotOutline(FlexibleBase):
    """Detailed plot - LLM adds richness appropriate to length/genre"""
    title: str
    premise: str
    required_character_roles: List[str] = Field(
        description="Roles needed: protagonist, antagonist, mentor, etc. MUST include ALL roles."
    )
    act_summaries: List[str] = Field(description="Rich prose summaries per act FOR EACH ACT")
    narrative_flow: str = Field(description="Complete plot in flowing prose")
    

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



# In src/story_engines/classic_adventure/agents/story_author.py

class ConnectedNarrativeAgent(FlexibleBase):
    """Agent after plot integration - flexible depth"""
    name: str
    role: str
    gender: str  # Explicitly added as Director often requires it for pronouns
    essence: str
    plot_function: str = Field(description="How this agent drives the plot")
    agent_arc: str = Field(description="How this agent changes through story")
    
    # CRITICAL FIX for Director: Explicitly require relationships
    relationships: Dict[str, str] = Field(
        default_factory=dict,
        description="Key relationship dynamics (e.g., {'Hero': 'Rivalry', 'Mentor': 'Respect'}). Essential for scene tension."
    )
    # LLM adds: distinctive_voice, key_moments, etc.

class ConnectedNarrativeAgentList(FlexibleBase):
    agents: List[ConnectedNarrativeAgent]


class IntegratedWorld(FlexibleBase):
    """World after plot integration"""
    foundation: str = Field(description="Core world elements")
    plot_integration: str = Field(
        description="How world rules enable/complicate conflicts"
    )
    
    # CRITICAL FIX for Director: Explicitly require world rules list
    world_rules: List[str] = Field(
        default_factory=list,
        description="Explicit list of 3-5 unbreakable constraints (magic costs, laws of physics, social taboos) for the Director to enforce."
    )


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
# ACT PLAN STRUCTURE (used by Director)
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
    
    def __init__(self, GENRE_CONFIGURATIONS):
        self.GENRE_CONFIGURATIONS = GENRE_CONFIGURATIONS
    
    async def generate_world_foundation(
        self,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[WorldFoundation, dict, dict]:
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
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=WorldFoundation
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        return json_data, author_tokens, utility_tokens
    
    async def integrate_with_conflict(
        self,
        world_foundation: WorldFoundation,
        conflict_matrix: ConflictMatrix,
        agents: List[ConnectedNarrativeAgent],
        model: str = "None"
    ) -> Tuple[IntegratedWorld, dict, dict]:
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
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=IntegratedWorld
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]

        return json_data, author_tokens, utility_tokens
    
    def _get_genre_config(self, genres: List[str]) -> Dict[str, Any]:
        """
        Get merged genre configuration for multi-genre stories.
        Primary genre takes precedence, secondary adds elements.
        """
        if not genres:
            return self.GENRE_CONFIGURATIONS.get("Drama", {})  # Default fallback
        
        primary_genre = genres[0]
        
        # Find matching configuration (handle parent genre extraction)
        config = None
        for genre_key in self.GENRE_CONFIGURATIONS.keys():
            if genre_key in primary_genre or primary_genre in genre_key:
                config = self.GENRE_CONFIGURATIONS[genre_key].copy()
                break
        
        if not config:
            config = self.GENRE_CONFIGURATIONS.get("Drama", {})
        
        # Merge secondary genres if present
        if len(genres) > 1:
            for secondary in genres[1:]:
                for genre_key in self.GENRE_CONFIGURATIONS.keys():
                    if genre_key in secondary or secondary in genre_key:
                        secondary_config = self.GENRE_CONFIGURATIONS[genre_key]
                        # Add unique elements from secondary
                        for key in ["character_focus", "world_focus", "subplot_types"]:
                            if key in secondary_config:
                                config[key] = list(set(config.get(key, []) + secondary_config[key]))
                        break
        
        return config
    
    async def expand_world_detail(
        self,
        integrated_world: IntegratedWorld,
        plot: ExpandedPlotOutline,
        seed: MinimalStorySeed,
        connected_agents: List[ConnectedNarrativeAgent],
        model: str = "None"
    ) -> Tuple[EnhancedWorldGuide, dict, dict]:
        """
        Create comprehensive world documentation for 80k+ novels.
        Genre-specific and plot-integrated.
        """
        
        genre_config = self._get_genre_config(seed.genre)
        
        system_prompt = f"""You are building a comprehensive world guide for a {'/'.join(seed.genre)} novel.

GENRE FOCUS AREAS: {', '.join(genre_config.get('world_focus', []))}

WORLD GUIDE REQUIREMENTS:

1. LOCATION DOSSIERS (5-10 key locations)
   For each location provide:
   - name: Evocative place name
   - description: Rich sensory details (200-300 words)
   - history: Background and significance
   - culture: Local customs, beliefs, social norms
   - secrets: Hidden aspects that could emerge
   - plot_relevance: How this location serves the story
   - genre_elements: {', '.join(genre_config.get('world_focus', []))}

2. SYSTEM DOCUMENTATION
   Detailed rules for:
   - Magic/technology systems with clear limitations
   - Social hierarchies and power structures
   - Economic systems and trade
   - Communication methods
   - Transportation
   - Governance and law
   (Focus on systems relevant to {'/'.join(seed.genre)})

3. HISTORICAL TIMELINE (5-10 key events)
   Events that inform current conflicts:
   - date/era: When it occurred
   - event: What happened
   - consequences: Lasting impact on world/characters
   - plot_connection: How this history matters now

4. CULTURAL DETAILS
   - Customs and traditions
   - Taboos and social rules
   - Celebrations and rituals
   - Beliefs and superstitions
   - Art, music, cuisine
   - Language quirks or slang

5. MINOR CHARACTER POOL (10-15 NPCs)
   Pre-generated characters ready for deployment:
   - name, role, brief description
   - Location they frequent
   - Potential plot uses

6. GENRE-SPECIFIC ELEMENTS
   {self._get_genre_specific_world_fields(seed.genre, genre_config)}

Age-appropriate for: {seed.target_audience_age} year olds

OUTPUT FORMAT (JSON):
{EnhancedWorldGuide.model_json_schema()}

Create immersive, internally consistent world documentation.
Output clean JSON with no markdown fences."""

        human_prompt = f"""Expand world detail for this story.

EXISTING WORLD FOUNDATION:
{integrated_world.model_dump_json(indent=2)}

PLOT CONTEXT:
{plot.model_dump_json(indent=2)}

MAIN CHARACTERS (to inform locations/NPCs):
{json.dumps([{"name": a.name, "role": a.role} for a in connected_agents], indent=2)}

Create comprehensive world guide."""

        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        world_guide, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=EnhancedWorldGuide
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        
        return world_guide, author_tokens, utility_tokens

    def _get_genre_specific_world_fields(self, genres: List[str], config: Dict) -> str:
        """Generate genre-specific field suggestions for world expansion"""
        suggestions = []
        
        if any(g in ["Fantasy", "High Fantasy", "Dark Fantasy"] for g in genres):
            suggestions.append("  - magic_system_details: Comprehensive magic rules, costs, limitations")
            suggestions.append("  - mystical_creatures: Beings and their roles in world")
            suggestions.append("  - ancient_artifacts: Objects of power and their histories")
            suggestions.append("  - prophecies: Known predictions affecting current events")
        
        if any(g in ["Sci-Fi", "Cyberpunk", "Space Opera"] for g in genres):
            suggestions.append("  - technology_tree: Available tech and research frontiers")
            suggestions.append("  - alien_species: Non-human civilizations and relations")
            suggestions.append("  - space_politics: Factions, alliances, conflicts")
            suggestions.append("  - scientific_laws: Physics, FTL, time travel rules")
        
        if any(g in ["Mystery", "Thriller", "Crime"] for g in genres):
            suggestions.append("  - criminal_networks: Underground organizations")
            suggestions.append("  - law_enforcement: Police structure, jurisdiction, methods")
            suggestions.append("  - surveillance_systems: How people are monitored")
            suggestions.append("  - information_sources: Where clues can be found")
        
        if any(g in ["Horror", "Psychological Horror", "Gothic Horror"] for g in genres):
            suggestions.append("  - supernatural_rules: How horror elements function")
            suggestions.append("  - haunted_locations: Places of dread and why")
            suggestions.append("  - isolation_factors: What keeps characters trapped")
            suggestions.append("  - historical_atrocities: Past horrors lingering")
        
        if any(g in ["Romance", "Historical Romance", "Contemporary Romance"] for g in genres):
            suggestions.append("  - social_expectations: Dating norms, marriage rules")
            suggestions.append("  - romantic_venues: Key meeting and dating locations")
            suggestions.append("  - relationship_obstacles: Cultural/social barriers")
            suggestions.append("  - community_dynamics: How gossip and reputation work")
        
        return "\n".join(suggestions) if suggestions else "  - Add genre-appropriate elements"

class AgentGenesis:
    """Generates independent narrative agent foundations with flexible depth"""
    
    def __init__(self, GENRE_CONFIGURATIONS):
        self.GENRE_CONFIGURATIONS = GENRE_CONFIGURATIONS

    def _get_genre_config(self, genres: List[str]) -> Dict[str, Any]:
        """
        Get merged genre configuration for multi-genre stories.
        Primary genre takes precedence, secondary adds elements.
        """
        if not genres:
            return self.GENRE_CONFIGURATIONS.get("Drama", {})  # Default fallback
        
        primary_genre = genres[0]
        
        # Find matching configuration (handle parent genre extraction)
        config = None
        for genre_key in self.GENRE_CONFIGURATIONS.keys():
            if genre_key in primary_genre or primary_genre in genre_key:
                config = self.GENRE_CONFIGURATIONS[genre_key].copy()
                break
        
        if not config:
            config = self.GENRE_CONFIGURATIONS.get("Drama", {})
        
        # Merge secondary genres if present
        if len(genres) > 1:
            for secondary in genres[1:]:
                for genre_key in self.GENRE_CONFIGURATIONS.keys():
                    if genre_key in secondary or secondary in genre_key:
                        secondary_config = self.GENRE_CONFIGURATIONS[genre_key]
                        # Add unique elements from secondary
                        for key in ["character_focus", "world_focus", "subplot_types"]:
                            if key in secondary_config:
                                config[key] = list(set(config.get(key, []) + secondary_config[key]))
                        break
        
        return config
    
    async def generate_narrative_agents(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        required_roles: List[str],
        model: str = "None"
    ) -> Tuple[List[NarrativeAgent], dict, dict]:
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
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        agents_json, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=NarrativeAgentList
        )

        if isinstance(agents_json, tuple):
            agents_json = agents_json[0]

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
        return agents, author_tokens, utility_tokens
    
    async def connect_agents_to_plot(
        self,
        narrative_agents: List[NarrativeAgent],
        conflict_matrix: ConflictMatrix,
        plot_outline: Any,
        model: str = "None"
    ) -> Tuple[List[ConnectedNarrativeAgent], dict, dict]:
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
        
        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8,
            model=model
        )
        print("Better Author Client Token Usage: ", author_tokens)
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        agents_json, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ConnectedNarrativeAgentList
        )

        if isinstance(agents_json, tuple):
            agents_json = agents_json[0]

        if hasattr(agents_json, "agents"):
            connected = agents_json.agents
        elif isinstance(agents_json, dict):
            raw = agents_json.get("agents") or next((v for v in agents_json.values() if isinstance(v, list)), None)
            connected = [ConnectedNarrativeAgent(**item) if isinstance(item, dict) else item for item in raw]
        else:
            raise ValueError("Failed to parse connected agents")

        return connected, author_tokens, utility_tokens

    async def generate_character_backstories(
        self,
        connected_agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        plot: ExpandedPlotOutline,
        seed: MinimalStorySeed,
        importance_threshold: str = "major",  # "major", "supporting", "all"
        model: str = "None"
    ) -> Tuple[List[DetailedCharacterBackstory], dict, dict]:
        """
        Generate deep backstories for characters in 80k+ novels.
        Genre-aware and importance-filtered.
        """
        
        genre_config = self._get_genre_config(seed.genre)
        
        # Filter characters by importance
        if importance_threshold == "major":
            roles_to_expand = ["protagonist", "antagonist", "mentor"]
        elif importance_threshold == "supporting":
            roles_to_expand = ["protagonist", "antagonist", "mentor", "ally", "rival", "love_interest"]
        else:  # all
            roles_to_expand = None  # include everyone
        
        agents_to_expand = [
            agent for agent in connected_agents
            if roles_to_expand is None or agent.role.lower() in roles_to_expand
        ]
        
        print(f"📖 Generating backstories for {len(agents_to_expand)} characters...")
        
        backstories = []
        
        for agent in agents_to_expand:
            system_prompt = f"""You are crafting a deep character backstory for a {'/'.join(seed.genre)} novel.

GENRE FOCUS AREAS: {', '.join(genre_config.get('character_focus', []))}
BACKSTORY EMPHASIS: {genre_config.get('backstory_emphasis', 'formative experiences')}

CHARACTER CONTEXT:
Name: {agent.name}
Role: {agent.role}
Essence: {agent.essence}
Plot Function: {agent.plot_function}
Character Arc: {agent.agent_arc}

BACKSTORY REQUIREMENTS:

1. FORMATIVE EVENTS (500-800 words of rich narrative)
   - Write 3-5 key life events as immersive scenes
   - Show how each shaped their worldview
   - Connect to their current desires/fears
   - Genre-specific traumas or triumphs
   
2. PSYCHOLOGICAL PROFILE
   - Internal contradictions
   - Coping mechanisms
   - Triggers and pressure points
   - Hidden strengths/weaknesses
   - {genre_config.get('backstory_emphasis', '')}
   
3. RELATIONSHIP HISTORY
   - Pre-story connections with other characters
   - Past relationships informing current behavior
   - Unresolved conflicts or debts
   
4. SECRETS (3-5 items)
   - Information they hide from others
   - Truths that could surface during story
   - Genre-appropriate revelations
   
5. VOICE PROFILE
   - Speech patterns and vocabulary
   - Mannerisms and body language
   - How they express emotion
   
6. SUBPLOT SEEDS
   - Personal story threads that could emerge
   - Internal conflicts needing resolution
   - Relationship arcs waiting to happen

7. GENRE-SPECIFIC DEPTH
   Add fields relevant to {'/'.join(seed.genre)}:
   {self._get_genre_specific_backstory_fields(seed.genre, genre_config)}

Age-appropriate for: {seed.target_audience_age} year olds

OUTPUT FORMAT (JSON):
{DetailedCharacterBackstory.model_json_schema()}

Write immersive, specific backstory. No placeholders.
Output clean JSON with no markdown fences."""

            human_prompt = f"""Create backstory for {agent.name}.

WORLD CONTEXT:
{world.model_dump_json(indent=2)}

PLOT CONTEXT:
Premise: {plot.premise}
Central Question: {plot.central_question}

OTHER CHARACTERS (for relationship history):
{json.dumps([{"name": a.name, "role": a.role, "essence": a.essence} for a in connected_agents if a.name != agent.name], indent=2)}

Generate rich, genre-appropriate backstory."""

            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.85,
                model=model
            )
            print("Better Author Client Token Usage: ", author_tokens)
            
            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)
            
            backstory_data, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=DetailedCharacterBackstory
            )
            if isinstance(backstory_data, tuple):
                backstory_data = backstory_data[0]
            
            backstories.append(backstory_data)
            print(f"  ✓ {agent.name} backstory created")
                
        return backstories, author_tokens, utility_tokens


    def _get_genre_specific_backstory_fields(self, genres: List[str], config: Dict) -> str:
        """Generate genre-specific field suggestions for backstories"""
        suggestions = []
        
        if any(g in ["Fantasy", "High Fantasy", "Urban Fantasy"] for g in genres):
            suggestions.append("  - magical_awakening: First encounter with magic")
            suggestions.append("  - magical_training: Teachers and methods")
            suggestions.append("  - power_origin: Source of abilities")
        
        if any(g in ["Sci-Fi", "Cyberpunk", "Space Opera"] for g in genres):
            suggestions.append("  - technological_expertise: Specialized knowledge")
            suggestions.append("  - first_space_experience: Impact of leaving Earth/home")
            suggestions.append("  - augmentations: Body modifications and why")
        
        if any(g in ["Mystery", "Thriller", "Crime"] for g in genres):
            suggestions.append("  - investigative_method: Unique approach to solving")
            suggestions.append("  - past_cases: Defining investigations")
            suggestions.append("  - criminal_connections: Underworld relationships")
        
        if any(g in ["Romance", "Romantic Comedy"] for g in genres):
            suggestions.append("  - past_heartbreaks: Failed relationships and lessons")
            suggestions.append("  - intimacy_barriers: Why they struggle with connection")
            suggestions.append("  - ideal_partner: What they think they want vs need")
        
        if any(g in ["Horror", "Psychological Horror"] for g in genres):
            suggestions.append("  - trauma_origin: Source of psychological damage")
            suggestions.append("  - fear_manifestation: How terror affects them")
            suggestions.append("  - sanity_threshold: What would break them")
        
        return "\n".join(suggestions) if suggestions else "  - Add genre-appropriate fields"

class ConflictArchitect:
    """Generates multi-dimensional conflict with genre-appropriate complexity"""
    
    def __init__(self):
        pass

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
    ) -> Tuple[ConflictMatrix, dict, dict]:
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
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ConflictMatrix
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        
        scaled_matrix = self._scale_conflict_complexity(
            target_length=seed.target_length,
            base_conflicts=json_data
        )

        return scaled_matrix, author_tokens, utility_tokens


class QualityController:
    """Validates story quality with focus on specificity"""
    
    def __init__(self):
        pass

    async def validate_story_elements(
        self,
        seed: MinimalStorySeed,
        plot: ExpandedPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        model: str = "None"
    ) -> Tuple[QualityReport, dict, dict]:
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
        
        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.7,
            model=model
        )
        print("Author Fast Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=QualityReport
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        
        return json_data, author_tokens, utility_tokens


class StoryTrackerManager:
    """Manages incremental story tracking"""
    
    def __init__(self):
        pass

    async def update_tracker_after_act(
        self,
        act_number: int,
        act_plan: ActPlan,
        author_context: str,
        model: str = "None"
    ) -> Tuple[StoryTracker, dict, dict]:
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

        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,
            model=model
        )
        print("Author Fast Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        # act_tracking_parser = PydanticOutputParser(pydantic_object=ActTracking)
        act_tracking_json, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ActTracking
        )

        if isinstance(act_tracking_json, tuple):
            act_tracking_json = act_tracking_json[0]
        
        if isinstance(act_tracking_json, dict):
            act_tracking = ActTracking(**act_tracking_json)
        else:
            act_tracking = act_tracking_json
        
        return act_tracking, author_tokens, utility_tokens
    
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
    def build_act_planner_context(
        self,
        story_data: Dict[str, Any],
        act_number: int,
        story_mode: str  # "compact", "standard", "epic"
    ) -> Dict[str, Any]:
        """
        Build optimized context for act planner based on story mode.
        """
        
        base_context = {
            "seed": story_data["seed"],
            "final_plot": story_data["final_plot"],
            "connected_agents": story_data["connected_agents"],
            "integrated_world": story_data["integrated_world"],
            "conflict_matrix": story_data["conflict_matrix"],
            "story_tracker": story_data["story_tracker"],
        }
        
        if story_mode == "compact":
            # Compact mode - base context is enough
            return base_context
        
        elif story_mode == "standard":
            # Standard mode - base context is enough
            return base_context
        
        elif story_mode == "epic":
            # Epic mode - add summarized enhancements
            backstories = story_data.get("character_backstories", [])
            world_guide = story_data.get("world_guide")
            subplot_arch = story_data.get("subplot_architecture")
            
            # Create backstory highlights (not full text)
            backstory_highlights = []
            for bs in backstories:
                backstory_highlights.append({
                    "name": bs.name,
                    "key_formative_events": bs.formative_events[:300] + "...",  # First 300 chars
                    "psychological_core": bs.psychological_profile[:200] + "...",
                    "secrets": bs.secrets[:3],  # Top 3 secrets
                    "subplot_seeds": bs.subplot_seeds[:2],  # Top 2 subplot ideas
                })
            
            base_context.update({
                "backstory_highlights": backstory_highlights,
                "key_locations": world_guide.location_dossiers if world_guide else [],
                "npc_pool": world_guide.minor_character_pool if world_guide else [],
                "subplot_architecture": subplot_arch,
                "subplot_integration_for_act": self._extract_subplot_beats_for_act(
                    subplot_arch, act_number
                ) if subplot_arch else []
            })
            
            return base_context
        
        return base_context


    def _extract_subplot_beats_for_act(
        self,
        subplot_arch: SubplotArchitecture,
        act_number: int
    ) -> List[Dict[str, str]]:
        """Extract only the subplot beats relevant to this act"""
        beats = []
        for subplot in subplot_arch.subplots:
            if act_number in subplot.act_integration:
                beats.append({
                    "subplot_title": subplot.title,
                    "character_owner": subplot.character_owner,
                    "beat_this_act": subplot.act_integration[act_number],
                    "thematic_connection": subplot.thematic_connection
                })
        return beats

    

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
    GENRE_CONFIGURATIONS = {
        "Thriller/Suspense": {
            "character_focus": ["hidden_agendas", "pressure_points", "betrayal_capacity", "paranoia_triggers"],
            "world_focus": ["power_structures", "surveillance_systems", "hidden_networks"],
            "subplot_types": ["trust_erosion", "parallel_investigation", "ticking_clock"],
            "backstory_emphasis": "psychological_vulnerabilities",
            "conflict_layers": ["external_threat", "internal_paranoia", "moral_compromise"],
        },
        "Mystery": {
            "character_focus": ["deductive_ability", "observation_skills", "biases", "hidden_connections"],
            "world_focus": ["clue_distribution", "red_herrings", "locked_spaces"],
            "subplot_types": ["suspect_development", "investigator_personal_stake", "secondary_mystery"],
            "backstory_emphasis": "relevant_expertise_and_trauma",
            "conflict_layers": ["intellectual_puzzle", "personal_danger", "ethical_dilemma"],
        },
        "Horror": {
            "character_focus": ["primal_fears", "trauma_history", "denial_mechanisms", "breaking_points"],
            "world_focus": ["atmospheric_dread", "supernatural_rules", "isolation_factors"],
            "subplot_types": ["sanity_degradation", "relationship_breakdown", "origin_mystery"],
            "backstory_emphasis": "past_encounters_with_darkness",
            "conflict_layers": ["supernatural_threat", "psychological_breakdown", "survival_instinct"],
        },
        "Romance": {
            "character_focus": ["emotional_wounds", "intimacy_barriers", "love_language", "relationship_patterns"],
            "world_focus": ["meeting_spaces", "romantic_obstacles", "social_expectations"],
            "subplot_types": ["rival_romance", "family_approval", "career_vs_love", "friendship_dynamics"],
            "backstory_emphasis": "past_relationships_and_heartbreak",
            "conflict_layers": ["external_obstacles", "internal_fears", "misunderstanding"],
        },
        "Comedy": {
            "character_focus": ["comedic_flaws", "misunderstanding_prone", "timing_issues", "verbal_style"],
            "world_focus": ["absurd_situations", "social_conventions_to_mock", "comedic_setpieces"],
            "subplot_types": ["mistaken_identity", "escalating_lie", "rival_suitor"],
            "backstory_emphasis": "embarrassing_history_and_quirks",
            "conflict_layers": ["social_embarrassment", "misunderstanding", "comedic_stakes"],
        },
        "Drama": {
            "character_focus": ["moral_struggles", "relationship_complexity", "identity_crisis", "growth_potential"],
            "world_focus": ["social_pressures", "cultural_context", "institutional_forces"],
            "subplot_types": ["family_tension", "friendship_evolution", "career_struggle"],
            "backstory_emphasis": "formative_relationships_and_choices",
            "conflict_layers": ["interpersonal", "societal", "internal_transformation"],
        },
        "Tragedy": {
            "character_focus": ["fatal_flaw", "hubris", "inevitable_downfall", "noble_qualities"],
            "world_focus": ["fate_mechanisms", "social_judgment", "inescapable_circumstances"],
            "subplot_types": ["doomed_relationship", "failed_redemption", "legacy_destruction"],
            "backstory_emphasis": "seeds_of_downfall",
            "conflict_layers": ["character_vs_fate", "tragic_irony", "moral_consequences"],
        },
        "Adventure": {
            "character_focus": ["courage", "resourcefulness", "loyalty", "adaptability", "personal_quest"],
            "world_focus": ["exotic_locations", "physical_challenges", "discovery_opportunities"],
            "subplot_types": ["treasure_hunt", "rescue_mission", "rival_adventurer"],
            "backstory_emphasis": "origin_of_wanderlust",
            "conflict_layers": ["physical_obstacles", "rival_forces", "personal_growth"],
        },
        "Crime": {
            "character_focus": ["moral_ambiguity", "criminal_expertise", "code_of_honor", "past_crimes"],
            "world_focus": ["underworld_hierarchy", "law_enforcement", "criminal_networks"],
            "subplot_types": ["double_cross", "redemption_arc", "turf_war"],
            "backstory_emphasis": "criminal_origin_story",
            "conflict_layers": ["heist_execution", "law_vs_outlaw", "honor_among_thieves"],
        },
        "Fantasy": {
            "character_focus": ["magical_ability", "destiny_connection", "ancient_lineage", "chosen_status"],
            "world_focus": ["magic_systems", "mythical_creatures", "ancient_prophecies", "realm_politics"],
            "subplot_types": ["magical_training", "political_intrigue", "artifact_quest"],
            "backstory_emphasis": "magical_heritage_and_training",
            "conflict_layers": ["good_vs_evil", "magical_power", "destiny_vs_choice"],
        },
        "Sci-Fi": {
            "character_focus": ["technological_aptitude", "adaptation_to_future", "ethical_stance", "scientific_mind"],
            "world_focus": ["technology_systems", "alien_species", "future_politics", "scientific_laws"],
            "subplot_types": ["tech_malfunction", "first_contact", "AI_awakening"],
            "backstory_emphasis": "scientific_background_and_specialization",
            "conflict_layers": ["human_vs_technology", "exploration_danger", "ethical_dilemma"],
        },
    }
    
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        
        # Initialize component systems
        self.world_builder = WorldBuilder(GENRE_CONFIGURATIONS=self.GENRE_CONFIGURATIONS)
        self.agent_genesis = AgentGenesis(GENRE_CONFIGURATIONS=self.GENRE_CONFIGURATIONS)
        self.conflict_architect = ConflictArchitect()
        self.quality_controller = QualityController()
        self.tracker_manager = StoryTrackerManager()

    def validate_act_count(self, act_count: int, structure_name: str) -> int:
        if structure_name == "Three Act Structure":
            return 3
        elif structure_name == "Fichtean Curve":
            return 3
        elif structure_name == "Save the Cat Beat Sheet":
            return 3
        elif structure_name == "Freytag's Pyramid":
            return 5
        elif structure_name == "The Hero's Journey":
            return 3
        elif act_count < 3:
            return 3
        elif act_count > 5:
            return 5
        else:
            return act_count
        
    def create_minimal_seed(self, user_context: Dict[str, Any]) -> MinimalStorySeed:
        """Extract and normalize minimal story seed from user context"""

        # ---------- GENRES ----------
        genres = user_context.get("Genre") or ["Fiction"]
        if not isinstance(genres, list):
            genres = [genres]

        sub_genre = user_context.get("Sub-Genre") or []
        if not isinstance(sub_genre, list):
            sub_genre = [sub_genre]

        themes = user_context.get("Additional Themes") or []
        if not isinstance(themes, list):
            themes = [themes]

        # ---------- PROTAGONIST ----------
        protagonist_specs = {
            k: v for k, v in {
                "name": user_context.get("protagonist_name"),
                "age": user_context.get("protagonist_age"),
                "gender": user_context.get("protagonist_gender"),
                "archetype": user_context.get("protagonist_archetype"),
                "core_trait": user_context.get("protagonist_core_trait"),
                "background": user_context.get("protagonist_background"),
                "desire": user_context.get("protagonist_desire"),
                "fear": user_context.get("protagonist_fear"),
                "relationships": user_context.get("protagonist_relationships"),
                "physical_description": user_context.get("protagonist_physical_description"),
            }.items()
            if v not in (None, "", [])
        }

        # ---------- TARGET LENGTH ----------
        word_count_map = {
            "Short Long Story (7,500 - 15,000 words)": 12000,
            "Novelette (15,000 - 25,000 words)": 20000,
            "Novella (25,000 - 40,000 words)": 36000,
            "Novel Chapter (40,000 - 60,000 words)": 50000,
            "Full Novel (60,000 - 90,000 words)": 80000,
            "Epic / Series (90,000 - 150,000+ words)": 125000,
        }

        raw_length = user_context.get("Length")
        target_length = word_count_map.get(raw_length, 20000)

        if isinstance(raw_length, int) and raw_length > 5000:
            target_length = raw_length

        # ---------- TITLE ----------
        excluded_titles = {
            None, "", " ", "None", "NULL", "Null", "null",
            "Untitled Story", "Unitled story", "untitled story",
            "Nill", "NILL", "nill"
        }

        title = user_context.get("Title")
        title = None if title in excluded_titles else title

        # ---------- TONE ----------
        tone_map = {
            0: "Very dark",
            20: "Dark",
            40: "Balanced",
            60: "Hopeful",
            80: "Light",
            100: "Whimsical",
        }

        raw_tone = user_context.get("Tone", "Balanced")
        tone = tone_map.get(raw_tone, raw_tone if isinstance(raw_tone, str) else "Balanced")

        # ---------- POV ----------
        pov = user_context.get("POV", "Third-person")

        # ---------- PROSE STYLE ----------
        guide_prose = user_context.get("Guide Prose") or []
        prose_style = (
            guide_prose[0].strip()
            if isinstance(guide_prose, list)
            and guide_prose
            and isinstance(guide_prose[0], str)
            and guide_prose[0].strip()
            else "Standard narrative"
        )

        # ---------- AUDIENCE ----------
        target_audience_age = user_context.get("target_audience_age", 13)
        try:
            target_audience_age = int(target_audience_age)
        except (TypeError, ValueError):
            target_audience_age = 13

        target_audience_age = max(8, min(target_audience_age, 18))

        # ---------- ACT COUNT ----------
        act_count = 3 if target_length <= 75000 else 5

        story_structure = find_best_structure_name(genres)
        act_count = self.validate_act_count(
            act_count=act_count,
            structure_name=story_structure
        )

        # ---------- FINAL SEED ----------
        return MinimalStorySeed(
            title=title,
            pov=pov,
            tone=tone,
            genre=genres,
            sub_genre=sub_genre,
            setting=user_context.get("Setting") or None,
            prose_style=prose_style,
            themes=themes,
            target_audience_age=target_audience_age,
            target_length=target_length,
            protagonist_specs=protagonist_specs,
            act_count=act_count,
            story_structure=story_structure,
        )


    async def generate_minimal_plot_outline(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        model: str = "None",
        max_retries: int = 4
    ) -> Tuple[MinimalPlotOutline, dict, dict]:
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

        system_prompt = f"""Create minimal plot outline for {'/'.join(seed.genre)} novel.

REQUIRED FIELDS:
- title: Evocative, thematic
- premise: 1-2 sentence hook with character desire + impossible stakes
- central_question: Moral/emotional dilemma (not plot mechanics)
- narrative_arc: Flowing prose summary (200-300 words) - paint emotional journey
- required_character_roles: {dynamic_min_roles}+ specific roles (NOT generic labels)
  Example: "guilt-ridden former expedition leader" not "mentor"

RULES:
- Roles MUST be specific to world/premise
- Each role = function + psychology + relationship potential
- Target age: {seed.target_audience_age} - adjust complexity accordingly
- Structure: {structure_template}

OUTPUT: Valid MinimalPlotOutline JSON. No markdown fences.

{MinimalPlotOutline.model_json_schema()}"""


        human_prompt = f"""Seed: {seed.model_dump_json(indent=2)}
World: {world_foundation.model_dump_json(indent=2)}

Generate plot outline."""

        # --- 2. Retry loop ---
        attempt = 0
        last_valid_response = None
        
        while attempt <= max_retries:
            attempt += 1
            
            # Only regenerate if no valid partial result
            if attempt > 1 and last_valid_response:
                fix_prompt = f"""Previous attempt had {len(last_valid_response.required_character_roles)} roles.
    Minimum required: {dynamic_min_roles}

    Add {dynamic_min_roles - len(last_valid_response.required_character_roles)} more specific roles.

    Previous roles: {json.dumps(last_valid_response.required_character_roles)}
    Previous plot: {last_valid_response.model_dump_json(indent=2)}

    Expand roles list ONLY. Keep rest unchanged."""
                
                response, author_tokens = await better_author_client(
                    system_prompt="Add missing character roles to existing plot.",
                    human_prompt=fix_prompt,
                    llm_temp=0.7,
                    model=model
                )
            else:
                # First attempt - full generation
                response, author_tokens = await better_author_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=0.9,
                    model=model
                )
            last_tokens = author_tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                outline, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=MinimalPlotOutline
                )

                if isinstance(outline, tuple):
                    outline = outline[0]
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            role_count = len(outline.required_character_roles)
            if role_count >= dynamic_min_roles:
                print("Better Author Client Token Usage: ", author_tokens)
                return outline, author_tokens, utility_tokens
        
            last_valid_response = outline
            print(f"→ Produced {role_count} roles: {outline.required_character_roles}")

            # --- 4. Check role thresholds ---
            if role_count >= dynamic_min_roles:
                print("Better Author Client Token Usage: ", author_tokens)
                print(f"✓ Accepted. Role count meets threshold ({dynamic_min_roles}).")
                return outline, author_tokens, utility_tokens
            else:
                print(f"⚠️ Insufficient roles ({role_count} < {dynamic_min_roles}). Retrying...")
                await asyncio.sleep(0.8 * attempt)

        # --- 5. If max retries exhausted ---
        print("🚨 Max retries exhausted. Returning last valid attempt (even if insufficient roles).")
        return outline, last_tokens, utility_tokens
    
    async def expand_plot_with_enhancements(
        self,
        minimal_plot: MinimalPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        integrated_world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        seed: MinimalStorySeed,
        backstories: List[DetailedCharacterBackstory],
        world_guide: EnhancedWorldGuide,
        subplot_architecture: SubplotArchitecture,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict, dict]:
        """
        Expand plot AFTER enhancements are generated.
        Act summaries and narrative flow can now reference:
        - Specific backstory events
        - Documented locations from world guide
        - Named NPCs from character pool
        - Subplot threads and their integration points
        """
        
        target_act_count = seed.act_count
        structure_template = find_best_structure(seed.story_structure)
        
        # Extract key details from enhancements for prompt
        backstory_highlights = []
        for bs in backstories[:5]:  # Top 5 characters
            backstory_highlights.append({
                "name": bs.name,
                "key_events": bs.formative_events,
                "secrets": bs.secrets,
                "subplot_seeds": bs.subplot_seeds
            })
        
        location_names = [loc.get("name", "Unknown") for loc in world_guide.location_dossiers[:10]]
        npc_pool = [npc.get("name", "Unknown") for npc in world_guide.minor_character_pool[:15]]
        
        subplot_summaries = []
        for sp in subplot_architecture.subplots:
            subplot_summaries.append({
                "title": sp.title,
                "premise": sp.premise,
                "owner": sp.character_owner,
                "act_moments": sp.act_integration
            })
        
        system_prompt = f"""You are expanding a minimal plot into a COMPLETE story blueprint for a {'/'.join(seed.genre)} novel.

YOU NOW HAVE ACCESS TO RICH STORY ENHANCEMENTS:

🎭 CHARACTER BACKSTORIES:
{json.dumps(backstory_highlights, indent=2)}

🗺️ DOCUMENTED LOCATIONS:
{', '.join(location_names)}

👥 NPC POOL (ready to deploy):
{', '.join(npc_pool[:10])}

🎬 SUBPLOT THREADS:
{json.dumps(subplot_summaries, indent=2)}

YOUR MISSION:
Write act summaries and narrative flow that NATURALLY REFERENCE these enhancements.

EXAMPLES OF INTEGRATION:
❌ BAD: "The hero faces a challenge and grows stronger"
✅ GOOD: "At the Crimson Observatory, Kael confronts the memory of his father's betrayal (backstory event from age 12), while Merchant Yara (NPC) offers cryptic warnings about the Convergence"

❌ BAD: "A romantic subplot develops"
✅ GOOD: "The romance subplot ('Fractured Trust') reaches its midpoint when Elara discovers Kael's secret mission, forcing them to decide between love and duty at the abandoned Temple of Echoes"

STRUCTURE: {structure_template}
TARGET: {target_act_count} acts

ACT SUMMARIES (200-300 words each):
- Reference specific LOCATIONS by name
- Mention BACKSTORY events when relevant
- Deploy NPCs from character pool organically
- Integrate SUBPLOT beats at specified act moments
- Use character NAMES constantly (not "the protagonist")
- Show concrete EVENTS with consequences

NARRATIVE FLOW (500-800 words):
- Weave all subplots into main arc
- Reference how backstories inform current choices
- Show characters moving through documented locations
- Include NPCs in the story fabric
- Complete emotional journey with all threads resolved

Age-appropriate: {seed.target_audience_age}+

OUTPUT FORMAT (JSON):
{ExpandedPlotOutline.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Expand this plot WITH all enhancements integrated.

MINIMAL PLOT:
{minimal_plot.model_dump_json(indent=2)}

CHARACTERS:
{json.dumps([{"name": a.name, "role": a.role, "arc": a.agent_arc} for a in connected_agents], indent=2)}

WORLD:
{integrated_world.model_dump_json(indent=2)}

CONFLICTS:
{conflict_matrix.model_dump_json(indent=2)}

THEMES: {', '.join(seed.themes)}

Write act summaries that feel like a complete story bible - specific, vivid, integrated."""

        max_retries = 4
        attempt = 0

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 Enhanced plot expansion attempt {attempt}/{max_retries + 1}")

            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.85,
                model=model
            )
            print("Better Author Client Token Usage: ", author_tokens)

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )
 
                if isinstance(json_data, tuple):
                    json_data = json_data[0]
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {target_act_count})")

            if generated_count == target_act_count:
                print("✓ Accepted. Act count matches + enhancements integrated.")
                return json_data, author_tokens, utility_tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {target_act_count}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        print("🚨 Max retries exhausted. Returning last attempt.")
        return json_data, author_tokens, utility_tokens
    
    async def expand_plot_outline(
        self,
        minimal_plot: MinimalPlotOutline,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        seed: MinimalStorySeed,
        model: str = "None",
        max_retries: int = 4
    ) -> Tuple[ExpandedPlotOutline, dict, dict]:
        """Expand minimal plot with creative flexibility and enforce act count matching."""

        target_act_count = seed.act_count
        print(f"→ Target act count: {target_act_count}")

        structure_template = find_best_structure(seed.story_structure)

        system_prompt = f"""
You are a disciplined narrative generator.
Expand this minimal plot outline into a FULL story blueprint (for a story novel).
You have full creative freedom within these boundaries, make it unique and compelling. 
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

Your Plot is the Story Bible that will be followed to produce a World Class Novel.
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
            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.9,
                model=model
            )
            last_tokens = author_tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON output ---
            try:
                json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )

                if isinstance(json_data, tuple):
                    json_data = json_data[0]
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            # --- VALIDATION: ACT COUNT ---
            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {target_act_count})")

            if generated_count == target_act_count:
                print("Better Author Client Token Usage: ", author_tokens)
                print("✓ Accepted. Act summary count matches the target.")
                return json_data, author_tokens, utility_tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {target_act_count}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        # --- MAX RETRIES EXHAUSTED ---
        print("🚨 Max retries exhausted. Returning last generated outline (mismatch unresolved).")
        return json_data, last_tokens, utility_tokens

    async def generate_subplot_architecture(
        self,
        plot: ExpandedPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        seed: MinimalStorySeed,
        act_count: int,
        model: str = "None"
    ) -> Tuple[SubplotArchitecture, dict, dict]:
        """
        Generate 2-4 interwoven subplots for 80k+ novels.
        Genre-aware and character-driven.
        """
        
        genre_config = self._get_genre_config(seed.genre)
        
        # Determine subplot count based on length
        if seed.target_length < 90000:
            subplot_count = "2-3"
        else:
            subplot_count = "3-4"
        
        system_prompt = f"""You are architecting subplots for a {'/'.join(seed.genre)} novel.

GENRE SUBPLOT TYPES: {', '.join(genre_config.get('subplot_types', []))}

Create {subplot_count} compelling subplots that:
1. Emerge from character backstories/desires
2. Reinforce main themes
3. Create organic intersections with main plot
4. Have clear setup → development → resolution arcs
5. Feel genre-appropriate

SUBPLOT STRUCTURE:

Each subplot needs:
- subplot_id: Unique identifier (e.g., "subplot_1")
- title: Evocative subplot name
- premise: 1-2 sentence hook
- character_owner: Primary driver (must be from main cast)
- supporting_characters: Others involved
- thematic_connection: How it reinforces main themes
- act_integration: Specific moments per act
  {{
    1: "Setup: What gets introduced",
    2: "Development: How it complicates",
    3: "Midpoint: Major turn or reveal",
    4: "Crisis: Subplot stakes peak" (if 5 acts),
    {act_count}: "Resolution: How it concludes"
  }}
- resolution_type: triumph/tragedy/transformation/bittersweet/etc.

INTEGRATION STRATEGY:
Explain how subplots weave together without overwhelming main plot.

GENRE GUIDANCE:
{self._get_genre_subplot_guidance(seed.genre, genre_config)}

Age-appropriate for: {seed.target_audience_age} year olds

OUTPUT FORMAT (JSON):
{SubplotArchitecture.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Design {subplot_count} subplots for this story.

MAIN PLOT:
{plot.model_dump_json(indent=2)}

AVAILABLE CHARACTERS:
{json.dumps([{"name": a.name, "role": a.role, "arc": a.agent_arc} for a in connected_agents], indent=2)}

THEMES: {', '.join(seed.themes)}
TOTAL ACTS: {act_count}

Create interwoven, genre-appropriate subplots."""

        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        print("Author Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        subplot_arch, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=SubplotArchitecture
        )

        if isinstance(subplot_arch, tuple):
            subplot_arch = subplot_arch[0]
        
        return subplot_arch, author_tokens, utility_tokens


    def _get_genre_subplot_guidance(self, genres: List[str], config: Dict) -> str:
        """Provide genre-specific subplot guidance"""
        guidance = []
        
        if any(g in ["Mystery", "Thriller/Suspense"] for g in genres):
            guidance.append("- Subplots can introduce red herrings or parallel investigations")
            guidance.append("- One subplot should heighten personal stakes for protagonist")
        
        if any(g in ["Romance"] for g in genres):
            guidance.append("- At least one subplot should involve relationship obstacles")
            guidance.append("- Consider rival romance or family approval subplot")
        
        if any(g in ["Fantasy", "Sci-Fi", "Adventure"] for g in genres):
            guidance.append("- Subplots can explore world-building or magic/tech systems")
            guidance.append("- Consider political intrigue or discovery subplots")
        
        if any(g in ["Drama", "Tragedy"] for g in genres):
            guidance.append("- Subplots should deepen character relationships")
            guidance.append("- Focus on interpersonal conflicts and growth")
        
        if any(g in ["Horror"] for g in genres):
            guidance.append("- Subplots can isolate characters or reveal supernatural rules")
            guidance.append("- Build paranoia through relationship breakdowns")
        
        return "\n".join(guidance) if guidance else "- Create character-driven subplots"
    
    async def _enforce_age_appropriateness(
        self,
        final_plot: Any,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Tuple[AgeAppropriatenessReport, dict, dict]:
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

Finally rate the plot for target Audience: 'A' for ages 8+, 'T' for ages 13+, 'M' for ages 18+, regardless of given Target audience age, it may be lower than given.

Output Format:
{AgeAppropriatenessReport.model_json_schema()}"""

        human_prompt = f"""Target age: {target_age}

Plot to check:
{final_plot.model_dump_json(indent=2)}

Review and rate."""

        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.6,
            model=model
        )
        print("Author Fast Client Token Usage: ", author_tokens)

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=AgeAppropriatenessReport
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]

        return json_data, author_tokens, utility_tokens
    
    async def refine_plot_with_quality_feedback(
        self,
        plot: ExpandedPlotOutline,
        quality_report: QualityReport,
        agents: List[ConnectedNarrativeAgent],
        world: IntegratedWorld,
        conflict_matrix: ConflictMatrix,
        act_count_target: int,
        model: str = "None"
    ) -> Tuple[ExpandedPlotOutline, dict, dict]:
        """Refine plot - ONLY send what needs fixing"""
        
        if not quality_report.concerns:
            return plot, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, None
        
        # Extract ONLY problematic sections
        vague_acts = []
        if any('vague' in c.lower() for c in quality_report.concerns):
            # Identify which acts are vague
            for i, act_summary in enumerate(plot.act_summaries, 1):
                if len(act_summary.split()) < 150:  # Too short = vague
                    vague_acts.append(i)
        
        # Build minimal context - ONLY relevant agents/locations
        mentioned_chars = set()
        mentioned_locs = set()
        for summary in plot.act_summaries:
            for agent in agents:
                if agent.name.lower() in summary.lower():
                    mentioned_chars.add(agent.name)
            # Extract location mentions (rough heuristic)
            words = summary.split()
            for loc_word in world.model_dump().get('key_locations', []):
                if any(loc_word.lower() in w.lower() for w in words):
                    mentioned_locs.add(loc_word)
        
        relevant_agents = [a.model_dump() for a in agents if a.name in mentioned_chars]
        
        system_prompt = f"""Refine plot based on editorial feedback. Target: {act_count_target} acts.

ADDRESS THESE CONCERNS:
{chr(10).join(f"- {c}" for c in quality_report.concerns)}

APPLY THESE FIXES:
{chr(10).join(f"- {r}" for r in quality_report.recommendations)}

OUTPUT: Same ExpandedPlotOutline structure. JSON only:
{ExpandedPlotOutline.model_json_schema()}"""

        human_prompt = f"""ACTS NEEDING SPECIFICITY: {vague_acts if vague_acts else 'All'}

Current plot:
Title: {plot.title}
Premise: {plot.premise}
Act summaries: {json.dumps(plot.act_summaries, indent=2)}

Relevant agents: {json.dumps(relevant_agents[:10], indent=2)}  # Cap at 10
Central conflict: {conflict_matrix.central_conflict}

Refine with concrete names/locations/events."""
        max_retries = 2
        attempt = 0
        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 LLM Refined outline expansion attempt {attempt}/{max_retries + 1}")

            # --- LLM CALL ---
            response, author_tokens = await author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.75,
                model=model
            )
            print("Author Client Token Usage: ", author_tokens)
            last_tokens = author_tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON output ---
            try:
                json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )

                if isinstance(json_data, tuple):
                    json_data = json_data[0]
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            # --- VALIDATION: ACT COUNT ---
            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {act_count_target})")

            if generated_count == act_count_target:
                print("✓ Accepted. Act summary count matches the target.")
                return json_data, author_tokens, utility_tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {act_count_target}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        # --- MAX RETRIES EXHAUSTED ---
        print("🚨 Max retries exhausted. Returning last generated outline (mismatch unresolved).")
        return plot, last_tokens, utility_tokens
    
    async def _create_story_tracker(
        self,
        act_count: int,
        final_plot: ExpandedPlotOutline,
        connected_agents: List[ConnectedNarrativeAgent],
        conflict_matrix: ConflictMatrix,
        model: str
    ) -> Tuple[StoryTracker, dict, dict]:
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

        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
        )
        print("Author Fast Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        json_data, utility_tokens = await StoryHelpers.load_json_with_retry(text=clean_resp, parser=StoryTracker)

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        return json_data, author_tokens, utility_tokens
    


    async def _compact_planning_flow(
        self,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Dict[str, Any]:
        """
        Streamlined planning for shorter works (10k-40k):
        - Single-pass plot generation
        - Simplified conflict matrix
        - Skip quality refinement
        - Minimal backstory
        """
        utility_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print("\n🌍 Phase 1: World foundation...")
        world_foundation, world_tokens, utility_tokens = await self.world_builder.generate_world_foundation(
            seed, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ World created")
        
        print("\n📖 Phase 2: Compact plot generation...")
        # Use simplified plot generation (merge minimal + expanded)
        compact_plot, plot_tokens, utility_tokens = await self._generate_compact_plot(
            seed, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Plot: '{compact_plot.title}' ({seed.act_count} acts)")
        
        print("\n👥 Phase 3: Character generation...")
        narrative_agents, agent_tokens, utility_tokens = await self.agent_genesis.generate_narrative_agents(
            seed, world_foundation, compact_plot.required_character_roles, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ {len(narrative_agents)} characters created")
        
        print("\n⚔️ Phase 4: Simplified conflicts...")
        conflict_matrix, conflict_tokens, utility_tokens = await self._generate_simplified_conflict(
            compact_plot, seed, narrative_agents, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Core conflict established")
        
        print("\n🔗 Phase 5: Connect elements...")
        connected_agents, connect_tokens, utility_tokens = await self.agent_genesis.connect_agents_to_plot(
            narrative_agents, conflict_matrix, compact_plot, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        integrated_world, integrate_tokens, utility_tokens = await self.world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Elements connected")
        
        print("\n📊 Phase 6: Age filter...")
        age_report, age_tokens, utility_tokens = await self._enforce_age_appropriateness(
            final_plot=compact_plot,
            seed=seed,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        if not age_report.plot_outline_pass:
            raise ValueError("Age filter failed")
        
        if age_report.age_rating == 'A':
            seed.target_audience_age = 8
        elif age_report.age_rating == 'T':
            seed.target_audience_age = 13
        elif age_report.age_rating == 'M':
            seed.target_audience_age = 18
        print(f"✓ Content validated for age {seed.target_audience_age}+")

        # Skip quality refinement for compact mode
        quality_report = QualityReport(
            strengths=["Compact narrative suitable for length"],
            concerns=[],
            recommendations=[],
            overall_assessment="Streamlined for shorter work"
        )

        print("\n📈 Phase 7: Story tracker...")
        story_tracker, tracker_tokens, utility_tokens = await self._create_story_tracker(
            act_count=seed.act_count,
            final_plot=compact_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix,
            model=model
        )
        StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)

        # Store in memory
        await self._store_planning_documents(
            seed, None, compact_plot, world_foundation,
            narrative_agents, conflict_matrix, connected_agents,
            integrated_world, quality_report, compact_plot.title, story_tracker
        )

        total_tokens = self._calculate_total_tokens([
            world_tokens, agent_tokens, plot_tokens, conflict_tokens,
            connect_tokens, integrate_tokens, age_tokens, tracker_tokens
        ])

        print("\n" + "=" * 60)
        print("✅ COMPACT PLANNING COMPLETE")
        print(f"📊 Total tokens: {total_tokens['total_tokens']:,}")
        print("=" * 60)

        return {
            "seed": seed,
            "minimal_plot": compact_plot,  # Same as final for compact
            "final_plot": compact_plot,
            "author_tokens": total_tokens,
            "utility_tokens": utility_token_usage,
        }
    
    async def orchestrate_story_planning(
        self,
        user_context: Dict[str, Any],
        model: str = "None"
    ) -> Dict[str, Any]:
        """
        Main orchestration with length-based routing:
        - 10k-40k: Compact flow
        - 40k-80k: Standard flow
        - 80k+: Epic flow
        """
        
        print("🎬 ORCHESTRATING WORLD-CLASS STORY PLANNING")
        print("=" * 60)
        
        seed = self.create_minimal_seed(user_context)
        
        # Route based on target length
        if seed.target_length <= 40000:
            print("📘 COMPACT MODE (10k-40k words)")
            return await self._compact_planning_flow(seed, model)
        elif seed.target_length >= 80000:
            print("📕 EPIC MODE (80k+ words)")
            return await self._epic_planning_flow(seed, model)
        else:
            print("📗 STANDARD MODE (40k-80k words)")
            return await self._standard_planning_flow(seed, model)
        
    def _get_genre_config(self, genres: List[str]) -> Dict[str, Any]:
        """
        Get merged genre configuration for multi-genre stories.
        Primary genre takes precedence, secondary adds elements.
        """
        if not genres:
            return self.GENRE_CONFIGURATIONS.get("Drama", {})  # Default fallback
        
        primary_genre = genres[0]
        
        # Find matching configuration (handle parent genre extraction)
        config = None
        for genre_key in self.GENRE_CONFIGURATIONS.keys():
            if genre_key in primary_genre or primary_genre in genre_key:
                config = self.GENRE_CONFIGURATIONS[genre_key].copy()
                break
        
        if not config:
            config = self.GENRE_CONFIGURATIONS.get("Drama", {})
        
        # Merge secondary genres if present
        if len(genres) > 1:
            for secondary in genres[1:]:
                for genre_key in self.GENRE_CONFIGURATIONS.keys():
                    if genre_key in secondary or secondary in genre_key:
                        secondary_config = self.GENRE_CONFIGURATIONS[genre_key]
                        # Add unique elements from secondary
                        for key in ["character_focus", "world_focus", "subplot_types"]:
                            if key in secondary_config:
                                config[key] = list(set(config.get(key, []) + secondary_config[key]))
                        break
        
        return config
    
    ## 9. Generate Compact Plot (for 10k-40k)
    async def _generate_compact_plot(
        self,
        seed: MinimalStorySeed,
        world_foundation: WorldFoundation,
        model: str = "None"
    ) -> Tuple[ExpandedCompactPlotOutline, dict, dict]:
        """
        Single-pass plot generation for compact stories.
        Combines minimal + expanded into one streamlined output.
        """
        
        genre_config = self._get_genre_config(seed.genre)
        structure_template = find_best_structure(seed.story_structure)
        target_act_count = seed.act_count
        # Calculate required roles dynamically (fewer for short stories)
        target_len = seed.target_length
        min_len, max_len = 10000, 40000
        norm = max(0.0, min(1.0, (target_len - min_len) / (max_len - min_len)))
        dynamic_min_roles = int(3 + norm * (7 - 3))  # 3-7 roles for compact
        
        system_prompt = f"""Create a complete plot outline for a {'/'.join(seed.genre)} story ({seed.target_length} words).

GENRE FOCUS: {', '.join(genre_config.get('conflict_layers', []))}
STRUCTURE: {structure_template}

REQUIRED OUTPUT:
- title: Evocative, thematic
- premise: 1-2 sentence hook
- central_question: Core dilemma
- required_character_roles: {dynamic_min_roles}+ specific roles (not generic)
- act_summaries: [{seed.act_count} summaries, 150-200 words each in flowing prose]
- narrative_flow: 300-500 words of complete emotional arc

RULES FOR COMPACT STORIES:
- Focus on single clear conflict thread
- Limit subplot complexity
- Every character must earn their presence
- Streamlined but emotionally complete
- Age-appropriate: {seed.target_audience_age}+

OUTPUT FORMAT (JSON):
{ExpandedCompactPlotOutline.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Generate complete plot.

Seed: {seed.model_dump_json(indent=2)}
World: {world_foundation.model_dump_json(indent=2)}

Create tight, focused narrative."""
        attempt = 0
        last_tokens = {}
        max_retries = 4
        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 LLM outline expansion attempt {attempt}/{max_retries + 1}")

            # --- LLM CALL ---
            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.9,
                model=model
            )
            last_tokens = author_tokens

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            # --- Parse JSON output ---
            try:
                json_data, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ExpandedPlotOutline
                )

                if isinstance(json_data, tuple):
                    json_data = json_data[0]
            except Exception as e:
                print(f"⚠️ Parsing failed ({e}). Retrying...")
                await asyncio.sleep(0.6 * attempt)
                continue

            # --- VALIDATION: ACT COUNT ---
            generated_count = len(json_data.act_summaries)
            print(f"→ Generated {generated_count} act summaries (target: {target_act_count})")

            if generated_count == target_act_count:
                print("Better Author Client Token Usage: ", author_tokens)
                print("✓ Accepted. Act summary count matches the target.")
                return json_data, author_tokens, utility_tokens

            print(f"⚠️ Act count mismatch ({generated_count} != {target_act_count}). Retrying...")
            await asyncio.sleep(0.8 * attempt)

        # --- MAX RETRIES EXHAUSTED ---
        print("🚨 Max retries exhausted. Returning last generated outline (mismatch unresolved).")
        return json_data, last_tokens, utility_tokens

        # response, author_tokens = await better_author_client(
        #     system_prompt=system_prompt,
        #     human_prompt=human_prompt,
        #     llm_temp=0.9,
        #     model=model
        # )
        
        # clean_resp = StoryHelpers._extract_content(response)
        # clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        # plot, utility_tokens = await StoryHelpers.load_json_with_retry(
        #     text=clean_resp,
        #     parser=ExpandedCompactPlotOutline
        # )
        # if isinstance(plot, tuple):
        #     plot = plot[0]
        
        # return plot, author_tokens, utility_tokens


    ## 10. Simplified Conflict (for 10k-40k)
    async def _generate_simplified_conflict(
        self,
        plot: ExpandedCompactPlotOutline,
        seed: MinimalStorySeed,
        narrative_agents: List[NarrativeAgent],
        world_foundation: WorldFoundation,
        model: str = "None"
    ) -> Tuple[ConflictMatrix, dict, dict]:
        """
        Streamlined conflict for compact stories - focus on core tension only.
        """
        
        genre_config = self._get_genre_config(seed.genre)
        
        system_prompt = f"""Analyze core conflict for {'/'.join(seed.genre)} story (compact length).

FOCUS: Single clear conflict with 1-2 layers maximum.

GENRE CONFLICT TYPE: {', '.join(genre_config.get('conflict_layers', [])[:2])}

OUTPUT FORMAT (JSON):
{ConflictMatrix.model_json_schema()}

Keep it focused - one central conflict, one escalation path.
Age-appropriate: {seed.target_audience_age}+

Output clean JSON with no markdown fences."""

        human_prompt = f"""Identify core conflict.

Plot: {plot.model_dump_json(indent=2)}
Agents: {json.dumps([a.model_dump() for a in narrative_agents], indent=2)}
World: {world_foundation.model_dump_json(indent=2)}

Streamlined conflict analysis."""

        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85,
            model=model
        )
        print("Author Fast Client Token Usage: ", author_tokens)
        
        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)
        
        conflict, utility_tokens = await StoryHelpers.load_json_with_retry(
            text=clean_resp,
            parser=ConflictMatrix
        )
        if isinstance(conflict, tuple):
            conflict = conflict[0]
        
        return conflict, author_tokens, utility_tokens

    ## 11. Epic Planning Flow (80k+)
    async def _epic_planning_flow(
        self,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Dict[str, Any]:
        """
        Enhanced planning for epic novels (80k+):
        Order: foundations → enhancements → integrated expansion
        """
        # structure_template = find_best_structure_name(seed.genre)
        print("\n📋 Phase 1: Story seed ready")
        utility_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print(f"✓ Genres: {seed.genre} | Target: {seed.target_length} words")
        
        # ===== FOUNDATION LAYER =====
        print("\n🌍 Phase 2: World foundation...")
        world_foundation, world_tokens, utility_tokens = await self.world_builder.generate_world_foundation(
            seed, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ World foundation created")
        
        print("\n📖 Phase 3: Minimal plot outline...")
        minimal_plot, plot_tokens, utility_tokens = await self.generate_minimal_plot_outline(
            seed, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Minimal plot: '{minimal_plot.title}' ({seed.act_count} acts)")
        print(f"✓ Requires {len(minimal_plot.required_character_roles)} roles")
        
        print("\n👥 Phase 4: Narrative agents...")
        narrative_agents, agent_tokens, utility_tokens = await self.agent_genesis.generate_narrative_agents(
            seed, world_foundation, minimal_plot.required_character_roles, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ {len(narrative_agents)} narrative agents created")
        
        print("\n⚔️ Phase 5: Conflict matrix...")
        conflict_matrix, conflict_tokens, utility_tokens = await self.conflict_architect.generate_conflict_layers(
            minimal_plot, seed, narrative_agents, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Multi-dimensional conflict matrix created")
        
        print("\n🔗 Phase 6: Connecting base elements...")
        connected_agents, connect_tokens, utility_tokens = await self.agent_genesis.connect_agents_to_plot(
            narrative_agents=narrative_agents,
            conflict_matrix=conflict_matrix,
            plot_outline=minimal_plot,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        integrated_world, integrate_tokens, utility_tokens = await self.world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Base elements connected")
        
        # ===== EPIC ENHANCEMENT LAYER =====
        print("\n" + "=" * 60)
        print("🌟 EPIC ENHANCEMENTS (Before Final Plot Expansion)")
        print("=" * 60)
        
        print("\n📖 Enhancement 1: Character backstories...")
        backstories, backstory_tokens, utility_tokens = await self.agent_genesis.generate_character_backstories(
            connected_agents=connected_agents,
            world=integrated_world,
            plot=minimal_plot,
            seed=seed,
            importance_threshold="supporting",
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ {len(backstories)} detailed backstories created")
        
        print("\n🗺️ Enhancement 2: World expansion...")
        world_guide, world_guide_tokens, utility_tokens = await self.world_builder.expand_world_detail(
            integrated_world=integrated_world,
            plot=minimal_plot,
            seed=seed,
            connected_agents=connected_agents,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Comprehensive world guide created")
        print(f"  - {len(world_guide.location_dossiers)} locations documented")
        print(f"  - {len(world_guide.minor_character_pool)} NPCs ready")
        
        print("\n🎭 Enhancement 3: Subplot architecture...")
        subplot_arch, subplot_tokens, utility_tokens = await self.generate_subplot_architecture(
            plot=minimal_plot,
            connected_agents=connected_agents,
            seed=seed,
            act_count=seed.act_count,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ {len(subplot_arch.subplots)} subplots architected")
        
        # ===== INTEGRATED EXPANSION =====
        print("\n" + "=" * 60)
        print("📈 INTEGRATED PLOT EXPANSION (With All Enhancements)")
        print("=" * 60)
        
        print("\n📈 Phase 7: Expanding plot WITH enhancements...")
        expanded_plot, expand_tokens, utility_tokens = await self.expand_plot_with_enhancements(
            minimal_plot=minimal_plot,
            connected_agents=connected_agents,
            integrated_world=integrated_world,
            conflict_matrix=conflict_matrix,
            seed=seed,
            backstories=backstories,
            world_guide=world_guide,
            subplot_architecture=subplot_arch,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Plot expanded with ALL epic detail integrated")
        
        print("\n👶 Phase 8: Age appropriateness...")
        age_report, age_filter_tokens, utility_tokens = await self._enforce_age_appropriateness(
            final_plot=expanded_plot,
            seed=seed,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        if not age_report.plot_outline_pass:
            raise ValueError("Age filter failed")
        
        if age_report.age_rating == 'A':
            seed.target_audience_age = 8
        elif age_report.age_rating == 'T':
            seed.target_audience_age = 13
        elif age_report.age_rating == 'M':
            seed.target_audience_age = 18
        
        print(f"✓ Content validated for age {seed.target_audience_age}+")
        
        print("\n🔍 Phase 9: Quality validation...")
        quality_report, quality_tokens, utility_tokens = await self.quality_controller.validate_story_elements(
            seed, expanded_plot, connected_agents, integrated_world, conflict_matrix, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if quality_report.concerns:
            final_plot, refine_tokens, utility_tokens = await self.refine_plot_with_quality_feedback(
                expanded_plot, quality_report, connected_agents,
                integrated_world, conflict_matrix, seed.act_count, model
            )
            utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        else:
            final_plot = expanded_plot
        
        print("\n📊 Phase 10: Story tracker...")
        story_tracker, tracker_tokens, utility_tokens = await self._create_story_tracker(
            act_count=seed.act_count,
            final_plot=final_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Story tracker initialized")
        
        # Merge backstories into connected_agents
        backstory_dict = {b.name: b for b in backstories}
        enriched_agents = []
        for agent in connected_agents:
            agent_dict = agent.model_dump() if hasattr(agent, 'model_dump') else agent
            if agent_dict["name"] in backstory_dict:
                agent_dict["detailed_backstory"] = backstory_dict[agent_dict["name"]].model_dump()
            enriched_agents.append(agent_dict)
        
        # Store everything
        await self._store_planning_documents(
            seed=seed, minimal_plot=minimal_plot, final_plot=final_plot, world_foundation=world_foundation,
            narrative_agents=narrative_agents, conflict_matrix=conflict_matrix, connected_agents=connected_agents,
            integrated_world=integrated_world, quality_report=quality_report, story_title=final_plot.title, story_tracker=story_tracker
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json([b.model_dump() for b in backstories]),
            metadata={"type": "character_backstories", "story_title": final_plot.title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(world_guide.model_dump()),
            metadata={"type": "world_guide", "story_title": final_plot.title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(subplot_arch.model_dump()),
            metadata={"type": "subplot_architecture", "story_title": final_plot.title}
        )
        
        total_tokens = self._calculate_total_tokens([
            world_tokens, agent_tokens, plot_tokens, conflict_tokens,
            connect_tokens, integrate_tokens, backstory_tokens,
            world_guide_tokens, subplot_tokens, expand_tokens,
            quality_tokens, refine_tokens, age_filter_tokens, tracker_tokens
        ])
        
        print("\n" + "=" * 60)
        print("✅ EPIC PLANNING COMPLETE")
        print(f"📊 Total tokens: {total_tokens['total_tokens']:,}")
        print(f"📖 Backstories: {len(backstories)}")
        print(f"🗺️ Locations: {len(world_guide.location_dossiers)}")
        print(f"🎭 Subplots: {len(subplot_arch.subplots)}")
        print("=" * 60)
        
        return {
            "seed": seed,
            "minimal_plot": minimal_plot,
            "final_plot": final_plot,
            "author_tokens": total_tokens,
            "utility_tokens": utility_token_usage,
        }

    ## 12. NEW METHOD: Standard Planning Flow (40k-80k)
    async def _standard_planning_flow(
        self,
        seed: MinimalStorySeed,
        model: str = "None"
    ) -> Dict[str, Any]:
        """
        Standard planning flow - this is your current implementation.
        Extracted into separate method for routing.
        """
        
        # This is essentially your current orchestrate_story_planning logic
        # Copy phases 1-10 from your existing implementation
        utility_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        print("\n📋 Phase 1: Story seed ready")
        # structure_template = find_best_structure_name(seed.story_structure)
        print(f"✓ Genres: {seed.genre} | Target: {seed.target_length} words")
        
        print("\n🌍 Phase 2: World foundation...")
        world_foundation, world_tokens, utility_tokens = await self.world_builder.generate_world_foundation(
            seed, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ World foundation created")
        
        print("\n📖 Phase 3: Minimal plot outline...")
        minimal_plot, plot_tokens, utility_tokens = await self.generate_minimal_plot_outline(
            seed, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Minimal plot: '{minimal_plot.title}' ({seed.act_count} acts)")
        print(f"✓ Requires {len(minimal_plot.required_character_roles)} roles")
        
        print("\n👥 Phase 4: Narrative agents...")
        narrative_agents, agent_tokens, utility_tokens = await self.agent_genesis.generate_narrative_agents(
            seed, world_foundation, minimal_plot.required_character_roles, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ {len(narrative_agents)} narrative agents created")
        
        print("\n⚔️ Phase 5: Conflict matrix...")
        conflict_matrix, conflict_tokens, utility_tokens = await self.conflict_architect.generate_conflict_layers(
            minimal_plot, seed, narrative_agents, world_foundation, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Multi-dimensional conflict matrix created")
        
        print("\n🔗 Phase 6: Connecting elements...")
        connected_agents, connect_tokens, utility_tokens = await self.agent_genesis.connect_agents_to_plot(
            narrative_agents=narrative_agents,
            conflict_matrix=conflict_matrix,
            plot_outline=minimal_plot,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        integrated_world, integrate_tokens, utility_tokens = await self.world_builder.integrate_with_conflict(
            world_foundation, conflict_matrix, connected_agents, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        print(f"✓ Agents connected to plot")
        print(f"✓ World integrated with conflicts")
        
        print("\n📈 Phase 7: Expanding plot...")
        expanded_plot, expand_tokens, utility_tokens = await self.expand_plot_outline(
            minimal_plot, connected_agents, integrated_world,
            conflict_matrix, seed, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Plot expanded with full detail")
        
        print("\n👶 Phase 8: Age appropriateness...")
        age_report, age_filter_tokens, utility_tokens = await self._enforce_age_appropriateness(
            final_plot=expanded_plot,
            seed=seed,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        if not age_report.plot_outline_pass:
            raise ValueError("Age filter failed")
        
        if age_report.age_rating == 'A':
            seed.target_audience_age = 8
        elif age_report.age_rating == 'T':
            seed.target_audience_age = 13
        elif age_report.age_rating == 'M':
            seed.target_audience_age = 18
        
        print(f"✓ Content validated for age {seed.target_audience_age}+")
        
        print("\n🔍 Phase 9: Quality validation...")
        quality_report, quality_tokens, utility_tokens = await self.quality_controller.validate_story_elements(
            seed, expanded_plot, connected_agents, integrated_world, conflict_matrix, model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        
        refine_tokens = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if quality_report.concerns:
            final_plot, refine_tokens, utility_tokens = await self.refine_plot_with_quality_feedback(
                expanded_plot, quality_report, connected_agents,
                integrated_world, conflict_matrix, seed.act_count, model
            )
            utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        else:
            final_plot = expanded_plot
        
        print("\n📊 Phase 10: Story tracker...")
        story_tracker, tracker_tokens, utility_tokens = await self._create_story_tracker(
            act_count=seed.act_count,
            final_plot=final_plot,
            connected_agents=connected_agents,
            conflict_matrix=conflict_matrix,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        print(f"✓ Story tracker initialized")
        
        # Store in memory
        await self._store_planning_documents(
            seed, minimal_plot, final_plot, world_foundation,
            narrative_agents, conflict_matrix, connected_agents,
            integrated_world, quality_report, final_plot.title, story_tracker
        )
        
        total_tokens = self._calculate_total_tokens([
            world_tokens, agent_tokens, plot_tokens, conflict_tokens,
            connect_tokens, integrate_tokens, expand_tokens,
            quality_tokens, refine_tokens, age_filter_tokens, tracker_tokens
        ])
        
        print("\n" + "=" * 60)
        print("✅ STANDARD PLANNING COMPLETE")
        print(f"📊 Total tokens: {total_tokens['total_tokens']:,}")
        print("=" * 60)
        
        return {
            "seed": seed,
            "minimal_plot": minimal_plot,
            "final_plot": final_plot,
            "author_tokens": total_tokens,
            "utility_tokens": utility_token_usage,
        }


    ## 13. HELPER METHODS
    def _calculate_total_tokens(self, token_dicts: List[dict]) -> dict:
        """Sum all token counts"""
        total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for tokens in token_dicts:
            total["prompt_tokens"] += tokens.get("prompt_tokens", 0)
            total["completion_tokens"] += tokens.get("completion_tokens", 0)
        total["total_tokens"] = total["prompt_tokens"] + total["completion_tokens"]
        return total


    async def _store_planning_documents(
        self,
        seed: MinimalStorySeed,
        minimal_plot: Optional[MinimalPlotOutline],
        final_plot: Any,
        world_foundation: WorldFoundation,
        narrative_agents: List[NarrativeAgent],
        conflict_matrix: ConflictMatrix,
        connected_agents: List[ConnectedNarrativeAgent],
        integrated_world: IntegratedWorld,
        quality_report: QualityReport,
        story_title: str,
        story_tracker: Optional[StoryTracker]
    ):
        """Store all planning documents in memory"""
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(seed.model_dump()),
            metadata={"type": "story_seed", "story_title": story_title}
        )
        
        if minimal_plot:
            await self.memory.add_long_term_document(
                text=StoryHelpers.compress_json(minimal_plot.model_dump()),
                metadata={"type": "minimal_plot_outline", "story_title": story_title}
            )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(final_plot.model_dump()),
            metadata={"type": "expanded_plot_outline", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(world_foundation.model_dump()),
            metadata={"type": "world_foundation", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json([a.model_dump() for a in narrative_agents]),
            metadata={"type": "narrative_agents", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(conflict_matrix.model_dump()),
            metadata={"type": "conflict_matrix", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json([a.model_dump() for a in connected_agents]),
            metadata={"type": "connected_agents", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(integrated_world.model_dump()),
            metadata={"type": "integrated_world", "story_title": story_title}
        )
        
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(quality_report.model_dump()),
            metadata={"type": "quality_report", "story_title": story_title}
        )

        if story_tracker:
            await self.memory.add_long_term_document(
                text=StoryHelpers.compress_json(story_tracker.model_dump()),
                metadata={"type": "story_tracker", "story_title": story_title}
            )


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
    
    def _get_genre_specific_anchoring(self, genres: List[str]) -> str:
        """Explicit genre constraints for act planning"""
        genre_constraints = {
            "Fantasy": (
                "- Magic must have clear costs/rules\n"
                "- World-building in EVERY anchor\n"
                "- Quest structure maintained"
            ),
            "Mystery": (
                "- Every anchor must advance investigation OR deepen character motive\n"
                "- Clues and red herrings in each chapter\n"
                "- Never lose sight of the mystery question"
            ),
            "Romance": (
                "- Romantic tension in every anchor\n"
                "- Emotional beats > plot mechanics\n"
                "- Relationship arc is the PRIMARY plot"
            ),
            "Thriller/Suspense": (
                "- Escalating danger in each anchor\n"
                "- Protagonist under constant pressure\n"
                "- Pacing cannot slow"
            ),
            "Comedy": (
                "- A comedic beat in EVERY anchor\n"
                "- Character flaws drive humor\n"
                "- Escalation through misunderstanding or irony"
            ),
            "Horror": (
                "- Sustained atmosphere of dread in every anchor\n"
                "- Threat must feel personal and inescapable\n"
                "- Tension > exposition"
            ),
            "Sci-Fi": (
                "- Speculative concept affects EVERY anchor\n"
                "- Internal logic of technology must remain consistent\n"
                "- Human consequences of the concept are foregrounded"
            ),
            "Crime": (
                "- Criminal objective or consequence in every anchor\n"
                "- Cause-and-effect realism\n"
                "- Moral pressure escalates continuously"
            ),
            "Drama": (
                "- Character choice drives every anchor\n"
                "- Emotional consequences are explicit\n"
                "- External events exist to pressure inner conflict"
            ),
            "Adventure": (
                "- Momentum and forward motion in every anchor\n"
                "- Clear external objective at all times\n"
                "- Set pieces advance plot, not spectacle alone"
            ),
            "Tragedy": (
                "- Inevitable downfall reinforced in every anchor\n"
                "- Character flaws actively cause harm\n"
                "- Hope is present but progressively undermined"
            ),
        }

        primary_genre = genres[0]
        for key in genre_constraints:
            if key in primary_genre:
                return genre_constraints[key]

        return (
            f"- Maintain {primary_genre} conventions\n"
            "- Core genre conflict in every anchor"
        )

    
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
        structure_config: dict,
        seed: dict
    ) -> str:
        """Directive system prompt for act planning - EMPHASIZE CREATIVITY"""
        
        return f"""You are crafting ACT {act_number} of {total_acts} for a {'/'.join(seed['genre'])} story.

**CRITICAL - GENRE ADHERENCE FOR COMPACT STORIES**
This is a {seed['target_length']}-word {'/'.join(seed['genre'])} story. EVERY anchor must:
1. Serve the core {'/'.join(seed['genre'])} conflict
2. Use genre-appropriate tropes and beats
3. Maintain genre tone: {seed['tone']}
4. No genre drift - stay laser-focused on {'/'.join(seed['genre'])} elements

Genre-specific requirements:
{self._get_genre_specific_anchoring(seed['genre'])}

YOUR ROLE: Sculpt a {'/'.join(seed['genre'])} story spine that never wavers from genre conventions - dynamic turning points that pulse with creativity, emotion, and surprise.
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
        
        # if act_number == total_acts:
        #     suggested_word_count = remaining_words
        # else:
        suggested_word_count = int(target_total * word_percentage)

        structure_config = self.calculate_chapter_structure(
            total_story_length=target_total,
            structure_name=structure_name
        )

        print(f"Act {act_number} guidance: {act_guidance}")
        print(f"Target: ~{suggested_word_count:,} ")

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
            structure_config=structure_config,
            seed=seed
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
        utility_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 Act Planning LLM Attempt {attempt}/{max_retries + 1}")

            response, author_tokens = await better_author_client(
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
                act_plan, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=ActPlan
                )


                if isinstance(act_plan, tuple):
                    act_plan = act_plan[0]
                utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
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

            missing = [c for c in act_plan.chapter_outlines if not c.target_word_count]

            if missing:
                per_chapter = int(suggested_word_count / len(act_plan.chapter_outlines))
                for c in missing:
                    c.target_word_count = per_chapter

            # --- Passed all checks ---
            print("Better Author Client Token Usage: ", author_tokens)
            print("✓ Act validated. Correct chapter count + valid word counts.")
            break

        # If retries exhausted but we still have something, return last valid plan
        if attempt > max_retries and last_plan:
            print("🚨 Max retries reached — returning last generated act plan (may be incomplete).")
            act_plan = last_plan

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
        
        act_tracking, tracker_tokens, utility_tokens = await self.tracker_manager.update_tracker_after_act(
            act_number=act_number,
            act_plan=act_plan,
            author_context=story_so_far,
            model=model
        )
        utility_token_usage = StoryHelpers._add_tokens_to_total(utility_token_usage, utility_tokens)
        tracker[act_number] = act_tracking     
        await self.tracker_manager.save_tracker(
            self.memory, story_title, tracker
        )
        total_tokens = {
            "prompt_tokens": author_tokens.get("prompt_tokens", 0) + tracker_tokens.get("prompt_tokens", 0),
            "completion_tokens": author_tokens.get("completion_tokens", 0) + tracker_tokens.get("completion_tokens", 0),
            "total_tokens": 0
        }
        total_tokens["total_tokens"] = total_tokens["prompt_tokens"] + total_tokens["completion_tokens"]
        
        # Store in memory
        await self.memory.add_long_term_document(
            text=StoryHelpers.compress_json(act_plan.model_dump()),
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
        
        return act_plan, total_tokens, utility_token_usage



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

