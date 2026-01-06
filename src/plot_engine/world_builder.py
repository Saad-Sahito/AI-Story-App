# src/plot_engine/world_builder.py

import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_helpers import StoryHelpers



class FlexibleBase(BaseModel):
    """Base class allowing LLM to add arbitrary creative fields"""
    model_config = ConfigDict(extra="allow")

class IntegratedWorld(FlexibleBase):
    """World after plot integration"""
    foundation: str = Field(description="Core world elements")
    plot_integration: str = Field(
        description="How world rules enable/complicate conflicts"
    )
    world_rules: List[str] = Field(
        default_factory=list,
        description="Explicit list of 3-5 unbreakable constraints (magic costs, laws of physics, social taboos) for the Director to enforce."
    )
    
class WorldFoundation(FlexibleBase):
    """Minimal world building - LLM adds genre-appropriate details"""
    core_concept: str = Field(
        description="1-3 sentence essence of this world"
    )
    # LLM adds dynamically: magic_system, tech_level, social_structure, 
    # geography, culture, threats, etc. based on genre


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



class WorldBuilder:
    """Generates independent world foundations with creative flexibility"""
    
    def __init__(self):
        self.GENRE_CONFIGURATIONS = {
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
    
    async def generate_world_foundation(
        self, seed: dict
    ) -> Tuple[WorldFoundation, dict, dict]:
        """Generate world with genre-appropriate flexibility"""
        
        system_prompt = f"""You are a master world-builder creating immersive, internally consistent worlds for a novel.

Build a world for: {', '.join(seed['genre'])}
Tone: {seed['tone']}
Setting hint: {seed['setting'] or 'Create something fitting'}
Themes: {', '.join(seed['themes'])}

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
Age-appropriate for: {seed['target_audience_age']} year old audience

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Create a rich world foundation for this story.

Seed details:
{json.dumps(seed)}

Build a world that could support many stories, with inherent conflicts and memorable details.
Add genre-appropriate fields that bring this world to life."""
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9
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
        world_foundation: dict,
        conflict_matrix: dict,
        agents: List[dict]
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
{json.dumps(world_foundation, indent=2)}

Story Conflicts:
{json.dumps(conflict_matrix, indent=2)}

Agents:
{json.dumps([json.dumps(a) for a in agents], indent=2, ensure_ascii=False)}

Show how this world naturally enables and complicates these conflicts."""
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85
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
        integrated_world: dict,
        plot: dict,
        seed: dict,
        connected_agents: List[dict]
    ) -> Tuple[EnhancedWorldGuide, dict, dict]:
        """
        Create comprehensive world documentation for 80k+ novels.
        Genre-specific and plot-integrated.
        """
        
        genre_config = self._get_genre_config(seed['genre'])
        
        system_prompt = f"""You are building a comprehensive world guide for a {'/'.join(seed['genre'])} novel.

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
   (Focus on systems relevant to {'/'.join(seed['genre'])})

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
   {self._get_genre_specific_world_fields(seed['genre'], genre_config)}

Age-appropriate for: {seed['target_audience_age']} year olds

OUTPUT FORMAT (JSON):
{EnhancedWorldGuide.model_json_schema()}

Create immersive, internally consistent world documentation.
Output clean JSON with no markdown fences."""

        human_prompt = f"""Expand world detail for this story.

EXISTING WORLD FOUNDATION:
{json.dumps(integrated_world, indent=2)}

PLOT CONTEXT:
{json.dumps(plot, indent=2)}

MAIN CHARACTERS (to inform locations/NPCs):
{json.dumps([{"name": a['name'], "role": a['role']} for a in connected_agents], indent=2)}

Create comprehensive world guide."""

        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85
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