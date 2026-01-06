# src/plot_engine/agent_genesis.py

import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_helpers import StoryHelpers



class FlexibleBase(BaseModel):
    """Base class allowing LLM to add arbitrary creative fields"""
    model_config = ConfigDict(extra="allow")


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


class AgentGenesis:
    """Generates independent narrative agent foundations with flexible depth"""
    
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
        seed: dict,
        world_foundation: dict,
        required_roles: List[str]
    ) -> Tuple[List[NarrativeAgent], dict, dict]:
        """Generate agents with genre-appropriate detail"""
        
        agent_count = len(required_roles)
        
        system_prompt = f"""You are designing narrative agents for a {'/'.join(seed['genre'])} novel.

YOU MUST OUTPUT EXACTLY {agent_count} AGENTS IN A SINGLE RESPONSE.
DO NOT STOP AFTER A FEW - OUTPUT ALL {agent_count} AGENTS.

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
Age-appropriate for: {seed['target_audience_age']} year old audience.

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Generate {agent_count} narrative agents for these roles:
{json.dumps(required_roles, indent=2)}

Seed:
{json.dumps(seed, indent=2)}

World Context:
{json.dumps(world_foundation, indent=2)}

Protagonist specs (if any):
{json.dumps(seed['protagonist_specs'], indent=2)}

Create agents with independent desires and fears shaped by their world.
Add detail appropriate to each role's importance."""
        
        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
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
        narrative_agents: List[dict],
        conflict_matrix: dict,
        plot_outline: dict
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
{json.dumps([json.dumps(a) for a in narrative_agents], indent=2)}

Plot:
{json.dumps(plot_outline, indent=2)}

Conflicts:
{json.dumps(conflict_matrix, indent=2)}

Show how each agent's existing desires/fears naturally intersect with the plot."""
        
        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.8
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
        connected_agents: List[dict],
        world: dict,
        plot: dict,
        seed: dict,
        importance_threshold: str = "major",  # "major", "supporting", "all"
    ) -> Tuple[List[DetailedCharacterBackstory], dict, dict]:
        """
        Generate deep backstories for characters in 80k+ novels.
        Genre-aware and importance-filtered.
        """
        
        genre_config = self._get_genre_config(seed['genre'])
        
        # Filter characters by importance
        if importance_threshold == "major":
            roles_to_expand = ["protagonist", "antagonist", "mentor"]
        elif importance_threshold == "supporting":
            roles_to_expand = ["protagonist", "antagonist", "mentor", "ally", "rival", "love_interest"]
        else:  # all
            roles_to_expand = None  # include everyone
        
        agents_to_expand = [
            agent for agent in connected_agents
            if roles_to_expand is None or agent['role'].lower() in roles_to_expand
        ]
        
        print(f"📖 Generating backstories for {len(agents_to_expand)} characters...")
        
        backstories = []
        
        for agent in agents_to_expand:
            system_prompt = f"""You are crafting a deep character backstory for a {'/'.join(seed['genre'])} novel.

GENRE FOCUS AREAS: {', '.join(genre_config.get('character_focus', []))}
BACKSTORY EMPHASIS: {genre_config.get('backstory_emphasis', 'formative experiences')}

CHARACTER CONTEXT:
Name: {agent['name']}
Role: {agent['role']}
Essence: {agent['essence']}
Plot Function: {agent['plot_function']}
Character Arc: {agent['agent_arc']}

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
   Add fields relevant to {'/'.join(seed['genre'])}:
   {self._get_genre_specific_backstory_fields(seed['genre'], genre_config)}

Age-appropriate for: {seed['target_audience_age']} year olds

OUTPUT FORMAT (JSON):
{DetailedCharacterBackstory.model_json_schema()}

Write immersive, specific backstory. No placeholders.
Output clean JSON with no markdown fences."""

            human_prompt = f"""Create backstory for {agent['name']}.

WORLD CONTEXT:
{json.dumps(world, indent=2)}

PLOT CONTEXT:
Premise: {plot['premise']}
Central Question: {plot['central_question']}

OTHER CHARACTERS (for relationship history):
{json.dumps([{"name": a['name'], "role": a['role'], "essence": a['essence']} for a in connected_agents if a['name'] != agent['name']], indent=2)}

Generate rich, genre-appropriate backstory."""

            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.85
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
            print(f"  ✓ {agent['name']} backstory created")
                
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