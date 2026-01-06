# src/plot_engine/story_author.py

import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_structure_decider import find_best_structure, find_best_structure_name
from .agent_genesis import AgentGenesis
from .world_builder import WorldBuilder
from .act_planner import ActPlan
from src.utilities.story_helpers import StoryHelpers
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
    target_medium: str
    protagonist_specs: Dict[str, Any] = Field(default_factory=dict)
    act_count: int
    story_structure: str


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
# COMPONENT CLASSES
# ============================================================================
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
        plot_outline: dict,
        seed: dict,
        narrative_agents: List[dict],
        world_foundation: dict
    ) -> Tuple[ConflictMatrix, dict, dict]:
        """Generate conflict analysis with flexible depth"""
        
        complexity_guidance = ""
        if seed['target_length'] < 15000:
            complexity_guidance = """
SHORT STORY MODE: Focus primarily on central conflict.
Keep layers minimal - one clear tension is enough."""
        elif seed['target_length'] < 40000:
            complexity_guidance = """
NOVELLA MODE: Central conflict + 1-2 character/thematic layers."""
        else:
            complexity_guidance = """
NOVEL MODE: Multiple interwoven conflict layers encouraged."""
        
        system_prompt = f"""You are analyzing conflicts for a {'/'.join(seed['genre'])} story (novel).

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
Age-appropriate for: {seed['target_audience_age']} years old.

Output clean JSON with no markdown fences."""
        
        human_prompt = f"""Build conflict matrix for this story:

Themes: {', '.join(seed['themes'])}
Tone: {seed['tone']}

Plot Foundation:
{json.dumps(plot_outline, indent=2)}

Agents:
{json.dumps([json.dumps(a) for a in narrative_agents], indent=2)}

World:
{json.dumps(world_foundation, indent=2)}

Create conflicts that emerge naturally from agents and world."""
        
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
            parser=ConflictMatrix
        )

        if isinstance(json_data, tuple):
            json_data = json_data[0]
        
        scaled_matrix = self._scale_conflict_complexity(
            target_length=seed['target_length'],
            base_conflicts=json_data
        )

        return scaled_matrix, author_tokens, utility_tokens


class QualityController:
    """Validates story quality with focus on specificity"""
    
    def __init__(self):
        pass

    async def validate_story_elements(
        self,
        seed: dict,
        plot: dict,
        agents: List[dict],
        world: dict,
        conflict_matrix: dict
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
   ✓ Act summaries match the total act count of {seed['act_count']}, if less suggest spreading content appropriately.
   
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

Seed: {json.dumps(seed, indent=2)}
Plot: {json.dumps(plot, indent=2)}
Agents: {json.dumps([json.dumps(a) for a in agents], indent=2)}
World: {json.dumps(world, indent=2)}
Conflicts: {json.dumps(conflict_matrix, indent=2)}

Focus on: Is the plot SPECIFIC enough for execution?"""
        
        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.7
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



# ============================================================================
# MAIN STORY AUTHOR CLASS
# ============================================================================

class StoryAuthor:
    """
    World-class story planning system using snowflake expansion method.
    Now with flexible creative output while maintaining structural integrity.
    """
        
    def __init__(self):
        #self.memory = memory_system
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
        # Initialize component systems
        self.world_builder = WorldBuilder()
        self.agent_genesis = AgentGenesis()
        self.conflict_architect = ConflictArchitect()
        self.quality_controller = QualityController()
        self.tracker_manager = StoryTrackerManager()
        
    async def create_minimal_seed(self, user_context: Dict[str, Any]) -> MinimalStorySeed:
        """Extract and normalize minimal story seed from user context"""
        print("user_context: ",  user_context)
        # ---------- GENRES ----------
        genres = user_context.get("genre") or ["Fiction"]
        if not isinstance(genres, list):
            genres = [genres]

        sub_genre = user_context.get("sub_genre") or []
        if not isinstance(sub_genre, list):
            sub_genre = [sub_genre]

        themes = user_context.get("themes") or []
        if not isinstance(themes, list):
            themes = [themes]
        protagonist_specs_json = user_context.get("protagonist_specs")
        # ---------- PROTAGONIST ----------
        protagonist_specs = {
            k: v for k, v in {
                "name": protagonist_specs_json.get("name"),
                "age": protagonist_specs_json.get("age"),
                "gender": protagonist_specs_json.get("gender"),
                "archetype": protagonist_specs_json.get("archetype"),
                "core_trait": protagonist_specs_json.get("core_trait"),
                "background": protagonist_specs_json.get("background"),
                "desire": protagonist_specs_json.get("desire"),
                "fear": protagonist_specs_json.get("fear"),
                "relationships": protagonist_specs_json.get("relationships"),
                "physical_description": protagonist_specs_json.get("physical_description"),
            }.items()
            if v not in (None, "", [])
        }

        # ---------- TARGET LENGTH ----------
        target_length = user_context.get("target_length")


        # ---------- TITLE ----------
        excluded_titles = {
            None, "", " ", "None", "NULL", "Null", "null",
            "Untitled Story", "Unitled story", "untitled story",
            "Nill", "NILL", "nill"
        }

        title = user_context.get("title")
        title = None if title in excluded_titles else title

        target_medium = user_context.get("target_medium")

        # ---------- TONE ----------
        # tone_map = {
        #     0: "Very dark",
        #     20: "Dark",
        #     40: "Balanced",
        #     60: "Hopeful",
        #     80: "Light",
        #     100: "Whimsical",
        # }

        raw_tone = user_context.get("tone", "Balanced")
        # tone = tone_map.get(raw_tone, raw_tone if isinstance(raw_tone, str) else "Balanced")

        # ---------- POV ----------
        pov = user_context.get("pov", "Third-person")

        # ---------- PROSE STYLE ----------
        guide_prose = user_context.get("prose_style") 
        # ---------- AUDIENCE ----------
        target_audience_age = user_context.get("target_audience_age", 13)
        try:
            target_audience_age = int(target_audience_age)
        except (TypeError, ValueError):
            target_audience_age = 13

        target_audience_age = max(8, min(target_audience_age, 18))

        # ---------- ACT COUNT ----------
        act_count = user_context.get("act_count")

        story_structure = user_context.get("story_structure")
        # act_count = self.validate_act_count(
        #     act_count=act_count,
        #     structure_name=story_structure
        # )

        # ---------- FINAL SEED ----------
        return MinimalStorySeed(
            title=title,
            pov=pov,
            tone=raw_tone,
            genre=genres,
            sub_genre=sub_genre,
            setting=user_context.get("setting") or None,
            prose_style=guide_prose,
            themes=themes,
            target_medium=target_medium,
            target_audience_age=target_audience_age,
            target_length=target_length,
            protagonist_specs=protagonist_specs,
            act_count=act_count,
            story_structure=story_structure,
        )


    async def generate_minimal_plot_outline(
        self,
        seed: dict,
        world_foundation: dict,
    ) -> Tuple[MinimalPlotOutline, dict, dict]:
        """Generate initial plot with flexible character roles and auto-retries."""

        # --- 1. Compute dynamic role thresholds based on target length ---
        # print(f"→ Target length: {target_len}, requiring at least {dynamic_min_roles} roles.")

        structure_template = find_best_structure(seed['story_structure'])

        system_prompt = f"""Create minimal plot outline for {'/'.join(seed['genre'])} novel.

REQUIRED FIELDS:
- title: Evocative, thematic
- premise: 1-2 sentence hook with character desire + impossible stakes
- central_question: Moral/emotional dilemma (not plot mechanics)
- narrative_arc: Flowing prose summary (200-300 words) - paint emotional journey
- required_character_roles: specific roles (NOT generic labels)
  Example: "guilt-ridden former expedition leader" not "mentor"

RULES:
- Roles MUST be specific to world/premise
- Each role = function + psychology + relationship potential
- Target age: {seed['target_audience_age']} - adjust complexity accordingly
- Structure: {structure_template}

OUTPUT: Valid MinimalPlotOutline JSON. No markdown fences.

{MinimalPlotOutline.model_json_schema()}"""


        human_prompt = f"""Seed: {json.dumps(seed, indent=2)}
World: {json.dumps(world_foundation, indent=2)}

Generate plot outline."""

        # --- 2. Retry loop ---
        # attempt = 0
        # last_valid_response = None
        
        # while attempt <= 3:
        #     attempt += 1
            
            # Only regenerate if no valid partial result
    #         if attempt > 1 and last_valid_response:
    #             fix_prompt = f"""Previous attempt had {len(last_valid_response.required_character_roles)} roles.
    # Minimum required: {dynamic_min_roles}

    # Add {dynamic_min_roles - len(last_valid_response.required_character_roles)} more specific roles.

    # Previous roles: {json.dumps(last_valid_response.required_character_roles)}
    # Previous plot: {last_valid_response.model_dump_json(indent=2)}

    # Expand roles list ONLY. Keep rest unchanged."""
                
                # response, author_tokens = await better_author_client(
                #     system_prompt="Add missing character roles to existing plot.",
                #     human_prompt=fix_prompt,
                #     llm_temp=0.7
                # )
            # else:
                # First attempt - full generation
        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9
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
            raise
            # await asyncio.sleep(0.6 * attempt)
            # continue

        role_count = len(outline.required_character_roles)
        # if role_count >= dynamic_min_roles:
        #     print("Better Author Client Token Usage: ", author_tokens)
        #     return outline, author_tokens, utility_tokens
    
        # last_valid_response = outline
        print(f"→ Produced {role_count} roles: {outline.required_character_roles}")

            # --- 4. Check role thresholds ---
            # if role_count >= dynamic_min_roles:
            #     print("Better Author Client Token Usage: ", author_tokens)
            #     print(f"✓ Accepted. Role count meets threshold ({dynamic_min_roles}).")
            #     return outline, author_tokens, utility_tokens
            # else:
            #     print(f"⚠️ Insufficient roles ({role_count} < {dynamic_min_roles}). Retrying...")
            #     await asyncio.sleep(0.8 * attempt)

        # --- 5. If max retries exhausted ---
        print("🚨 Max retries exhausted. Returning last valid attempt (even if insufficient roles).")
        return outline, last_tokens, utility_tokens
    
    async def expand_plot_with_enhancements(
        self,
        minimal_plot: dict,
        connected_agents: List[dict],
        integrated_world: dict,
        conflict_matrix: dict,
        seed: dict,
        backstories: List[dict],
        world_guide: dict,
        subplot_architecture: dict
    ) -> Tuple[ExpandedPlotOutline, dict, dict]:
        """
        Expand plot AFTER enhancements are generated.
        Act summaries and narrative flow can now reference:
        - Specific backstory events
        - Documented locations from world guide
        - Named NPCs from character pool
        - Subplot threads and their integration points
        """
        
        target_act_count = seed['act_count']
        structure_template = find_best_structure(seed['story_structure'])
        
        # Extract key details from enhancements for prompt
        backstory_highlights = []
        for bs in backstories:
            backstory_highlights.append({
                "name": bs['name'],
                "key_events": bs['formative_events'],
                "secrets": bs['secrets'],
                "subplot_seeds": bs['subplot_seeds']
            })
        
        location_names = [loc.get("name", "Unknown") for loc in world_guide['location_dossiers']]
        npc_pool = [npc.get("name", "Unknown") for npc in world_guide['minor_character_pool']]
        
        subplot_summaries = []
        for sp in subplot_architecture['subplots']:
            subplot_summaries.append({
                "title": sp['title'],
                "premise": sp['premise'],
                "owner": sp['character_owner'],
                "act_moments": sp['act_integration']
            })
        
        system_prompt = f"""You are expanding a minimal plot into a COMPLETE story blueprint for a {'/'.join(seed['genre'])} novel.

YOU NOW HAVE ACCESS TO RICH STORY ENHANCEMENTS:

🎭 CHARACTER BACKSTORIES:
{json.dumps(backstory_highlights, indent=2)}

🗺️ DOCUMENTED LOCATIONS:
{', '.join(location_names)}

👥 NPC POOL (ready to deploy):
{', '.join(npc_pool)}

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

Age-appropriate: {seed['target_audience_age']}+

OUTPUT FORMAT (JSON):
{ExpandedPlotOutline.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Expand this plot WITH all enhancements integrated.

MINIMAL PLOT:
{json.dumps(minimal_plot, indent=2)}

CHARACTERS:
{json.dumps([{"name": a['name'], "role": a['role'], "arc": a['agent_arc']} for a in connected_agents], indent=2)}

WORLD:
{json.dumps(integrated_world, indent=2)}

CONFLICTS:
{json.dumps(conflict_matrix, indent=2)}

THEMES: {', '.join(seed['themes'])}

Write act summaries that feel like a complete story bible - specific, vivid, integrated."""

        max_retries = 4
        attempt = 0

        while attempt <= max_retries:
            attempt += 1
            print(f"\n📘 Enhanced plot expansion attempt {attempt}/{max_retries + 1}")

            response, author_tokens = await better_author_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=0.85
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
        minimal_plot: dict,
        agents: List[dict],
        world: dict,
        conflict_matrix: dict,
        seed: dict,
        max_retries: int = 4
    ) -> Tuple[ExpandedPlotOutline, dict, dict]:
        """Expand minimal plot with creative flexibility and enforce act count matching."""

        target_act_count = seed['act_count']
        print(f"→ Target act count: {target_act_count}")

        structure_template = find_best_structure(seed['story_structure'])

        system_prompt = f"""
You are a disciplined narrative generator.
Expand this minimal plot outline into a FULL story blueprint (for a story novel).
You have full creative freedom within these boundaries, make it unique and compelling. 
Use only information provided through variables.

════════ STORY INPUT ════════
Premise: {minimal_plot['premise']}
Central Question: {minimal_plot['central_question']}
Structure Inspiration: {structure_template}
Emotional Core: {seed['tone']}
Prose Voice: {seed['prose_style']}
Target Length: {seed['target_length']} words across {target_act_count} acts
Title: {minimal_plot['title']}
Genre: {'/'.join(seed['genre'])}
Audience Age: {seed['target_audience_age']}

════════ Creative Mandate ════════

  title: {minimal_plot['title']},
  premise: {minimal_plot['premise']},

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
  // Do not copy samples blindly. Invent fields appropriate to { '/'.join(seed['genre']) }.

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

        
        human_prompt = f"""Weave the full tapestry of this {'/'.join(seed['genre'])} story.

═══════════════════════════════════════════════════════════════════════════════
THE RAW MATERIALS
═══════════════════════════════════════════════════════════════════════════════

MINIMAL PLOT SKELETON:
{json.dumps(minimal_plot, indent=2)}

THE CAST (breathe life into these):
{json.dumps([json.dumps(a) for a in agents], indent=2)}

THE WORLD (make it pulse):
{json.dumps(world, indent=2)}

THE CONFLICTS (make them bleed):
{json.dumps(conflict_matrix, indent=2)}

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
Target Length: {seed['target_length']} words across {target_act_count} acts

HONOR THEMES: {', '.join(seed['themes'])}
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
                llm_temp=0.9
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
        plot: dict,
        connected_agents: List[dict],
        seed: dict,
        act_count: int
    ) -> Tuple[SubplotArchitecture, dict, dict]:
        """
        Generate 2-4 interwoven subplots for 80k+ novels.
        Genre-aware and character-driven.
        """
        
        genre_config = self._get_genre_config(seed['genre'])
        
        # Determine subplot count based on length
        if seed['target_length'] < 90000:
            subplot_count = "2-3"
        else:
            subplot_count = "3-4"
        
        system_prompt = f"""You are architecting subplots for a {'/'.join(seed['genre'])} novel.

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
{self._get_genre_subplot_guidance(seed['genre'], genre_config)}

Age-appropriate for: {seed['target_audience_age']} year olds

OUTPUT FORMAT (JSON):
{SubplotArchitecture.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Design {subplot_count} subplots for this story.

MAIN PLOT:
{json.dumps(plot, indent=2)}

AVAILABLE CHARACTERS:
{json.dumps([{"name": a['name'], "role": a['role'], "arc": a['agent_arc']} for a in connected_agents], indent=2)}

THEMES: {', '.join(seed['themes'])}
TOTAL ACTS: {act_count}

Create interwoven, genre-appropriate subplots."""

        response, author_tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85
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
    
    
    async def _create_story_tracker(
        self,
        act_count: int,
        final_plot: dict,
        connected_agents: List[dict],
        conflict_matrix: dict
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
{json.dumps(final_plot, indent=2)}

Agents:
{json.dumps([json.dumps(a)for a in connected_agents], indent=2)}

Conflicts:
{json.dumps(conflict_matrix, indent=2)}

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
        seed: dict,
        world_foundation: dict
    ) -> Tuple[ExpandedCompactPlotOutline, dict, dict]:
        """
        Single-pass plot generation for compact stories.
        Combines minimal + expanded into one streamlined output.
        """
        
        genre_config = self._get_genre_config(seed['genre'])
        structure_template = find_best_structure(seed['story_structure'])
        target_act_count = seed['act_count']
        # Calculate required roles dynamically (fewer for short stories)
        # target_len = seed['target_length']
        # min_len, max_len = 10000, 40000
        # norm = max(0.0, min(1.0, (target_len - min_len) / (max_len - min_len)))
        # dynamic_min_roles = int(3 + norm * (7 - 3))  # 3-7 roles for compact
        
        system_prompt = f"""Create a complete plot outline for a {'/'.join(seed['genre'])} story ({seed['target_length']} words).

GENRE FOCUS: {', '.join(genre_config.get('conflict_layers', []))}
STRUCTURE: {structure_template}

REQUIRED OUTPUT:
- title: Evocative, thematic
- premise: 1-2 sentence hook
- central_question: Core dilemma
- required_character_roles: specific roles (not generic)
- act_summaries: [{seed['act_count']} summaries, 150-200 words each in flowing prose]
- narrative_flow: 300-500 words of complete emotional arc

RULES FOR COMPACT STORIES:
- Focus on single clear conflict thread
- Limit subplot complexity
- Every character must earn their presence
- Streamlined but emotionally complete
- Age-appropriate: {seed['target_audience_age']}+

OUTPUT FORMAT (JSON):
{ExpandedCompactPlotOutline.model_json_schema()}

Output clean JSON with no markdown fences."""

        human_prompt = f"""Generate complete plot.

Seed: {json.dumps(seed, indent=2)}
World: {json.dumps(world_foundation, indent=2)}

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
                llm_temp=0.9
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

    ## 10. Simplified Conflict (for 10k-40k)
    async def _generate_simplified_conflict(
        self,
        plot: dict,
        seed: dict,
        narrative_agents: List[dict],
        world_foundation: dict
    ) -> Tuple[ConflictMatrix, dict, dict]:
        """
        Streamlined conflict for compact stories - focus on core tension only.
        """
        
        genre_config = self._get_genre_config(seed['genre'])
        
        system_prompt = f"""Analyze core conflict for {'/'.join(seed['genre'])} story (compact length).

FOCUS: Single clear conflict with 1-2 layers maximum.

GENRE CONFLICT TYPE: {', '.join(genre_config.get('conflict_layers', []))}

OUTPUT FORMAT (JSON):
{ConflictMatrix.model_json_schema()}

Keep it focused - one central conflict, one escalation path.
Age-appropriate: {seed['target_audience_age']}+

Output clean JSON with no markdown fences."""

        human_prompt = f"""Identify core conflict.

Plot: {json.dumps(plot, indent=2)}
Agents: {json.dumps([json.dumps(a) for a in narrative_agents], indent=2)}
World: {json.dumps(world_foundation, indent=2)}

Streamlined conflict analysis."""

        response, author_tokens = await author_fast_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.85
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

