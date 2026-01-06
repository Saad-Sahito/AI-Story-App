# src/plot_engine/act_planner.py

import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict
from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from .act_context_builder import ActContextBuilder
from src.utilities.story_helpers import StoryHelpers
import asyncio




class FlexibleBase(BaseModel):
    """Base class allowing LLM to add arbitrary creative fields"""
    model_config = ConfigDict(extra="allow")


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


class ActPlanner:
    def __init__(self):
        pass
    
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

