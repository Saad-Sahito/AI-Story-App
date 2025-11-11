import base64
import json
from typing import Any, Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.llm_client.llm_client import author_client, utility_client, ingestor_client
from src.utilities.story_helpers import StoryHelpers
from config_vars import author_story_rules_negative
from src.memory.memory_system import StoryMemorySystem
from src.utilities.story_structure_decider import find_best_structure
import gc


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StorySeed(BaseModel):
    """Initial story foundation - lean and flexible"""
    title: str = Field(..., description="Story title")
    premise: str = Field(..., description="Core story concept in 3-4 sentences")
    protagonist: Dict[str, str] = Field(
        ..., 
        description="Main character relevant description"
    )
    world_essentials: Dict[str, str] = Field(
        ...,
        description="Setting info with keys: setting, time_period, key_rule"
    )
    central_conflict: str = Field(..., description="Primary story tension")
    themes: List[str] = Field(..., description="1-3 core themes")
    genre: List[str] = Field(..., description="Primary genres (list of 1-3 genres)")
    tone: str = Field(..., description="Emotional tone (e.g., dark, hopeful, epic)")
    style_guide: Dict[str, str] = Field(
        ...,
        description="Keys: prose_style, pov, tense, narrative_voice"
    )
    target_length: int = Field(..., description="Target word count for entire story")
    act_count: int = Field(..., description="Number of acts (typically 3-5)")


class ChapterOutline(BaseModel):
    """Detailed plan for a single chapter within an act"""
    chapter_number: int = Field(..., description="Sequential chapter number in story")
    chapter_title: str = Field(..., description="Chapter title")
    chapter_goal: str = Field(..., description="What this chapter accomplishes narratively")
    key_scenes: List[str] = Field(
        ..., 
        description="3-5 scene descriptions that comprise this chapter"
    )
    emotional_beats: str = Field(
        ..., 
        description="Emotional progression (e.g., 'tension → fear → resolve')"
    )
    ends_when: str = Field(
        ..., 
        description="Clear closure condition (e.g., 'Hero escapes the castle')"
    )
    target_word_count: int = Field(..., description="Target words for this chapter")


class ActPlan(BaseModel):
    """Complete plan for one act of the story"""
    act_number: int = Field(..., description="Which act this is (1, 2, 3, etc)")
    act_title: str = Field(..., description="Evocative name for this act")
    act_purpose: str = Field(
        ..., 
        description="What this act accomplishes in 2-3 sentences"
    )
    emotional_trajectory: str = Field(
        ..., 
        description="Overall emotional shift across this act"
    )
    key_developments: List[str] = Field(
        ..., 
        description="3-5 major story beats that must occur in this act"
    )
    chapter_outlines: List[ChapterOutline] = Field(
        ..., 
        description="Detailed outlines for each chapter in this act"
    )
    character_arcs_this_act: Dict[str, str] = Field(
        ..., 
        description="How main characters evolve during this act"
    )
    act_closure_condition: str = Field(
        ..., 
        description="Condition that signals this act is complete"
    )



class StoryMinimumAge(BaseModel):
    min_age: int = Field(description="minimum age limit to read this story")

# ============================================================================
# OUTPUT PARSERS
# ============================================================================

story_seed_parser = PydanticOutputParser(pydantic_object=StorySeed)
act_plan_parser = PydanticOutputParser(pydantic_object=ActPlan)
min_age_parser = PydanticOutputParser(pydantic_object=StoryMinimumAge)
# ============================================================================
# STORY AUTHOR CLASS
# ============================================================================

class StoryAuthor:
    """
    Manages story creation through act-based, adaptive planning.
    
    Workflow:
    1. create_story_seed() - Initial lean foundation
    2. generate_blurb() - Marketing copy from seed
    3. generate_cover_image() - Visual based on blurb
    4. plan_act() - Detailed chapter plans per act (called just-in-time)
    """
    
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system


    async def create_story_seed(
        self, 
        user_context: Dict[str, Any], 
        model: str = "None"
    ) -> Tuple[StorySeed, int]:
        """
        Create initial story foundation from user context.
        
        Args:
            user_context: Dict with keys: POV, Tone, Genre, Title, Length, Setting, 
                         Guide Prose, Additional Themes, target_audience_age
            story_title: Story title from user_context['Title']
            model: LLM model to use
        
        Returns:
            Tuple of (StorySeed object, token_count)
        """
        print("🌱 Creating story seed...")
        
        # Parse length into word count
        word_count_map = {
            "Short Long Story (7,500 - 15,000 words)": 12000,
            "Novelette (15,000 - 25,000 words)": 20000,
            "Novella (25,000 - 40,000 words)": 32000,
            "Novel Chapter (40,000 - 60,000 words)": 50000,
            "Full Novel (60,000 - 90,000 words)": 75000,
            "Epic / Series (90,000 - 150,000+ words)": 120000
        }
        target_length = word_count_map.get(user_context.get('Length', ''), 20000)
        
        # Determine act count based on length
        if target_length <= 15000:
            act_count = 3
        elif target_length <= 40000:
            act_count = 3
        elif target_length <= 90000:
            act_count = 4
        else:
            act_count = 5
        # Handle genres as a list
        genres = user_context.get('Genre', ['Fiction'])
        genres_str = ", ".join(genres)
        themes = user_context.get('Additional Themes', [''])
        themes_str = ", ".join(themes)
        title = user_context.get('Title')
        if title in [None, "None", "", " ", "Untitled Story", "Unitled story", "untitled story", "Null", "NULL", "Nill", "NILL", "null", "nill"]:
            title_str = """- title: Create a fitting Story Title for the Story"""
        else:
            title_str = f"""- title: Use "{user_context.get("Title")}" exactly as given"""
        protagonist_specs = {
            'name': user_context.get('protagonist_name'),
            'age': user_context.get('protagonist_age'),
            'gender': user_context.get('protagonist_gender'),
            'archetype': user_context.get('protagonist_archetype'),
            'core_trait': user_context.get('protagonist_core_trait'),
            'background': user_context.get('protagonist_background'),
            'desire': user_context.get('protagonist_desire'),
            'fear': user_context.get('protagonist_fear'),
            'relationships': user_context.get('protagonist_relationships'),
            'physical description': user_context.get('protagonist_physical_description')
        }

        # Filter out None values
        protagonist_specs = {k: v for k, v in protagonist_specs.items() if v}

        # Build protagonist specification string
        if protagonist_specs:
            protagonist_spec_str = "\n".join([f"  - {k.replace('_', ' ').title()}: {v}" 
                                            for k, v in protagonist_specs.items()])
            protagonist_instruction = f"""- Protagonist Specifications (MUST USE):
        {protagonist_spec_str}
        For any unspecified traits, create details that complement the given specifications."""
        else:
            protagonist_instruction = "- Protagonist: Create from scratch (age-appropriate) Keep gender limited to male or female"

        # Then update the system_prompt:
        system_prompt = f"""You are the Story Architect. Create a flexible story foundation for a classic narrative.

        Your goal: Provide a creative seed—NOT a rigid blueprint. This seed guides act planning, which happens progressively as the story unfolds.

        Create a story foundation that:
        1. Respects ALL user specifications above (POV, tone, genre are FIXED)
        2. STRICTLY follows any protagonist specifications provided - these are MANDATORY
        3. Develops a premise that fits the setting, themes, and protagonist
        4. For unspecified protagonist details, create age-appropriate traits that complement given specs
        5. Establishes world rules that enable interesting conflict
        6. Identifies the central dramatic question

        Output Structure:
        {title_str}
        - premise: 3-4 sentences establishing setup and conflict (featuring the specified protagonist)
        - protagonist: Dict with name, age, gender, core_trait, desire, fear, background
        * Use EXACT values for any user-specified protagonist details
        * Generate only the missing details to complete the character
        * Ensure all traits are age-appropriate for {user_context.get('target_audience_age', 'general')} audience
        * Maintain internal consistency between specified and generated traits
        - world_essentials: Dict with setting (use user's setting), time_period, key_rule
        - central_conflict: What's at stake (must relate to protagonist's desire/fear)
        - themes: Use exactly if any: {themes}
        - genre: Use exactly: {genres}
        - tone: Use exactly: {user_context.get('Tone', 'Balanced')}
        - style_guide: Dict with prose_style (from Prose Style), pov (from POV), tense, narrative_voice
        - target_length: {target_length}
        - act_count: {act_count}

        Rules:
        - CRITICAL: User-specified protagonist details are MANDATORY and CANNOT be changed
        - Strict adherence to user's POV, tone, genre, setting, prose style
        - Age-appropriate content for {user_context.get('target_audience_age', 'general')} year old audience
        - Be concise—this is a seed, not full planning
        - Focus on emotional core and world rules, not detailed plot
        - Internal consistency is critical (especially protagonist traits must align)
        - {author_story_rules_negative}

        {story_seed_parser.get_format_instructions()}
        """

        
        # Format user context for prompt
        human_prompt = f"""Create a story seed with these specifications:
{title_str}
- POV: {user_context.get('POV', 'Third-person')}
- Tone: {user_context.get('Tone', 'Balanced')}
- Genre: {genres_str}
- Setting: {user_context.get('Setting', 'To be determined')}
- Prose Style: {user_context.get('Guide Prose', 'Standard narrative')}
- Themes: {themes_str}
- Target Audience Age: {user_context.get('target_audience_age', 'General')}
- Target Length: {target_length} words ({act_count} acts recommended)
{protagonist_instruction}

Generate a compelling story foundation that brings these elements together."""
        
        response, tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        
        # Parse into Pydantic model
        story_seed = story_seed_parser.parse(response.content.strip())
        del response
        max_age = min(user_context.get('target_audience_age', 18), 18)
        response = await utility_client(
            system_prompt=f"""Output the minimum age required to read a story with the given story seed, output an integer ranging from 9 - {max_age}, according to: 
{min_age_parser.get_format_instructions()}""",
            human_prompt=f"""
story seed: 
{story_seed}
"""
        )
        min_age = min_age_parser.parse(response.content.strip())
        del response
        print(f"✅ Story seed created: '{story_seed.title}' ({story_seed.act_count} acts, {target_length} words)")
        return story_seed, tokens, min_age.min_age

    async def plan_act(
        self, 
        story_title: str,
        act_number: int,
        model: str = "None"
    ) -> Tuple[ActPlan, int]:
        """
        Generate detailed chapter plans for the specified act.
        Called just-in-time as story progresses.
        
        Args:
            story_title: Title of the story
            act_number: Which act to plan (1, 2, 3, etc.)
            model: LLM model to use
            
        Returns:
            Tuple of (ActPlan object, token_count)
        """
        print(f"📋 Planning Act {act_number}...")
        
        # Get story seed
        story_seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': story_title}
        )
        story_seed = json.loads(story_seed_json)
        
        # Get story progress
        progress = await self.memory.get_story_progress()
        current_word_count = progress.get('story_word_count', 0)
        target_total = story_seed.get('target_length', 50000)
        remaining_words = target_total - current_word_count
        
        # Build context based on act number
        if act_number == 1:
            context_prompt = self._build_act_1_context(story_seed)
        else:
            context_prompt = await self._build_later_act_context(
                story_title=story_title,
                story_seed=story_seed,
                act_number=act_number,
                current_word_count=current_word_count,
                remaining_words=remaining_words
            )
        
        # Construct system prompt
        #genres_str = ", ".join(story_seed.get('genre', ['Fiction']))  # Updated to handle list of genres

        system_prompt = f"""You are the Act Planner. Design Act {act_number} based on the story foundation and what has happened so far.

Your output must include:
1. act_number (int)
2. act_title (string) - Evocative name for this act
3. act_purpose (string) - What this act accomplishes narratively (2-3 sentences)
4. emotional_trajectory (string) - How emotions shift across this act
5. key_developments (list of strings) - 3-5 major story beats that must occur
6. chapter_outlines (list of ChapterOutline objects):
   - chapter_number: Sequential number in story
   - chapter_title: Evocative title
   - chapter_goal: What this chapter achieves
   - key_scenes: List of 3-5 scene descriptions
   - emotional_beats: Emotion progression (e.g., "hope → doubt → acceptance")
   - ends_when: Clear closure condition
   - target_word_count: Typically 2000-4000 per chapter
7. character_arcs_this_act (dict) - How main characters evolve in THIS act
8. act_closure_condition (string) - Clear condition signaling act completion

Guidelines:
- Make chapter outlines detailed enough for Director to create scene plans
- Keep flexibility for story to adapt organically
- Respect established tone, world rules, and character motivations
- For Act 1: Establish world, introduce protagonist, present inciting incident
- For Act 2+: Build on previous developments, escalate conflict
- Distribute remaining word count intelligently across chapters

{act_plan_parser.get_format_instructions()}
"""
        
        response, tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=context_prompt,
            llm_temp=0.8,
            model=model
        )
        
        # Parse into Pydantic model
        act_plan = act_plan_parser.parse(response.content.strip())
        
        # Store in memory
        await self.memory.add_long_term_document(
            text=act_plan.model_dump_json(indent=2),
            metadata={
                "type": "act_plan",
                "act_id": act_number,
                "story_title": story_title
            }
        )
        
        chapter_count = len(act_plan.chapter_outlines)
        print(f"✅ Act {act_number} planned: '{act_plan.act_title}' ({chapter_count} chapters)")
        
        return act_plan, tokens


    def _build_act_1_context(self, story_seed: dict) -> str:
        """Build context prompt for Act 1 planning."""
        # Calculate Act 1 target (typically 20-25% of total story)
        act_1_target = int(story_seed['target_length'] * 0.22)
        
        return f"""Story Seed:
{json.dumps(story_seed, indent=2)}

This is Act 1. Your goals:
- Establish the world and its rules (setting: {story_seed['world_essentials'].get('setting', 'TBD')})
- Introduce the protagonist in their ordinary world, from the description: {story_seed['protagonist']}
- Present the inciting incident that disrupts normalcy
- Set up the central conflict: {story_seed['central_conflict']}
- End when the protagonist commits to their journey
- Maintain {story_seed['style_guide']['pov']} POV and {story_seed['tone']} tone
- Keep content appropriate for {story_seed.get('target_audience_age', 'general')} year old readers

Target word count for Act 1: ~{act_1_target} words (approximately 22% of story)
Suggested chapter count: {max(2, act_1_target // 3000)} chapters (~3000 words each)
"""


    async def _build_later_act_context(
        self,
        story_title: str,
        story_seed: dict,
        act_number: int,
        current_word_count: int,
        remaining_words: int
    ) -> str:
        """Build context prompt for Act 2+ planning."""
        
        # Get all previous act plans
        previous_acts = []
        for i in range(1, act_number):
            try:
                act_plan_json = await self.memory.get_long_term_document(
                    metadata={'type': 'act_plan', 'act_id': i, 'story_title': story_title}
                )
                previous_acts.append(f"=== Act {i} Plan ===\n{act_plan_json}")
            except:
                pass  # Act plan might not exist
        
        previous_acts_text = "\n\n".join(previous_acts) if previous_acts else "No previous acts"
        
        # Get rolling summary using director context function
        # We set current_chapter + 1 to avoid excluding current chapter
        progress = await self.memory.get_story_progress()
        latest_chapter = progress.get('latest_chapter_id', 0)
        
        rolling_summary = await self.memory.get_director_context(
            current_act_number=act_number,
            current_chapter_number=latest_chapter + 1,  # +1 to include latest chapter
            query="",  # Empty query gets general context
            k=5  # Last 5 relevant pieces
        )
        
        # Determine act purpose based on number
        total_acts = story_seed.get('act_count', 3)
        if act_number == 2:
            act_guidance = "Build rising tension, complicate the protagonist's journey, introduce reversals"
            word_percentage = 0.40 if total_acts == 3 else 0.30
        elif act_number == total_acts:
            act_guidance = "Resolve central conflict, deliver thematic payoff, provide closure"
            word_percentage = 0.35
        else:
            act_guidance = "Escalate conflict, develop character arcs, move toward resolution"
            word_percentage = 0.25
        
        suggested_word_count = int(remaining_words * word_percentage) if remaining_words > 0 else 5000
        suggested_chapters = max(2, suggested_word_count // 3000)
        
        # Get target audience for content filtering
        target_age = story_seed.get('target_audience_age', 'general')
        genres_str = ", ".join(story_seed.get('genre', ['Fiction']))  # Updated to handle list of genres

        return f"""Story Seed:
{json.dumps(story_seed, indent=2)}

Previous Acts Plans:
{previous_acts_text}

Story So Far (Rolling Summary) + Relevant Context:
{rolling_summary}

Current Story Stats:
- Word count so far: {current_word_count}
- Estimated remaining words: {remaining_words}
- Latest completed chapter: {latest_chapter}
- Target audience: {target_age} years old

This is Act {act_number} of {total_acts}.
Your goals: {act_guidance}

Maintain throughout:
- POV: {story_seed['style_guide']['pov']}
- Tone: {story_seed['tone']}
- Prose style: {story_seed['style_guide'].get('prose_style', 'Standard')}
- Genre conventions: {genres_str}
- Age-appropriate content for {target_age} year old readers

Suggested word count for this act: ~{suggested_word_count} words
Suggested chapter count: {suggested_chapters} chapters (~3000 words each)
"""


    async def check_act_completion(
        self, 
        story_title: str, 
        act_number: int
    ) -> Dict[str, Any]:
        """
        Check if the current act's closure condition has been met.
        
        Args:
            story_title: Story title
            act_number: Current act number
            
        Returns:
            Dict with keys: condition_met (bool), reasoning (str), next_action (str)
        """
        print(f"🔍 Checking Act {act_number} completion...")
        
        # Get act plan
        act_plan_json = await self.memory.get_long_term_document(
            metadata={'type': 'act_plan', 'act_id': act_number, 'story_title': story_title}
        )
        act_plan = json.loads(act_plan_json)
        closure_condition = act_plan.get('act_closure_condition', '')
        
        # Get story progress
        progress = await self.memory.get_story_progress()
        latest_chapter = progress.get('latest_chapter_id', 0)
        
        # Get rolling summary
        rolling_summary = await self.memory.get_director_context(
            current_act_number=act_number,
            current_chapter_number=latest_chapter + 1,
            query="",
            k=5
        )
        
        system_prompt = """You are the Act Completion Checker.

Evaluate if the act's closure condition has been clearly met based on the story so far.

Return JSON with:
{
  "condition_met": true or false,
  "reasoning": "Brief explanation (2-3 sentences)",
  "confidence": "high/medium/low"
}

Be strict: the condition must be CLEARLY fulfilled, not just implied."""
        
        human_prompt = f"""Act {act_number} Closure Condition:
{closure_condition}

Story So Far:
{rolling_summary}

Has the closure condition been met?"""
        
        response = await utility_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt
        )
        
        try:
            result = json.loads(response.content.strip())
            
            # Determine next action
            if result['condition_met']:
                story_seed_json = await self.memory.get_long_term_document(
                    metadata={'type': 'story_seed', 'story_title': story_title}
                )
                story_seed = json.loads(story_seed_json)
                total_acts = story_seed.get('act_count', 3)
                
                if act_number >= total_acts:
                    result['next_action'] = 'COMPLETE_STORY'
                else:
                    result['next_action'] = 'START_NEXT_ACT'
            else:
                result['next_action'] = 'CONTINUE_CURRENT_ACT'
            
            print(f"✅ Act completion check: {result['next_action']}")
            return result
            
        except json.JSONDecodeError:
            # Fallback
            return {
                'condition_met': False,
                'reasoning': 'Could not parse completion check',
                'confidence': 'low',
                'next_action': 'CONTINUE_CURRENT_ACT'
            }
