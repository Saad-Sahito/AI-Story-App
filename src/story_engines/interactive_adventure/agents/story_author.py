
import json
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
import gc
from langchain_core.output_parsers import PydanticOutputParser
from src.llm_client.llm_client import author_client, utility_client, ingestor_client
from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem


# ============================================================================
# PYDANTIC MODELS FOR INTERACTIVE STORIES
# ============================================================================

class StorySeed(BaseModel):
    """Initial story foundation - lean and flexible for interactive narratives"""
    title: str = Field(..., description="Story title")
    premise: str = Field(..., description="Core story concept in 3-4 sentences")
    protagonist: Dict[str, str] = Field(
        ..., 
        description="Main character with keys: name, core_trait, desire, fear"
    )
    world_essentials: Dict[str, str] = Field(
        ...,
        description="Setting info with keys: setting, time_period, key_rule"
    )
    initial_conflict: str = Field(..., description="Opening tension/conflict")
    themes: List[str] = Field(..., description="1-3 core themes")
    genre: List[str] = Field(..., description="Primary genres (list of 1-3 genres)")
    tone: str = Field(..., description="Emotional tone (e.g., dark, hopeful, epic)")
    style_guide: Dict[str, str] = Field(
        ...,
        description="Keys: prose_style, pov, tense, narrative_voice"
    )
    target_length: int = Field(..., description="Target word count for entire story")
    act_count: int = Field(..., description="Number of acts (typically 3 for interactive)")


class ActPlan(BaseModel):
    """
    Lightweight act plan for interactive stories.
    No rigid chapter outlines - story flows with user choices.
    """
    act_number: int = Field(..., description="Which act this is (1, 2, 3)")
    act_title: str = Field(..., description="Evocative name for this act")
    story_style_guide: str = Field(..., description="prose style, pov, tense, narrative voice")
    act_purpose: str = Field(
        ..., 
        description="What this act accomplishes in 2-3 sentences"
    )
    key_themes: List[str] = Field(
        ..., 
        description="2-4 themes to emphasize in this act"
    )
    potential_branches: List[str] = Field(
        ...,
        description="3-5 possible story directions based on user choices"
    )
    emotional_trajectory: str = Field(
        ..., 
        description="Overall emotional shift across this act"
    )
    key_moments: List[str] = Field(
        ...,
        description="3-5 pivotal moments or revelations that should occur"
    )
    target_word_count: int = Field(
        ...,
        description="Suggested word count for this act"
    )
    act_closure_suggestion: str = Field(
        ..., 
        description="Natural stopping point for this act (e.g., 'Hero makes irreversible choice')"
    )

class ActSummary(BaseModel):
    act_summary: str = Field(description="Detailed summary of the entire Act")

# ============================================================================
# OUTPUT PARSERS
# ============================================================================

story_seed_parser = PydanticOutputParser(pydantic_object=StorySeed)
act_plan_parser = PydanticOutputParser(pydantic_object=ActPlan)
act_ingestor_parser = PydanticOutputParser(pydantic_object=ActSummary)

# ============================================================================
# STORY AUTHOR CLASS (INTERACTIVE)
# ============================================================================

class StoryAuthor:
    """
    Manages interactive story creation through act-based, choice-driven planning.
    
    Workflow:
    1. create_story_seed() - Initial lean foundation
    2. generate_blurb() - Marketing copy from seed
    3. generate_cover_image() - Visual based on blurb
    4. plan_act() - Lightweight act plans (no rigid chapter outlines)
    """
    
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system


    async def create_story_seed(
        self, 
        user_context: Dict[str, Any],
        model: str = "None"
    ) -> Tuple[StorySeed, int]:
        """
        Create initial story foundation from user context for interactive narrative.
        
        Args:
            user_context: Dict with keys: POV, Tone, Genre (List[str]), Title, Length, 
                         Setting, Guide Prose, Additional Themes, target_audience_age
            story_title: Story title from user_context['Title']
            model: LLM model to use
        
        Returns:
            Tuple of (StorySeed object, token_count)
        """
        print("🌱 Creating story seed for interactive story...")
        
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
        
        # Interactive stories typically shorter - use 3 acts for most lengths
        if target_length <= 25000:
            act_count = 3
        elif target_length <= 50000:
            act_count = 3
        else:
            act_count = 4  # Only very long interactive stories get 4 acts
        
        # Handle genres as a list
        genres = user_context.get('Genre', ['Fiction'])
        genres_str = ", ".join(genres)
        themes = user_context.get('Additional Themes', [''])
        themes_str = ", ".join(themes)
        title = user_context.get('Title')
        if title in [None, "None", "", "Untitled Story", "Unitled story", "untitled story", "Null", "NULL", "Nill", "NILL", "null", "nill"]:
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
            protagonist_instruction = "- Protagonist: Create from scratch (age-appropriate), keep gender limited to male or female"

        # Then update the system_prompt:
        system_prompt = f"""
        You are the Story Architect for an interactive, choice-driven narrative.
        Your goal: Create a flexible story foundation - NOT a rigid blueprint. This seed guides act planning, which happens progressively as the user makes choices.

        The user has specified:
        - POV: {user_context.get('POV', 'Third-person')}
        - Tone: {user_context.get('Tone', 'Balanced')}
        - Genre: {genres_str}
        - Setting: {user_context.get('Setting', 'To be determined')}
        - Prose Style: {user_context.get('Guide Prose', 'Standard narrative')}
        - Themes: {themes_str}
        - Target Audience: {user_context.get('target_audience_age', 'General')} years old
        - Target Length: {target_length} words ({act_count} acts recommended)
        {protagonist_instruction}

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

        {story_seed_parser.get_format_instructions()}
        """

        human_prompt = f"""Create an interactive story seed with these specifications:

Setting: {user_context.get('Setting', 'Create an appropriate setting')}
Genres: {genres_str}
POV: {user_context.get('POV', 'Third-person')}
Tone: {user_context.get('Tone', 'Balanced')}
Prose Style: {user_context.get('Guide Prose', 'Standard')}
Themes: {themes_str}
Target Audience Age: {user_context.get('target_audience_age', 'General audience')}
Length: {user_context.get('Length', 'Standard')} ({target_length} words)

Generate a compelling foundation that enables meaningful user choices and branching narratives, blending the specified genres ({genres_str}) appropriately."""
        
        response, tokens = await author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=0.9,
            model=model
        )
        
        # Parse into Pydantic model
        story_seed = story_seed_parser.parse(response.content.strip())
        
        del response
        response = await utility_client(
            system_prompt="Check if the user context matches the story_seed produced generally, " \
            "important things to check are the protagonist vital details like name, age, etc. " \
            f"Output the same seed back if correct, if not make changes and output accordingly:  {story_seed_parser.get_format_instructions()}",
            human_prompt=f"user context: {user_context} " \
            f"story seed: {story_seed}"
        )
        story_seed = story_seed_parser.parse(response.content.strip())
        del response
        
        print(f"✅ Interactive story seed created: '{story_seed.title}' ({story_seed.act_count} acts, {target_length} words)")
        return story_seed, tokens


    async def generate_blurb(self, story_seed: StorySeed) -> str:
        """
        Create compelling back-cover blurb from story seed.
        
        Args:
            story_seed: The StorySeed object
            
        Returns:
            Marketing blurb as string
        """
        print("📖 Generating book blurb...")
        
        system_prompt = """You are a professional book marketer specializing in interactive fiction back-cover copy.

Write a short captivating blurb (<50 words) that:
- Hooks readers emotionally
- Establishes atmosphere and stakes
- Teases the protagonist's journey
- Hints at meaningful choices without spoiling them
- Matches the story's genres and tone
- Emphasizes agency: "Your choices shape the story"

Style: Professional, marketable, similar to what you'd find on interactive fiction or game narratives.

Output only the blurb—no commentary or formatting."""
        
        genres_str = ", ".join(story_seed.genre)  # Updated to handle list of genres
        
        seed_summary = f"""
Title: {story_seed.title}
Genres: {genres_str}
Tone: {story_seed.tone}
Premise: {story_seed.premise}
Protagonist: {story_seed.protagonist['name']} - {story_seed.protagonist['core_trait']}
Initial Conflict: {story_seed.initial_conflict}
Themes: {', '.join(story_seed.themes)}
Setting: {story_seed.world_essentials.get('setting', 'Unknown')}

Note: This is an INTERACTIVE story where user choices matter.
"""
        
        response = await utility_client(
            system_prompt=system_prompt,
            human_prompt=f"Create a blurb for this interactive story:\n\n{seed_summary}"
        )
        
        blurb = response.content.strip()
        print("✅ Blurb generated")
        return blurb


    

    async def plan_act(
        self, 
        story_title: str,
        act_number: int,
        model: str = "None"
    ) -> Tuple[ActPlan, int]:
        """
        Generate lightweight act plan for interactive story.
        No rigid chapter outlines - story flows with user choices.
        
        Args:
            story_title: Title of the story
            act_number: Which act to plan (1, 2, 3, etc.)
            model: LLM model to use
            
        Returns:
            Tuple of (ActPlan object, token_count)
        """
        print(f"📋 Planning Act {act_number} for interactive story...")
        
        # Get story seed
        story_seed_json = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': story_title}
        )
        story_seed = json.loads(story_seed_json) if story_seed_json else {}
        
        # Get story progress
        progress = await self.memory.get_story_progress()
        current_word_count = progress.get('story_word_count', 0) if progress else 0
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
        genres_str = ", ".join(story_seed.get('genre', ['Fiction']))  # Updated to handle list of genres
        
        system_prompt = f"""You are the Act Planner for an interactive, choice-driven story.

Design Act {act_number} as a FLEXIBLE framework, not a rigid blueprint.

Your output must include:
1. act_number (int)
2. act_title (string) - Evocative name
3. story_style_guide (string)
4. act_purpose (string) - What this act accomplishes (2-3 sentences)
5. key_themes (list of strings) - 2-4 themes to emphasize
6. potential_branches (list of strings) - 3-5 possible story directions based on user choices
7. emotional_trajectory (string) - How emotions shift across this act
8. key_moments (list of strings) - 3-5 pivotal moments or revelations that COULD occur (not must)
9. target_word_count (int) - Suggested word count for this act
10. act_closure_suggestion (string) - Natural stopping point (e.g., "Hero makes irreversible choice")

CRITICAL for Interactive:
- Do NOT plan fixed plot points or predetermined outcomes
- Focus on themes, tensions, and potential conflicts
- Suggest possible branches, not required paths
- Leave room for user agency and emergent storytelling
- Respect established genres: {genres_str}

Guidelines:
- Keep planning loose and adaptive
- Respect established tone, world rules, and character motivations
- For Act 1: Establish world, introduce choices, present initial conflict
- For Act 2+: Build on previous developments, escalate stakes, expand choices
- Allow story to evolve based on actual user decisions

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
        
        print(f"✅ Act {act_number} planned: '{act_plan.act_title}' (~{act_plan.target_word_count} words)")
        print(f"   Potential branches: {len(act_plan.potential_branches)}")
        
        return act_plan, tokens


    def _build_act_1_context(self, story_seed: dict) -> str:
        """Build context prompt for Act 1 planning."""
        act_1_target = int(story_seed['target_length'] * 0.30)  # 30% for Act 1 (interactive)
        genres_str = ", ".join(story_seed.get('genre', ['Fiction']))  # Updated to handle list of genres
        
        return f"""Story Seed:
{json.dumps(story_seed, indent=2)}

This is Act 1 of an INTERACTIVE story. Your goals:
- Establish the world and its rules (setting: {story_seed['world_essentials'].get('setting', 'TBD')})
- Introduce {story_seed['protagonist']['name']} and their ordinary world
- Present the inciting incident
- Offer initial meaningful choices to the user
- Set up the initial conflict: {story_seed['initial_conflict']}
- End when user makes a defining choice that commits them to the journey
- Maintain {story_seed['style_guide']['pov']} POV and {story_seed['tone']} tone
- Blend genres: {genres_str}
- Keep content appropriate for {story_seed.get('target_audience_age', 'general')} year old readers

Target word count for Act 1: ~{act_1_target} words

Remember: Plan themes and tensions, not fixed outcomes. User choices will shape the actual story.
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
                pass
        
        previous_acts_text = "\n\n".join(previous_acts) if previous_acts else "No previous acts"
        
        # Get rolling summary using director context
        progress = await self.memory.get_story_progress()
        latest_chapter = progress.get('latest_chapter_id', 0) if progress else 0
        
        rolling_summary = await self.memory.get_director_context(
            current_act_number=act_number,
            current_chapter_number=latest_chapter + 1,
            query="",
            k=5
        )
        
        # Determine act purpose
        total_acts = story_seed.get('act_count', 3)
        if act_number == 2:
            act_guidance = "Escalate stakes, complicate choices, introduce new tensions based on user decisions"
            word_percentage = 0.40
        elif act_number == total_acts:
            act_guidance = "Bring story toward resolution based on accumulated choices, deliver thematic payoff"
            word_percentage = 0.30
        else:
            act_guidance = "Develop consequences, expand world, present harder choices"
            word_percentage = 0.30
        
        suggested_word_count = int(remaining_words * word_percentage) if remaining_words > 0 else 5000
        
        target_age = story_seed.get('target_audience_age', 'general')
        genres_str = ", ".join(story_seed.get('genre', ['Fiction']))
        
        return f"""Story Seed:
{json.dumps(story_seed, indent=2)}

Previous Acts Plans:
{previous_acts_text}

Story So Far (What Actually Happened Based on User Choices) + Relevant Context:
{rolling_summary}

Current Story Stats:
- Word count so far: {current_word_count}
- Estimated remaining words: {remaining_words}
- Latest completed chapter: {latest_chapter}
- Target audience: {target_age} years old

This is Act {act_number} of {total_acts} for an INTERACTIVE story.
Your goals: {act_guidance}

Maintain throughout:
- POV: {story_seed['style_guide']['pov']}
- Tone: {story_seed['tone']}
- Prose style: {story_seed['style_guide'].get('prose_style', 'Standard')}
- Genre conventions: {genres_str}
- Age-appropriate content for {target_age} year old readers

Suggested word count for this act: ~{suggested_word_count} words

CRITICAL: Base your planning on what ACTUALLY happened in previous acts (see rolling summary).
User choices may have taken the story in unexpected directions - adapt your plan accordingly.
Suggest potential branches, but don't force predetermined outcomes.
"""

    async def ingest_act(self, current_act: int, chapters: int, story_title: str, max_retries: int = 3):
        """
        Ingests entire act using act chapter summaries.
        """
        text = self.memory.get_entire_act_chapters_for_act_ingestion_episodic_story(current_act=current_act, chapters=chapters)
        combined_text = '\n'.join(text)

        system_prompt = f"""
You are the Act Breakdown Agent. Extract relevant and absolutely important info from the Act, including relevant references to user decisions wherever indicated.
Always include chapter number reference alongside relevant text.

Respond ONLY in JSON with this schema:
{act_ingestor_parser.get_format_instructions()}
"""

        human_prompt = f"""
Current Act: {current_act}

Act Chapters Summaries:
{combined_text}
"""

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, ActSummary, act_ingestor_parser)
            
            if success:
                await self.memory.add_post_act_bundle(
                    scene_bundle=result,
                    metadata={
                        "act_id": current_act,
                        "story_title": story_title,
                        "type": "act summary"
                    }
                )
                return "success"

            else:
                print(f"[Attempt {attempt}] Scene ingestion validation failed:", exc)
            
            try:
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, ActSummary, act_ingestor_parser)
                del fixed_clean
                
                if success:
                    await self.memory.add_post_act_bundle(
                        scene_bundle=result,
                        metadata={
                            "act_id": current_act,
                            "story_title": story_title,
                            "type": "act summary"
                        }
                    )
                    return "success"
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised:", inner_e)
            
            if attempt < max_retries:
                print(f"Retrying ingest_scene... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted; returning minimal structure.")
                del system_prompt, human_prompt
                return "failed"