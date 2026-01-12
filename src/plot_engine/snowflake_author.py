
import json
from typing import Any, Dict, List, Optional, Tuple, Literal
from pydantic import BaseModel, Field, ConfigDict

from src.llm_client.llm_client import author_client, better_author_client, author_fast_client
from src.utilities.story_helpers import StoryHelpers


class FlexibleBase(BaseModel):
    """Base class allowing LLM to add arbitrary creative fields"""
    model_config = ConfigDict(extra="allow")


class OneTextForm(BaseModel):
    text: str


class CharacterSpecifications(FlexibleBase):
    """Base for all character types"""
    name: str
    age: int
    gender: str
    archetype: str
    core_trait: str
    background: str
    desire: str
    fear: str
    physical_description: str
    role_in_story: str
    character_arc: str
    relationship_to_protagonist: Optional[str] = None


class ProtagonistSpecifications(CharacterSpecifications):
    """Protagonist with additional fields"""
    internal_conflict: str
    external_conflict: str
    moral_weakness: str
    epiphany: str


class CharacterList(FlexibleBase):
    """Collection of all story characters"""
    protagonist: ProtagonistSpecifications
    antagonist: CharacterSpecifications
    supporting_characters: List[CharacterSpecifications] = Field(default_factory=list)
    minor_characters: List[CharacterSpecifications] = Field(default_factory=list)


class CharacterOnePager(FlexibleBase):
    """Detailed one-page character description"""
    character_name: str
    one_page_description: str
    backstory: str
    motivation: str
    growth_arc: str
    key_relationships: Dict[str, str]
    defining_moments: List[str]
    voice_and_mannerisms: str


class SceneSummary(FlexibleBase):
    """Individual scene breakdown"""
    scene_number: int
    scene_title: str
    pov_character: str
    location: str
    time: str
    characters_present: List[str]
    scene_purpose: str
    conflict: str
    outcome: str
    emotional_beat: str
    one_sentence_summary: str


class ChapterSummary(FlexibleBase):
    """Chapter-level organization"""
    chapter_number: int
    chapter_title: str
    scenes: List[SceneSummary]
    chapter_goal: str
    opening_hook: str
    closing_hook: str


class StoryOutline(FlexibleBase):
    """Complete story structure"""
    title: str
    logline: str
    acts: List[Dict[str, Any]]
    chapters: List[ChapterSummary]
    major_plot_points: Dict[str, str]
    subplots: List[Dict[str, str]]


class Scene(FlexibleBase):
    """Full scene with prose"""
    scene_number: int
    scene_title: str
    full_scene_text: str
    word_count: int


class SnowflakeWorkflow:
    """Complete Snowflake Method implementation"""
    
    def __init__(self):
        pass
        
    async def step_1_one_sentence_generate(self, one_sentence_form: str) -> Tuple[OneTextForm, dict, dict]:
        """Step 1: One-sentence summary"""
        system_prompt = f"""You are the beginning of a snow-flake method workflow for creating story plots.
Rewrite one sentence describing the plot for the user given one-sentence.

OUTPUT: Valid JSON. No markdown fences.
{OneTextForm.model_json_schema()}"""

        human_prompt = f"""User's One Sentence to improve: {one_sentence_form}
"""

        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.2
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            outline, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=OneTextForm
            )
            if isinstance(outline, tuple):
                outline = outline[0]
        except Exception as e:
            print(f"⚠️ Parsing failed ({e}).")
            raise
        print("Created One Sentence story...")
        return outline, author_tokens, utility_tokens
    
    async def step_1_one_sentence_feedback(self, one_sentence_form: str) -> Tuple[OneTextForm, dict]:
        """Step 1: One-sentence summary"""
        system_prompt = f"""You are an honest reviewer for a snowflake method based AI writer assistant app, review the one sentence, that the user wrote and give honest feedback that includes strengths, weaknesses, etc.
output in markdown format. No tables just bullet points, no artificial lines.
Be as concise as you can.
        """

        human_prompt = f"""User's One Sentence to review: {one_sentence_form}
"""

        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.2
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)


        print("Created One Sentence story...")
        return clean_resp, author_tokens
    
    async def step_2_one_paragraph_generate(self, one_sentence_context: str, user_one_paragraph: str = None) -> Tuple[OneTextForm, dict, dict]:
        """Step 2: Expand to one paragraph"""
        system_prompt = f"""You are part of a snow-flake method workflow for creating story plots.
Create/re-write one paragraph describing the plot for the given user paragraph describing the story and the given one sentence for the story.

Keep the given parameters under story context constant.
OUTPUT: Valid JSON. No markdown fences.
{OneTextForm.model_json_schema()}"""

        human_prompt = f"""Story One Sentence: {one_sentence_context}

{f"User Paragraph to rewrite: {user_one_paragraph}" if user_one_paragraph else "Create One Paragraph for the following one sentence."}
"""

        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.4
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            outline, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=OneTextForm
            )
            if isinstance(outline, tuple):
                outline = outline[0]
        except Exception as e:
            print(f"⚠️ Parsing failed ({e}).")
            raise

        print("Created One Paragraph story...")
        return outline, author_tokens, utility_tokens
    
    async def step_2_one_paragraph_feedback(self, one_paragraph_form: str, one_senetence_form: str) -> Tuple[OneTextForm, dict]:
        """Step 2: Expand to one paragraph"""
        system_prompt = f"""You are an honest reviewer for a snowflake method based AI writer assistant app, review the one Paragraph, that the user wrote and give honest feedback that includes strengths, weaknesses, etc.
output in markdown format. No tables just bullet points, no artificial lines.
Be as concise as you can.
        """

        human_prompt = f"""Story One Sentence for reference: {one_senetence_form}

User's One Paragraph to review: {one_paragraph_form}
"""

        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.2
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)


        print("Created One Sentence story...")
        return clean_resp, author_tokens
    
    async def step_3_character_summaries(self, one_paragraph_context: dict) -> Tuple[CharacterList, dict, dict]:
        """Step 3: Create major character summaries"""
        system_prompt = f"""You are part of a snow-flake method workflow for creating story characters.
Create the protagonist, antagonist, and 2-4 supporting characters based on the story context.

Each character should be well-developed with clear motivations, conflicts, and arcs.

OUTPUT: Valid JSON. No markdown fences.
{CharacterList.model_json_schema()}"""

        human_prompt = f"""Story Context: {json.dumps(one_paragraph_context, indent=2)}"""

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.3
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            characters, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=CharacterList
            )
            if isinstance(characters, tuple):
                characters = characters[0]
        except Exception as e:
            print(f"⚠️ Character creation failed ({e}).")
            raise
            
        print("Created character summaries...")
        return characters, tokens, utility_tokens
    
    async def step_4_one_page_plot(self, one_paragraph_context: dict) -> Tuple[OneTextForm, dict, dict]:
        """Step 4: Expand to one-page plot summary"""
        system_prompt = f"""You are part of a snow-flake method workflow for creating story plots.
Create one page describing the plot for the given paragraph describing the story and other specs given along.

Dont change the given parameters under story context, keep them the same.
OUTPUT: Valid JSON. No markdown fences.
{OneTextForm.model_json_schema()}"""

        human_prompt = f"""Story Context: {json.dumps(one_paragraph_context, indent=2)}
"""

        response, author_tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.6
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            outline, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=OneTextForm
            )
            if isinstance(outline, tuple):
                outline = outline[0]
        except Exception as e:
            print(f"⚠️ Parsing failed ({e}).")
            raise

        print("Created One page story...")
        return outline, author_tokens, utility_tokens
    
    async def _character_one_pagers(self, character: dict, story_context: dict) -> Tuple[CharacterOnePager, dict, dict]:
        """Create one-page description for each major character"""
        system_prompt = f"""You are part of a snow-flake method workflow for deepening character development.
Create a detailed one-page description for the given character within the story context.

Focus on making the character three-dimensional, compelling, and integral to the story.

OUTPUT: Valid JSON. No markdown fences.
{CharacterOnePager.model_json_schema()}"""

        human_prompt = f"""Character Summary: {json.dumps(character, indent=2)}

Story Context: {json.dumps(story_context, indent=2)}"""

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.2
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            one_pager, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=CharacterOnePager
            )
            if isinstance(one_pager, tuple):
                one_pager = one_pager[0]
        except Exception as e:
            print(f"⚠️ Character one-pager creation failed ({e}).")
            raise
            
        print(f"Created one-pager for {character.name}...")
        return one_pager, tokens, utility_tokens
    
    async def step_5_four_page_outline(self, one_page_context: dict, characters: List[dict]) -> Tuple[StoryOutline, dict, dict]:
        """Step 5: Expand to four-page story outline with major plot points"""
        system_prompt = f"""You are part of a snow-flake method workflow for story structure.
Create a detailed four-page story outline including:
- Complete act structure with major plot points
- Key turning points (inciting incident, plot points, midpoint, climax)
- Subplots
- Story progression

OUTPUT: Valid JSON. No markdown fences.
{StoryOutline.model_json_schema()}"""

        human_prompt = f"""Story Context: {json.dumps(one_page_context, indent=2)}

Characters: {json.dumps(characters, indent=2)}"""

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.4
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            outline, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=StoryOutline
            )
            if isinstance(outline, tuple):
                outline = outline[0]
        except Exception as e:
            print(f"⚠️ Outline creation failed ({e}).")
            raise
            
        print("Created four-page story outline...")
        return outline, tokens, utility_tokens
    
    async def step_6_character_charts(self, characters: list[dict], outline: dict) -> Tuple[Dict[str, CharacterOnePager], dict, dict]:
        """Step 6: Create detailed character charts for all major characters"""
        character_charts = {}
        total_tokens = {}
        total_utility = {}
        
        for char in characters:
            chart, tokens, utility = await self._character_one_pagers(
                character=char, 
                story_context=outline
            )
            character_charts[char['name']] = chart
            
            for key, val in tokens.items():
                total_tokens[key] = total_tokens.get(key, 0) + val
            for key, val in utility.items():
                total_utility[key] = total_utility.get(key, 0) + val
        
        print(f"Created character charts for {len(characters)} characters...")
        return character_charts, total_tokens, total_utility
    
    async def step_7_scene_list(self, outline: StoryOutline, characters: Dict[str, CharacterOnePager], target_length: int) -> Tuple[List[SceneSummary], dict, dict]:
        """Step 7: Create detailed scene-by-scene breakdown"""
        
        system_prompt = f"""You are part of a snow-flake method workflow for scene planning.
Create a detailed scene-by-scene breakdown for the story. Estimate the number of scenes it would take to execute this chapter then output plans for them accordingly.

Each scene should:
- Advance the plot or develop character
- Have clear conflict and purpose
- Connect logically to surrounding scenes
- Build toward story climax

Return a list of scene summaries.

OUTPUT: Valid JSON array of scenes. No markdown fences.
Example: [{{"scene_number": 1, "scene_title": "...", ...}}]"""

        human_prompt = f"""Story Outline: {json.dumps(outline.model_dump(), indent=2)}

Character Details: {json.dumps({k: v.model_dump() for k, v in characters.items()}, indent=2)}

Target Story Length: {target_length} words"""

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.3
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            scenes_data = json.loads(clean_resp)
            scenes = [SceneSummary(**scene) for scene in scenes_data]
            utility_tokens = {}
        except Exception as e:
            print(f"⚠️ Scene list creation failed ({e}). Retrying with parser...")
            # Fallback: try to parse as single object
            try:
                from pydantic import TypeAdapter
                adapter = TypeAdapter(List[SceneSummary])
                scenes = adapter.validate_python(json.loads(clean_resp))
                utility_tokens = {}
            except Exception as e2:
                print(f"⚠️ Second parse attempt failed ({e2}).")
                raise
        
        print(f"Created scene list with {len(scenes)} scenes...")
        return scenes, tokens, utility_tokens
    
    async def step_8_scene_details(self, scenes: List[SceneSummary], outline: StoryOutline, characters: Dict[str, CharacterOnePager]) -> Tuple[List[SceneSummary], dict, dict]:
        """Step 8: Expand each scene summary with more detail (optional enhancement)"""
        # This step typically involves the writer manually expanding scenes
        # For automation, we can add more narrative details to each scene
        enhanced_scenes = []
        total_tokens = {}
        total_utility = {}
        
        for scene in scenes[:5]:  # Demo: only process first 5 to avoid token limits
            system_prompt = f"""You are part of a snow-flake method workflow for scene development.
Enhance the given scene summary with richer detail, including:
- Specific character actions and reactions
- Dialogue snippets or key lines
- Sensory details
- Emotional progression
- Scene beats

OUTPUT: Valid JSON matching the scene structure. No markdown fences.
{SceneSummary.model_json_schema()}"""

            human_prompt = f"""Scene to enhance: {json.dumps(scene.model_dump(), indent=2)}

Full Story Context: {json.dumps(outline.model_dump(), indent=2)}

Characters: {json.dumps({k: v.model_dump() for k, v in characters.items()}, indent=2)}"""

            response, tokens = await author_fast_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=1.2
            )

            clean_resp = StoryHelpers._extract_content(response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                enhanced, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean_resp,
                    parser=SceneSummary
                )
                if isinstance(enhanced, tuple):
                    enhanced = enhanced[0]
                enhanced_scenes.append(enhanced)
            except Exception as e:
                print(f"⚠️ Scene enhancement failed for scene {scene.scene_number} ({e}). Using original.")
                enhanced_scenes.append(scene)
            
            for key, val in tokens.items():
                total_tokens[key] = total_tokens.get(key, 0) + val
        
        # Add remaining unprocessed scenes
        enhanced_scenes.extend(scenes[5:])
        
        print(f"Enhanced {min(5, len(scenes))} scenes with details...")
        return enhanced_scenes, total_tokens, total_utility
    
    async def step_9_write_scenes(self, scene: SceneSummary, outline: StoryOutline, characters: Dict[str, CharacterOnePager], previous_scene: Optional[str] = None) -> Tuple[Scene, dict, dict]:
        """Step 9: Write full prose for each scene"""
        system_prompt = f"""You are a creative fiction writer completing a snowflake method workflow.
Write the full prose for the given scene. 

Requirements:
- Write in vivid, engaging prose appropriate for the genre and tone
- Include dialogue, action, description, and internal thoughts
- Show, don't tell
- Maintain consistent voice and style
- Aim for 1000-2000 words
- Create an immersive, compelling scene

OUTPUT: Valid JSON. No markdown fences.
{Scene.model_json_schema()}"""

        context_addition = ""
        if previous_scene:
            context_addition = f"\n\nPrevious Scene Text (for continuity): {previous_scene[-500:]}"

        human_prompt = f"""Scene to write: {json.dumps(scene.model_dump(), indent=2)}

Story Context: {json.dumps(outline.model_dump(), indent=2)}

Characters: {json.dumps({k: v.model_dump() for k, v in characters.items()}, indent=2)}{context_addition}"""

        response, tokens = await better_author_client(
            system_prompt=system_prompt,
            human_prompt=human_prompt,
            llm_temp=1.5
        )

        clean_resp = StoryHelpers._extract_content(response)
        clean_resp = StoryHelpers._strip_code_fences(clean_resp)

        try:
            full_scene, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean_resp,
                parser=Scene
            )
            if isinstance(full_scene, tuple):
                full_scene = full_scene[0]
        except Exception as e:
            print(f"⚠️ Scene writing failed for scene {scene.scene_number} ({e}).")
            raise
        
        print(f"Wrote scene {scene.scene_number}: {scene.scene_title}...")
        return full_scene, tokens, utility_tokens


# class BasicAuthor:
#     """Original implementation for backward compatibility"""
    
#     def __init__(self):
#         pass
        
    
#     async def protagonist_generation(self, one_page_context: dict) -> Tuple[ProtagonistSpecifications, dict, dict]:
#         system_prompt = f"""You are part of a snow-flake method workflow for creating story characters.
# Create a protagonist given the story context, follow the json format, add any other field as required.

# OUTPUT: Valid JSON. No markdown fences.
# {ProtagonistSpecifications.model_json_schema()}"""

#         human_prompt = f"""Story Context: {json.dumps(one_page_context, indent=2)}
# """

#         response, author_tokens = await better_author_client(
#             system_prompt=system_prompt,
#             human_prompt=human_prompt,
#             llm_temp=1.2
#         )

#         clean_resp = StoryHelpers._extract_content(response)
#         clean_resp = StoryHelpers._strip_code_fences(clean_resp)

#         try:
#             outline, utility_tokens = await StoryHelpers.load_json_with_retry(
#                 text=clean_resp,
#                 parser=ProtagonistSpecifications
#             )
#             if isinstance(outline, tuple):
#                 outline = outline[0]
#         except Exception as e:
#             print(f"⚠️ Parsing failed ({e}).")
#             raise
#         print("Created Protagonist Specifications...")
#         return outline, author_tokens, utility_tokens