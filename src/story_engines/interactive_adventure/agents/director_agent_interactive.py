# src/story_engines/interactive_adventure/agents/director_agent.py
import asyncio
import json
import gc
import re
import traceback
from typing import Any, Dict, List, Literal, Optional, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
from .shared_scene_planner import UserSceneContext
from config_vars import tier_2_monthly_words_limit, tier_1_monthly_words_limit
import src.story_engines.interactive_adventure.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import director_client
from src.utilities.ingestor import Ingestor


# ============================================================================
# STATE AND MODELS
# ============================================================================
@dataclass
class StoryState:
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    story_word_count: int = 0
    current_chapter_word_count: int = 0
    current_act_id: int = 1
    messages: List[Any] = field(default_factory=list)
    next_action: str = ""


class SceneDirectorOutput(BaseModel):
    instructions: str = Field(
        description="Detailed scene instructions including chapter_id, scene_id, recap, characters, detailed_scene_blueprint, user_decision_points, screenplay_notes, and style guide. Provided as a plain string, not a nested dict."
    )
    target_scene_word_count: int = Field(
        description="Target word count for the scene, guiding the Scene Writer on the expected length. If ending chapter leave empty."
    )
    action: Literal["generate_and_ingest", "END"] = Field(
        description="Action to take now for instructions: 'generate_and_ingest' to continue or 'END' if the chapter closure condition is met and chapter should end NOW."
    )


scene_director_parser = PydanticOutputParser(pydantic_object=SceneDirectorOutput)


class ChapterDirectorOutput(BaseModel):
    instructions: str = Field(
        description="Detailed chapter instructions including style guide for the story. Plain string, not nested dict. Include chapter word count target as 'word count: ' Leave empty if story should end NOW."
    )
    action: Literal["continue", "END"] = Field(
        description="Output continue if story not complete, END otherwise"
    )


chapter_director_parser = PydanticOutputParser(pydantic_object=ChapterDirectorOutput)


# ============================================================================
# DIRECTOR GRAPH (REFACTORED FOR ACT-BASED INTERACTIVE)
# ============================================================================
class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("chapter_director_node")
        self.graph.add_node("chapter_director_node", self.chapter_director_node)
        self.graph.add_node("scene_director_node", self.scene_director_node)
        self.graph.add_node("generate_and_ingest", self.generate_and_ingest_node)
        self.graph.add_node("ingest_chapter", self.ingest_chapter)
        self.graph.add_node("story_complete", self.story_complete)

        self.graph.add_edge("chapter_director_node", "scene_director_node")
        self.graph.add_conditional_edges(
            "chapter_director_node",
            lambda state: state.next_action,
            {
                "continue": "scene_director_node",
                "END": "story_complete",
            },
        )
        self.graph.add_conditional_edges(
            "scene_director_node",
            lambda state: state.next_action,
            {
                "generate_and_ingest": "generate_and_ingest",
                "END": "ingest_chapter",
                "ERROR": END
            },
        )

        def route_after_generate(state: StoryState):
            if state.next_action == "END":
                return END
            return "scene_director_node"

        self.graph.add_conditional_edges(
            "generate_and_ingest",
            route_after_generate,
            {
                "scene_director_node": "scene_director_node",
                END: END
            }
        )

        self.graph.add_edge("ingest_chapter", END)
        self.graph.add_edge("story_complete", END)
        self.compiled = self.graph.compile()

        self.act_title = ""
        self.current_chap_summary = ""
        self.llm_temp = 0.7
        self.model = ""
        self.target_scene_word_count = 0

    async def story_complete(self, state: StoryState):
        """Mark story as complete"""
        await self.memory.mark_story_complete()
        status_payload = {"type": "status", "message": "story complete"}
        try:
            self.scene_chunk_callback(status_payload)
        except Exception as e:
            print(f"scene_chunk_callback raised: {e}")
        return state

    def _get_latest_director_message(self, state: StoryState) -> str:
        """Extract the latest director instructions from state messages"""
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    async def generate_and_ingest_node(self, state: StoryState):
        """Generate scene using shared scene planner and ingest."""
        print("Called Generate and Ingest Node!")
        if scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE is None:
            print("ERROR: INTERACTIVE_SCENE_PLANNER_SERVICE is None!")
            state.next_action = "FATAL_ERROR"
            raise RuntimeError("Scene planner service not initialized")

        director_instructions = self._get_latest_director_message(state)
        user_context = UserSceneContext.create_for_user(
            user_id=self.memory.user_id,
            story_id=self.memory.story_id,
            director_instructions=director_instructions,
            scene_chunk_callback=self.scene_chunk_callback,
            llm_temp=self.llm_temp,
            model=self.model,
            token_usage=self.writer_token_usage,
            target_length=self.target_length,
            scene_target_length=self.target_scene_word_count
        )

        print(f"DEBUG: Calling run_scene for {user_context.user_id}/{user_context.story_id}")
        scene_text, scene_cluster, status, writer_tokens, scene_word_count = await scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE.run_scene(
            user_context=user_context,
            stop_event=self.stop_event
        )
        print("Scene generation complete.")

        if status == "CANCELLED":
            state.next_action = "END"
            return {
                "scene_id": state.scene_id,
                "current_chapter_id": state.current_chapter_id,
                "current_act_id": state.current_act_id,
                "next_action": "END"
            }
        elif status.startswith("FATAL:") or status.startswith("EXCEPTION:"):
            print(f"Fatal error from scene planner: {status}")
            state.next_action = "FATAL_ERROR"
            try:
                self.scene_chunk_callback({"type": "status", "ERROR": status})
            except Exception as e:
                print(f"scene_chunk_callback raised: {e}")
            raise RuntimeError(f"Scene planner error: {status}")
        elif status == "SUCCESS":
            await self.memory.update_user_monthly_word_count(word_count=scene_word_count)
            del user_context
            gc.collect()
            print("INSIDE DIRECTOR Writer token usage: ", writer_tokens)

            # Handle user continue/stop choice
            from setup.shared_redis_pool import get_redis_client
            try:
                redis_client = await get_redis_client()
                queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"
                resume_payload = {"type": "save"}
                try:
                    self.scene_chunk_callback(resume_payload)
                except Exception as e:
                    print(f"ERROR: scene_chunk_callback raised: {e}")
                    import traceback
                    traceback.print_exc()

                user_choice = False
                for _ in range(600):
                    choice = await redis_client.lpop(queue_key)
                    if choice in (b'1', 1, '1'):
                        user_choice = True
                        print("User chose to continue")
                        break
                    elif choice is not None:
                        print(f"User chose not to continue ({choice})")
                        break
                    await asyncio.sleep(1.0)

                if not user_choice:
                    print("User chose not to continue")
                    return {
                        "scene_id": state.scene_id,
                        "current_chapter_id": state.current_chapter_id,
                        "current_act_id": state.current_act_id,
                        "next_action": "END"
                    }
            except Exception as e:
                print(f"ERROR in user input handling: {e}")
                import traceback
                traceback.print_exc()
                state.next_action = "FATAL_ERROR"
                raise RuntimeError(f"User input error: {e}")

            # Ingest scene
            self.scene_chunk_callback({"type": "status", "message": "saving story"})
            self.writer_token_usage = writer_tokens
            del writer_tokens

            chars = await self.memory.get_long_term_characters_names()
            worlds = await self.memory.get_long_term_worlds_names()
            scene_bundle = await Ingestor.ingest_scene(
                chapter_id=state.current_chapter_id,
                scene_id=state.scene_id,
                scene_text=scene_text,
                chars=chars,
                worlds=worlds
            )
            gc.collect()

            if scene_bundle:
                state.current_chapter_word_count += scene_word_count
                state.story_word_count += scene_word_count
                await self.memory.update_story_progress(
                    metadata={
                        "latest_chapter_id": state.current_chapter_id,
                        "continue_scene_id": state.scene_id + 1,
                        "story_word_count": state.story_word_count,
                        "chapter_word_count": state.current_chapter_word_count,
                        "writer_token_usage": self.writer_token_usage
                    }
                )
            else:
                print("Scene ingestion failed; returning minimal structure.")
                scene_bundle = {
                    "story_summary": "",
                    "character_details": {},
                    "world_details": {}
                }

            await self.memory.add_story_scene_cluster(
                text=scene_cluster,
                metadata={
                    "act_id": state.current_act_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "scene_id": state.scene_id,
                    "chapter_word_count": state.current_chapter_word_count,
                    "act_title": self.act_title
                }
            )

            await self.memory.add_post_scene_bundle(
                scene_bundle=scene_bundle,
                metadata={
                    "act_id": state.current_act_id,
                    "scene_id": state.scene_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "type": "scene summary"
                }
            )

            del scene_bundle, scene_cluster, scene_text
            gc.collect()

            return {
                "scene_id": state.scene_id + 1,
                "current_chapter_id": state.current_chapter_id,
                "current_act_id": state.current_act_id,
                "story_word_count": state.story_word_count,
                "current_chapter_word_count": state.current_chapter_word_count
            }
        else:
            state.next_action = "FATAL_ERROR"
            raise RuntimeError(f"Unknown scene planner status: {status}")

    async def ingest_chapter(self, state: StoryState):
        """Ingest completed chapter"""
        world_details = await self.memory.get_long_term_recent_worlds(state.current_chapter_id)
        char_details = await self.memory.get_long_term_recent_characters(state.current_chapter_id)
        self.scene_chunk_callback({"type": "status", "message": "saving chapter"})

        chapter_bundle = await Ingestor.ingest_chapter(
            chapter_id=state.current_chapter_id,
            current_chap_summary=self.current_chap_summary,
            char_details=char_details,
            world_details=world_details
        )

        result = "failure"
        if chapter_bundle:
            await self.memory.add_post_chapter_bundle(
                parts=chapter_bundle,
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "act_id": state.current_act_id
                }
            )
            await self.memory.increment_chapter(word_count_delta=0, scene_id=1)
            state.current_chapter_id += 1
            state.scene_id = 1
            print("Chapter Complete!")
            result = "success"

        if result == "success":
            try:
                self.scene_chunk_callback({"chapter_complete": True})
                await self.memory.update_story_progress({"chapter_word_count": 0})
            except Exception as e:
                print(f"ERROR: scene_chunk_callback raised: {e}")
                import traceback
                traceback.print_exc()

        self.current_chap_summary = ""
        gc.collect()
        return result

    async def chapter_director_node(self, state: StoryState) -> Dict:
        """Chapter Director refactored for act-based interactive generation."""
        print(f"\nChapter Director: Chapter {state.current_chapter_id}, Act {state.current_act_id}")

        # Get current act plan
        try:
            act_plan_json = await self.memory.get_long_term_document(
                metadata={
                    'type': 'act_plan',
                    'act_id': state.current_act_id,
                    'story_title': state.story_title
                }
            )
            if not act_plan_json:
                print(f"No act plan found for act {state.current_act_id}")
                act_context = "No act plan available - proceed with story premise only"
            else:
                act_plan = json.loads(act_plan_json)
                self.act_title = act_plan.get('act_title', 'Unknown')
                act_context = f"""Current Act: {self.act_title} (Act {state.current_act_id})
Act Purpose: {act_plan.get('act_purpose', 'N/A')}
Key Themes: {', '.join(act_plan.get('key_themes', []))}
Potential Branches: {', '.join(act_plan.get('potential_branches', []))}
Emotional Trajectory: {act_plan.get('emotional_trajectory', 'N/A')}
Key Moments: {', '.join(act_plan.get('key_moments', []))}
Target Word Count for Act: {act_plan.get('target_word_count', 'N/A')}"""
        except Exception as e:
            print(f"Error loading act plan: {e}")
            act_context = "Error loading act plan - proceed with caution"

        # Get recent context (last 2-3 chapters)
        if state.current_chapter_id > 1:
            self.current_chap_summary = await self.memory.search_single_episodic_story(
                act_number=state.current_act_id,
                chapter_number=state.current_chapter_id - 1,
                summary_type="chapter summary"
            )
            director_context = await self.memory.get_director_context(
                current_act_number=state.current_act_id,
                current_chapter_number=state.current_chapter_id,
                query=self.current_chap_summary if self.current_chap_summary else "",
                k=5
            )
        else:
            director_context = "Start of Story"

        system_prompt = f"""You are the Chapter Director for an interactive story.
Your job is to plan the overall direction and structure of THIS chapter ONLY. You design a compact blueprint that the Scene Director will follow scene-by-scene.
Act Context: {act_context}
Use the story so far and the user's most recent choice to determine how this chapter should develop emotionally, thematically, and narratively.
Your output must be in JSON format and include:

* instructions - A string containing:

  * chapter_title - A short, descriptive title

  * narrative_goal - What this chapter must accomplish (e.g., 'Sam discovers betrayal')

  * emotional_arc - The emotional progression (e.g., 'tension → shock → resolve')

  * key_conflicts - The central struggles or decisions faced

  * closure_condition - The condition under which the chapter should end

  * tone_guidelines - Notes on tone, pacing, and atmosphere

  * style_guide - prose_style, pov, tense, narrative_voice"

  * expected_scenes - Estimated number of scenes for pacing reference

  * word_count - Target word count for the chapter

  * act_alignment - How this chapter advances the current act's themes/purpose

* action - 'continue' to proceed with scenes or 'END' if the story is complete

If the story should end now then 'action' should be set to 'END' and the 'instructions' should be an empty string ''. If 'instructions' for this chapter are set then 'action' MUST be 'continue'
CRITICAL for Interactive Stories:

* Respect user choices that have been made

* Set up meaningful decision points

* Align with act themes: {act_context}

* Don't force predetermined outcomes

* Leave room for emergent storytelling

Keep this concise but detailed enough that a Scene Director can plan and execute each scene from it.
{chapter_director_parser.get_format_instructions()}"""

        context = f"""Recent Story Context:- context: {director_context}
last chapter: {self.current_chap_summary}
Current Chapter Number: {state.current_chapter_id} Current Act: {state.current_act_id} Story Word Count so far: {state.story_word_count}"""
        human_prompt = context
        max_retries = 3
        scenario, action = None, None

        # Check if chapter plan already exists
        existing_plan = await self.memory.get_long_term_document(
            metadata={
                "type": "chapter_plan",
                "chapter_id": state.current_chapter_id,
                "act_id": state.current_act_id,
                "story_title": state.story_title
            }
        )
        if existing_plan:
            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                existing_plan, ChapterDirectorOutput, chapter_director_parser
            )
            if success:
                scenario = result.get("instructions")
                action = result.get("action")
                print(f"Using existing chapter plan for chapter {state.current_chapter_id}")
            else:
                print(f"Existing chapter plan validation failed: {exc}")
                scenario = None

        if not existing_plan:
            # Generate new plan
            for attempt in range(1, max_retries + 1):
                resp, director_tokens = await director_client(
                    system_prompt=system_prompt,
                    human_prompt=human_prompt,
                    llm_temp=self.llm_temp,
                    model=self.model
                )
                if resp is None:
                    print(f"Retrying chapter_director_node... (attempt {attempt+1})")
                    continue

                raw_text = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(raw_text)
                self.director_token_usage["prompt_tokens"] += director_tokens["prompt_tokens"]
                self.director_token_usage["completion_tokens"] += director_tokens["completion_tokens"]
                self.director_token_usage["total_tokens"] += director_tokens["total_tokens"]
                del raw_text, resp, director_tokens
                gc.collect()

                if isinstance(clean_resp, dict):
                    clean_resp = json.dumps(clean_resp)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    clean_resp, ChapterDirectorOutput, chapter_director_parser
                )
                if success:
                    scenario = result.get("instructions")
                    action = result.get("action")
                    await self.memory.add_long_term_document(
                        text=clean_resp,
                        metadata={
                            "chapter_id": state.current_chapter_id,
                            "act_id": state.current_act_id,
                            "type": "chapter_plan",
                            "story_title": state.story_title
                        }
                    )
                    await self.memory.update_story_progress({"director_token_usage": self.director_token_usage})
                    print(f"Generated new chapter plan for chapter {state.current_chapter_id}")
                    break

                print(f"[Attempt {attempt}] First-pass validation failed: {exc}")
                try:
                    fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                    fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                    del clean_resp, fixed_resp
                    gc.collect()

                    if isinstance(fixed_clean, dict):
                        fixed_clean = json.dumps(fixed_clean)

                    success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                        fixed_clean, ChapterDirectorOutput, chapter_director_parser
                    )
                    del fixed_clean
                    if success:
                        scenario = result.get("instructions")
                        action = result.get("action")
                        await self.memory.add_long_term_document(
                            text=json.dumps(result.model_dump()),
                            metadata={
                                "chapter_id": state.current_chapter_id,
                                "act_id": state.current_act_id,
                                "type": "chapter_plan",
                                "story_title": state.story_title
                            }
                        )
                        await self.memory.update_story_progress({"director_token_usage": self.director_token_usage})
                        break
                    print(f"[Attempt {attempt}] json_fixer validation failed: {exc}")
                except Exception as inner_e:
                    print(f"[Attempt {attempt}] json_fixer raised: {inner_e}")

                if attempt < max_retries:
                    print(f"Retrying chapter_director_node... (attempt {attempt+1})")
                    continue
                else:
                    print("All retries exhausted; falling back to raw response and continue.")
                    scenario = clean_resp if 'clean_resp' in locals() else "Error generating instructions"
                    action = "continue"

        if 'clean_resp' in locals():
            del clean_resp
        del system_prompt, human_prompt, context
        gc.collect()

        messages = state.messages or []
        messages.append(AIMessage(content=scenario))
        print("chapter_director_node, next_action: ", action)
        self.scene_chunk_callback({"type": "act_title", "message": self.act_title})

        return {
            "messages": messages,
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "current_act_id": state.current_act_id,
            "next_action": action,
        }

    async def scene_director_node(self, state: StoryState) -> Dict:
        """Scene Director refactored for act-based interactive generation."""
        print(f"\nScene Director: Scene {state.scene_id}, Chapter {state.current_chapter_id}, Act {state.current_act_id}")

        # Ensure user monthly word count compatibility
        monthly_wc_data = await self.memory.get_monthly_word_count()
        if monthly_wc_data.get('tier') == 1:
            if monthly_wc_data.get('monthly_word_count') >= tier_1_monthly_words_limit:
                status = "User monthly word count limit reached for tier 'free'"
                state.next_action = "ERROR"
                raise RuntimeError(status)
        elif monthly_wc_data.get('tier') == 2:
            if monthly_wc_data.get('monthly_word_count') >= tier_2_monthly_words_limit:
                status = "User monthly word count limit reached for tier 'scribe'"
                state.next_action = "ERROR"
                raise RuntimeError(status)
        del monthly_wc_data

        # Get act plan for thematic context
        try:
            act_plan_json = await self.memory.get_long_term_document(
                metadata={
                    'type': 'act_plan',
                    'act_id': state.current_act_id,
                    'story_title': state.story_title
                }
            )
            if act_plan_json:
                act_plan = json.loads(act_plan_json)
                act_context = f"""Act Context:

* Act Title: {act_plan.get('act_title', 'Unknown')}

* Act Purpose: {act_plan.get('act_purpose', 'N/A')}

* Key Themes: {', '.join(act_plan.get('key_themes', []))}

* Emotional Trajectory: {act_plan.get('emotional_trajectory', 'N/A')}"""
            else:
                act_context = "Act context unavailable"
        except Exception as e:
            print(f"Error loading act plan: {e}")
            act_context = "Act context unavailable"

        # Get current chapter summary and director context
        self.current_chap_summary = await self.memory.search_single_episodic_story(
            act_number=state.current_act_id,
            chapter_number=state.current_chapter_id,
            summary_type="scene summary"
        )
        director_context = await self.memory.get_director_context(
            current_act_number=state.current_act_id,
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=3
        )

        # Get chapter plan
        chapter_plan_text = await self.memory.get_long_term_document(
            metadata={
                "type": "chapter_plan",
                "chapter_id": state.current_chapter_id,
                "act_id": state.current_act_id,
                "story_title": state.story_title
            }
        )

        # Extract chapter target word count and expected scenes from chapter plan
        chapter_word_target = 1200  # Default
        expected_scenes = 3  # Default
        if chapter_plan_text:
            # Try to extract word_count from chapter plan
            wc_match = re.search(r'word[_ ]count[:\s]+(\d+)', chapter_plan_text, re.IGNORECASE)
            if wc_match:
                chapter_word_target = int(wc_match.group(1))

            # Try to extract expected_scenes
            scenes_match = re.search(r'expected[_ ]scenes[:\s]+(\d+)', chapter_plan_text, re.IGNORECASE)
            if scenes_match:
                expected_scenes = int(scenes_match.group(1))

        # Calculate target word count for THIS scene
        remaining_chapter_words = chapter_word_target - state.current_chapter_word_count
        scenes_remaining = expected_scenes - (state.scene_id - 1)
        if scenes_remaining <= 0 or remaining_chapter_words < 100:
            return {"next_action": "END"}

        target_scene_word_count = max(200, int(remaining_chapter_words / scenes_remaining))
        target_scene_word_count = min(target_scene_word_count, 800)  # Cap

        print(f"Word Count Calculation:")
        print(f" Chapter target: {chapter_word_target} words")
        print(f" Chapter so far: {state.current_chapter_word_count} words")
        print(f" Remaining: {remaining_chapter_words} words")
        print(f" Expected scenes: {expected_scenes}")
        print(f" Scenes remaining: {scenes_remaining}")
        print(f" Target for THIS scene: {target_scene_word_count} words")

        # Get story seed
        story_seed_raw = await self.memory.get_long_term_document(
            metadata={'type': 'story_seed', 'story_title': state.story_title}
        )
        if story_seed_raw:
            try:
                story_seed = json.loads(story_seed_raw)
            except json.JSONDecodeError as e:
                print(f"Error parsing story_seed: {e}")
                story_seed = {}
        else:
            story_seed = {}

        system_prompt = f"""You are the Scene Director for an interactive story. You work under a Chapter Blueprint that defines the chapter's purpose, emotional arc, and closure condition.

{act_context}
Your job is to create detailed, prescriptive instructions for the Scene Writer to follow for THIS scene only. You must include all relevant details — characters, actions, setting, beats, and decision points — because the Scene Writer has no memory of previous scenes.
Base your plan on:

* The story so far

* The user's most recent choice

* The current Chapter Blueprint

* The current Act's themes and emotional trajectory

╔═══════════════════════════════════════════════════════════════════════╗
║ WORD COUNT MANAGEMENT (CRITICAL)                                      ║
╠═══════════════════════════════════════════════════════════════════════╣
║ Chapter Target: {chapter_word_target:>5} words                                 ║
║ Chapter So Far: {state.current_chapter_word_count:>5} words                            ║
║ Remaining: {remaining_chapter_words:>5} words                                 ║
║ Scenes Remaining: {scenes_remaining:>5}                                        ║
║ THIS Scene: {target_scene_word_count:>5} words (STRICT TARGET)                     ║
╚═══════════════════════════════════════════════════════════════════════╝

WORD COUNT DISCIPLINE:

* The Scene Writer MUST stay within {target_scene_word_count} words (±50 words tolerance)

* If remaining chapter words < 400, this should be the FINAL scene

* Balance narrative beats with word count constraints

* Prioritize essential story beats over lengthy descriptions

* If a decision point would exceed word count, save it for next scene

Story Style Guidelines (to be passed to scene writer):

* POV: {story_seed.get('style_guide', {}).get('pov', 'Third-person')}

* Prose Style: {story_seed.get('style_guide', {}).get('prose_style', '')}

* Tense: {story_seed.get('style_guide', {}).get('tense', '')}

* Narrative Voice: {story_seed.get('style_guide', {}).get('narrative_voice', '')}

* Tone: {story_seed.get('tone', 'Balanced')}

* Genre: {story_seed.get('genre', 'Fiction')}

Your output must be in JSON format with:

* instructions: A string containing:

  * chapter_id: {state.current_chapter_id}

  * scene_id: {state.scene_id}

  * recap - A short recap of events so far relevant to this scene (max 100 words)

  * narrative style guide - Relevant guide prose for this scene

  * characters - Detailed character list with traits, motivations, and current emotions

  * detailed_scene_blueprint - A numbered, beat-by-beat breakdown of the scene's structure (actions, dialogue, setting, etc.)

    * Keep blueprint CONCISE - this scene has only {target_scene_word_count} words to work with

    * Each beat should be 80-130 words of prose

  * user_decision_points - 1 or more explicit decision moments (dialogue or actions) that the Scene Writer must present as choices

    * ONLY include if word count allows (current words + 150 < {target_scene_word_count})

  * screenplay_notes - Strict creative constraints (e.g., 'include one metaphor about light and shadow')

* target_scene_word_count: {target_scene_word_count} (THIS IS MANDATORY - Scene Writer will enforce this strictly)

* action: 'generate_and_ingest' to continue or 'END' if the chapter closure_condition is fulfilled.

CRITICAL for Interactive Stories:

* Decision points must offer MEANINGFUL choices

* Each option should lead to different narrative consequences

* Align decision points with act themes

* Don't force predetermined outcomes

* Respect user agency

* RESPECT WORD COUNT LIMITS - Better a complete short scene than an incomplete long one

Do not go beyond the chapter's emotional arc or closure condition. Only end the chapter if the closure condition is clearly met OR if remaining chapter words < 200.
{scene_director_parser.get_format_instructions()}"""

        context = f"""Chapter Number: {state.current_chapter_id} Scene Number: {state.scene_id} Act Number: {state.current_act_id}
Chapter Blueprint: {chapter_plan_text}
Relevant Chapter Context: {director_context}
Current Chapter So Far Summary: {self.current_chap_summary}
Current Chapter word count: {state.current_chapter_word_count} / {chapter_word_target} words"""
        human_prompt = context
        max_retries = 3
        scenario, target_scene_word_count_result, action = None, None, None

        for attempt in range(1, max_retries + 1):
            resp, director_tokens = await director_client(
                system_prompt=system_prompt,
                human_prompt=human_prompt,
                llm_temp=self.llm_temp,
                model=self.model
            )
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            self.director_token_usage["prompt_tokens"] += director_tokens["prompt_tokens"]
            self.director_token_usage["completion_tokens"] += director_tokens["completion_tokens"]
            self.director_token_usage["total_tokens"] += director_tokens["total_tokens"]
            del raw_text, resp, director_tokens
            gc.collect()

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                clean_resp, SceneDirectorOutput, scene_director_parser
            )
            if success:
                scenario = result.get("instructions")
                target_scene_word_count_result = result.get("target_scene_word_count")
                action = result.get("action")

                # Validate the returned word count
                if target_scene_word_count_result is None or target_scene_word_count_result < 100:
                    print(f"Invalid target_scene_word_count ({target_scene_word_count_result}), using calculated: {target_scene_word_count}")
                    target_scene_word_count_result = target_scene_word_count
                elif abs(target_scene_word_count_result - target_scene_word_count) > 300:
                    print(f"Scene Director suggested {target_scene_word_count_result} words, but calculated {target_scene_word_count}. Using calculated value.")
                    target_scene_word_count_result = target_scene_word_count
                break

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)
            try:
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                    fixed_clean, SceneDirectorOutput, scene_director_parser
                )
                del fixed_clean, fixed_resp
                gc.collect()

                if success:
                    scenario = result.get("instructions")
                    target_scene_word_count_result = result.get("target_scene_word_count")
                    action = result.get("action")

                    if target_scene_word_count_result is None or target_scene_word_count_result < 100:
                        print(f"Invalid target_scene_word_count ({target_scene_word_count_result}), using calculated: {target_scene_word_count}")
                        target_scene_word_count_result = target_scene_word_count
                    elif abs(target_scene_word_count_result - target_scene_word_count) > 300:
                        print(f"Scene Director suggested {target_scene_word_count_result} words, but calculated {target_scene_word_count}. Using calculated value.")
                        target_scene_word_count_result = target_scene_word_count
                    break
                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised:", inner_e)
                del clean_resp
                gc.collect()

            if attempt < max_retries:
                print(f"Retrying scene_director_node... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted; falling back to defaults.")
                scenario = clean_resp if 'clean_resp' in locals() else "Error generating instructions"
                target_scene_word_count_result = target_scene_word_count
                action = "END"

        if 'clean_resp' in locals():
            del clean_resp
        del system_prompt, human_prompt
        gc.collect()

        messages = state.messages or []
        messages.append(AIMessage(content=scenario))
        self.target_scene_word_count = target_scene_word_count_result

        print(f"Scene Director Output:")
        print(f" Target scene word count: {self.target_scene_word_count}")
        print(f" Action: {action}")

        return {
            "messages": messages,
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "current_act_id": state.current_act_id,
            "story_word_count": state.story_word_count,
            "current_chapter_word_count": state.current_chapter_word_count,
            "next_action": "generate_and_ingest" if action == "generate_and_ingest" else "END",
        }

    async def run(self, scene_chunk_callback, stop_event: asyncio.Event | None = None):
        """Run the interactive director agent with act-based story generation."""
        print("Running act-based interactive director agent...")
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()

        try:
            # Load story progress
            story_progress = await self.memory.get_story_progress()
            chapter_word_count = story_progress.get("chapter_word_count", 0)
            story_word_count = story_progress.get("story_word_count", 0)
            if not story_progress:
                print("No story progress found")
                return "No story progress found"

            self.llm_temp = story_progress.get("tone_temp", 0.7)
            self.model = story_progress.get("model", "gpt-4")

            # Helper function to parse token usage
            def parse_token_usage(value):
                default = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                if not value:
                    return default
                try:
                    if isinstance(value, str):
                        parsed = json.loads(value)
                        if not isinstance(parsed, dict) or not all(key in parsed for key in ["prompt_tokens", "completion_tokens", "total_tokens"]):
                            return default
                        return parsed
                    elif isinstance(value, dict):
                        if not all(key in value for key in ["prompt_tokens", "completion_tokens", "total_tokens"]):
                            return default
                        return value
                    return default
                except json.JSONDecodeError:
                    return default

            # Initialize token usage fields
            self.director_token_usage = parse_token_usage(story_progress.get("director_token_usage"))
            self.writer_token_usage = parse_token_usage(story_progress.get("writer_token_usage"))
            self.target_length = story_progress.get("target_length", 50000)

            # Check if story already complete
            complete = story_progress.get("complete", False)
            if complete:
                print("Story already complete")
                return "Story already complete."

            # Initialize state with act tracking
            initialized_state = StoryState(
                current_chapter_id=story_progress.get("latest_chapter_id", 1),
                scene_id=story_progress.get("continue_scene_id", 1),
                story_title=story_progress.get("story_title", "Untitled Story"),
                story_word_count=story_word_count,
                current_chapter_word_count=chapter_word_count,
                current_act_id=story_progress.get("current_act_id", 1),
                messages=[],
                next_action=""
            )

            print(f"Interactive Story State: Chapter {initialized_state.current_chapter_id}, Scene {initialized_state.scene_id}, Act {initialized_state.current_act_id}")
            print(f"Current Chapter Word Count: {initialized_state.current_chapter_word_count}")

            # Run LangGraph inside a cancellable task
            task = asyncio.create_task(
                self.compiled.ainvoke(
                    initialized_state,
                    {"recursion_limit": 50, "stop_event": self.stop_event}
                )
            )

            while not task.done():
                if self.stop_event.is_set():
                    print("Stop event received — cancelling DirectorGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("DirectorGraph task cancelled cleanly")
                        return None
                await asyncio.sleep(0.2)

            result = await task

            # Check for fatal errors
            if result.get("next_action") == "FATAL_ERROR":
                print("Fatal error detected in graph execution")
                self.stop_event.set()
                return None

            print("Interactive Director Node Finished:", result)
            return result

        except asyncio.CancelledError:
            print("DirectorGraph CancelledError caught")
            self.stop_event.set()
            return None
        except Exception as e:
            print(f"Error in DirectorGraph.run: {e}")
            import traceback
            traceback.print_exc()
            self.stop_event.set()
            return None
        finally:
            print("DirectorGraph stopped gracefully")
            gc.collect()