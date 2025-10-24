# src/story_engines/classic_narrative/agents/shared_scene_planner.py

import json
import re
import gc
import asyncio
import traceback
from typing import List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import story_client, llm_for_scene_planner_client
from langgraph.graph import StateGraph, END
from src.utilities.story_helpers import StoryHelpers
from langchain_core.output_parsers import PydanticOutputParser

# Global shared instance
CLASSIC_SCENE_PLANNER_SERVICE = None


class SceneMemory(BaseModel):
    DirectorInstructions: Any = Field(description="Director's instructions for this scene.")
    scene_so_far: str = Field(default="", description="Accumulated text of the scene.")
    scene_cluster: List = Field(default=[], description="Scene text, questions, and user choices as dicts.")
    word_count: int = Field(..., description="Word count of the scene.")
    story_id: Optional[str] = Field(default="", description="Story ID for this scene.")


class SceneState(BaseModel):
    scene_memory: Optional[SceneMemory] = None
    next_node: Optional[str] = None
    user_context_id: Optional[str] = Field(default=None, description="User ID for this scene")
    iteration_count: int = 0
    scene_chunk_callback: Optional[Callable] = Field(default=None, description="Callback for frontend updates")
    batch_counter: int = 0  # Track paragraphs in batch
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fatal: bool = False

    class Config:
        extra = "allow"


class SceneWriterOutput(BaseModel):
    scene: str = Field(description="One paragraph of narrative text.")


class ScenePlannerOutput(BaseModel):
    action: str = Field(description="'Complete' if scene is done, 'Not Complete' otherwise.")


@dataclass
class UserSceneContext:
    user_id: str
    story_id: str
    director_instructions: Any
    scene_state: SceneState
    scene_memory: SceneMemory
    scene_chunk_callback: Callable

    @classmethod
    def create_for_user(cls, user_id: str, story_id: str, director_instructions: str, scene_chunk_callback):
        scene_memory = SceneMemory(
            DirectorInstructions=director_instructions.strip(),
            scene_so_far="",
            story_id=story_id,
            word_count=0
        )

        scene_state = SceneState(
            scene_memory=scene_memory,
            next_node=None,
            user_context_id=user_id,
            scene_chunk_callback=scene_chunk_callback
        )

        return cls(
            user_id=user_id,
            story_id=story_id,
            director_instructions=director_instructions,
            scene_state=scene_state,
            scene_memory=scene_memory,
            scene_chunk_callback=scene_chunk_callback
        )


class SharedScenePlannerService:
    def __init__(self):
        self.scene_writer_parser = PydanticOutputParser(pydantic_object=SceneWriterOutput)
        self.scene_planner_parser = PydanticOutputParser(pydantic_object=ScenePlannerOutput)

        self.graph = StateGraph(SceneState)
        self.graph.add_node("Initializer", self._initializer)
        self.graph.add_node("SceneWriter", self._scene_writer_agent)
        self.graph.add_node("ScenePlanner", self._scene_planner_agent)

        self.graph.set_entry_point("Initializer")
        self.graph.add_edge("Initializer", "SceneWriter")

        def planner_decider(state: SceneState):
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            if state.batch_counter >= self.batch_size:  # Check after every 3 paragraphs
                return "ScenePlanner"
            return "SceneWriter"

        self.graph.add_conditional_edges(
            "SceneWriter",
            planner_decider,
            {
                "SceneWriter": "SceneWriter",
                "ScenePlanner": "ScenePlanner",
                "FATAL_ERROR": END,
            },
        )

        def final_decider(state: SceneState):
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            return getattr(state, "next_node", "Complete")

        self.graph.add_conditional_edges(
            "ScenePlanner",
            final_decider,
            {
                "Not Complete": "SceneWriter",
                "Complete": END,
                "FATAL_ERROR": END,
            },
        )

        self.compiled = self.graph.compile()

    def _handle_fatal_error(self, state: SceneState, error: Exception, stage: str) -> SceneState:
        state.fatal = True
        state.next_node = "FATAL_ERROR"
        state.error_type = stage
        state.error_message = f"{type(error).__name__}: {str(error)}"
        print(f"❌ [FATAL] {stage} failed — {state.error_message}")
        traceback.print_exc()
        return state

    async def run_scene(self, user_context: UserSceneContext, token_usage: dict, target_length: int, stop_event: asyncio.Event | None = None, llm_temp: float = 0.7, model: str = "None") -> Tuple[str, list, str]:
        print(f"🎭 Starting run_scene for {user_context.user_id}/{user_context.story_id}")
        self.llm_temp = llm_temp
        self.model = model
        self.token_usage = token_usage
        if target_length < 25000:
            self.batch_size = 1
        elif target_length < 75000:
            self.batch_size = 2
        else:
            self.batch_size = 3
        del llm_temp, model, token_usage
        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ Failed to set user_context_id: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", token_usage

        try:
            task = asyncio.create_task(
                self.compiled.ainvoke(user_context.scene_state, {"recursion_limit": 25, "stop_event": stop_event})
            )

            while not task.done():
                if stop_event and stop_event.is_set():
                    print("🛑 Stop event received — cancelling SceneGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ SceneGraph task cancelled cleanly")
                    return "", [], "CANCELLED", token_usage
                await asyncio.sleep(0.2)

            result = await task

            next_node = getattr(result, "next_node", None) if not isinstance(result, dict) else result.get("next_node")
            error_message = getattr(result, "error_message", None) if not isinstance(result, dict) else result.get("error_message")
            scene_memory = getattr(result, "scene_memory", None) if not isinstance(result, dict) else result.get("scene_memory")

            if getattr(result, "fatal", False) or next_node == "FATAL_ERROR":
                fatal_msg = error_message or "Unknown fatal error"
                print(f"❌ Scene aborted due to fatal error: {fatal_msg}")
                return "", [], f"FATAL: {fatal_msg}", token_usage

            if scene_memory is None:
                print("⚠️ Warning: LangGraph returned with no scene_memory.")
                return "", [], "EXCEPTION: No scene_memory returned", token_usage

            scene_so_far = getattr(scene_memory, "scene_so_far", "") if not isinstance(scene_memory, dict) else scene_memory.get("scene_so_far", "")
            scene_cluster = getattr(scene_memory, "scene_cluster", []) if not isinstance(scene_memory, dict) else scene_memory.get("scene_cluster", [])

            print("✅ Scene completed normally")
            return scene_so_far, scene_cluster, "SUCCESS", self.token_usage

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            return "", [], "CANCELLED", token_usage
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", token_usage
        finally:
            print("🎭 SceneGraph stopped gracefully")
            gc.collect()

    def _initializer(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: Initializer node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 Initializer skipping because fatal flag is set.")
            return state
        state.next_node = "SceneWriter"
        state.batch_counter = 0
        return state

    async def _scene_planner_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: ScenePlanner node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 ScenePlanner skipping because fatal flag is set.")
            return state

        scene_memory: SceneMemory = state.scene_memory
        system_prompt = """
You are the Scene Planner Agent. Evaluate the scene text against the Director's Scene Plan to check if all required events, 
emotional beats, and target word count are met. Return JSON with 'action': 'Complete' if done, or 'Not Complete' if more is needed. 
Output only valid JSON matching the schema.
"""

        human_prompt = f"""
Director's Instructions: {scene_memory.DirectorInstructions}
Scene so far: {scene_memory.scene_so_far}
Current scene word count: {scene_memory.word_count}
{self.scene_planner_parser.get_format_instructions()}
Output ONLY valid JSON, no extra text or markdown.
"""

        try:
            timeout = 60.0 if self.model in ["gpt-4", "large_model"] else 30.0
            llm_response = await asyncio.wait_for(
                llm_for_scene_planner_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=timeout
            )
            clean_resp = StoryHelpers._extract_content(llm_response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                parsed = self.scene_planner_parser.parse(clean_resp)
            except Exception as e:
                print(f"❌ ScenePlanner parsing failed: {e}")
                fixed_json = await StoryHelpers._json_fixer(clean_resp)
                parsed = self.scene_planner_parser.parse(fixed_json)

            print(f"🔍 DEBUG: ScenePlanner action: {parsed.action}")

            if parsed.action == "Not Complete":
                state.iteration_count = getattr(state, 'iteration_count', 0) + 1
                state.batch_counter = 0  # Reset batch counter
                if state.iteration_count > 10:
                    print("⚠️ Forcing Complete to avoid infinite loop")
                    parsed.action = "Complete"

            state.next_node = parsed.action

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "ScenePlanner Timeout")
        except Exception as e:
            return self._handle_fatal_error(state, e, "ScenePlanner")
        finally:
            state.scene_memory = scene_memory
            del llm_response, clean_resp
            gc.collect()
        return state

    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: SceneWriter node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 SceneWriter skipping because fatal flag is set.")
            return state

        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id

        system_prompt = """
You are the Scene Writer Agent. Write one concise paragraph (70-120 words) of story text that executes the Scene Plan exactly, 
focusing on clarity, emotional impact, and tight pacing. Use only the plan's characters, location, and emotional beats. 
Do not invent new elements or resolve chapter closure early. Output only valid JSON matching the schema.
"""

        human_prompt = f"""
Director's Instructions:
{scene_memory.DirectorInstructions}

Scene so far (DO NOT rewrite, only continue): {scene_memory.scene_so_far}
Current scene word count: {scene_memory.word_count}
{self.scene_writer_parser.get_format_instructions()}
Output ONLY valid JSON, no extra text or markdown.
"""

        try:
            timeout = 60.0 if self.model in ["gpt-4", "large_model"] else 30.0
            llm_response, tokens = await asyncio.wait_for(
                story_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp, model=self.model),
                timeout=timeout
            )
            print(f"🔍 Token Usage: Prompt={tokens['prompt_tokens']}, Completion={tokens['completion_tokens']}, Total={tokens['total_tokens']}")
            self.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            self.token_usage["completion_tokens"] += tokens["completion_tokens"]
            self.token_usage["total_tokens"] += tokens["total_tokens"]
            clean_resp = StoryHelpers._extract_content(llm_response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                parsed = self.scene_writer_parser.parse(clean_resp)
                scene_text = parsed.scene
            except Exception as e:
                print(f"❌ SceneWriter parsing failed: {e}")
                fixed_json = await StoryHelpers._json_fixer(clean_resp)
                try:
                    parsed = self.scene_writer_parser.parse(fixed_json)
                    scene_text = parsed.scene
                except Exception as repair_e:
                    print(f"❌ JSON fixer failed: {repair_e}")
                    match = re.search(r'"scene"\s*:\s*"([^"]*)"', clean_resp)
                    scene_text = match.group(1) if match else ""

            new_word_count = StoryHelpers._count_words_split(scene_text)
            scene_memory.word_count += new_word_count
            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text,
                "word_count": new_word_count
            })
            scene_memory.scene_so_far += (" " + scene_text + "\n\n") if scene_text else ""

            state.scene_memory = scene_memory
            state.batch_counter += 1

            if user_context_id and state.scene_chunk_callback:
                try:
                    print(f"🔍 DEBUG: Sending scene text to frontend for {user_context_id}/{scene_memory.story_id}: {scene_text[:50]}...")
                    state.scene_chunk_callback({
                        "type": "text",
                        "scene_text": scene_text
                    })
                    state.scene_chunk_callback({
                        "type": "status",
                        "word_count": new_word_count
                    })
                except Exception as e:
                    print(f"❌ ERROR: Failed to send scene text to frontend: {e}")
                    traceback.print_exc()

            del llm_response, clean_resp, tokens
            gc.collect()
            return state

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "SceneWriter Timeout")
        except Exception as e:
            return self._handle_fatal_error(state, e, "SceneWriter")