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
    DirectorInstructions: Any = Field(description="The director's detailed instructions for this scene.")
    scene_so_far: str = Field(default="", description="Accumulated text of the scene written so far.")
    scene_cluster: List = Field(default=[], description="Combination of scene text, questions and user choices stored as dicts inside the list.")
    word_count: int = Field(..., description="Word count of the scene so far.")
    story_id: Optional[str] = Field(default="", description="The story ID associated with this scene.")


class SceneState(BaseModel):
    # messages: List[BaseMessage] = []
    scene_memory: Optional[SceneMemory] = None
    next_node: Optional[str] = None
    user_context_id: Optional[str] = Field(default=None, description="User ID for this scene")
    iteration_count: int = 0  # Added for recursion limit
    scene_chunk_callback: Optional[Callable] = Field(default=None, description="Callback to send scene chunks to frontend")

    # Error propagation fields
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fatal: bool = False  # explicit boolean flag for fatal errors

    class Config:
        extra = "allow"


class SceneWriterOutput(BaseModel):
    scene: str = Field(description="One paragraph of continuing narrative text. Do Not write the question here, only in the 'question' field.")


class ScenePlannerOutput(BaseModel):
    action: str = Field(description="Either 'Complete' if the scene has all scene blueprint events, or 'Not Complete' otherwise.")


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
            # messages=[AIMessage(content=director_instructions)],
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
        # nodes
        self.graph.add_node("Initializer", self._initializer)
        self.graph.add_node("SceneWriter", self._scene_writer_agent)
        self.graph.add_node("ScenePlanner", self._scene_planner_agent)

        self.graph.set_entry_point("Initializer")
        # flow initializer -> writer -> planner -> (writer or END)
        self.graph.add_edge("Initializer", "SceneWriter")
        self.graph.add_edge("SceneWriter", "ScenePlanner")

        # conditional edge returns must include possibility of fatal error
        def planner_decider(state: SceneState):
            # If a fatal error has been flagged by a node, propagate it.
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            # otherwise use the next_node from planner which should be "Not Complete" or "Complete"
            return getattr(state, "next_node", "Complete")

        self.graph.add_conditional_edges(
            "ScenePlanner",
            planner_decider,
            {
                "Not Complete": "SceneWriter",
                "Complete": END,
                "FATAL_ERROR": END,
            },
        )

        self.compiled = self.graph.compile()

    # -------------------------
    # Helper: fatal error handling
    # -------------------------
    def _handle_fatal_error(self, state: SceneState, error: Exception, stage: str) -> SceneState:
        """
        Mark state as fatal, attach error metadata, and ensure graph sees FATAL_ERROR.
        """
        state.fatal = True
        state.next_node = "FATAL_ERROR"
        state.error_type = stage
        # make message concise but informative
        state.error_message = f"{type(error).__name__}: {str(error)}"
        print(f"❌ [FATAL] {stage} failed — {state.error_message}")
        traceback.print_exc()
        return state

    # -------------------------
    # Public: run_scene
    # -------------------------
    async def run_scene(self, user_context: UserSceneContext, token_usage: dict, stop_event: asyncio.Event | None = None, llm_temp: float = 0.7, model: str = "None") -> Tuple[str, list, str]:
        """
        Process scene for a specific user using their context.
        Returns (scene_text_so_far, scene_cluster, status).
        status is one of:
          - "SUCCESS"
          - "CANCELLED"
          - "FATAL: <error message>"
          - "EXCEPTION: <error message>"
        """
        print(f"🎭 Starting run_scene for {user_context.user_id}/{user_context.story_id}")
        self.llm_temp = llm_temp
        self.model = model
        self.token_usage = token_usage
        del llm_temp, model, token_usage
        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ Failed to set user_context_id: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", {}

        try:
            # Run LangGraph in a cancellable task
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
                    return "", [], "CANCELLED", {}
                await asyncio.sleep(0.2)

            result = await task  # result is expected to be a mapping-like state snapshot

            # result may be SceneState-like or dict; attempt safe extraction
            next_node = None
            error_message = None
            scene_memory = None
            try:
                if isinstance(result, dict):
                    next_node = result.get("next_node")
                    error_message = result.get("error_message") or result.get("errorMessage")
                    scene_memory = result.get("scene_memory")
                else:
                    # try attribute-style access
                    next_node = getattr(result, "next_node", None)
                    error_message = getattr(result, "error_message", None)
                    scene_memory = getattr(result, "scene_memory", None)
            except Exception:
                # fallback - attempt to find scene_memory via key access
                try:
                    scene_memory = result["scene_memory"]
                except Exception:
                    scene_memory = None

            # If a fatal error was flagged by a node, return it as FATAL
            if getattr(result, "fatal", False) or next_node == "FATAL_ERROR" or getattr(result, "next_node", None) == "FATAL_ERROR":
                fatal_msg = error_message or (getattr(result, "error_message", None) if hasattr(result, "error_message") else "Unknown fatal error")
                print(f"❌ Scene aborted due to fatal error: {fatal_msg}")
                return "", [], f"FATAL: {fatal_msg}", {}

            # Normal completion path
            if scene_memory is None:
                # no scene memory returned — interpret as alarming, but not fatal
                print("⚠️ Warning: LangGraph returned with no scene_memory. Returning empty results.")
                return "", [], "EXCEPTION: No scene_memory returned", {}

            # scene_memory might be a dict or model
            if isinstance(scene_memory, dict):
                scene_so_far = scene_memory.get("scene_so_far", "")
                scene_cluster = scene_memory.get("scene_cluster", [])
            else:
                # assume it's a pydantic model
                scene_so_far = getattr(scene_memory, "scene_so_far", "")
                scene_cluster = getattr(scene_memory, "scene_cluster", [])

            print("✅ Scene completed normally")
            return scene_so_far, scene_cluster, "SUCCESS", self.token_usage

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            return "", [], "CANCELLED", {}
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", {}
        finally:
            print("🎭 SceneGraph stopped gracefully")
            gc.collect()

    # -------------------------
    # Node implementations
    # -------------------------
    def _initializer(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: Initializer node for user_context_id={state.user_context_id}")
        # short-circuit if fatal already flagged
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 Initializer skipping because fatal flag is set.")
            return state
        state.next_node = "SceneWriter"
        return state

    async def _scene_planner_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: ScenePlanner node for user_context_id={state.user_context_id}")
        # short-circuit if fatal already flagged
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 ScenePlanner skipping because fatal flag is set.")
            return state

        scene_memory: SceneMemory = state.scene_memory
        system_prompt = """
You are the Scene Planner Agent. Your job is to evaluate the current scene text
against the Director's Scene Plan and determine if all required scene events, emotional beats,
and target word count have been addressed.

You are checking only this specific scene — do not consider the overall chapter closure condition
or story-wide goals.

Return a JSON object with a single field 'action':
- 'Complete' if all events and emotional beats from the Scene Plan have been sufficiently covered in the scene text.
- 'Not Complete' if any required events or beats are missing or incomplete, or if the word count is significantly less than the target.

Do not include any explanations or extra fields in your response — just the JSON.
"""


        human_prompt = f"""
            Director's Instructions: {scene_memory.DirectorInstructions}
            Scene so far: {scene_memory.scene_so_far}
            {self.scene_planner_parser.get_format_instructions()}
            {f"Current scene word count: {scene_memory.word_count}" if scene_memory.word_count else ""}
        """

        try:
            llm_response = await asyncio.wait_for(
                llm_for_scene_planner_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=30.0
            )
            raw_resp = getattr(llm_response, "content", str(llm_response)).strip()
            clean_resp = StoryHelpers._strip_code_fences(raw_resp)

            # 🔎 Extract the first JSON object from the response
            match = re.search(r"\{[\s\S]*?\}", clean_resp)
            if match:
                json_str = match.group(0)
            else:
                print(f"⚠️ No JSON object found in response:\n{clean_resp}")
                json_str = '{"action": "Complete"}'  # fallback to avoid crash

            parsed = self.scene_planner_parser.parse(json_str)
            print(f"🔍 DEBUG: ScenePlanner action: {parsed.action}")

            # prevent infinite loops by counting iterations
            if parsed.action == "Not Complete":
                state.iteration_count = getattr(state, 'iteration_count', 0) + 1
                if state.iteration_count > 10:
                    print("⚠️ Forcing Complete to avoid infinite loop")
                    parsed.action = "Complete"

            state.next_node = parsed.action

        except asyncio.TimeoutError as e:
            # treat timeout as fatal for planner (so caller can decide to retry)
            return self._handle_fatal_error(state, e, "ScenePlanner Timeout")
        except Exception as e:
            # network failures, APIConnectionError, parsing errors, etc.
            return self._handle_fatal_error(state, e, "ScenePlanner")
        finally:
            state.scene_memory = scene_memory
            gc.collect()
        return state

    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: SceneWriter node for user_context_id={state.user_context_id}")
        # short-circuit if fatal already flagged
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 SceneWriter skipping because fatal flag is set.")
            return state

        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id

        system_prompt = """
        You are the Scene Writer Agent. Your job is to write a single, coherent paragraph of story text 
        that realizes the provided Scene Plan exactly as written. The Scene Plan already contains all context 
        you need: its goal, tone, emotional beats, thematic notes, location, time context, and chapter purpose.

        ⚙️ RULES:
        - Treat the Scene Plan as the complete truth. Do not invent new events, characters, or locations beyond it.
        - Stay consistent with the listed emotional and thematic intentions.
        - Write naturally but do not contradict or exceed the blueprint.
        - Focus on clarity and emotional impact — avoid long, flowery, or overly descriptive sentences.
        - Keep pacing tight: describe only what matters to the current beat.
        - If the Scene Plan contains a chapter closure condition, build toward it naturally but do not resolve it early.
        - Never summarize or restate the plan — write story prose only.
        - Write ONE paragraph only per turn (not the full scene).
        - Keep the paragraph concise (around 70–120 words).
        - Combined scene word count should roughly match the indicated target.
        - Keep content appropriate for all ages.

        Your goal is to bring the Scene Plan to life faithfully and vividly, as if you are animating its blueprint with focused, natural storytelling.
        """


        #max_scene_length = 2500  # Reduced for memory efficiency
        truncated_scene = scene_memory.scene_so_far#[-max_scene_length:] if len(scene_memory.scene_so_far) > max_scene_length else scene_memory.scene_so_far
        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {truncated_scene}" if truncated_scene else ""}
        {f"Current scene word count: {scene_memory.word_count}" if scene_memory.word_count else ""}

        {self.scene_writer_parser.get_format_instructions()}
        """
        print("SCENE WRITER CONTEXT: ", human_prompt)
        try:
            llm_response, tokens = await asyncio.wait_for(
                story_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp, model=self.model),
                timeout=30.0
            )
            self.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            self.token_usage["completion_tokens"] += tokens["completion_tokens"]
            self.token_usage["total_tokens"] += tokens["total_tokens"]
            clean_resp = getattr(llm_response, "content", str(llm_response)).strip()
            # try parse result
            try:
                parsed = self.scene_writer_parser.parse(clean_resp)
                scene_text = parsed.scene

            except Exception as e:
                print(f"❌ SceneWriter parsing failed: {e}")
                # retry with explicit instruction
                retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
                try:
                    retry_resp = await asyncio.wait_for(
                        story_client(system_prompt=system_prompt, human_prompt=retry_prompt, llm_temp=self.llm_temp, model=self.model),
                        timeout=30.0
                    )

                    retry_clean = StoryHelpers._strip_code_fences(retry_resp.content)
                    parsed = self.scene_writer_parser.parse(retry_clean)
                    scene_text = parsed.scene
                except Exception as retry_e:
                    print(f"❌ SceneWriter retry parsing failed: {retry_e}")
                    # attempt to recover JSON blob from retry_clean if available
                    try:
                        match = re.search(r'(\{[\s\S]*\})', retry_clean)
                        if match:
                            recovered = json.loads(match.group(1))
                            scene_text = recovered.get("scene", "")
                        else:
                            scene_text = retry_clean
                    except Exception as inner_e:
                        print(f"❌ SceneWriter JSON recovery failed: {inner_e}")
                        scene_text = retry_clean if 'retry_clean' in locals() else ""
                    # ensure we continue but consider that a partial success
                finally:
                    if 'retry_clean' in locals():
                        del retry_clean
                    if 'retry_resp' in locals():
                        del retry_resp

            # update memory
            scene_memory.word_count += StoryHelpers._count_words_split(scene_text)
            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text
            })
            # keep spacing consistent
            scene_memory.scene_so_far += (" " + scene_text + "\n\n") if scene_text else ""

            state.scene_memory = scene_memory

            # send chunk to frontend if callback present
            if user_context_id and state.scene_chunk_callback:
                try:
                    print(f"🔍 DEBUG: Sending scene text to frontend for {user_context_id}/{scene_memory.story_id}: {scene_text[:50]}...")
                    state.scene_chunk_callback({
                        "type": "text",
                        "scene_text": scene_text
                    })
                    state.scene_chunk_callback({
                        "type": "status",
                        "word_count": StoryHelpers._count_words_split(scene_text)
                    })
                except Exception as e:
                    print(f"❌ ERROR: Failed to send scene text to frontend: {e}")
                    traceback.print_exc()

            gc.collect()
            return state

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "SceneWriter Timeout")
        except Exception as e:
            # treat network/API errors as fatal so top-level run_scene can return FATAL
            return self._handle_fatal_error(state, e, "SceneWriter")
