# src/agents/scene_creation_subgraph/shared_scene_planner.py

import json
import re
import gc
import asyncio
from typing import List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import story_client, llm_for_scene_planner_client
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from src.utilities.story_helpers import StoryHelpers
from langchain_core.output_parsers import PydanticOutputParser

# Global shared instance
INTERACTIVE_SCENE_PLANNER_SERVICE = None

class SceneMemory(BaseModel):
    DirectorInstructions: str = Field(description="The director's detailed instructions for this scene.")
    #scene_so_far: str = Field(default="", description="Accumulated text of the scene written so far.")
    ai_question: Optional[str] = Field(default="", description="Most recent decision point question, if any.")
    UserInput: Optional[str] = Field(default="", description="The latest user input choice, if any.")
    scene_so_far_for_scene_planner: str = Field(default="", description="Accumulated text of the scene, ai questions and user responses.")
    number_of_options: Optional[int] = Field(default=0, description="Number of options available at the decision point.")
    scene_cluster: List = Field(default=[], description="Combination of scene text, questions and user choices stored as dicts inside the list.")
    story_id: Optional[str] = Field(default="", description="The story ID associated with this scene.")
    word_count: int = Field(..., description="Word count of the scene so far.")


class SceneState(BaseModel):
    messages: List[BaseMessage] = []
    scene_memory: SceneMemory | None = None
    next_node: str | None = None
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
    question: str = Field(description="Decision prompt for the user if this is the marked decision point, otherwise empty string.")
    number_of_options: Optional[int] = Field(description="If there is a question, how many options are provided (0 if no question).")


class ScenePlannerOutput(BaseModel):
    action: str = Field(description="Either 'Complete' if the scene has all scene blueprint events, or 'Not Complete' otherwise.")


@dataclass
class UserSceneContext:
    user_id: str
    story_id: str
    director_instructions: str
    scene_state: SceneState
    scene_memory: SceneMemory
    scene_chunk_callback: Callable
    
    @classmethod
    def create_for_user(cls, user_id: str, story_id: str, director_instructions: str, scene_chunk_callback):
        scene_memory = SceneMemory(
            DirectorInstructions=director_instructions.strip(),
            #scene_so_far="",
            story_id=story_id,
            word_count=0
        )
        
        scene_state = SceneState(
            messages=[AIMessage(content=director_instructions)],
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
        self.graph.add_node("ScenePlanner", self._scene_planner_agent)
        self.graph.add_node("SceneWriter", self._scene_writer_agent)
        
        self.graph.set_entry_point("Initializer")
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
        import traceback
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
            import traceback
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}"

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
                    return "", [], "CANCELLED"
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
                return "", [], f"FATAL: {fatal_msg}"

            # Normal completion path
            if scene_memory is None:
                # no scene memory returned — interpret as alarming, but not fatal
                print("⚠️ Warning: LangGraph returned with no scene_memory. Returning empty results.")
                return "", [], "EXCEPTION: No scene_memory returned"

            # scene_memory might be a dict or model
            if isinstance(scene_memory, dict):
                scene_so_far = scene_memory.get("scene_so_far_for_scene_planner", "")
                scene_cluster = scene_memory.get("scene_cluster", [])
            else:
                # assume it's a pydantic model
                scene_so_far = getattr(scene_memory, "scene_so_far_for_scene_planner", "")
                scene_cluster = getattr(scene_memory, "scene_cluster", [])

            print("✅ Scene completed normally")
            return scene_so_far, scene_cluster, "SUCCESS", self.token_usage

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            return "", [], "CANCELLED"
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            import traceback
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}"
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
        system_prompt = f"""
You are the Scene Planner Agent. Your job is to analyze whether the Scene Writer's output so far
has covered all the events, beats, and decision points specified in the Director's Scene Blueprint.

You must only consider the current scene, not the entire chapter or story.
Do not confuse 'scene completion' with 'chapter completion' — those are handled separately.

Check if all required beats from the Director's instructions (actions, emotions, locations, dialogue moments,
and any specified user decision points) have been fulfilled.

If the last sentence expects a user decision then let the scene continue.

Return a valid JSON object in this format:
{self.scene_planner_parser.get_format_instructions()}

If even one major event or decision point is missing, mark action as 'Not Complete'.
Only return 'Complete' when you are confident that the entire scene blueprint has been faithfully covered.
"""


        human_prompt = (
            f"Director's Instructions: {scene_memory.DirectorInstructions}\n"
            f"Scene so far: {scene_memory.scene_so_far_for_scene_planner}\n"
        )

        try:
            llm_response = await asyncio.wait_for(
                llm_for_scene_planner_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=30.0
            )
            raw_resp = llm_response.content.strip()
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
            gc.collect()

        try:
            # Only trigger decision flow if planner says Not Complete AND there's an ai_question awaiting answer
            if parsed.action == "Not Complete" and scene_memory and scene_memory.ai_question.strip():
                print(f"🔍 DEBUG: Need user input for question: {scene_memory.ai_question}")
                from setup.shared_redis_pool import get_redis_client
                try:
                    # Try to find the user/session-level structures
                    user_context_id = state.user_context_id
                    redis_client = await get_redis_client()
                    queue_key = f"input_queue:{user_context_id}:{scene_memory.story_id}"

                    # Build callback payload
                    decision_payload = {
                        "type": "decision",
                        "question": scene_memory.ai_question.strip(),
                        "options": scene_memory.number_of_options,
                        "user_choice": ""
                    }

                    # Prefer state.scene_chunk_callback if provided
                    scene_chunk_cb = state.scene_chunk_callback if getattr(state, "scene_chunk_callback", None) else None

                    # Send decision to frontend
                    if scene_chunk_cb:
                        try:
                            #print(f"🔍 DEBUG: Sending decision prompt to frontend for {user_context_id}/{scene_memory.story_id}: {decision_payload}")
                            scene_chunk_cb(decision_payload)
                        except Exception as e:
                            print(f"❌ ERROR: scene_chunk_callback raised: {e}")
                            import traceback; traceback.print_exc()
                    else:
                        print(f"❌ ERROR: No scene_chunk_callback found for {user_context_id}/{scene_memory.story_id}")

                    # Wait for user input from Redis List
                    try:
                        print(f"🔍 DEBUG: Waiting for user input from Redis queue {queue_key}")
                        user_choice = None
                        for _ in range(3600):
                            choice = await redis_client.lpop(queue_key)
                            if choice:
                                user_choice = choice
                                break
                            await asyncio.sleep(1.0)
                        if user_choice:
                            print(f"✅ DEBUG: Received user choice: {user_choice} from Redis queue {queue_key}")
                            # Append decision entry to scene_cluster and update memory
                            scene_memory.scene_cluster.append({
                                "type": "decision",
                                "question": scene_memory.ai_question.strip(),
                                "options": scene_memory.number_of_options,
                                "user_choice": user_choice.strip() if isinstance(user_choice, str) else user_choice
                            })
                            scene_memory.UserInput = user_choice
                            scene_memory.scene_so_far_for_scene_planner += f"(The user chose: {user_choice})\n"
                        else:
                            print(f"❌ TIMEOUT: No user input received within 5000 seconds for {user_context_id}/{scene_memory.story_id}")
                            scene_memory.UserInput = ""
                    except Exception as e:
                        print(f"❌ ERROR in Redis queue handling: {e}")
                        import traceback; traceback.print_exc()
                        scene_memory.UserInput = ""
                except Exception as e:
                    print(f"❌ ERROR in user input handling: {e}")
                    import traceback; traceback.print_exc()
                    scene_memory.UserInput = ""
            else:
                # No decision outstanding
                scene_memory.UserInput = ""
        except Exception as e:
            # Safety: ensure any unexpected exception in decision handling won't break planner
            print(f"❌ ERROR after ScenePlanner LLM call while handling user input: {e}")
            import traceback; traceback.print_exc()
            scene_memory.UserInput = ""

        # Persist updated scene memory back into state
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
        
        system_prompt = f"""
You are the Scene Writer Agent for an interactive text-based story.
Your task is to write exactly one concise, coherent paragraph that continues the scene
according to the Director's detailed scene blueprint.

⚙️ Rules:
1. Follow the Director's instructions precisely — do not invent, alter, or omit planned details.
2. You only know what the Director tells you — you have no memory of past scenes.
3. Write in a natural, vivid style that captures tone, setting, emotions, and action as described.
4. Avoid long, overly descriptive sentences — keep pacing tight and focused on key beats.
5. Do NOT summarize or repeat previous content; continue naturally from the last paragraph.
6. Include all required creative elements from the 'screenplay_notes' section (e.g., imagery, metaphors, pacing cues).
7. Keep the paragraph around 80–130 words for balanced pacing.

🎭 Decision Points:
- If the Director's instructions specify a user decision point, end your paragraph with that question.
- Offer 2–4 clearly distinct options labeled (A), (B), (C), (D) within the question.
- Each option must represent a meaningful narrative branch.
- Output the number of available options as an integer in the field 'number_of_options'.

🧩 Output JSON format:
{self.scene_writer_parser.get_format_instructions()}

Stay strictly within the Director's plan.
Do not create new storylines, characters, or settings.
End naturally if the scene has no pending decision points.
"""


        max_scene_length = 3500  # Reduced for memory efficiency
        truncated_scene = scene_memory.scene_so_far_for_scene_planner[-max_scene_length:] if len(scene_memory.scene_so_far_for_scene_planner) > max_scene_length else scene_memory.scene_so_far_for_scene_planner
        human_prompt = f"""Director's Instructions:
{scene_memory.DirectorInstructions}

{f'Scene so far (DO NOT rewrite this, only output text that continues from here): {truncated_scene}' if truncated_scene else ''}
{f'Current scene word count: {scene_memory.word_count}' if scene_memory.word_count else ''}"""
        print("SCENE WRITER CONTEXT: ", human_prompt)
        try:
            llm_response, tokens = await asyncio.wait_for(
                story_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp, model=self.model),
                timeout=30.0
            )
            self.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            self.token_usage["completion_tokens"] += tokens["completion_tokens"]
            self.token_usage["total_tokens"] += tokens["total_tokens"]
            clean_resp = llm_response.content.strip()
            del llm_response, tokens

            try:
                parsed = self.scene_writer_parser.parse(clean_resp)
                del clean_resp
                scene_text = parsed.scene
                question_text = parsed.question
                number_of_options = parsed.number_of_options
            except Exception as e:
                print(f"❌ SceneWriter parsing failed: {e}")
                retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
                try:
                    retry_resp = await asyncio.wait_for(
                        story_client(system_prompt=system_prompt, human_prompt=retry_prompt, llm_temp=self.llm_temp, model=self.model),
                        timeout=30.0
                    )
                    retry_clean = StoryHelpers._strip_code_fences(retry_resp.content)
                    del retry_resp
                    parsed = self.scene_writer_parser.parse(retry_clean)
                    scene_text = parsed.scene
                    question_text = parsed.question
                    number_of_options = parsed.number_of_options
                except Exception as retry_e:
                    print(f"❌ SceneWriter retry parsing failed: {retry_e}")
                    match = re.search(r'(\{[\s\S]*\})', retry_clean)
                    if match:
                        try:
                            recovered = json.loads(match.group(1))
                            del match
                            scene_text = recovered.get("scene", "")
                            question_text = recovered.get("question", "")
                            number_of_options = recovered.get("number_of_options", 0)
                            del recovered
                        except Exception as inner_e:
                            print(f"❌ SceneWriter JSON recovery failed: {inner_e}")
                            scene_text = retry_clean
                            question_text = ""
                            number_of_options = 0
                    else:
                        scene_text = retry_clean
                        question_text = ""
                        number_of_options = 0
                del retry_clean, human_prompt, retry_prompt

            print(f"🔍 DEBUG: SceneWriter parsed - scene_text: {scene_text[:50]}..., question: {question_text}, options: {number_of_options}")

            if scene_memory.UserInput:
                scene_memory.scene_so_far_for_scene_planner += f"(The user chose: {scene_memory.UserInput})\n"

            scene_memory.word_count += StoryHelpers._count_words_split(scene_text)
            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text
            })

            #scene_memory.scene_so_far += " " + scene_text + "\n\n"
            scene_memory.scene_so_far_for_scene_planner += scene_text + "\n"

            if question_text.strip():
                scene_memory.scene_so_far_for_scene_planner += f"(The scene writer asked: {question_text.strip()})\n"
                scene_memory.ai_question = question_text.strip()
                scene_memory.number_of_options = number_of_options if number_of_options > 0 else 1
            else:
                scene_memory.ai_question = ""
                scene_memory.number_of_options = 0

            state.scene_memory = scene_memory

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
                    import traceback
                    traceback.print_exc()

            gc.collect()
            return state

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "SceneWriter Timeout")
        except Exception as e:
            # treat network/API errors as fatal so top-level run_scene can return FATAL
            return self._handle_fatal_error(state, e, "SceneWriter")