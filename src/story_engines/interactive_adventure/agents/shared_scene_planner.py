# src/agents/scene_creation_subgraph/shared_scene_planner.py

import json
import re
import gc
import asyncio
from typing import List, Optional, Callable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import groq_client, groq_zero_temp_client
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from src.utilities.story_helpers import StoryHelpers
from langchain.output_parsers import PydanticOutputParser

# Global shared instance
INTERACTIVE_SCENE_PLANNER_SERVICE = None

class SceneMemory(BaseModel):
    DirectorInstructions: str = Field(description="The director's detailed instructions for this scene.")
    scene_so_far: str = Field(default="", description="Accumulated text of the scene written so far.")
    ai_question: Optional[str] = Field(default="", description="Most recent decision point question, if any.")
    UserInput: Optional[str] = Field(default="", description="The latest user input choice, if any.")
    scene_so_far_for_scene_planner: Optional[str] = Field(default="", description="Accumulated text of the scene, ai questions and user responses for the scene planner.")
    number_of_options: Optional[int] = Field(default=0, description="Number of options available at the decision point.")
    scene_cluster: List = Field(default=[], description="Combination of scene text, questions and user choices stored as dicts inside the list.")
    story_id: Optional[str] = Field(default="", description="The story ID associated with this scene.")

class SceneState(BaseModel):
    messages: List[BaseMessage] = []
    scene_memory: SceneMemory | None = None
    next_node: str | None = None
    user_context_id: Optional[str] = Field(default=None, description="User ID for this scene")
    iteration_count: int = 0  # Added for recursion limit
    scene_chunk_callback: Optional[Callable] = Field(default=None, description="Callback to send scene chunks to frontend")

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
            scene_so_far="",
            story_id=story_id
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
        
        self.graph.add_conditional_edges(
            "ScenePlanner",
            lambda state: "Not Complete" if state.next_node == "Not Complete" else "Complete",
            {
                "Not Complete": "SceneWriter",
                "Complete": END,
            },
        )
        
        self.compiled = self.graph.compile()
    
    async def run_scene(self, user_context: UserSceneContext, stop_event: asyncio.Event | None = None) -> tuple[str, list]:
        """Process scene for a specific user using their context, cancellable via stop_event."""
        print(f"🔍 DEBUG: Starting run_scene with user_id={user_context.user_id}, story_id={user_context.story_id}")

        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ ERROR: Failed to set user_context_id: {e}")
            import traceback
            traceback.print_exc()
            return "", []

        try:
            print(f"🔍 DEBUG: Invoking graph for {user_context.user_id}/{user_context.story_id}")

            # ✅ Run LangGraph inside a cancellable task
            task = asyncio.create_task(
                self.compiled.ainvoke(
                    user_context.scene_state,
                    {"recursion_limit": 25, "stop_event": stop_event}
                )
            )

            # ✅ Monitor for cancellation
            while not task.done():
                if stop_event and stop_event.is_set():
                    print("🛑 Stop event received — cancelling SceneGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ SceneGraph task cancelled cleanly")
                    return "", []
                await asyncio.sleep(0.2)

            # ✅ Get final result
            result = await task

            print(f"🔍 DEBUG: Graph result: {result}")

            scene_memory: SceneMemory = result["scene_memory"]
            print(f"🔍 DEBUG: Scene memory: so_far={scene_memory.scene_so_far[:50]}..., cluster_len={len(scene_memory.scene_cluster)}")

            if result.get("next_node") == "END":
                print("✅ Scene reached END normally")
                return scene_memory.scene_so_far, scene_memory.scene_cluster

            print("✅ Scene completed normally")
            return scene_memory.scene_so_far, scene_memory.scene_cluster

        except asyncio.CancelledError:
            print("🛑 SceneGraph run_scene() cancelled cleanly (asyncio.CancelledError caught)")
            return "", []
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            import traceback
            traceback.print_exc()
            return "", []

    
    def _initializer(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: Initializer node for user_context_id={state.user_context_id}")
        state.next_node = "SceneWriter"
        return state
    

    async def _scene_planner_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: ScenePlanner node for user_context_id={state.user_context_id}")
        scene_memory: SceneMemory = state.scene_memory
        system_prompt = (
            "You are the Scene Planner Agent. Your job is to review the scene so far and determine "
            "if all events in the Director's Instructions have been covered. "
            "Return JSON with a single field 'action' set to 'Complete' if all events are covered, "
            "or 'Not Complete' otherwise."
        )

        human_prompt = (
            f"Director's Instructions: {scene_memory.DirectorInstructions}\n"
            f"Scene so far: {scene_memory.scene_so_far_for_scene_planner}\n"
            f"{self.scene_planner_parser.get_format_instructions()}"
        )

        try:
            llm_response = await asyncio.wait_for(
                groq_zero_temp_client(system_prompt=system_prompt, human_prompt=human_prompt),
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

        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT: LLM call in ScenePlanner timed out after 30 seconds")
            state.next_node = "Complete"
        except Exception as e:
            print(f"❌ ERROR in ScenePlanner: {e}")
            import traceback; traceback.print_exc()
            state.next_node = "Complete"  # fallback
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
                        for _ in range(5000):
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
        state.next_node = parsed.action
        state.scene_memory = scene_memory
        gc.collect()
        return state

        
    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: SceneWriter node for user_context_id={state.user_context_id}")
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id
        
        system_prompt = (
            "You are the Scene Writer Agent. Write one paragraph continuing the scene based on the Director's Instructions. "
            "The scene blueprint that you need to follow strictly for this scene. "
            "You do not know anything beyond what the Director tells you, so be sure to include all relevant context in your writing. "
            "Write in a vivid, engaging style, with rich descriptions and immersive details. "
            "If the Director's Instructions include a decision point question for the user, end your paragraph output with that question. "
            "Do not produce more user decision points than what the Director includes. "
            "Do not make up any new characters or worlds that the Director has not mentioned. "
            "Do not repeat the entire scene so far, only continue it with one new paragraph. "
            "You may give up to two - four options for the user to choose from, include them in the question, mark each with letters. "
            "Include the number of options in the 'number_of_options' field."
        )

        max_scene_length = 2500  # Reduced for memory efficiency
        truncated_scene = scene_memory.scene_so_far[-max_scene_length:] if len(scene_memory.scene_so_far) > max_scene_length else scene_memory.scene_so_far
        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {truncated_scene}" if truncated_scene else ""}

        {f"Your Question: {scene_memory.ai_question}" if scene_memory.ai_question else ""}
        {f"(The user chose: {scene_memory.UserInput})" if scene_memory.UserInput else ""}

        {self.scene_writer_parser.get_format_instructions()}
        """
        print("SCENE WRITER CONTEXT: ", human_prompt)
        try:
            llm_response = await asyncio.wait_for(
                groq_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=30.0
            )
            clean_resp = llm_response.content.strip()
            del llm_response

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
                        groq_client(system_prompt=system_prompt, human_prompt=retry_prompt),
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

            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text
            })

            scene_memory.scene_so_far += " " + scene_text + "\n\n"
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
                except Exception as e:
                    print(f"❌ ERROR: Failed to send scene text to frontend: {e}")
                    import traceback
                    traceback.print_exc()

            gc.collect()
            return state
        except Exception as e:
            print(f"❌ ERROR in SceneWriter: {e}")
            import traceback
            traceback.print_exc()
            return state

