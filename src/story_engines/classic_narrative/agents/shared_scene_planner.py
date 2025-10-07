# src/agents/scene_creation_subgraph/shared_scene_planner.py

import json
import re
import gc
import asyncio
from typing import List, Optional, Callable
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import groq_client
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from src.utilities.story_helpers import StoryHelpers
from langchain.output_parsers import PydanticOutputParser

# Global shared instance
CLASSIC_SCENE_PLANNER_SERVICE = None

class SceneMemory(BaseModel):
    DirectorInstructions: str = Field(description="The director's detailed instructions for this scene.")
    scene_so_far: str = Field(default="", description="Accumulated text of the scene written so far.")
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
        """Process scene for a specific user using their context"""
        print(f"🎭 Starting run_scene for {user_context.user_id}/{user_context.story_id}")
        

        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ Failed to set user_context_id: {e}")
            import traceback; traceback.print_exc()
            return "", []

        try:
            # Run LangGraph in a cancellable task
            task = asyncio.create_task(
                self.compiled.ainvoke(user_context.scene_state, {"recursion_limit": 25, "stop_event": stop_event})
            )

            while not task.done():
                if stop_event.is_set():
                    print("🛑 Stop event received — cancelling SceneGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ SceneGraph task cancelled cleanly")
                    return "", []
                await asyncio.sleep(0.2)

            result = await task
            scene_memory: SceneMemory = result["scene_memory"]
            print("✅ Scene completed normally")
            return scene_memory.scene_so_far, scene_memory.scene_cluster

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            raise
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            import traceback; traceback.print_exc()
            return "", []
        finally:
            print("🎭 SceneGraph stopped gracefully")

    
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
            f"Scene so far: {scene_memory.scene_so_far}\n"
            f"{self.scene_planner_parser.get_format_instructions()}"
        )

        try:
            llm_response = await asyncio.wait_for(
                groq_client(system_prompt=system_prompt, human_prompt=human_prompt),
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

        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT: LLM call in ScenePlanner timed out after 30 seconds")
            state.next_node = "Complete"
        except Exception as e:
            print(f"❌ ERROR in ScenePlanner: {e}")
            import traceback; traceback.print_exc()
            state.next_node = "Complete"  # fallback
        finally:
            gc.collect()

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
            "Do not make up any new characters or worlds that the Director has not mentioned. "
            "Do not repeat the entire scene so far, only continue it with one new paragraph. "
        )

        max_scene_length = 2500  # Reduced for memory efficiency
        truncated_scene = scene_memory.scene_so_far[-max_scene_length:] if len(scene_memory.scene_so_far) > max_scene_length else scene_memory.scene_so_far
        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {truncated_scene}" if truncated_scene else ""}

        {self.scene_writer_parser.get_format_instructions()}
        """

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
                except Exception as retry_e:
                    print(f"❌ SceneWriter retry parsing failed: {retry_e}")
                    match = re.search(r'(\{[\s\S]*\})', retry_clean)
                    if match:
                        try:
                            recovered = json.loads(match.group(1))
                            del match
                            scene_text = recovered.get("scene", "")
                            del recovered
                        except Exception as inner_e:
                            print(f"❌ SceneWriter JSON recovery failed: {inner_e}")
                            scene_text = retry_clean
                    else:
                        scene_text = retry_clean
                        # question_text = ""
                        # number_of_options = 0
                del retry_clean, human_prompt, retry_prompt

            #print(f"🔍 DEBUG: SceneWriter parsed - scene_text: {scene_text[:50]}..., question: {question_text}, options: {number_of_options}")

            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text
            })

            scene_memory.scene_so_far += " " + scene_text + "\n\n"

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

