# src/agents/scene_creation_subgraph/shared_scene_planner.py

import json
import re
import gc
import asyncio
from typing import List, Optional
from pydantic import BaseModel, Field
from dataclasses import dataclass

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from src.llm_client.llm_client import LLMClient
from src.utilities.story_helpers import StoryHelpers
from langchain.output_parsers import PydanticOutputParser

# Global shared instance (will be initialized in APIBackend)
SHARED_SCENE_PLANNER_SERVICE = None

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
    user_context_id: Optional[str] = Field(default=None, description="User ID for this scene")  # Add as explicit field

    class Config:
        extra = "allow"  # Allow additional dynamic attributes if needed

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
    scene_chunk_callback: callable
    
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
            user_context_id=user_id  # Set user_context_id here
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
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
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
    
    async def run_scene(self, user_context: UserSceneContext) -> tuple[str, list]:
        """Process scene for a specific user using their context"""
        print(f"🔍 DEBUG: Starting scene for user {user_context.user_id}/{user_context.story_id}")
        print(f"🔍 DEBUG: Director instructions length: {len(user_context.director_instructions)}")
        from connection.api_backend import SESSIONS
        print(f"🔍 DEBUG: SESSIONS content: {SESSIONS}")
        
        print(f"🔍 DEBUG: Scene state before assignment: {user_context.scene_state}")
        try:
            user_context.scene_state.user_context_id = user_context.user_id
            print(f"🔍 DEBUG: Set user_context_id: {user_context.scene_state.user_context_id}")
        except Exception as e:
            print(f"❌ ERROR: Failed to set user_context_id: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"🔍 DEBUG: Scene state after assignment: {user_context.scene_state}")
            
        try:
            print(f"🔍 DEBUG: Invoking graph for {user_context.user_id}/{user_context.story_id}")
            result = await self.compiled.ainvoke(user_context.scene_state, {"recursion_limit": 50})
            print(f"🔍 DEBUG: Scene planner finished with result keys: {result.keys()}")
            print(f"🔍 DEBUG: Final state after ainvoke: {result}")
            
            scene_memory: SceneMemory = result["scene_memory"]
            
            if result.get("next_node") == "END":
                print("✅ Scene reached END normally")
                return scene_memory.scene_so_far, scene_memory.scene_cluster
            
            print("✅ Scene completed normally")
            return scene_memory.scene_so_far, scene_memory.scene_cluster
            
        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT: Graph invocation timed out after 120 seconds")
            return "", []
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            import traceback
            traceback.print_exc()
            return "", []
    
    def _initializer(self, state: SceneState) -> SceneState:
        user_context_id = state.user_context_id
        print(f"🔍 DEBUG: Initializer called with user_context_id: {user_context_id}")
        director_instructions = ""
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                director_instructions = msg.content
                break
        
        state.scene_memory = SceneMemory(
            DirectorInstructions=director_instructions.strip(),
            scene_so_far="",
            story_id=state.scene_memory.story_id if state.scene_memory else ""
        )
        state.user_context_id = user_context_id  # Preserve user_context_id
        return state
    
    async def _scene_planner_agent(self, state: SceneState) -> SceneState:
        print("Running Scene Planner...")
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id
        print(f"🔍 DEBUG: ScenePlanner user_context_id: {user_context_id}")
        print(f"🔍 DEBUG: Scene planner agent called for scene: {scene_memory.scene_so_far_for_scene_planner[:100]}...")
    
        system_prompt = "You are a scene context guard. See if the scene blueprint is completed, including the decision points. Output strictly according to schema."
        human_prompt = f"""
        Director Instructions:
        {scene_memory.DirectorInstructions}

        Scene So Far:
        {scene_memory.scene_so_far_for_scene_planner}

        {self.scene_planner_parser.get_format_instructions()}
        """
        
        print(f"🔍 DEBUG: Calling llm.groq_client in ScenePlanner")
        try:
            llm_response = await asyncio.wait_for(
                self.llm.groq_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT: LLM call in ScenePlanner timed out after 60 seconds")
            state.next_node = "Complete"
            return state
        print(f"🔍 DEBUG: ScenePlanner got LLM response: {llm_response.content[:100]}...")
        
        strip = StoryHelpers._strip_code_fences(llm_response.content)
        match = re.search(r'(\{[\s\S]*?\})', strip)

        if match:
            try:
                extracted_str = match.group(1)
                clean_resp_parsed = json.loads(extracted_str)
                next_node = "Complete" if clean_resp_parsed.get("action", "").lower() == "complete" else "Not Complete"
            except Exception as e:
                print("ScenePlanner JSON parsing failed:", e)
                next_node = "Not Complete"
        else:
            print("ScenePlanner: No valid JSON found, retrying...")
            retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
            try:
                retry_resp = await asyncio.wait_for(
                    self.llm.groq_client(system_prompt=system_prompt, human_prompt=retry_prompt),
                    timeout=60.0
                )
                retry_clean = StoryHelpers._strip_code_fences(retry_resp.content)
                retry_json = json.loads(retry_clean)
                next_node = "Complete" if retry_json.get("action", "").lower() == "complete" else "Not Complete"
            except asyncio.TimeoutError:
                print(f"❌ TIMEOUT: LLM retry call in ScenePlanner timed out after 60 seconds")
                next_node = "Complete"
            except Exception:
                next_node = "Not Complete"

        print(f"🔍 DEBUG: ScenePlanner determined next_node: {next_node}")
        
        if next_node == "Not Complete" and scene_memory.ai_question.strip():
            print(f"🔍 DEBUG: Need user input for question: {scene_memory.ai_question}")
            try:
                from connection.api_backend import SESSIONS
                text = {
                    "type": "decision",
                    "question": scene_memory.ai_question.strip(),
                    "options": scene_memory.number_of_options,
                    "user_choice": ""
                }
                if user_context_id:
                    print(f"🔍 DEBUG: Sending question callback for user {user_context_id}/{scene_memory.story_id}")
                    user_data = SESSIONS.get(user_context_id, {})
                    story_data = user_data.get(scene_memory.story_id, {})
                    scene_chunk_callback = story_data.get("director", {}).scene_chunk_callback
                    if scene_chunk_callback:
                        print(f"🔍 DEBUG: Sending decision prompt to frontend: {text}")
                        scene_chunk_callback(text)
                    else:
                        print(f"❌ ERROR: No scene_chunk_callback found for {user_context_id}/{scene_memory.story_id}")
                    user_input_queue = story_data.get("user_input_queue")
                    if user_input_queue:
                        print(f"🔍 DEBUG: Waiting for user input from queue for {user_context_id}/{scene_memory.story_id}")
                        try:
                            user_choice = await asyncio.wait_for(user_input_queue.get(), timeout=30.0)
                            print(f"✅ DEBUG: Received user choice: {user_choice}")
                            scene_memory.scene_cluster.append({
                                "type": "decision",
                                "question": scene_memory.ai_question.strip(),
                                "options": scene_memory.number_of_options,
                                "user_choice": user_choice.strip()
                            })
                            scene_memory.UserInput = user_choice
                        except asyncio.TimeoutError:
                            print(f"❌ TIMEOUT: No user input received within 30 seconds for {user_context_id}/{scene_memory.story_id}")
                            scene_memory.UserInput = ""
                    else:
                        print(f"❌ ERROR: No user input queue found for {user_context_id}/{scene_memory.story_id}")
                        scene_memory.UserInput = ""
                else:
                    print("❌ ERROR: No user context ID available")
                    scene_memory.UserInput = ""
            except Exception as e:
                print(f"❌ ERROR in user input handling: {e}")
                import traceback
                traceback.print_exc()
                scene_memory.UserInput = ""
        else:
            scene_memory.UserInput = ""
        
        state.scene_memory = scene_memory
        state.next_node = next_node
        gc.collect()
        return state
    
    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print("Running Scene Writer...")
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id
        print(f"🔍 DEBUG: SceneWriter user_context_id: {user_context_id}")
        print(f"🔍 DEBUG: SceneWriter called for scene: {scene_memory.scene_so_far[:100]}...")
        
        system_prompt = (
            "You are the Scene Writer Agent. Follow schema strictly. "
            "The Director's Instructions will include a recap of the story so far, "
            "relevant characters/worlds to the current scene, and the scene blueprint that you need to follow strictly for this scene. "
            "You do not know anything beyond what the Director tells you, so be sure to include all relevant context in your writing. "
            "Write in a vivid, engaging style, with rich descriptions and immersive details. "
            "If the Director's Instructions include a decision point question for the user, end your paragraph output with that question. "
            "Do not produce more user decision points than what the Director includes. "
            "Do not make up any new characters or worlds that the Director has not mentioned. "
            "Do not repeat the entire scene so far, only continue it with one new paragraph. "
            "You may give up to two - four options for the user to choose from, include them in the question, mark each with letters. "
            "Include the number of options in the 'number_of_options' field."
        )

        # Truncate scene_so_far to prevent excessive prompt size
        max_scene_length = 1000
        truncated_scene = scene_memory.scene_so_far[-max_scene_length:] if len(scene_memory.scene_so_far) > max_scene_length else scene_memory.scene_so_far
        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {truncated_scene}" if truncated_scene else ""}

        {f"Your Question: {scene_memory.ai_question}" if scene_memory.ai_question else ""}
        {f"(The user chose: {scene_memory.UserInput})" if scene_memory.UserInput else ""}

        {self.scene_writer_parser.get_format_instructions()}
        """

        print(f"🔍 DEBUG: Calling llm.groq_client in SceneWriter")
        try:
            llm_response = await asyncio.wait_for(
                self.llm.groq_client(system_prompt=system_prompt, human_prompt=human_prompt),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            print(f"❌ TIMEOUT: LLM call in SceneWriter timed out after 60 seconds")
            return state
        print(f"🔍 DEBUG: SceneWriter got LLM response: {llm_response.content[:100]}...")
        
        clean_resp = llm_response.content.strip()
        del llm_response, human_prompt

        try:
            parsed = self.scene_writer_parser.parse(clean_resp)
            del clean_resp
            scene_text = parsed.scene
            question_text = parsed.question
            number_of_options = parsed.number_of_options
        except Exception as e:
            print("SceneWriter parsing failed:", e)
            retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
            try:
                retry_resp = await asyncio.wait_for(
                    self.llm.groq_client(system_prompt=system_prompt, human_prompt=retry_prompt),
                    timeout=60.0
                )
                retry_clean = StoryHelpers._strip_code_fences(retry_resp.content)
                del retry_resp
                parsed = self.scene_writer_parser.parse(retry_clean)
                scene_text = parsed.scene
                question_text = parsed.question
                number_of_options = parsed.number_of_options
            except Exception as retry_e:
                print("SceneWriter retry parsing failed:", retry_e)
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
                        print("SceneWriter JSON recovery failed:", inner_e)
                        scene_text = retry_clean
                        question_text = ""
                        number_of_options = 0
                else:
                    scene_text = retry_clean
                    question_text = ""
                    number_of_options = 0
            del retry_clean

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

        if user_context_id:
            try:
                from connection.api_backend import SESSIONS
                user_data = SESSIONS.get(user_context_id, {})
                story_data = user_data.get(scene_memory.story_id, {})
                scene_chunk_callback = story_data.get("director", {}).scene_chunk_callback
                if scene_chunk_callback:
                    print(f"🔍 DEBUG: Sending scene text to frontend for {user_context_id}/{scene_memory.story_id}: {scene_text[:50]}...")
                    scene_chunk_callback({
                        "type": "text",
                        "scene_text": scene_text
                    })
                else:
                    print(f"❌ ERROR: No scene_chunk_callback found for {user_context_id}/{scene_memory.story_id}")
            except Exception as e:
                print(f"❌ ERROR: Failed to send scene text to frontend: {e}")
                import traceback
                traceback.print_exc()

        gc.collect()
        return state