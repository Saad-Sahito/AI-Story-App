import asyncio
import json
import re
import gc
from typing import List, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.store.memory import InMemoryStore
from src.llm_client.llm_client import LLMClient
#from src.memory.memory_system import StoryMemorySystem
from src.utilities.story_helpers import StoryHelpers

from langchain.output_parsers import PydanticOutputParser

class SceneMemory(BaseModel):
    DirectorInstructions: str = Field(
        description="The director's detailed instructions for this scene."
    )
    scene_so_far: str = Field(
        default="", description="Accumulated text of the scene written so far."
    )
    ai_question: Optional[str] = Field(
        default="", description="Most recent decision point question, if any."
    )
    UserInput: Optional[str] = Field(
        default="", description="The latest user input choice, if any."
    )
    scene_so_far_for_scene_planner: Optional[str] = Field(
        default="", description="Accumulated text of the scene, ai questions and user responses for the scene planner."
    )
    number_of_options: Optional[int] = Field(
        default=0, description="Number of options available at the decision point."
    )
    scene_cluster: List = Field(
        default=[], description="Combination of scene text, questions and user choices stored as dicts inside the list."
    )


class SceneState(BaseModel):
    messages: List[BaseMessage] = []
    scene_memory: SceneMemory | None = None
    next_node: str | None = None


class SceneWriterOutput(BaseModel):
    scene: str = Field(
        description="One paragraph of continuing narrative text. Do Not write the question here, only in the 'question' field."
    )
    question: str = Field(
        description="Decision prompt for the user if this is the marked decision point, otherwise empty string."
    )
    number_of_options: Optional[int] = Field(
        description="If there is a question, how many options are provided (0 if no question)."
    )

class ScenePlannerOutput(BaseModel):
    action: str = Field(
        description="Either 'Complete' if the scene has all scene blueprint events, or 'Not Complete' otherwise."
    )


scene_writer_parser = PydanticOutputParser(pydantic_object=SceneWriterOutput)
scene_planner_parser = PydanticOutputParser(pydantic_object=ScenePlannerOutput)


class ScenePlannerGraph:
    def __init__(self, llm_client: LLMClient):
        #self.memory_system = memory_system
        self.memory_store = InMemoryStore()
        self.llm = llm_client
        # The state is a dictionary, so we don't need a custom lambda
        self.graph = StateGraph(SceneState, checkpointer=self.memory_store)

        
        # Nodes
        self.graph.add_node("Initializer", self.initializer)
        self.graph.add_node("ScenePlanner", self.scene_planner_agent)
        self.graph.add_node("SceneWriter", self.scene_writer_agent)

        # Entry point
        self.graph.set_entry_point("Initializer")

        self.graph.add_edge("Initializer", "SceneWriter")

        self.graph.add_edge("SceneWriter", "ScenePlanner")
        # The SceneWriter determines the next step based on the LLM's JSON output
        # self.graph.add_conditional_edges(
        #     "SceneWriter",
        #     lambda state: "SceneWriter" if state.next_node == "SceneWriter" else "ScenePlanner",
        #     {
        #         "SceneWriter": "SceneWriter",
        #         "ScenePlanner": "ScenePlanner",
        #     },
        # )

        self.graph.add_conditional_edges(
            "ScenePlanner",
            lambda state: "Not Complete" if state.next_node == "Not Complete" else "Complete",
            {
                "Not Complete": "SceneWriter",
                "Complete": END,
            },
        )

        self.compiled = self.graph.compile()
        self.question_boolean = False
        #self._user_input_future = None
        self.user_input_queue = asyncio.Queue()

    
    async def wait_for_user_input(self):
        return await self.user_input_queue.get()

    # def receive_user_input(self, choice: str):
    #     if self._user_input_future and not self._user_input_future.done():
    #         self._user_input_future.set_result(choice)

    # -------------------------
    # Initializes Context
    # -------------------------
    def _initializer(self, state: dict) -> dict:
        user_context_id = getattr(state, 'user_context_id', None)
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
        # Preserve user_context_id
        if user_context_id:
            state.user_context_id = user_context_id
        return state


    # ------------------------
    # Scene Planner
    # ------------------------
    async def _scene_writer_agent(self, state: dict) -> dict:
        print("Running Scene Writer...")
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = getattr(state, 'user_context_id', None)
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

        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {scene_memory.scene_so_far}" if scene_memory.scene_so_far else ""}

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

        # Stream the scene chunk back through user's callback
        if user_context_id:
            try:
                from connection.api_backend import SESSIONS
                user_data = SESSIONS.get(user_context_id, {})
                story_data = user_data.get(scene_memory.story_id, {})
                scene_chunk_callback = story_data.get("director", {}).scene_chunk_callback
                if scene_chunk_callback:
                    print(f"🔍 DEBUG: Sending scene text to frontend for {user_context_id}/{scene_memory.story_id}")
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

    async def _scene_planner_agent(self, state: dict) -> dict:
        print("Running Scene Planner...")
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = getattr(state, 'user_context_id', None)
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
                    print(f"🔍 DEBUG: Sending question callback for user {user_context_id}")
                    user_data = SESSIONS.get(user_context_id, {})
                    story_data = user_data.get(scene_memory.story_id, {})
                    scene_chunk_callback = story_data.get("director", {}).scene_chunk_callback
                    if scene_chunk_callback:
                        print(f"🔍 DEBUG: Sending decision prompt to frontend for {user_context_id}/{scene_memory.story_id}")
                        scene_chunk_callback(text)
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


    # ------------------------
    # Run full scene
    # ------------------------
    async def run(self, state: SceneState, scene_chunk_callback) -> str:
        # initialize scene memory only if it's missing
        # if not state.scene_memory:
        state.scene_memory = SceneMemory(
                DirectorInstructions="",
                scene_so_far="",
                ai_question="",
                UserInput=""
            )

        print("Running scene planner...")
        self.scene_chunk_callback = scene_chunk_callback
        result = await self.compiled.ainvoke(state, {"recursion_limit": 50})

        scene_memory: SceneMemory = result["scene_memory"]

        if result.get("next_node") == "END":
            print("Reached END. Stopping execution.")
            return scene_memory.scene_so_far, scene_memory.scene_cluster
        
        # free instance internals
        # del self.graph, self.compiled, self.llm, self.memory_store  
        # gc.collect()
        return scene_memory.scene_so_far, scene_memory.scene_cluster


