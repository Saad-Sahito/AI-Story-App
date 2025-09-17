import asyncio
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.store.memory import InMemoryStore
from src.llm_client.llm_client import LLMClient

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

class ScenePlannerOutput(BaseModel):
    action: str = Field(
        description="Either 'Complete' if the scene has all scene blueprint events, or 'Not Complete' otherwise."
    )


scene_writer_parser = PydanticOutputParser(pydantic_object=SceneWriterOutput)
scene_planner_parser = PydanticOutputParser(pydantic_object=ScenePlannerOutput)


class ScenePlannerGraph:
    def __init__(self, llm_client: LLMClient):
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
        self._user_input_future = None

    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```"):
            # remove leading/trailing ```json ... ```
            return re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        return text
    
    async def wait_for_user_input(self):
        self._user_input_future = asyncio.Future()
        return await self._user_input_future

    def receive_user_input(self, choice: str):
        if self._user_input_future and not self._user_input_future.done():
            self._user_input_future.set_result(choice)

    # -------------------------
    # Initializes Context
    # -------------------------
    def initializer(self, state: dict) -> dict:  
        director_instructions = ""
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                director_instructions = msg.content
                break

        # Initialize SceneMemory properly
        state.scene_memory = SceneMemory(
            DirectorInstructions=director_instructions.strip(),
            scene_so_far=""   # will accumulate later
        )
        return state


    # ------------------------
    # Scene Planner
    # ------------------------
    async def scene_planner_agent(self, state: dict) -> dict:
        """
        Determines if the SceneWriter has completed the scene based on Director's instructions.
        """
        scene_memory: SceneMemory = state.scene_memory

        system_prompt = "You are a scene context guard. See if the scene blueprint is completed, including the decision points. " \
        "Output strictly according to schema."
        human_prompt = f"""
        Director Instructions:
        {scene_memory.DirectorInstructions}

        Scene So Far:
        {scene_memory.scene_so_far_for_scene_planner}

        {scene_planner_parser.get_format_instructions()}
        """
        print("SCENE PLANNER HUMAN PROMPT: ", human_prompt)
        llm_response = self.llm.groq_client(system_prompt=system_prompt, human_prompt=human_prompt)
        strip = self._strip_code_fences(llm_response.content)
        match = re.search(r'(\{[\s\S]*?\})', strip)

        if match:
            try:
                extracted_str = match.group(1)
                clean_resp_parsed = json.loads(extracted_str)
                next_node = "Complete" if clean_resp_parsed.get("action", "").lower() == "complete" else "Not Complete"
            except Exception as e:
                print("ScenePlanner JSON parsing failed:", e)
                # fallback: treat as incomplete
                next_node = "Not Complete"
        else:
            print("ScenePlanner: No valid JSON found, retrying...")
            # retry with stricter format instructions
            retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
            retry_resp = self.llm.groq_client(system_prompt=system_prompt, human_prompt=retry_prompt)
            try:
                retry_clean = self._strip_code_fences(retry_resp.content)
                retry_json = json.loads(retry_clean)
                next_node = "Complete" if retry_json.get("action", "").lower() == "complete" else "Not Complete"
            except:
                next_node = "Not Complete"


        print("SCENE PLANNER RESPONSE: ", next_node)
        #print("QUESTION: ", scene_memory.ai_question)
        if next_node == "Not Complete":
            if self.question_boolean:
                
                self.scene_chunk_callback(json.dumps({"decision_point": scene_memory.ai_question.strip()}) + "\n")

                user_choice = await self.wait_for_user_input()
                print("USER CHOICE RECEIVED: ", user_choice)
                scene_memory.UserInput = user_choice
                state.scene_memory = scene_memory

            else:
                # No question to ask, just continue
                scene_memory.UserInput = ""
                state.scene_memory = scene_memory
                
        state.next_node = next_node
        return state

    # ------------------------
    # Scene Writer
    # ------------------------
    async def scene_writer_agent(self, state: dict) -> dict:
        scene_memory = state.scene_memory
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
            "If the scene is complete, just write the next paragraph of the scene."
        )

        human_prompt = f"""
        Director's Instructions:
        {scene_memory.DirectorInstructions}

        {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {scene_memory.scene_so_far}" if scene_memory.scene_so_far else ""}

        {f"Your Question: {scene_memory.ai_question}" if scene_memory.ai_question else ""}
        {f"(The user chose: {scene_memory.UserInput})" if scene_memory.UserInput else ""}

        {scene_writer_parser.get_format_instructions()}
        """
        print("SCENE WRITER HUMAN PROMPT: ", human_prompt)

        llm_response = self.llm.groq_client(system_prompt=system_prompt, human_prompt=human_prompt)
        clean_resp = llm_response.content.strip()

        # Try parsing
        try:
            parsed = scene_writer_parser.parse(clean_resp)
            scene_text = parsed.scene
            question_text = parsed.question
        except Exception as e:
            print("SceneWriter parsing failed:", e)

            # Retry with stricter reminder
            retry_prompt = human_prompt + "\n\nREMEMBER: Output ONLY valid JSON strictly matching schema."
            retry_resp = self.llm.groq_client(system_prompt=system_prompt, human_prompt=retry_prompt)
            retry_clean = retry_resp.content.strip()
            try:
                parsed = scene_writer_parser.parse(retry_clean)
                scene_text = parsed.scene
                question_text = parsed.question
            except Exception as retry_e:
                print("SceneWriter retry parsing failed:", retry_e)

                # Try regex-based JSON recovery
                match = re.search(r'(\{[\s\S]*\})', retry_clean)
                if match:
                    try:
                        recovered = json.loads(match.group(1))
                        scene_text = recovered.get("scene", "")
                        question_text = recovered.get("question", "")
                    except Exception as inner_e:
                        print("SceneWriter JSON recovery failed:", inner_e)
                        scene_text = retry_clean
                        question_text = ""
                else:
                    # fallback: plain text
                    scene_text = retry_clean
                    question_text = ""


        # --- update scene memory ---
        if scene_memory.UserInput:
            scene_memory.scene_so_far_for_scene_planner += f"(The user chose: {scene_memory.UserInput})\n"

        scene_memory.scene_so_far += " " + scene_text + "\n\n"
        scene_memory.scene_so_far_for_scene_planner += scene_text + "\n"

        if question_text.strip():
            scene_memory.scene_so_far_for_scene_planner += f"(The scene writer asked: {question_text.strip()})\n"
            scene_memory.ai_question = question_text.strip()
            self.question_boolean = True
        else:
            scene_memory.ai_question = ""
            self.question_boolean = False

        state.scene_memory = scene_memory

        # Stream the scene chunk back
        self.scene_chunk_callback(json.dumps({"scene_chunk": scene_text}) + "\n")

        return state


    # ------------------------
    # Run full scene
    # ------------------------
    async def run(self, state: dict, scene_chunk_callback) -> str:
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

        scene_memory = result["scene_memory"]

        if result.get("next_node") == "END":
            print("Reached END. Stopping execution.")
            return scene_memory.scene_so_far

        return scene_memory.scene_so_far


