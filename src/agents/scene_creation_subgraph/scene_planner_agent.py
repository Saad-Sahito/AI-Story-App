import asyncio
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.store.memory import InMemoryStore
from llm_client.llm_client import LLMClient

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

        system_prompt = "You are a scene context guard. See if the scene blueprint is completed, disregard the decision points. " \
        "Output strictly according to schema."
        human_prompt = f"""
        Director Instructions:
        {scene_memory.DirectorInstructions}

        Scene So Far:
        {scene_memory.scene_so_far}

        {scene_planner_parser.get_format_instructions()}
        """

        llm_response = self.llm.gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
        clean_resp = llm_response.content.strip()
        
        try:
            parsed = scene_planner_parser.parse(clean_resp)
            next_node = "Complete" if parsed.action.lower() == "complete" else "Not Complete"
        except Exception as e:
            print("ScenePlanner parsing failed:", e)
            next_node = "Not Complete"

        print("SCENE PLANNER RESPONSE: ", next_node)
        #print("QUESTION: ", scene_memory.ai_question)
        if next_node == "Not Complete":
            if self.question_boolean:
                
                await self.scene_chunk_callback(f"<DECISION_POINT>{scene_memory.ai_question.strip()}")
                user_choice = await self.wait_for_user_input()
                scene_memory.UserInput = user_choice
                state.scene_memory = scene_memory.model_dump()

            else:
                # No question to ask, just continue
                scene_memory.UserInput = ""
                state.scene_memory = scene_memory.model_dump()
                
        state.next_node = next_node
        return state

    # ------------------------
    # Scene Writer
    # ------------------------
    async def scene_writer_agent(self, state: dict) -> dict:
        scene_memory = state.scene_memory
        #print("SCENE MEMORY AT START OF SCENE WRITER: ", scene_memory.scene_so_far)
        system_prompt = "You are the Scene Writer Agent. Follow schema strictly. " \
        "The Director's Instructions will include a recap of the story so far, " \
        "relevant characters/worlds to the current scene, and the scene blueprint that you need to follow strictly for this scene. " \
        "You do not know anything beyond what the Director tells you, so be sure to include all relevant context in your writing. " \
        "Write in a vivid, engaging style, with rich descriptions and immersive details. " \
        "If the Director's Instructions include a decision point question for the user, end your paragraph output with that question. " \
        "Do not make up any new characters or worlds that the Director has not mentioned. " \
        "Do not repeat the entire scene so far, only continue it with one new paragraph. " \
        "If the scene is complete, just write the next paragraph of the scene. " 


        human_prompt = f"""
    Director's Instructions:
    {scene_memory.DirectorInstructions}

    {f"Scene so far (DO NOT rewrite this, only output text that continues from here): {scene_memory.scene_so_far}" if scene_memory.scene_so_far else ""}

    {f"Your Question: {scene_memory.ai_question}" if scene_memory.ai_question else ""}
    {f"(The user chose: {scene_memory.UserInput})" if scene_memory.UserInput else ""}

    {scene_writer_parser.get_format_instructions()}
    """
        #print("SCENE WRITER HUMAN PROMPT: ", human_prompt)
        llm_response = self.llm.gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
        clean_resp = llm_response.content.strip()
        #print("SCENE WRITER RESPONSE: ", clean_resp)

        try:
            parsed = scene_writer_parser.parse(clean_resp)
            scene_text = parsed.scene
            question_text = parsed.question
        except Exception as e:
            print("SceneWriter parsing failed:", e)
            scene_text = clean_resp
            question_text = ""

        scene_memory: SceneMemory = state.scene_memory

        #if scene_text.strip():
        scene_memory.scene_so_far += " " + scene_text + "\n"
        scene_memory.ai_question = question_text if question_text.strip() else ""
        state.scene_memory = scene_memory   # keep object, not dict
        await self.scene_chunk_callback(scene_text)

        if question_text.strip():
            self.question_boolean = True
        #     await self.scene_chunk_callback(f"<DECISION_POINT>{question_text.strip()}")
        #     user_choice = await self.wait_for_user_input()
        #     scene_memory.UserInput = user_choice
        #     state.scene_memory = scene_memory.model_dump()
            #state.messages.append(HumanMessage(content=f"(User chose: {user_choice})"))
            # next_node = "SceneWriter"
        else:
            self.question_boolean = False
        #     next_node = "ScenePlanner"
        #     scene_memory.UserInput = ""
        #     state.scene_memory = scene_memory.model_dump()

        # state.next_node = next_node
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


