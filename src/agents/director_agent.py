# src/agents/director_agent.py - MODIFIED VERSION

import json
import gc
from typing import Any, Dict, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from dataclasses import dataclass, field

from src.memory.memory_system import StoryMemorySystem

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from src.utilities.story_helpers import StoryHelpers

# Import the shared scene planner service and context
from .scene_creation_subgraph.shared_scene_planner import ( 
    UserSceneContext
)
import src.agents.scene_creation_subgraph.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import groq_client, gemini_client

# Keep existing state and models unchanged
@dataclass
class StoryState:
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    word_count: int = 0
    messages: List[Any] = field(default_factory=list)
    next_action: str = ""

class SceneBundle(BaseModel):
    story_summary: str = Field(
        description="Detailed but short summary of the scene."
    )
    character_details: Dict[str, str] = Field(
        description="Dictionary: {character_name: details about traits/actions/motivations this scene inline all of it str not dict, also mention the chapter/scene number} "
    )
    world_details: Dict[str, str] = Field(
        description="Dictionary: {world_element: atmosphere, culture, or environment details this scene inline all of it str not dict, also mention the chapter/scene number} "
    )

scene_parser = PydanticOutputParser(pydantic_object=SceneBundle)

class ChapterBundle(BaseModel):
    summary: str = Field(
        description="A detailed but short summary of the entire chapter."
    )
    character_summary: Dict[str, str] = Field(
        description="Dictionary: {character_name: summary of the character regarding their traits/actions/motivations inline all of it str not dict.} "
    )
    world_summary: Dict[str, str] = Field(
        description="Dictionary: {world_element: summary of the world element regarding its atmosphere/culture/environment inline all of it str not dict.} "
    )

chapter_parser = PydanticOutputParser(pydantic_object=ChapterBundle)

class DirectorOutput(BaseModel):
    instructions: str = Field(
        description="200–500 words of detailed scene instructions. Do not make a nested dictionary, just plain string type text."
    )
    action: Literal["generate_and_ingest", "END"] = Field(
        description="Action to take after instructions."
    )

director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)

# Keep Ingestor class unchanged (it's already fine)
class Ingestor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    # ---- Scene Ingestion (this is part of the Ingestor class, not DirectorGraph) ----
    def ingest_scene(self, state: StoryState, scene_text: str, max_retries: int = 3) -> Dict[str, str]:
        """Ingest scene text and extract structured JSON using schema + parser."""
        system_prompt = "You are the Scene Breakdown Agent. Extract structured info from the scene. " \
        "Always include chapter and scene id in character and world details, in order to keep track later. " \
        "Make sure the character and world names are exactly as the keys presented to you under Character and World Names, " \
        "if any need to be changed then create new entry for that entity mentioning previous name in the new entry, " \
        "if not present then create new names as needed."

        human_prompt = f"""
        Current Chapter: {state.current_chapter_id}, Current Scene: {state.scene_id}

        Scene:
        {scene_text}

        Character and World Names:
        Characters:
        {self.memory.get_long_term_characters().keys()}

        Worlds:
        {self.memory.get_long_term_worlds().keys()}

        Respond ONLY in JSON with this schema:
        {scene_parser.get_format_instructions()}
        """
        
        print("INGEST SCENE HUMAN PROMPT: ", human_prompt)

        for attempt in range(1, max_retries + 1):
            resp = gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()
            print(f"[Attempt {attempt}] RAW INGEST SCENE RESPONSE:", clean_resp)

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            print("TYPE OF CLEAN_RESP:", type(clean_resp))

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, SceneBundle, scene_parser)
            
            if success:
                state.word_count += StoryHelpers._count_words_split(scene_text)
                state.scene_id += 1
                self.memory.update_story_progress(metadata={
                    "latest_chapter_id": state.current_chapter_id, 
                    "continue_scene_id": state.scene_id, 
                    "word_count": state.word_count, 
                    "story_title": state.story_title
                })
                del system_prompt, human_prompt
                return result

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            try:
                fixed_resp = StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, SceneBundle, scene_parser)
                del fixed_clean
                if success:
                    state.word_count += StoryHelpers._count_words_split(scene_text)
                    state.scene_id += 1
                    self.memory.update_story_progress(metadata={
                        "latest_chapter_id": state.current_chapter_id, 
                        "continue_scene_id": state.scene_id, 
                        "word_count": state.word_count, 
                        "story_title": state.story_title
                    })
                    del system_prompt, human_prompt, scene_text
                    return result

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised an exception:", inner_e)

            if attempt < max_retries:
                print(f"Retrying ingest_scene... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted for ingest_scene; returning minimal safe structure.")
                del system_prompt, human_prompt, scene_text
                return {
                    "summary": "",
                    "character_summary": {},
                    "world_summary": {}
                }
        
        del system_prompt, human_prompt

    def ingest_chapter(self, state: StoryState, current_chap_summary, max_retries: int = 3) -> Dict[str, Any]:
        """Summarize and extract structured details about a full chapter.
        Retries with LLM if parse_obj + parse + json_fixer all fail.
        """
        print("Ingesting chapter...")

        #chapter_content = self.memory.get_current_chapter()
        world_details = self.memory.get_long_term_worlds()
        char_details = self.memory.get_long_term_characters()
        
        # self.memory.add_story_chapter(
        #     text=chapter_content,
        #     metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title}
        # )

        system_prompt = (
            "You are the Chapter Breakdown Agent. Extract structured info from the "
            "chapter content and character details and world details. "
            "The character and world names should be exactly as the keys presented to you under Character and World Details, "
            "separated by Character and Worlds respectively. "
            "Always include chapter id in character and world details, in order to keep track later."
        )

        human_prompt = f"""
        Current Chapter: {state.current_chapter_id}
        Chapter Text:
        {current_chap_summary}

        Character Details:
        {char_details}

        World Details:
        {world_details}

        Respond ONLY in JSON with this schema:
        {chapter_parser.get_format_instructions()}
        """

        for attempt in range(1, max_retries + 1):
            resp = gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            #print(f"[Attempt {attempt}] RAW INGEST CHAPTER RESPONSE:", clean_resp)
            del raw_text, resp
            gc.collect()
            # make sure we have a string
            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            #print("TYPE OF CLEAN_RESP:", type(clean_resp))

            # 1) try model_validate/parse_obj first, then parser
            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, ChapterBundle, chapter_parser)
            if success:
                self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                state.current_chapter_id += 1
                state.scene_id = 1
                self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count})
                #self.memory.close()
                print("Chapter Complete!")
                result.update({
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    "word_count": state.word_count,
                })
                del system_prompt, human_prompt
                return result

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            # 2) try json_fixer, then same validation sequence
            try:
                fixed_resp = StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del clean_resp, fixed_resp
                gc.collect()
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, ChapterBundle, chapter_parser)
                del fixed_clean
                if success:
                    self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                    state.current_chapter_id += 1
                    state.scene_id = 1
                    self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count})
                    #self.memory.close()
                    print("Chapter Complete!")
                    result.update({
                        "current_chapter_id": state.current_chapter_id,
                        "scene_id": state.scene_id,
                        "word_count": state.word_count,
                    })
                    del system_prompt, human_prompt
                    return result

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised an exception:", inner_e)

            # if not last attempt, retry LLM
            if attempt < max_retries:
                print(f"Retrying ingest_chapter... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted for ingest_chapter; returning minimal safe structure.")
                del system_prompt, human_prompt
                return {
                    "summary": "",
                    "character_summary": {},
                    "world_summary": {},
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    "word_count": state.word_count,
                }
        del system_prompt, human_prompt

    


# Modified DirectorGraph to use shared scene planner
class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system
        
        # NO MORE sceneplanner instance per user!
        # We'll use the global shared instance
        
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("director_node")
        self.graph.add_node("director_node", self.director_node)
        self.graph.add_node("generate_and_ingest", self.generate_and_ingest_node)
        self.graph.add_node("ingest_chapter", self.ingest_chapter)

        self.graph.add_conditional_edges(
            "director_node",
            lambda state: state.next_action,
            {
                "generate_and_ingest": "generate_and_ingest",
                "END": "ingest_chapter",
            },
        )

        self.graph.add_edge("generate_and_ingest", "director_node")
        self.graph.add_edge("ingest_chapter", END)

        self.compiled = self.graph.compile()
        self.current_chap_summary = ""

    def _get_latest_director_message(self, state: StoryState) -> str:
        """Extract the latest director instructions from state messages"""
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                return msg.content
        return ""

    async def generate_and_ingest_node(self, state: StoryState):
        """Generates a new scene using shared scene planner and ingests it"""
        print("Called Generate and Ingest Node!")
        
        # Get the latest director instructions
        director_instructions = self._get_latest_director_message(state)
        
        # Create user-specific context for this scene generation
        user_context = UserSceneContext.create_for_user(
            user_id=self.memory.user_id,
            story_id=self.memory.story_id,
            director_instructions=director_instructions,
            scene_chunk_callback=self.scene_chunk_callback
        )
        
        

        
        if scene_planner_module.SHARED_SCENE_PLANNER_SERVICE is None:
            print("❌ ERROR: SHARED_SCENE_PLANNER_SERVICE is None!")
            raise
        
        director_instructions = self._get_latest_director_message(state)

        
        user_context = UserSceneContext.create_for_user(
            user_id=self.memory.user_id,
            story_id=self.memory.story_id,
            director_instructions=director_instructions,
            scene_chunk_callback=self.scene_chunk_callback
        )
        
        print(f"🔍 DEBUG: Calling run_scene for {user_context.user_id}/{user_context.story_id}")
        scene_text, scene_cluster = await scene_planner_module.SHARED_SCENE_PLANNER_SERVICE.run_scene(user_context)
    
        # Clean up user context (it's no longer needed)
        del user_context
        gc.collect()

        # Create Ingestor only for this ingestion
        ingestor = Ingestor(self.memory)
        scene_bundle = ingestor.ingest_scene(state, scene_text)
        del ingestor
        gc.collect()
        
        self.memory.add_story_scene_cluster(text=scene_cluster, metadata={
            "chapter_id": state.current_chapter_id, 
            "story_title": state.story_title, 
            "scene_id": state.scene_id, 
            "word_count": state.word_count
        })
        
        self.memory.add_post_scene_bundle(
            scene_bundle=scene_bundle,
            full_scene_text=scene_text,
            metadata={
                "scene_id": state.scene_id,
                "chapter_id": state.current_chapter_id,
                "story_title": state.story_title
            }
        )
        
        del scene_bundle, scene_cluster, scene_text
        gc.collect()

        # Return the final state for the next node
        yield {
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
        }

    def ingest_chapter(self, state: StoryState):
        ingestor = Ingestor(self.memory)
        result = ingestor.ingest_chapter(state, self.current_chap_summary)
        self.current_chap_summary = ""
        del ingestor
        gc.collect()
        return result

    async def director_node(self, state: StoryState) -> Dict:
        """Decide the next scene or end the chapter, using schema parsing with retries."""
        system_prompt = (
            "You are the Director Agent for an interactive text-based novel. "
            "You must create exhaustive, prescriptive instructions for the Scene Writer agent. "
            "The Scene Writer will write ONLY what you specify — it has no memory of past scenes and no freedom to improvise. "
            "Therefore, you must make EVERY creative decision. "
            "Do not use vague descriptions, do not leave placeholders, and do not rely on the Scene Writer to 'fill in the gaps.' "
            "All beats, dialogue, character reactions, and world details must be fully defined by you. "
            "If something is unclear, you must decide it yourself. "
            "The Scene Writer should NEVER invent characters, settings, dialogue, decision points, or events. "
            "Prohibit generic phrasing such as 'mundane small talk,' 'subtle hints,' or 'something happens.' Always give exact lines or examples. "
            "Your output must be in the given JSON format. "
            "The 'instructions' value must be 200-500 words string, not a dictionary and contain the following sections:\n\n"
            "1. **Recap** - A concise summary of the story so far. Be concrete, include all key facts the Scene Writer needs. \n"
            "2. **Characters** - List all relevant characters with names, ages, traits, and current state of mind. If a side character appears, provide their exact role and tone. \n"
            "3. **Detailed Scene Blueprint** - A numbered, step-by-step breakdown of the scene's beats in strict order. Each beat must describe: location, action, at least one visual detail, at least one sound detail, suggest general dialogue idea. "
            "Do not allow ambiguity. Do not say 'the writer should show this.' You must say 'this happens, in this way.' \n"
            "4. **Main Character's (User) Decision Points** - Describe 1-2 explicit points in the scene where the scene writer prompts the user to make a choice, either dialogue or action.\n"
            "5. **Screenplay Notes** - A strict checklist of required elements (e.g., 'Include one description of neon reflection on glass,' 'Include two internal monologue lines showing anxiety'). "
            "These are mandatory, not suggestions. \n"
            "6. **Chapter/Scene ID** - Exact chapter and scene number. \n\n"
            "Always be concrete, exhaustive, and prescriptive. "
            "Never leave the Scene Writer to guess or invent. "
            "Call END if you think the chapter should end now. "
            "Make sure the output JSON is valid and contains no extra text."
        )

        # Gather story context
        self.current_chap_summary = self.memory.search_episodic_story_summary(chapter_number=state.current_chapter_id)
        director_context = self.memory.get_director_context(
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=5
        )
        context = (
            f"Story Premise: {self.memory.get_long_term_document('story_premise')}\n"
            f"Relevant Chapter Context: {director_context}\n"
            f"Chapter Number: {state.current_chapter_id}\n"
            f"Scene Number: {state.scene_id}\n"
            f"Current Chapter So Far Summary: {self.current_chap_summary}\n"
            
            
        )
        
        print("CONTEXT TO DIRECTOR:", context)
        human_prompt = f"""
        {context}

        {director_parser.get_format_instructions()}
        """
        max_retries = 3
        scenario, action = None, None

        for attempt in range(1, max_retries + 1):
            resp = await groq_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()
            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            # Try validate/parse and get dict back
            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, DirectorOutput, director_parser)
            if success:
                scenario = result.get("instructions")
                action = result.get("action")
                break

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            # json_fixer attempt
            try:
                fixed_resp = StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, DirectorOutput, director_parser)
                del fixed_clean, fixed_resp
                gc.collect()
                if success:
                    scenario = result.get("instructions")
                    action = result.get("action")
                    break

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised an exception:", inner_e)
                del clean_resp
                gc.collect()

            if attempt < max_retries:
                print(f"Retrying director_node... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted for director_node; falling back to raw response and END.")
                scenario = clean_resp if 'clean_resp' in locals() else "Error generating instructions"
                action = "END"
        
        # Clean up variables
        if 'clean_resp' in locals():
            del clean_resp
        del system_prompt, human_prompt
        gc.collect()
        
        if state.scene_id == 2:  # DEBUGGING Code
            action = "DEBUG"  # DEBUGGING Code

        # finalize and return
        messages = state.messages or []
        messages.append(AIMessage(content=scenario))

        return {
            "messages": messages,
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
            "next_action": "generate_and_ingest" if action == "generate_and_ingest" else "END",
        }

    async def run(self, scene_chunk_callback):
        print("Running director agent...")
        self.scene_chunk_callback = scene_chunk_callback
        story_progress = self.memory.get_story_progress()
        
        if story_progress:
            initialized_state = StoryState(
                current_chapter_id=story_progress.get("latest_chapter_id", 1),
                scene_id=story_progress.get("continue_scene_id", 1),
                story_title=story_progress.get("metadata", {}).get("story_title", "None"),
                word_count=story_progress.get("word_count", 0),
                messages=[],
                next_action=""
            )
        else:
            initialized_state = StoryState(
                current_chapter_id=1,
                scene_id=1,
                story_title="None",
                word_count=0,
                messages=[],
                next_action=""
            )
        
        result = await self.compiled.ainvoke(initialized_state, {"recursion_limit": 50})
        print("Director Node Finished: ", result)
        return result