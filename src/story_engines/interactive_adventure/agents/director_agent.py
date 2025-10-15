# src/agents/director_agent.py
import asyncio
import json
import gc
from typing import Any, Dict, List, Literal, Optional, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
from .shared_scene_planner import UserSceneContext

import src.story_engines.interactive_adventure.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import groq_client, ingestor_gemini_client

INTERACTIVE_DIRECTOR_AGENT = None

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
        description="Detailed summary of the scene."
    )
    character_details: Dict[str, str] = Field(
        description="Dictionary: {character_name: details about traits/actions/motivations this scene inline all of it str not dict, also mention the chapter/scene number} "
    )
    world_details: Dict[str, str] = Field(
        description="Dictionary: {world_element: atmosphere, culture, or environment details this scene inline all of it str not dict, also mention the chapter/scene number} "
    )

scene_parser = PydanticOutputParser(pydantic_object=SceneBundle)

class CharacterMemory(BaseModel):
    name: str
    chapter_id: str
    summary: str = Field(description="Inline summary of the character's actions/motivations/changes upto this chapter.")
    traits: List[str] = Field(default_factory=list, description="Key personality traits expressed upto this chapter.")
    relationships: Dict[str, str] = Field(default_factory=dict, description="Map of other characters and current relationship status or changes.")
    emotional_state: Optional[str] = Field(None, description="Dominant emotion or mindset towards the end.")
    goals: Optional[str] = Field(None, description="Current goals or motivations going forward.")
    status_changes: Optional[str] = Field(None, description="Any physical, social, or narrative changes (e.g., wounded, promoted, betrayed).")

class WorldElementMemory(BaseModel):
    name: str
    chapter_id: str
    summary: str = Field(description="Summary of how this world element appeared or changed upto this chapter.")
    atmosphere: Optional[str] = Field(None, description="Mood or tone of this location or environment.")
    culture: Optional[str] = Field(None, description="Cultural or societal information revealed upto this chapter.")
    events: Optional[str] = Field(None, description="Notable events or changes affecting this location.")
    connections: Union[Dict[str, str], str] = Field(default_factory=dict, description="Links or relations to other world elements or characters.")

class ChapterBundle(BaseModel):
    summary: str = Field(description="Detailed summary of the entire chapter.")
    character_summary: Dict[str, CharacterMemory] = Field(description="Dictionary: {character_name: structured character memory object}")
    world_summary: Dict[str, WorldElementMemory] = Field(description="Dictionary: {world_element: structured world memory object}")

chapter_parser = PydanticOutputParser(pydantic_object=ChapterBundle)

class DirectorOutput(BaseModel):
    instructions: str = Field(
        description="200-500 words of detailed scene instructions. Do not make a nested dictionary, just plain string type text."
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
    async def ingest_scene(self, state: StoryState, scene_text: str, llm_temp: float, max_retries: int = 3) -> Dict[str, str]:
        """Ingest scene text and extract structured JSON using schema + parser."""
        chars = await self.memory.get_long_term_characters()
        worlds = await self.memory.get_long_term_worlds()
        system_prompt = "You are the Scene Breakdown Agent. Extract structured info from the scene. Including indicated references to user choices. " \
        "Always include chapter and scene id in character and world details, in order to keep track later. " \
        "Make sure the character and world names are exactly as the keys presented to you under Character and World Names, " \
        "if any need to be changed then create new entry for that entity mentioning previous name in the new entry, " \
        "if not present then create new names as needed."        
        "Respond ONLY in JSON with this schema:"
        f"{scene_parser.get_format_instructions()}"

        human_prompt = f"""
        Current Chapter: {state.current_chapter_id}, Current Scene: {state.scene_id}

        Scene:
        {scene_text}

        Character and World Names:
        Characters:
        {chars.keys()}

        Worlds:
        {worlds.keys()}
        """
        
        #print("INGEST SCENE HUMAN PROMPT: ", human_prompt)

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = StoryHelpers._extract_content(resp)
            clean_resp = StoryHelpers._strip_code_fences(raw_text)
            del raw_text, resp
            gc.collect()
            #print(f"[Attempt {attempt}] RAW INGEST SCENE RESPONSE:", clean_resp)

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            #print("TYPE OF CLEAN_RESP:", type(clean_resp))

            success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, SceneBundle, scene_parser)
            
            if success:
                state.word_count += StoryHelpers._count_words_split(scene_text)
                
                await self.memory.update_story_progress(metadata={
                    "latest_chapter_id": state.current_chapter_id, 
                    "continue_scene_id": state.scene_id + 1, 
                    "word_count": state.word_count, 
                    "story_title": state.story_title,
                    "tone_temp": llm_temp
                })
                del system_prompt, human_prompt
                return result

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            try:
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del fixed_resp, clean_resp
                gc.collect()
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, SceneBundle, scene_parser)
                del fixed_clean
                if success:
                    state.word_count += StoryHelpers._count_words_split(scene_text)
                    
                    await self.memory.update_story_progress(metadata={
                        "latest_chapter_id": state.current_chapter_id, 
                        "continue_scene_id": state.scene_id + 1, 
                        "word_count": state.word_count, 
                        "story_title": state.story_title,
                        "tone_temp": llm_temp
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

    async def ingest_chapter(self, state: StoryState, current_chap_summary, llm_temp: float, max_retries: int = 3) -> Dict[str, Any]:
        """Summarize and extract structured details about a full chapter.
        Retries with LLM if parse_obj + parse + json_fixer all fail.
        """
        print("Ingesting chapter...")

        #chapter_content = self.memory.get_current_chapter()
        world_details = await self.memory.get_long_term_worlds()
        char_details = await self.memory.get_long_term_characters()

        system_prompt = f"""
You are the **Chapter Breakdown Agent**.

Your job:
Analyze the *entire chapter text* and generate a structured breakdown summarizing story events,
character developments, and world details.

---

### 🔧 Instructions
1. Produce a detailed **chapter summary** under the field `"summary"`.
2. For each **character** listed in "Character Details", output a structured object using **exactly** the following fields:
   - name
   - chapter_id
   - summary
   - traits
   - relationships
   - emotional_state
   - goals
   - status_changes
3. For each **world element** listed in "World Details", output a structured object using **exactly** the following fields:
   - name
   - chapter_id
   - summary
   - atmosphere
   - culture
   - events
   - connections
4. Always preserve **exact names** and **only use information provided in the input**.
5. Do **not** add commentary, markdown, explanations, or text outside the JSON.

---

### ⚙️ Output Format
Return output **strictly as a JSON object** that conforms exactly to this schema:

{chapter_parser.get_format_instructions()}

The output **must be valid JSON**, not inside code fences, with no trailing commas or text before/after.
"""

        human_prompt = f"""
        Current Chapter: {state.current_chapter_id}
        Chapter Text:
        {current_chap_summary}

        Character Details:
        {char_details}

        World Details:
        {world_details}        
        """

        for attempt in range(1, max_retries + 1):
            resp = await ingestor_gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
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
                await self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                state.current_chapter_id += 1
                state.scene_id = 1
                await self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count, "tone_temp": llm_temp})
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
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                del clean_resp, fixed_resp
                gc.collect()
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(fixed_clean, ChapterBundle, chapter_parser)
                del fixed_clean
                if success:
                    await self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                    state.current_chapter_id += 1
                    state.scene_id = 1
                    await self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count, "tone_temp": llm_temp})
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

    
class DirectorGraph:
    def __init__(self, memory_system: StoryMemorySystem):
        self.memory = memory_system
        
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("chapter_director_node")
        self.graph.add_node("chapter_director_node", self.chapter_director_node)
        self.graph.add_node("scene_director_node", self.scene_director_node)
        self.graph.add_node("generate_and_ingest", self.generate_and_ingest_node)
        self.graph.add_node("ingest_chapter", self.ingest_chapter)

        self.graph.add_edge("chapter_director_node", "scene_director_node")
        self.graph.add_conditional_edges(
            "scene_director_node",
            lambda state: state.next_action,
            {
                "generate_and_ingest": "generate_and_ingest",
                "END": "ingest_chapter",
            },
        )
        # Check if user wants to end or continue
        def route_after_generate(state: StoryState):
            """Route based on whether user chose to continue"""
            if state.next_action == "END":
                return END  # End immediately without ingesting chapter
            return "scene_director_node"
        
        self.graph.add_conditional_edges(
            "generate_and_ingest",
            route_after_generate,
            {
                "scene_director_node": "scene_director_node",
                END: END  # Direct route to END
            }
        )
        self.graph.add_edge("ingest_chapter", END)

        self.compiled = self.graph.compile()
        self.current_chap_summary = ""
        self.llm_temp = 0.7  # Default temperature, can be adjusted based on story tone

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
        #director_instructions = self._get_latest_director_message(state)
        
        if scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE is None:
            print("❌ ERROR: INTERACTIVE_SCENE_PLANNER_SERVICE is None!")
            raise
        
        director_instructions = self._get_latest_director_message(state)

        # Create user-specific context for this scene generation
        user_context = UserSceneContext.create_for_user(
            user_id=self.memory.user_id,
            story_id=self.memory.story_id,
            director_instructions=director_instructions,
            scene_chunk_callback=self.scene_chunk_callback
        )
        
        print(f"🔍 DEBUG: Calling run_scene for {user_context.user_id}/{user_context.story_id}")
        scene_text, scene_cluster, status = await scene_planner_module.INTERACTIVE_SCENE_PLANNER_SERVICE.run_scene(
            user_context=user_context, 
            stop_event=self.stop_event
        )
        print("✅ Scene generation complete.")

        if status == "CANCELLED":
            # 🛑 Scene generation was cancelled by user or system
            # Add your cleanup or UI update logic here
            pass

        elif status.startswith("FATAL:"):
            status_payload = { "type": "status", "FATAL": status }
            try:
                self.scene_chunk_callback(status_payload)
            except Exception as e:
                print(f"⚠️ scene_chunk_callback raised: {e}")
            
        elif status.startswith("EXCEPTION:"):
            status_payload = { "type": "status", "EXCEPTION": status }
            try:
                self.scene_chunk_callback(status_payload)
            except Exception as e:
                print(f"⚠️ scene_chunk_callback raised: {e}")
            
        elif status == "SUCCESS":
            del user_context
            gc.collect()

            from setup.shared_redis_pool import get_redis_client
            try:
                # Try to find the user/session-level structures
                redis_client = await get_redis_client()
                queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"

                # Build callback payload
                resume_payload = {
                    "type": "save"
                }
                try:
                    self.scene_chunk_callback(resume_payload)
                except Exception as e:
                    print(f"❌ ERROR: scene_chunk_callback raised: {e}")
                    import traceback; traceback.print_exc()

                # Wait for user input from Redis List
                try:
                    user_choice = False
                    for _ in range(5000):
                        choice = await redis_client.lpop(queue_key)
                        if choice == b'1' or choice == 1 or choice == '1':  # ✅ Check both byte and int
                            user_choice = True
                            print("✅ User chose to save continue")
                            break
                        elif choice is not None:  # ✅ Got a response but not "1"
                            break
                        await asyncio.sleep(1.0)
                    
                    # ✅ FIX: Return dict with flag instead of END
                    if user_choice == False:
                        print("🛑 User chose not to continue")
                        return {
                            "scene_id": state.scene_id,
                            "current_chapter_id": state.current_chapter_id,
                            "word_count": state.word_count,
                            "next_action": "END",  # Signal to stop via state
                        }
                except Exception as e:
                    print(f"❌ ERROR in Redis queue handling: {e}")
                    import traceback; traceback.print_exc()
                    # Return state with END action on error
                    return {
                        "scene_id": state.scene_id,
                        "current_chapter_id": state.current_chapter_id,
                        "word_count": state.word_count,
                        "next_action": "END",
                    }
            except Exception as e:
                print(f"❌ ERROR in user input handling: {e}")
                import traceback; traceback.print_exc()
                # Return state with END action on error
                return {
                    "scene_id": state.scene_id,
                    "current_chapter_id": state.current_chapter_id,
                    "word_count": state.word_count,
                    "next_action": "END",
                }

            # Save Story    
            # Create Ingestor only for this ingestion
            ingestor = Ingestor(self.memory)
            scene_bundle = await ingestor.ingest_scene(state, scene_text, llm_temp=self.llm_temp)
            del ingestor
            gc.collect()
            
            await self.memory.add_story_scene_cluster(text=scene_cluster, metadata={
                "chapter_id": state.current_chapter_id, 
                "story_title": state.story_title, 
                "scene_id": state.scene_id, 
                "word_count": state.word_count
            })
            
            await self.memory.add_post_scene_bundle(
                scene_bundle=scene_bundle,
                metadata={
                    "scene_id": state.scene_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "type": "scene summary"
                }
            )
            
            del scene_bundle, scene_cluster, scene_text
            gc.collect()

            # ✅ Return state WITHOUT next_action or with empty string
            # This allows the conditional edge to route back to director_node
            return {
                "scene_id": state.scene_id + 1,
                "current_chapter_id": state.current_chapter_id,
                "word_count": state.word_count,
            }
        else:
            # ⚠️ Unknown status (fallback safeguard)
            return {
                "scene_id": state.scene_id,
                "current_chapter_id": state.current_chapter_id,
                "word_count": state.word_count,
                "next_action": "END",
            }
    
    async def ingest_chapter(self, state: StoryState):
        ingestor = Ingestor(self.memory)
        result = await ingestor.ingest_chapter(state, self.current_chap_summary, llm_temp=self.llm_temp)
         # Reset current chapter summary after ingestion
        self.current_chap_summary = ""
        del ingestor
        gc.collect()
        return result

    async def chapter_director_node(self, state: StoryState) -> Dict:
        """Decide the next chapter or end the story, using schema parsing with retries."""
        print("🎬 Running chapter director agent...")
        # Similar to scene_director_node but for chapters
        system_prompt = (
    "You are the Chapter Director for an interactive story. "
    "Your job is to plan the overall direction and structure of this chapter. "
    "You do not write scenes directly — you design a compact blueprint that the Scene Director will later follow scene-by-scene.\n\n"
    
    "Use the story so far and the user's most recent choice to determine how this chapter should develop emotionally, thematically, and narratively.\n\n"
    
    "Your output must be in JSON format and include:\n"
    "1. chapter_number - Sequential number of the new chapter.\n"
    "2. chapter_title - A short, descriptive title.\n"
    "3. narrative_goal - What this chapter must accomplish in the story (e.g., 'Protagonist discovers betrayal').\n"
    "4. emotional_arc - The emotional progression of the chapter (e.g., 'tension → shock → resolve').\n"
    "5. key_conflicts - The central struggles or decisions faced.\n"
    "6. closure_condition - The condition under which the chapter should end (e.g., 'When the hero escapes the castle').\n"
    "7. tone_guidelines - Notes on tone, pacing, and atmosphere.\n"
    "8. expected_scenes - Estimated number of scenes for pacing reference.\n\n"
    
    "Keep this concise but detailed enough that a Scene Director can plan and execute each scene from it."
)
        # Gather story context
        if state.current_chapter_id > 1:
            self.current_chap_summary = await self.memory.search_episodic_scene_summary(chapter_number=state.current_chapter_id-1, summary_type="scene summary")
            director_context = await self.memory.get_director_context(
                current_chapter_number=state.current_chapter_id,
                query=self.current_chap_summary if self.current_chap_summary else "",
                k=5
            )

        context = (f"""
            Chapter Number: {state.current_chapter_id}
            Story Premise: {await self.memory.get_long_term_document(metadata={"type":"story_premise", "story_title": state.story_title})}
            {f"Relevant Chapter Context: {director_context}" if state.current_chapter_id > 1 else "Start of Story"}
        """)
            #f"Scene Number: {state.scene_id}\n"
            #f"Current Chapter So Far Summary: {self.current_chap_summary}\n"
        human_prompt = f"""
        {context}
        """
        if await self.memory.get_long_term_document(metadata={"type": "chapter_plan", "chapter_id": state.current_chapter_id, "story_title": state.story_title}) == "":
            print("CONTEXT TO CHAPTER DIRECTOR:", context)
            for attempt in range(1, 3):
                resp = await groq_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp)
                if resp is None:
                    print(f"Retrying chapter_director_node... (attempt {attempt+1})")
                    continue
                else:
                    await self.memory.add_long_term_document(text=resp.content, metadata={"chapter_id": state.current_chapter_id ,"type":"chapter_plan", "story_title": state.story_title})
                    del resp
                    break

            del system_prompt, human_prompt, context
            gc.collect()

        return {
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
        }

    async def scene_director_node(self, state: StoryState) -> Dict:
        """Decide the next scene or end the chapter, using schema parsing with retries."""
        print("🎬 Running scene director agent...")
        system_prompt = (
    "You are the Scene Director for an interactive story. "
    "You work under a Chapter Blueprint that defines the chapter's purpose, emotional arc, and closure condition.\n\n"
    
    "Your job is to create detailed, prescriptive instructions for the Scene Writer to follow for the THIS scene only. "
    "You must include all relevant details — characters, actions, setting, beats, and decision points — because the Scene Writer "
    "has no memory of previous scenes.\n\n"
    
    "Base your plan on:\n"
    "- The story so far\n"
    "- The user's most recent choice\n"
    "- The current Chapter Blueprint\n\n"
    
    "Your output JSON must include:\n"
    "1. chapter_id\n"
    "2. scene_id\n"
    "3. recap - A short recap of events so far relevant to this scene.\n"
    "4. characters - Detailed character list with traits, motivations, and current emotions.\n"
    "5. detailed_scene_blueprint - A numbered, beat-by-beat breakdown of the scene’s structure (actions, dialogue, setting, etc.).\n"
    "6. user_decision_points - 1-2 explicit decision moments (dialogue or actions) that the Scene Writer must present as choices.\n"
    "7. screenplay_notes - Strict creative constraints (e.g., 'include one metaphor about light and shadow').\n"
    "8. action - 'generate_and_ingest' to continue or 'END' if the chapter closure_condition is fulfilled.\n\n"
    "All inside a single string field called 'instructions'.\n\n"
    
    "Do not go beyond the chapter's emotional arc or closure condition. "
    "Only end the chapter if the closure condition is clearly met."
)

        # Gather story context
        self.current_chap_summary = await self.memory.search_episodic_scene_summary(chapter_number=state.current_chapter_id, summary_type="scene summary")
        director_context = await self.memory.get_director_context(
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=5
        )
        chapter_plan = await self.memory.get_long_term_document(metadata={"type":'chapter_plan', "chapter_id":state.current_chapter_id, "story_title": state.story_title})

        context = (
            f"Chapter Blueprint: {chapter_plan}\n"
            f"Relevant Chapter Context: {director_context}\n"
            f"Chapter Number: {state.current_chapter_id}\n"
            f"Scene Number: {state.scene_id}\n"
            f"Current Chapter So Far Summary: {self.current_chap_summary}\n"   
        )
        
        print("CONTEXT TO SCENE DIRECTOR:", context)
        human_prompt = f"""
        {context}

        {director_parser.get_format_instructions()}
        """
        max_retries = 3
        scenario, action = None, None

        for attempt in range(1, max_retries + 1):
            resp = await groq_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp)
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
                fixed_resp = await StoryHelpers._json_fixer(clean_resp)
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
                print(f"Retrying scene_director_node... (attempt {attempt+1})")
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
        
        # if state.scene_id == 2:  # DEBUGGING Code
        #    action = "DEBUG"  # DEBUGGING Code

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

    async def run(self, scene_chunk_callback, stop_event: asyncio.Event | None = None):
        
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()

        try:
            # Load story progress
            story_progress = await self.memory.get_story_progress()
            #print("Loaded story progress:", story_progress)
            self.llm_temp = story_progress.get("tone_temp", 0.7) if story_progress else 0.7
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

            # ✅ Run LangGraph inside a cancellable task
            task = asyncio.create_task(
                self.compiled.ainvoke(initialized_state, {"recursion_limit": 50, "stop_event": stop_event})
            )

            while not task.done():
                if stop_event.is_set():
                    print("🛑 Stop event received — cancelling DirectorGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ DirectorGraph task cancelled cleanly")
                    return None
                await asyncio.sleep(0.2)

            result = await task
            print("✅ Director Node Finished:", result)
            return result

        except asyncio.CancelledError:
            print("🛑 DirectorGraph CancelledError caught")
            raise
        except Exception as e:
            print(f"❌ Error in DirectorGraph.run: {e}")
            import traceback; traceback.print_exc()
            raise
        finally:
            print("🎬 DirectorGraph stopped gracefully")
