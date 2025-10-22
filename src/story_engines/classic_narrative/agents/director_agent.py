# src/agents/director_agent.py

import json
import asyncio
import gc
from typing import Any, Dict, List, Literal, Optional, Union
from langgraph.graph import StateGraph, END
#from langchain_core.messages import AIMessage
from dataclasses import dataclass, field
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from src.utilities.story_helpers import StoryHelpers
from src.memory.memory_system import StoryMemorySystem
from .shared_scene_planner import UserSceneContext
import src.story_engines.classic_narrative.agents.shared_scene_planner as scene_planner_module
from src.llm_client.llm_client import story_client, ingestor_gemini_client

# Keep existing state and models unchanged
@dataclass
class StoryState:
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    word_count: int = 0
    #messages: List[Any] = field(default_factory=list)
    chapter_plan: Optional[Dict[str, Any]] = field(default_factory=dict)
    next_action: str = ""

class ScenePlan(BaseModel):
    scene_id: int = Field(..., description="Unique identifier for the scene, it should be labeled sequentially starting from 1")
    chapter_number: int = Field(..., description="Sequential chapter index")
    chapter_title: str = Field(..., description="Title or label of the chapter this scene belongs to")
    narrative_purpose: str = Field(
        ...,
        description="What this chapter is meant to accomplish in the overall story (e.g., 'Hero begins questioning loyalty.')"
    )
    emotional_arc: str = Field(
        ...,
        description="The emotional tone progression of this chapter (e.g., 'tense → sorrowful → hopeful')"
    )
    chapter_closure_condition: str = Field(
        ...,
        description="Condition for chapter resolution (e.g., 'Protagonist escapes the city.')"
    )

    # Scene-specific
    scene_goal: str = Field(..., description="Immediate purpose of the scene (e.g., introduce antagonist)")
    main_characters: List[str] = Field(..., description="List of main characters appearing in this scene")
    location: str = Field(..., description="Where the scene takes place")
    time_context: str = Field(..., description="When or under what condition the scene occurs (e.g., night, dream sequence)")
    key_events: List[str] = Field(..., description="List of key narrative events that must occur in this scene")
    emotional_beats: List[str] = Field(..., description="Target emotional moods or beats (e.g., fear, doubt, resolve)")
    thematic_notes: List[str] = Field(..., description="How this scene reinforces the story's themes")
    word_count: str = Field(..., description="Approximate word count target for the scene")


class DirectorOutput(BaseModel):
    scenes: List[ScenePlan] = Field(
        ...,
        description="List of detailed scene plans, each containing full contextual information for the scene writer."
    )
    action: Literal["generate_and_ingest", "END"] = Field(
        description="Output generate_and_ingest if the story is not complete yet, output END if the entire story is Completed."
    )

director_parser = PydanticOutputParser(pydantic_object=DirectorOutput)

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



# Keep Ingestor class unchanged (it's already fine)
class Ingestor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    # ---- Scene Ingestion (this is part of the Ingestor class, not DirectorGraph) ----
    async def ingest_scene(self, state: StoryState, scene_text: str, llm_temp: float, token_usage: dict, max_retries: int = 3) -> Dict[str, str]:
        """Ingest scene text and extract structured JSON using schema + parser."""
        chars = await self.memory.get_long_term_characters()
        worlds = await self.memory.get_long_term_worlds()
        system_prompt = f"""
You are the Scene Breakdown Agent. Extract structured info from the scene.
Always include chapter and scene id inside individual character and world details within the details, in order to keep track later.
Make sure the character and world names are exactly as the keys presented to you under Character and World Names.
If any need to be changed, create a new entry mentioning the previous name in the new entry.
If not present, create new names as needed.

Respond ONLY in JSON with this schema:
{scene_parser.get_format_instructions()}
"""

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
                new_word_count = StoryHelpers._count_words_split(scene_text)
                state.word_count += new_word_count
                
                await self.memory.update_story_progress(metadata={
                    "latest_chapter_id": state.current_chapter_id, 
                    "continue_scene_id": state.scene_id + 1, 
                    "word_count": state.word_count, 
                    "story_title": state.story_title,
                    "tone_temp": llm_temp,
                    "tokens_usage": token_usage
                })
                await self.memory._update_user_monthly_word_count(word_count=new_word_count)
                del system_prompt, human_prompt, new_word_count
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
                    new_word_count = StoryHelpers._count_words_split(scene_text)
                    state.word_count += new_word_count
                    
                    await self.memory.update_story_progress(metadata={
                        "latest_chapter_id": state.current_chapter_id, 
                        "continue_scene_id": state.scene_id + 1, 
                        "word_count": state.word_count, 
                        "story_title": state.story_title,
                        "tone_temp": llm_temp,
                        "tokens_usage": token_usage
                    })
                    await self.memory._update_user_monthly_word_count(word_count=new_word_count)
                    del system_prompt, human_prompt, scene_text, new_word_count
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

    async def ingest_chapter(self, state: StoryState, current_chap_summary, llm_temp: float, token_usage: dict, max_retries: int = 3) -> Dict[str, Any]:
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
                await self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count, "tone_temp": llm_temp, "tokens_usage": token_usage})
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
                    await self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count, "tokens_usage": token_usage})
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
        self.memory = memory_system
        
        self.graph = StateGraph(StoryState)
        self.graph.set_entry_point("director_node")
        self.graph.add_node("director_node", self.director_node)
        self.graph.add_node("generate_and_ingest", self.generate_and_ingest_node)
        self.graph.add_node("ingest_chapter", self.ingest_chapter)
        self.graph.add_node("story_complete", self.story_complete)

        self.graph.add_conditional_edges(
            "director_node",
            lambda state: state.next_action,
            {
                "generate_and_ingest": "generate_and_ingest",
                "END": "story_complete",
            },
        )
        # ✅ FIX: Add conditional edge from generate_and_ingest
        # Check if user wants to end or continue
        def route_after_generate(state: StoryState):
            """Route based on whether user chose to continue"""
            if state.next_action == "END":
                return END  # End immediately without ingesting chapter
            return "ingest_chapter"
        
        self.graph.add_conditional_edges(
            "generate_and_ingest",
            route_after_generate,
            {
                "ingest_chapter": "ingest_chapter",
                END: END  # Direct route to END
            }
        )
        #self.graph.add_edge("generate_and_ingest", "ingest_chapter")
        self.graph.add_edge("ingest_chapter", END)
        self.graph.add_edge("story_complete", END)
        self.compiled = self.graph.compile()
        self.current_chap_summary = ""
        self.llm_temp = 0.7  # default temperature

    async def story_complete(self, state: StoryState):
        await self.memory.update_story_progress(metadata={ "complete": True })
        status_payload = {"type": "status", "message":"story complete"}
        try:
            self.scene_chunk_callback(status_payload)
        except Exception as e:
            print(f"⚠️ scene_chunk_callback raised: {e}")
        return state

    async def generate_and_ingest_node(self, state: StoryState):
        """Generates all scenes using shared scene planner and ingests each sequentially."""
        print("Called Generate and Ingest Node!")

        if scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE is None:
            raise RuntimeError("❌ CLASSIC_SCENE_PLANNER_SERVICE is None!")

        # ✅ FIX: Check if chapter_plan is a list or dict, handle both cases
        scenes_list = []
        if isinstance(state.chapter_plan, list):
            scenes_list = state.chapter_plan
        elif isinstance(state.chapter_plan, dict):
            scenes_list = state.chapter_plan.get("scenes", [])
        
        if not scenes_list:
            print("⚠️ No valid chapter plan or scenes found.")
            return {
                "scene_id": state.scene_id,
                "current_chapter_id": state.current_chapter_id,
                "word_count": state.word_count,
                "next_action": "END",  # End immediately if no scenes
            }

        redis_client = None
        try:
            from setup.shared_redis_pool import get_redis_client
            redis_client = await get_redis_client()
        except Exception as e:
            print(f"⚠️ Could not initialize Redis client early: {e}")

        # 🚨 CRITICAL: Loop through scenes but BREAK IMMEDIATELY on non-SUCCESS
        for scene_dict in scenes_list:
            # ✅ FIX: Convert dict to ScenePlan object if needed
            if isinstance(scene_dict, dict):
                scene = ScenePlan(**scene_dict)
            else:
                scene = scene_dict
                
            # 🧱 Prepare scene input
            if scene.scene_id > state.scene_id:
                break
            if scene.scene_id != state.scene_id:
                print(f"Skipping scene {scene.scene_id}, already completed.")
                continue  # skip already completed scenes

            director_instructions = json.dumps(scene.model_dump(), indent=2)
            state.scene_id = scene.scene_id
            print(f"\n🎞️ Running Scene {scene.scene_id} | Goal: {scene.scene_goal}")

            # 🎯 Create scene context
            user_context = UserSceneContext.create_for_user(
                user_id=self.memory.user_id,
                story_id=self.memory.story_id,
                director_instructions=director_instructions,
                scene_chunk_callback=self.scene_chunk_callback
            )

            # ⚡ Async call — never use asyncio.run() inside an async def
            print(f"🔍 DEBUG: Calling run_scene for {user_context.user_id}/{user_context.story_id}")
            scene_text, scene_cluster, status, tokens = await scene_planner_module.CLASSIC_SCENE_PLANNER_SERVICE.run_scene(
                user_context=user_context,
                stop_event=self.stop_event,
                llm_temp=self.llm_temp,
                model=self.model,
                token_usage=self.token_usage
            )
            
            # 🚨 IMMEDIATE EXIT ON ANY NON-SUCCESS STATUS
            if status != "SUCCESS":
                print(f"🛑 Scene {scene.scene_id} failed with status: {status}")
                # Send status update to UI
                status_payload = {"type": "status", "ERROR": f"Scene {scene.scene_id} failed: {status}"}
                try:
                    self.scene_chunk_callback(status_payload)
                except Exception as e:
                    print(f"⚠️ scene_chunk_callback raised: {e}")
                
                # IMMEDIATELY END THE ENTIRE DIRECTOR GRAPH
                return {
                    "scene_id": state.scene_id,
                    "current_chapter_id": state.current_chapter_id,
                    "word_count": state.word_count,
                    "next_action": "END",
                }

            # ✅ SUCCESS: Continue with normal processing
            print(f"✅ Scene {scene.scene_id} completed successfully")
            self.token_usage = tokens
            del user_context, tokens
            gc.collect()

            # 🧠 Handle post-scene user choice
            if redis_client:
                try:
                    queue_key = f"continue_input_queue:{self.memory.user_id}:{self.memory.story_id}"
                    resume_payload = {"type": "save"}
                    try:
                        self.scene_chunk_callback(resume_payload)
                    except Exception as e:
                        print(f"⚠️ scene_chunk_callback raised: {e}")

                    user_choice = False
                    for _ in range(3600):
                        choice = await redis_client.lpop(queue_key)
                        if choice in (b"1", "1", 1):
                            user_choice = True
                            print("✅ User chose to continue")
                            break
                        elif choice is not None:
                            print(f"🛑 User chose not to continue ({choice})")
                            break
                        await asyncio.sleep(1.0)

                    if not user_choice:
                        print("🛑 Ending story after this scene per user choice")
                        return {
                            "scene_id": state.scene_id,
                            "current_chapter_id": state.current_chapter_id,
                            "word_count": state.word_count,
                            "next_action": "END",
                        }

                except Exception as e:
                    print(f"⚠️ Redis handling error: {e}")

            # 📚 Ingest the scene into memory
            ingestor = Ingestor(self.memory)
            scene_bundle = await ingestor.ingest_scene(state=state, scene_text=scene_text, llm_temp=self.llm_temp, token_usage=self.token_usage)
            del ingestor
            gc.collect()

            await self.memory.add_story_scene_cluster(
                text=scene_cluster,
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                    "scene_id": state.scene_id,
                    "word_count": state.word_count,
                }
            )

            await self.memory.add_post_scene_bundle(
                scene_bundle=scene_bundle,
                metadata={
                    "scene_id": state.scene_id,
                    "chapter_id": state.current_chapter_id,
                    "story_title": state.story_title,
                }
            )

            del scene_bundle, scene_cluster, scene_text
            gc.collect()

            # Move to next scene
            state.scene_id += 1

        # ✅ All scenes completed successfully
        print(f"✅ All scenes for chapter {state.current_chapter_id} completed!")
        return {
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
            "next_action": "END",  # End after all scenes to trigger chapter ingestion
        }
        
    async def ingest_chapter(self, state: StoryState):
        ingestor = Ingestor(self.memory)
        result = await ingestor.ingest_chapter(state=state, current_chap_summary=self.current_chap_summary, llm_temp=self.llm_temp, token_usage=self.token_usage)
         # reset current chapter summary after ingesting
        self.current_chap_summary = ""
        del ingestor
        gc.collect()
        return result

    async def director_node(self, state: StoryState) -> Dict:
        """Decide the next scene or end the chapter, using schema parsing with retries."""
        system_prompt = f"""
You are the Director Agent, responsible for orchestrating each chapter of the story.

You received the Story Bible created by the Story Author Agent, which defines the world, characters, tone, and story structure.  
You also received context from the memory system describing previous chapters, character developments, and world state.

Your task is to plan this chapter by:
- Determining what happens next according to the story bible and previous events.
- Outlining a set of coherent scenes that can be handed to the Scene Writer.
- Keeping tone, POV, and prose consistent with the Story Author's style guide.
- Maintaining continuity and character arcs.

---

### 🧠 You must:
- Respect established story logic and tone.
- Reflect ongoing character and world evolution.
- Progress the narrative toward its final resolution.
- Avoid rewriting scenes that already exist.
- Signal when the current chapter should end.
- Signal a targer word count for each scene.

---
### 🎯 Follow this structure exactly and respond ONLY in JSON::
{director_parser.get_format_instructions()}
---

Do NOT write narrative prose — only scene instructions and narrative planning.
If the story is complete then leave all fields empty except for 'action' which should be set to 'END'.
"""

        # Gather story context
        chapter_plan_combined = "Your chapter plans for previous chapters:\n"
        if state.current_chapter_id > 1:
            for i in range(state.current_chapter_id):
                chapter_plan_combined += "\n\n" + "chapter no.: " + i + "\n" + await self.memory.get_long_term_document(metadata={"type": "chapter_plan", "chapter_id": i, "story_title": state.story_title})

        self.current_chap_summary = await self.memory.search_episodic_scene_summary(chapter_number=state.current_chapter_id, summary_type="scene summary")
        director_context = await self.memory.get_director_context(
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=5
        )
        context = (f"""
            Story Bible: {await self.memory.get_long_term_document(metadata={'type': 'story_premise', 'story_title': state.story_title})}
            Chapter Number: {state.current_chapter_id}
            {f"Relevant Chapter Context: {director_context}" if state.current_chapter_id > 1 else ""}
            {f"{chapter_plan_combined}" if state.current_chapter_id > 1 else ""}
            {f"Story Word Count so far: {state.word_count}" if state.word_count > 0 else ""}
        """)

        
        #print("CONTEXT TO DIRECTOR:", context)
        human_prompt = context
        
        max_retries = 3
        action = None
        scenario = await self.memory.get_long_term_document(metadata={"type":'chapter_plan', "chapter_id":state.current_chapter_id, "story_title": state.story_title})

        if scenario != "":
            print(f"📋 Found existing chapter plan for chapter {state.current_chapter_id}")
            
            try:
                # Parse the scenario
                if isinstance(scenario, str):
                    scenario_dict = json.loads(scenario)
                else:
                    scenario_dict = scenario
                
                chapter_plan_scenes = scenario_dict.get("chapter_plan", [])
                
                if chapter_plan_scenes and len(chapter_plan_scenes) > 0:
                    # Find the highest scene_id in the plan
                    max_scene_id = max(int(s.get("scene_id", 0)) for s in chapter_plan_scenes)
                    
                    print(f"🎬 Current scene_id: {state.scene_id}, Max scene in plan: {max_scene_id}")
                    
                    # ✅ If current scene_id is beyond the last scene in plan, chapter is done
                    if state.scene_id > max_scene_id:
                        print(f"✅ All scenes completed for chapter {state.current_chapter_id}. Moving to ingest_chapter.")
                        action = "END"
                        # Keep the scenario for returning in state
                        scenario = scenario_dict
                    else:
                        print(f"▶️ Continuing with scene {state.scene_id}")
                        action = "generate_and_ingest"
                        scenario = scenario_dict
                else:
                    print("⚠️ Chapter plan exists but has no scenes. Regenerating...")
                    scenario = ""
                    
            except Exception as e:
                print(f"❌ Error parsing existing chapter plan: {e}")
                import traceback
                traceback.print_exc()
                scenario = ""
        
        # Only generate new plan if scenario is empty
        if scenario == "":
            print(f"🎭 Generating NEW chapter plan for chapter {state.current_chapter_id}")
            
            for attempt in range(1, max_retries + 1):
                resp, tokens = await story_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=self.llm_temp, model=self.model)
                raw_text = StoryHelpers._extract_content(resp)
                clean_resp = StoryHelpers._strip_code_fences(raw_text)
                self.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
                self.token_usage["completion_tokens"] += tokens["completion_tokens"]
                self.token_usage["total_tokens"] += tokens["total_tokens"]
                del raw_text, resp, tokens
                gc.collect()
                if isinstance(clean_resp, dict):
                    clean_resp = json.dumps(clean_resp)

                # Try validate/parse and get dict back
                success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, DirectorOutput, director_parser)
                print(result)
                if success:
                    # ✅ FIX: Use "scenes" field from DirectorOutput schema
                    scenario = {
                        "chapter_plan": result.get("scenes", [])
                    }
                    action = result.get("action", "generate_and_ingest")
                    print(f"✅ Generated {len(scenario['chapter_plan'])} scenes for chapter {state.current_chapter_id}")
                    break

                print(f"[Attempt {attempt}] First-pass validation failed:", exc)

                # json_fixer attempt
                try:
                    fixed_resp = await StoryHelpers._json_fixer(clean_resp)
                    fixed_clean = StoryHelpers._strip_code_fences(fixed_resp)
                    if isinstance(fixed_clean, dict):
                        fixed_clean = json.dumps(fixed_clean)

                    success, result, exc = StoryHelpers._try_validate_with_model_then_parser(
                        fixed_clean, DirectorOutput, director_parser
                    )
                    del fixed_clean, fixed_resp
                    gc.collect()
                    if success:
                        scenario = {
                            "chapter_plan": result.get("scenes", [])
                        }
                        action = result.get("action", "generate_and_ingest")
                        print(f"✅ Generated {len(scenario['chapter_plan'])} scenes for chapter {state.current_chapter_id}")
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
                    scenario = {"chapter_plan": []}
                    action = "END"
            
            # Save the newly generated plan
            await self.memory.add_long_term_document(
                text=json.dumps(scenario), 
                metadata={
                    "chapter_id": state.current_chapter_id,
                    "type": "chapter_plan", 
                    "story_title": state.story_title
                }
            )

        # Clean up variables
        if 'clean_resp' in locals():
            del clean_resp
        del system_prompt, human_prompt, context, director_context
        gc.collect()
        
        return {
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
            "chapter_plan": scenario.get("chapter_plan") if isinstance(scenario, dict) else None,
            "next_action": action,
        }

    # inside DirectorGraph class
    async def run(self, scene_chunk_callback, stop_event: asyncio.Event | None = None):
        print("🎬 Running director agent...")
        self.scene_chunk_callback = scene_chunk_callback
        self.stop_event = stop_event or asyncio.Event()

        try:
            # Load story progress
            story_progress = await self.memory.get_story_progress()
            self.llm_temp = story_progress.get("metadata", {}).get("tone_temp", 1) if story_progress else 1
            self.model = story_progress.get("metadata", {}).get("model","")
            self.token_usage = story_progress.get("metadata", {}).get("token_usage", {})
            complete = story_progress.get("complete", False) if story_progress else False
            if complete:
                return "Story already complete."
            if story_progress:
                initialized_state = StoryState(
                    current_chapter_id=story_progress.get("latest_chapter_id", 1),
                    scene_id=story_progress.get("continue_scene_id", 1),
                    story_title=story_progress.get("metadata", {}).get("story_title", "None"),
                    word_count=story_progress.get("word_count", 0),
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
                self.compiled.ainvoke(initialized_state, {"recursion_limit": 50, "stop_event": self.stop_event})
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
