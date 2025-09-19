import json
import re

from typing import Any, Dict, Tuple, Optional, List, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, ToolMessage
from dataclasses import dataclass, field

# Assuming a memory system is defined elsewhere
from src.memory.memory_system import StoryMemorySystem
from .scene_creation_subgraph.scene_planner_agent import ScenePlannerGraph
from src.llm_client.llm_client import LLMClient
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


# -----------------------------
# State model
# -----------------------------
@dataclass
class StoryState:
    current_chapter_id: int = 1
    scene_id: int = 1
    story_title: str = "None"
    word_count: int = 0
    messages: List[Any] = field(default_factory=list)
    next_action: str = ""
    # user_id: str = ""
    # story_id: str = ""

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

# -----------------------------
# Main director class
# -----------------------------
class DirectorGraph:
    def __init__(self, llm_client: LLMClient, memory_system: StoryMemorySystem, sceneplanner: ScenePlannerGraph):
        self.llm = llm_client
        self.scene_planner_agent = sceneplanner
        self.memory = memory_system
        


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
    
    # --- helper utilities ---
    def _count_words_split(self, text: str) -> int:
        return len(text.split())
    
    def _strip_code_fences(self, text: str) -> str:
        if text.startswith("```"):
            # remove leading/trailing ```json ... ```
            return re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        return text
    
    def _extract_content(self, resp: Any) -> str:
        """
        Normalize LLM responses to plain string content.
        Handles LangChain AIMessage, dicts, objects with .content,
        and plain strings.
        """
        # LangChain AIMessage
        try:
            from langchain_core.messages import AIMessage
            if isinstance(resp, AIMessage):
                return resp.content
        except ImportError:
            pass

        # dict response (some SDKs return {"content": "..."} or {"text": "..."})
        if isinstance(resp, dict):
            if "content" in resp and isinstance(resp["content"], str):
                return resp["content"]
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"]
            # fallback: dump dict
            return str(resp)

        # object with .content attribute
        if hasattr(resp, "content") and isinstance(resp.content, str):
            return resp.content

        # already a string
        if isinstance(resp, str):
            return resp

        # fallback: stringify anything else
        return str(resp)

    def _coerce_character_world_field(self, val: Any) -> Dict[str, str]:
        """
        Convert a variety of shapes into Dict[str,str]:
        - dict with non-str values => stringify values
        - list of "Name - summary" strings => convert to dict
        - list of dicts => merge into single dict
        - other => wrap as {"unknown": str(val)}
        """
        if isinstance(val, dict):
            out = {}
            for k, v in val.items():
                out[str(k)] = v if isinstance(v, str) else json.dumps(v)
            return out

        if isinstance(val, list):
            out = {}
            for item in val:
                if isinstance(item, str):
                    # try to split "Name – summary" or "Name: summary"
                    if "–" in item:
                        name, summary = item.split("–", 1)
                        out[name.strip()] = summary.strip()
                    elif ":" in item:
                        name, summary = item.split(":", 1)
                        out[name.strip()] = summary.strip()
                    else:
                        out[f"entry_{len(out)+1}"] = item
                elif isinstance(item, dict):
                    for k, v in item.items():
                        out[str(k)] = v if isinstance(v, str) else json.dumps(v)
                else:
                    out[f"entry_{len(out)+1}"] = json.dumps(item)
            return out

        # fallback: stringify
        return {"unknown": str(val)}

    def _normalize_payload_for_bundle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes top-level fields commonly used in Chapter/Scene bundles:
        - ensure 'summary' is a string
        - ensure 'character_summary' and 'world_summary' are Dict[str,str]
        This is intentionally conservative (non-destructive).
        """
        if not isinstance(payload, dict):
            return payload

        p = dict(payload)  # shallow copy

        # ensure summary is string
        if "summary" in p and not isinstance(p["summary"], str):
            p["summary"] = json.dumps(p["summary"])

        for field in ("character_summary", "world_summary"):
            if field in p:
                p[field] = self._coerce_character_world_field(p[field])

        return p

    def _model_validate_or_parse_obj(self, bundle_cls, data: Dict[str, Any]):
        """Try pydantic v2 model_validate, otherwise parse_obj (v1)."""
        if hasattr(bundle_cls, "model_validate"):
            # pydantic v2
            return bundle_cls.model_validate(data)
        elif hasattr(bundle_cls, "parse_obj"):
            return bundle_cls.parse_obj(data)
        else:
            raise RuntimeError("No pydantic validation method found on class")

    def _try_validate_with_model_then_parser(self, raw_str: str, bundle_cls, parser) -> Tuple[bool, Optional[Dict[str, Any]], Exception]:
        """
        Attempts:
        1) json.loads(raw_str) -> normalize -> model_validate/parse_obj
        2) parser.parse(raw_str) (LangChain PydanticOutputParser)
        Returns (success, result_dict_or_None, last_exception)
        """
        last_exc = None

        # 1) try to load JSON -> validate with pydantic model
        try:
            data = json.loads(raw_str)
            data = self._normalize_payload_for_bundle(data)
            model = self._model_validate_or_parse_obj(bundle_cls, data)
            return True, model.model_dump(), None
        except Exception as e:
            last_exc = e
            # fall through to step 2

        # 2) try langchain parser
        try:
            parsed = parser.parse(raw_str)
            try:
                # parser.parse often returns a pydantic-like object with `.model_dump()`
                return True, parsed.model_dump(), None
            except Exception:
                # if not, try to coerce to dict
                return True, dict(parsed), None
        except Exception as e:
            last_exc = e

        return False, None, last_exc
    
    def _json_fixer(self, text):
        system_prompt = """
You are a JSON repair agent. 
- Input may be malformed JSON text. 
- Output ONLY the corrected JSON, nothing else. 
- Never wrap JSON inside strings. 
- Never introduce additional nesting. 
- Always return a plain JSON object with the same top-level keys.
- If the input is correct, return it as is.
"""

        prompt = f"Fix the following json: {text}\n, if it is correct, then output as is, DO NOT add anything else, no leading or ending remarks."
        resp = self.llm.gemini_client(system_prompt=system_prompt, human_prompt=prompt)
        raw_text = self._extract_content(resp)
        clean_resp = self._strip_code_fences(raw_text)
        return clean_resp

    # Graph Functions
    async def generate_and_ingest_node(self, state: StoryState):
        """Generates a new scene and ingests it, with streaming chunks."""
        print("Called Generate and Ingest Node!")
        scene_text = await self.scene_planner_agent.run(state, self.scene_chunk_callback)
        #print("FINAL SCENE TEXT: ", scene_text)
        # After streaming is complete, store in memory
        #self.memory.set_current_chapter(scene_text)
        scene_bundle = self._ingest_scene(state, scene_text)
        #self.memory.add_story_chapter(text=scene_text, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title, "scene_id": state.scene_id, "word_count": state.word_count})
        #print("SCENE BUNDLE: ", scene_bundle)
        self.memory.add_post_scene_bundle(
            scene_bundle=scene_bundle,
            full_scene_text=scene_text,
            metadata={"scene_id": state.scene_id,
                      "chapter_id": state.current_chapter_id,
                      "story_title": state.story_title},
        )
        #state.scene_id += 1

        # Return the final state for the next node
        yield {
            "messages": [
                ToolMessage(
                    content=f"<scene_text>\n{scene_text}\n</scene_text>",
                    name="generate_and_ingest",
                    tool_call_id=f"scene_{state.scene_id}",
                )
            ],
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
        }


    def _ingest_scene(self, state: StoryState, scene_text: str, max_retries: int = 3) -> Dict[str, str]:
        """Ingest scene text and extract structured JSON using schema + parser.
        Retries with LLM if parse_obj + parse + json_fixer all fail.
        """

        system_prompt = "You are the Scene Breakdown Agent. Extract structured info from the scene. "
        "Always include chapter and scene id in character and world details, in order to keep track later. "
        "Make sure the character and world names are exactly as the keys presented to you under Character and World Names, "
        "if any need to be changed then create new entry for that entity mentioning previous name in the new entry, "
        "if not present then create new names as needed."

        human_prompt = f"""
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
            resp = self.llm.gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = self._extract_content(resp)
            clean_resp = self._strip_code_fences(raw_text)
            print(f"[Attempt {attempt}] RAW INGEST SCENE RESPONSE:", clean_resp)

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            print("TYPE OF CLEAN_RESP:", type(clean_resp))

            success, result, exc = self._try_validate_with_model_then_parser(clean_resp, SceneBundle, scene_parser)
            if success:
                state.word_count += self._count_words_split(scene_text)
                state.scene_id += 1
                self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "word_count": state.word_count, "story_title": state.story_title})
                return result

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            try:
                fixed_resp = self._json_fixer(clean_resp)
                fixed_clean = self._strip_code_fences(fixed_resp)
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = self._try_validate_with_model_then_parser(fixed_clean, SceneBundle, scene_parser)
                if success:
                    state.word_count += self._count_words_split(scene_text)
                    state.scene_id += 1
                    self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "word_count": state.word_count, "story_title": state.story_title})
                    return result

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised an exception:", inner_e)

            if attempt < max_retries:
                print(f"Retrying ingest_scene... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted for ingest_scene; returning minimal safe structure.")
                return {
                    "summary": "",
                    "character_summary": {},
                    "world_summary": {}
                }

    # -----------------------------
    # Maintain continuity, by adding episodic memory
    # -----------------------------
    def ingest_chapter(self, state: StoryState, max_retries: int = 3) -> Dict[str, Any]:
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
        Chapter Text:
        {self.current_chap_summary}

        Character Details:
        {char_details}

        World Details:
        {world_details}

        Respond ONLY in JSON with this schema:
        {chapter_parser.get_format_instructions()}
        """

        for attempt in range(1, max_retries + 1):
            resp = self.llm.gemini_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = self._extract_content(resp)
            clean_resp = self._strip_code_fences(raw_text)
            #print(f"[Attempt {attempt}] RAW INGEST CHAPTER RESPONSE:", clean_resp)

            # make sure we have a string
            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)
            #print("TYPE OF CLEAN_RESP:", type(clean_resp))

            # 1) try model_validate/parse_obj first, then parser
            success, result, exc = self._try_validate_with_model_then_parser(clean_resp, ChapterBundle, chapter_parser)
            if success:
                self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                state.current_chapter_id += 1
                state.scene_id = 1
                self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count})
                print("Chapter Complete!")
                result.update({
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    "word_count": state.word_count,
                })
                return result

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            # 2) try json_fixer, then same validation sequence
            try:
                fixed_resp = self._json_fixer(clean_resp)
                fixed_clean = self._strip_code_fences(fixed_resp)
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = self._try_validate_with_model_then_parser(fixed_clean, ChapterBundle, chapter_parser)
                if success:
                    self.memory.add_post_chapter_bundle(parts=result, metadata={"chapter_id": state.current_chapter_id, "story_title": state.story_title})
                    state.current_chapter_id += 1
                    state.scene_id = 1
                    self.memory.update_story_progress(metadata={"latest_chapter_id": state.current_chapter_id, "continue_scene_id": state.scene_id, "story_title": state.story_title, "word_count": state.word_count})
                    print("Chapter Complete!")
                    result.update({
                        "current_chapter_id": state.current_chapter_id,
                        "scene_id": state.scene_id,
                        "word_count": state.word_count,
                    })
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
                return {
                    "summary": "",
                    "character_summary": {},
                    "world_summary": {},
                    "current_chapter_id": state.current_chapter_id,
                    "scene_id": state.scene_id,
                    "word_count": state.word_count,
                }


    # -----------------------------
    # Director node
    # -----------------------------
    def director_node(self, state: StoryState) -> Dict:
        """Decide the next scene or end the chapter, using schema parsing with retries."""
        # story_dictionary = self.memory.get_story_progress()
        # if story_dictionary and "latest_chapter_id" in story_dictionary:
        #     state.current_chapter_id = story_dictionary["latest_chapter_id"]
        # if story_dictionary and "continue_scene_id" in story_dictionary:
        #     state.scene_id = story_dictionary["continue_scene_id"]
        # if story_dictionary and "story_title" in story_dictionary["metadata"]:
        #     state.story_title = story_dictionary["metadata"]["story_title"]
        # if story_dictionary and "word_count" in story_dictionary:
        #     state.word_count = story_dictionary["word_count"]

        # Short system prompt
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
        #print("SEARCHED CHAPTER SUMMARY: ", self.current_chap_summary)
        director_context = self.memory.get_director_context(
            current_chapter_number=state.current_chapter_id,
            query=self.current_chap_summary if self.current_chap_summary else "",
            k=5
        )
        context = (
            f"Premise: {self.memory.get_long_term_document('story_premise')}\n"
            f"Scene ID: {state.scene_id}\n"
            f"Current Chapter So Far: {self.current_chap_summary}\n"
            f"Relevant Chapter Context: {director_context}\n"
            f"Chapter Number: {state.current_chapter_id}\n"
        )
        print("CONTEXT TO DIRECTOR:", context)
        human_prompt = f"""
        {context}

        {director_parser.get_format_instructions()}
        """
        max_retries = 3
        scenario, action = None, None

        for attempt in range(1, max_retries + 1):
            resp = self.llm.groq_client(system_prompt=system_prompt, human_prompt=human_prompt)
            raw_text = self._extract_content(resp)
            clean_resp = self._strip_code_fences(raw_text)
            #print(f"[Attempt {attempt}] RAW DIRECTOR RESPONSE:", clean_resp)

            if isinstance(clean_resp, dict):
                clean_resp = json.dumps(clean_resp)

            # Try validate/parse and get dict back
            success, result, exc = self._try_validate_with_model_then_parser(clean_resp, DirectorOutput, director_parser)
            if success:
                # result is dict containing 'instructions' and 'action'
                scenario = result.get("instructions")
                action = result.get("action")
                break

            print(f"[Attempt {attempt}] First-pass validation failed:", exc)

            # json_fixer attempt
            try:
                fixed_resp = self._json_fixer(clean_resp)
                fixed_clean = self._strip_code_fences(fixed_resp)
                if isinstance(fixed_clean, dict):
                    fixed_clean = json.dumps(fixed_clean)

                success, result, exc = self._try_validate_with_model_then_parser(fixed_clean, DirectorOutput, director_parser)
                if success:
                    scenario = result.get("instructions")
                    action = result.get("action")
                    break

                print(f"[Attempt {attempt}] json_fixer validation failed:", exc)
            except Exception as inner_e:
                print(f"[Attempt {attempt}] json_fixer raised an exception:", inner_e)

            if attempt < max_retries:
                print(f"Retrying director_node... (attempt {attempt+1})")
                continue
            else:
                print("All retries exhausted for director_node; falling back to raw response and END.")
                scenario = clean_resp
                action = "END"

        if state.scene_id == 2: # DEBUGGING Code
            action = "DEBUG" # DEBUGGING Code

        # finalize and return (same as you had)
        messages = state.messages or []
        messages.append(AIMessage(content=scenario))

        return {
            "messages": messages,
            "scene_id": state.scene_id,
            "current_chapter_id": state.current_chapter_id,
            "word_count": state.word_count,
            "next_action": "generate_and_ingest" if action == "generate_and_ingest" else "END",
        }
    # -----------------------------
    # Public interface
    # -----------------------------
    async def run(self, scene_chunk_callback):
        print("Running director agent...")
        #self.memory.reset_current_chapter()
        self.scene_chunk_callback = scene_chunk_callback
        story_progress = self.memory.get_story_progress()
        if story_progress:
            # Use existing state from memory if available
            initialized_state = StoryState(
                current_chapter_id=story_progress.get("latest_chapter_id", 1),
                scene_id=story_progress.get("continue_scene_id", 1),
                story_title=story_progress.get("metadata", {}).get("story_title", "None"),
                word_count=story_progress.get("word_count", 0),
                messages=[],  # Messages can be reset per run
                next_action=""
            )
        else:
            # Otherwise, initialize a new state
            initialized_state = StoryState(
                current_chapter_id=1,
                scene_id=1,
                story_title="None",
                word_count=0,
                messages=[],
                next_action=""
            )
        result = await self.compiled.ainvoke(initialized_state, {"recursion_limit": 50})
        #text = result["scene_memory"].get("scene_so_far", "")
        print("Director Node Finished: ", result)
        return result

