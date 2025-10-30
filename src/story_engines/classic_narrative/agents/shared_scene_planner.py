# src/story_engines/classic_narrative/agents/shared_scene_planner.py

import json
import re
import gc
import asyncio
import traceback
from typing import List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import writer_client
from langgraph.graph import StateGraph, END
from src.utilities.story_helpers import StoryHelpers
from langchain_core.output_parsers import PydanticOutputParser

# Global shared instance
CLASSIC_SCENE_PLANNER_SERVICE = None


class SceneMemory(BaseModel):
    DirectorInstructions: Any = Field(description="Director's instructions for this scene.")
    scene_so_far: str = Field(default="", description="Accumulated text of the scene.")
    scene_cluster: List = Field(default=[], description="Scene text, questions, and user choices as dicts.")
    word_count: int = Field(..., description="Word count of the scene.")
    story_id: Optional[str] = Field(default="", description="Story ID for this scene.")


class SceneState(BaseModel):
    scene_memory: Optional[SceneMemory] = None
    next_node: Optional[str] = None
    user_context_id: Optional[str] = Field(default=None, description="User ID for this scene")
    iteration_count: int = 0
    scene_chunk_callback: Optional[Callable] = Field(default=None, description="Callback for frontend updates")
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    fatal: bool = False
    llm_temp: float = Field(default=0.7, description="Temperature for LLM")
    model: str = Field(default="None", description="Model name for LLM")
    token_usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, description="Token usage tracking")
    target_length: int = Field(default=500, description="Target length for the story")
    scene_target_length: int = Field(default=500, description="Target length for this scene")

    class Config:
        extra = "allow"


class SceneWriterOutput(BaseModel):
    scene: str = Field(description="One paragraph of continuing narrative text (80-130 words).")
    next_action: str = Field(description="'Continue' to write more paragraphs, 'End' if scene is complete.")


@dataclass
class UserSceneContext:
    user_id: str
    story_id: str
    director_instructions: Any
    scene_state: SceneState
    scene_memory: SceneMemory
    scene_chunk_callback: Callable

    @classmethod
    def create_for_user(cls, user_id: str, story_id: str, director_instructions: str, scene_chunk_callback: Callable,
                       llm_temp: float = 0.7, model: str = "None", token_usage: dict = None,
                       target_length: int = 500, scene_target_length: int = 500):
        # Initialize token_usage if not provided
        token_usage = token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        scene_memory = SceneMemory(
            DirectorInstructions=director_instructions.strip(),
            scene_so_far="",
            story_id=story_id,
            word_count=0
        )

        scene_state = SceneState(
            scene_memory=scene_memory,
            next_node=None,
            user_context_id=user_id,
            scene_chunk_callback=scene_chunk_callback,
            llm_temp=llm_temp,
            model=model,
            token_usage=token_usage,
            target_length=target_length,
            scene_target_length=scene_target_length
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
        
        # Cache static prompt components
        self.writer_schema = self.scene_writer_parser.get_format_instructions()
        
        self.writer_system_prompt = """You are the Scene Writer Agent for a classic narrative story. You write paragraphs AND decide when the scene is complete.

CRITICAL RULES:
- Write ONLY one paragraph per turn (80-130 words)
- Follow the Director's blueprint exactly (characters, location, emotional beats, style guide)
- Do NOT invent new elements or resolve the scene prematurely
- Create smooth, flowing narrative prose

WORD COUNT DISCIPLINE (HIGHEST PRIORITY):
1. If current word count >= target word count → set next_action: "End"
2. If current word count >= 90% of target → set next_action: "End"
3. If current word count > 110% of target → FORCE next_action: "End" (override everything)
4. If remaining words < 150 → This MUST be final paragraph, set next_action: "End"
5. Only set next_action: "Continue" if major story beats remain

Output Format:
- Valid JSON only, no markdown or extra text
- 'next_action': MUST be exactly "Continue" or "End" (case-sensitive)
- Check current word count before writing AND deciding next_action

"""

        self.graph = StateGraph(SceneState)
        self.graph.add_node("Initializer", self._initializer)
        self.graph.add_node("SceneWriter", self._scene_writer_agent)

        self.graph.set_entry_point("Initializer")
        self.graph.add_edge("Initializer", "SceneWriter")

        def writer_decider(state: SceneState):
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            
            # Check writer's decision
            next_action = getattr(state, "next_node", "End")
            if next_action == "Continue":
                return "SceneWriter"
            return "END"

        self.graph.add_conditional_edges(
            "SceneWriter",
            writer_decider,
            {
                "SceneWriter": "SceneWriter",
                "END": END,
                "FATAL_ERROR": END,
            },
        )

        self.compiled = self.graph.compile()

    def _handle_fatal_error(self, state: SceneState, error: Exception, stage: str) -> SceneState:
        state.fatal = True
        state.next_node = "FATAL_ERROR"
        state.error_type = stage
        state.error_message = f"{type(error).__name__}: {str(error)}"
        print(f"❌ [FATAL] {stage} failed – {state.error_message}")
        traceback.print_exc()
        return state

    async def run_scene(self, user_context: UserSceneContext, stop_event: asyncio.Event | None = None) -> Tuple[str, list, str, dict]:
        print(f"🎭 Starting run_scene for {user_context.user_id}/{user_context.story_id}")
        
        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ Failed to set user_context_id: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", user_context.scene_state.token_usage

        try:
            task = asyncio.create_task(
                self.compiled.ainvoke(user_context.scene_state, {"recursion_limit": 25, "stop_event": stop_event})
            )

            while not task.done():
                if stop_event and stop_event.is_set():
                    print("🛑 Stop event received – cancelling SceneGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ SceneGraph task cancelled cleanly")
                    return "", [], "CANCELLED", user_context.scene_state.token_usage
                await asyncio.sleep(0.2)

            result = await task
            # Update user_context.scene_state with the final result
            if not isinstance(result, dict):
                user_context.scene_state = result
            else:
                user_context.scene_state = SceneState(**result)
            next_node = getattr(result, "next_node", None) if not isinstance(result, dict) else result.get("next_node")
            error_message = getattr(result, "error_message", None) if not isinstance(result, dict) else result.get("error_message")
            scene_memory = getattr(result, "scene_memory", None) if not isinstance(result, dict) else result.get("scene_memory")

            if getattr(result, "fatal", False) or next_node == "FATAL_ERROR":
                fatal_msg = error_message or "Unknown fatal error"
                print(f"❌ Scene aborted due to fatal error: {fatal_msg}")
                return "", [], f"FATAL: {fatal_msg}", user_context.scene_state.token_usage

            if scene_memory is None:
                print("⚠️ Warning: LangGraph returned with no scene_memory.")
                return "", [], "EXCEPTION: No scene_memory returned", user_context.scene_state.token_usage

            scene_so_far = getattr(scene_memory, "scene_so_far", "") if not isinstance(scene_memory, dict) else scene_memory.get("scene_so_far", "")
            scene_cluster = getattr(scene_memory, "scene_cluster", []) if not isinstance(scene_memory, dict) else scene_memory.get("scene_cluster", [])

            print("✅ Scene completed normally")
            return scene_so_far, scene_cluster, "SUCCESS", user_context.scene_state.token_usage

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            return "", [], "CANCELLED", user_context.scene_state.token_usage
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", user_context.scene_state.token_usage
        finally:
            print("🎭 SceneGraph stopped gracefully")
            gc.collect()

    def _initializer(self, state: SceneState) -> SceneState:
        print(f"🔧 DEBUG: Initializer node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 Initializer skipping because fatal flag is set.")
            return state
        state.next_node = "SceneWriter"
        return state

    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print(f"🔧 DEBUG: SceneWriter node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 SceneWriter skipping because fatal flag is set.")
            return state

        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id
        
        # Circuit breaker: Force end if too many iterations
        if state.iteration_count > 15:
            print(f"🛑 FORCE END: Exceeded max iterations ({state.iteration_count})")
            state.next_node = "End"
            return state
        
        # Calculate word count metrics
        target_word_count = state.scene_target_length
        current_word_count = scene_memory.word_count
        remaining_words = max(0, target_word_count - current_word_count)
        completion_pct = (current_word_count / target_word_count * 100) if target_word_count > 0 else 0
        
        # Circuit breaker: Force end if severely over target
        if current_word_count >= target_word_count * 1.2:
            print(f"🛑 FORCE END: Scene at {current_word_count}/{target_word_count} words (120% exceeded)")
            state.next_node = "End"
            return state
        urgency_note = ""
        # Determine urgency level
        if remaining_words < 150:
            urgency_note = f"\n\nCRITICAL: Only {remaining_words} words remaining! This MUST be the FINAL paragraph. Set next_action: 'End'."
        elif remaining_words < 300:
            pass
            #urgency_note = f"\n\nWARNING: Only {remaining_words} words remaining. Prepare to end scene. Set next_action: 'End' after this paragraph."
        elif completion_pct >= 90:
            urgency_note = f"\n\nScene at {completion_pct:.1f}% completion. Consider setting next_action: 'End' soon."

        human_prompt = f"""
        Scene so far (continue from here):
{scene_memory.scene_so_far if scene_memory.scene_so_far else "[Scene starting now]"}
        
        Director's Instructions (Blueprint):
{scene_memory.DirectorInstructions}

╔═══════════════════════════════════════════════════════════╗
║ WORD COUNT STATUS (YOU DECIDE WHEN TO END)                ║
╠═══════════════════════════════════════════════════════════╣
║ Target Word Count:  {target_word_count:>5} words          ║
║ Current Word Count: {current_word_count:>5} words         ║
║ Remaining:          {remaining_words:>5} words            ║
║ Progress:           {completion_pct:>5.1f}%               ║
║ Iteration:          {state.iteration_count:>5}            ║
╚═══════════════════════════════════════════════════════════╝
{urgency_note}

{self.writer_schema}

Write the NEXT paragraph (80-130 words) and decide if scene should continue or end.
Output ONLY valid JSON, no extra text or markdown.
"""
# DECISION RULES FOR next_action:
# 1. Set 'End' if current_word_count >= {target_word_count} (at/over target)
# 2. Set 'End' if current_word_count >= {int(target_word_count * 0.9)} (within 10%, acceptable)
# 3. Set 'End' if remaining_words < 150 (not enough room for another paragraph)
# 4. Set 'End' if iteration_count > 12 (prevent infinite loops)
# 5. Set 'Continue' ONLY if current_word_count < {int(target_word_count * 0.8)} AND major story beats remain
        try:
            timeout = 60.0
            llm_response, tokens = await asyncio.wait_for(
                writer_client(system_prompt=self.writer_system_prompt, human_prompt=human_prompt, llm_temp=state.llm_temp, model=state.model),
                timeout=timeout
            )
            
            state.token_usage["prompt_tokens"] += tokens["prompt_tokens"]
            state.token_usage["completion_tokens"] += tokens["completion_tokens"]
            state.token_usage["total_tokens"] += tokens["total_tokens"]
            
            clean_resp = StoryHelpers._extract_content(llm_response)
            clean_resp = StoryHelpers._strip_code_fences(clean_resp)

            try:
                parsed = self.scene_writer_parser.parse(clean_resp)
                scene_text = parsed.scene
                next_action = parsed.next_action
            except Exception as e:
                print(f"❌ SceneWriter parsing failed: {e}")
                fixed_json = await StoryHelpers._json_fixer(clean_resp)
                try:
                    parsed = self.scene_writer_parser.parse(fixed_json)
                    scene_text = parsed.scene
                    next_action = parsed.next_action
                except Exception as repair_e:
                    print(f"❌ JSON fixer failed: {repair_e}")
                    match = re.search(r'"scene"\s*:\s*"([^"]*)"', clean_resp)
                    scene_text = match.group(1) if match else ""
                    next_action = "End"

            # Update word count and scene content
            new_word_count = StoryHelpers._count_words_split(scene_text)
            scene_memory.word_count += new_word_count
            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text,
                "word_count": new_word_count
            })
            scene_memory.scene_so_far += (" " + scene_text + "\n\n") if scene_text else ""
            
            print(f"📝 Written: {new_word_count} words | Total: {scene_memory.word_count}/{target_word_count} ({(scene_memory.word_count/target_word_count*100):.1f}%)")
            print(f"🎯 Writer decision: next_action='{next_action}'")

            # Override next_action based on word count (safety checks)
            if scene_memory.word_count >= target_word_count * 1.1:
                print(f"⚠️ OVERRIDE: Forcing 'End' - exceeded 110% of target")
                next_action = "End"
            elif scene_memory.word_count >= target_word_count * 0.9:
                if next_action == "Continue":
                    print(f"⚠️ OVERRIDE: Changing 'Continue' to 'End' - at 90% of target")
                next_action = "End"

            state.scene_memory = scene_memory
            state.iteration_count += 1
            state.next_node = next_action

            # Send text to frontend
            if user_context_id and state.scene_chunk_callback:
                try:
                    state.scene_chunk_callback({
                        "type": "text",
                        "scene_text": scene_text
                    })
                    state.scene_chunk_callback({
                        "type": "status",
                        "word_count": new_word_count,
                        "total_word_count": scene_memory.word_count,
                        "target_word_count": target_word_count,
                        "completion_percentage": int((scene_memory.word_count / target_word_count) * 100)
                    })
                except Exception as e:
                    print(f"❌ ERROR: Failed to send scene text to frontend: {e}")
                    traceback.print_exc()

            del llm_response, clean_resp, tokens
            gc.collect()
            return state

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "SceneWriter Timeout")
        except Exception as e:
            return self._handle_fatal_error(state, e, "SceneWriter")