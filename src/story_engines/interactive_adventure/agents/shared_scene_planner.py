import json
import re
import gc
import asyncio
from typing import List, Optional, Callable, Any, Tuple
from pydantic import BaseModel, Field
from dataclasses import dataclass
from src.llm_client.llm_client import writer_client
from langgraph.graph import StateGraph, END
from langchain_core.messages import AIMessage, BaseMessage
from src.utilities.story_helpers import StoryHelpers
from langchain_core.output_parsers import PydanticOutputParser

# Global shared instance
INTERACTIVE_SCENE_PLANNER_SERVICE = None

class SceneMemory(BaseModel):
    DirectorInstructions: str = Field(description="Director's instructions for this scene.")
    ai_question: Optional[str] = Field(default="", description="Most recent decision point question, if any.")
    UserInput: Optional[str] = Field(default="", description="Latest user input choice, if any.")
    scene_so_far: str = Field(default="", description="Accumulated scene text, questions, and user responses.")
    number_of_options: Optional[int] = Field(default=0, description="Number of options at the decision point.")
    scene_cluster: List = Field(default=[], description="Scene text, questions, and user choices as dicts.")
    story_id: Optional[str] = Field(default="", description="Story ID for this scene.")
    word_count: int = Field(default=0, description="Word count of the scene so far.")
    questions_asked: List[str] = Field(default=[], description="List of all questions already asked to prevent duplicates.")
    scene_texts_written: List[str] = Field(default=[], description="List of all scene texts already written to prevent duplicates.")


class SceneState(BaseModel):
    messages: List[BaseMessage] = []
    scene_memory: SceneMemory | None = None
    next_node: str | None = None
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
    scene_target_length: int = Field(default=500, description="Target length for the scene")
    decision_in_progress: bool = Field(default=False, description="Flag to prevent duplicate decision handling")

    class Config:
        extra = "allow"


class SceneWriterOutput(BaseModel):
    scene: str = Field(description="One paragraph of continuing narrative text (80-130 words).")
    question: str = Field(
        description='Decision prompt WITH options included. Format: "Prompt text (A) option (B) option (C) option". Empty string "" if no decision point.'
    )
    number_of_options: int = Field(
        description="Number of options in the question field (2-4 if question present, else 0)."
    )
    next_action: str = Field(
        description="'Continue' to write more paragraphs, 'End' if scene is complete."
    )


@dataclass
class UserSceneContext:
    user_id: str
    story_id: str
    director_instructions: str
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
            story_id=story_id,
            word_count=0,
            questions_asked=[],
            scene_texts_written=[]
        )
        
        scene_state = SceneState(
            messages=[AIMessage(content=director_instructions)],
            scene_memory=scene_memory,
            next_node=None,
            user_context_id=user_id,
            scene_chunk_callback=scene_chunk_callback,
            llm_temp=llm_temp,
            model=model,
            token_usage=token_usage,
            target_length=target_length,
            scene_target_length=scene_target_length,
            decision_in_progress=False
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
        
        self.writer_system_prompt = """You are the Scene Writer Agent for an interactive story. 
        You write paragraphs AND decide when the scene is complete.
        ZERO TOLERANCE FOR REPEATED TEXTS OR QUESTIONS. 

CRITICAL RULES:
- Write ONLY one paragraph per turn (80-130 words)
- You must decide when to end the scene
- Follow the Director's blueprint exactly (characters, location, emotional beats, style guide)
- If a decision point is specified AND word count allows, the question field MUST include the question that contains 2-4 labeled options in format: "(A) option text (B) option text (C) option text"
- IMPORTANT: Do NOT repeat scene text or decision points that have already been presented. Check the "Scene so far" section above - if you see "(The scene writer asked: ...)" followed by "(The user chose: ...)", that decision has already been handled. Move the story forward with NEW content.
- Make the descision points meaningful to the story direction.
- Do NOT invent new elements or resolve the scene prematurely

WORD COUNT DISCIPLINE (HIGHEST PRIORITY):
1. If current word count >= target word count → set next_action: "End"
2. If current word count >= 90% of target → set next_action: "End"
3. If current word count > 110% of target → FORCE next_action: "End" (override everything)
4. If remaining words < 150 → This MUST be final paragraph, set next_action: "End"
5. Only set next_action: "Continue" if word count < 80% of target AND major story beats remain

DECISION POINT RULES:
- If decision point pending AND word count < 80% of target → Continue writing
- If word count >= 80% of target → Skip decision points and wrap up scene

FORBIDDEN (DO NOT GENERATE):
- Repeat/rephrase ANY prior text
- Re-ask ANY prior questions
- Recap/summarize past
- Generic decisions
- Decisions at >=80% word count

QUESTION FIELD FORMAT:
- If no decision point: question = ""
- If decision point: question = "What do you do? (A) First option (B) Second option (C) Third option"
- Do NOT put just the prompt without options
- Do NOT put options in a separate field

Output Format:
- Valid JSON only, no markdown or extra text
- 'number_of_options': Count of options (2-4) if question present, 0 if empty question
- 'next_action': MUST be exactly "Continue" or "End" (case-sensitive)
- Check current word count before writing AND deciding next_action

"""
        
        self.graph = StateGraph(SceneState)
        self.graph.add_node("Initializer", self._initializer)
        self.graph.add_node("SceneWriter", self._scene_writer_agent)
        self.graph.add_node("DecisionHandler", self._decision_handler)
        
        self.graph.set_entry_point("Initializer")
        self.graph.add_edge("Initializer", "SceneWriter")
        
        def writer_decider(state: SceneState):
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            
            # Check if there's a pending decision (and not already in progress)
            scene_memory = state.scene_memory
            if (scene_memory and scene_memory.ai_question and scene_memory.ai_question.strip() 
                and not state.decision_in_progress):
                return "DecisionHandler"
            
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
                "DecisionHandler": "DecisionHandler",
                "END": END,
                "FATAL_ERROR": END,
            },
        )
        
        def decision_decider(state: SceneState):
            if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
                return "FATAL_ERROR"
            
            # After decision is handled, check if scene should continue
            next_action = getattr(state, "next_node", "End")
            if next_action == "Continue":
                return "SceneWriter"
            return "END"

        self.graph.add_conditional_edges(
            "DecisionHandler",
            decision_decider,
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
        print(f"❌ [FATAL] {stage} failed — {state.error_message}")
        import traceback
        traceback.print_exc()
        return state

    def _is_duplicate_text(self, new_text: str, existing_texts: List[str], threshold: float = 0.85) -> bool:
        """
        Check if new_text is a duplicate of any existing text using similarity comparison.
        
        Args:
            new_text: The new scene text to check
            existing_texts: List of already written scene texts
            threshold: Similarity threshold (0.0-1.0) - default 0.85 means 85% similar
        
        Returns:
            True if duplicate found, False otherwise
        """
        if not new_text.strip() or not existing_texts:
            return False
        
        new_text_normalized = new_text.strip().lower()
        new_text_words = set(new_text_normalized.split())
        
        for existing_text in existing_texts:
            existing_normalized = existing_text.strip().lower()
            
            # Exact match check
            if new_text_normalized == existing_normalized:
                print(f"🚫 EXACT DUPLICATE DETECTED")
                return True
            
            # Similarity check using Jaccard similarity
            existing_words = set(existing_normalized.split())
            
            if len(new_text_words) == 0 or len(existing_words) == 0:
                continue
                
            intersection = len(new_text_words.intersection(existing_words))
            union = len(new_text_words.union(existing_words))
            
            similarity = intersection / union if union > 0 else 0
            
            if similarity >= threshold:
                print(f"🚫 SIMILAR DUPLICATE DETECTED (similarity: {similarity:.2%})")
                return True
        
        return False

    async def run_scene(self, user_context: UserSceneContext, stop_event: asyncio.Event | None = None) -> Tuple[str, list, str, dict]:
        print(f"🎭 Starting run_scene for {user_context.user_id}/{user_context.story_id}")
        
        try:
            user_context.scene_state.user_context_id = user_context.user_id
        except Exception as e:
            print(f"❌ Failed to set user_context_id: {e}")
            import traceback
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", user_context.scene_state.token_usage, 0

        try:
            task = asyncio.create_task(
                self.compiled.ainvoke(user_context.scene_state, {"recursion_limit": 25, "stop_event": stop_event})
            )

            while not task.done():
                if stop_event and stop_event.is_set():
                    print("🛑 Stop event received — cancelling SceneGraph task...")
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        print("✅ SceneGraph task cancelled cleanly")
                    return "", [], "CANCELLED", user_context.scene_state.token_usage, 0

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
                return "", [], f"FATAL: {fatal_msg}", user_context.scene_state.token_usage, 0

            if scene_memory is None:
                print("⚠️ Warning: LangGraph returned with no scene_memory.")
                return "", [], "EXCEPTION: No scene_memory returned", user_context.scene_state.token_usage, 0

            scene_so_far = getattr(scene_memory, "scene_so_far", "") if not isinstance(scene_memory, dict) else scene_memory.get("scene_so_far", "")
            scene_cluster = getattr(scene_memory, "scene_cluster", []) if not isinstance(scene_memory, dict) else scene_memory.get("scene_cluster", [])
            scene_word_count = getattr(scene_memory, "word_count", 0) if not isinstance(scene_memory, dict) else scene_memory.get("word_count", 0)
            print("✅ Scene completed normally")
            print("SCENE WRITER Token usage: ", user_context.scene_state.token_usage)

            return scene_so_far, scene_cluster, "SUCCESS", user_context.scene_state.token_usage, scene_word_count

        except asyncio.CancelledError:
            print("🛑 SceneGraph CancelledError caught")
            return "", [], "CANCELLED", user_context.scene_state.token_usage, 0
        except Exception as e:
            print(f"❌ ERROR in run_scene: {e}")
            import traceback
            traceback.print_exc()
            return "", [], f"EXCEPTION: {type(e).__name__}: {e}", user_context.scene_state.token_usage, 0
        finally:
            print("🎭 SceneGraph stopped gracefully")
            gc.collect()

    def _initializer(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: Initializer node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 Initializer skipping because fatal flag is set.")
            return state
        state.next_node = "SceneWriter"
        return state

    async def _decision_handler(self, state: SceneState) -> SceneState:
        """Handle user decisions and determine if scene should continue."""
        print(f"🔍 DEBUG: DecisionHandler node for user_context_id={state.user_context_id}")
        if getattr(state, "fatal", False) or getattr(state, "next_node", None) == "FATAL_ERROR":
            print("🛑 DecisionHandler skipping because fatal flag is set.")
            return state

        # Prevent re-entry
        if state.decision_in_progress:
            print("⚠️ DecisionHandler already in progress, skipping duplicate call")
            state.next_node = "End"
            return state
        
        state.decision_in_progress = True
        scene_memory: SceneMemory = state.scene_memory
        user_context_id = state.user_context_id
        
        if not scene_memory.ai_question or not scene_memory.ai_question.strip():
            print("⚠️ DecisionHandler called but no question pending")
            state.decision_in_progress = False
            state.next_node = "End"
            return state

        try:
            from setup.shared_redis_pool import get_redis_client
            redis_client = await get_redis_client()
            queue_key = f"input_queue:{user_context_id}:{scene_memory.story_id}"
            
            current_question = scene_memory.ai_question.strip()
            
            decision_payload = {
                "type": "decision",
                "question": current_question,
                "options": scene_memory.number_of_options,
                "user_choice": ""
            }
            state.scene_chunk_callback(decision_payload)
            print(f"📤 Sent decision to frontend: {current_question[:50]}...")
            
            print(f"🔍 DEBUG: Waiting for user input from Redis queue {queue_key}")
            
            user_choice = None
            for _ in range(600):  # 10-minute timeout
                choice = await redis_client.lpop(queue_key)
                if choice:
                    user_choice = choice.decode() if isinstance(choice, bytes) else str(choice)
                    break
                await asyncio.sleep(1.0)
            
            if user_choice:
                print(f"✅ Received user choice: {user_choice}")
                scene_memory.scene_cluster.append({
                    "type": "decision",
                    "question": current_question,
                    "options": scene_memory.number_of_options,
                    "user_choice": user_choice
                })
                scene_memory.UserInput = user_choice
                scene_memory.scene_so_far += f"\n(The user chose: {user_choice})\n"
            else:
                print(f"❌ TIMEOUT: No user input received within 600 seconds")
                scene_memory.UserInput = "default_choice"
                scene_memory.scene_so_far += "\n(No user input; continuing with default choice)\n"
            
            # CRITICAL: Clear the question and flag IMMEDIATELY after handling
            scene_memory.ai_question = ""
            scene_memory.number_of_options = 0
            state.decision_in_progress = False
            
            # Check if scene should continue after decision
            target_word_count = state.scene_target_length
            current_word_count = scene_memory.word_count
            
            if current_word_count >= target_word_count * 0.9:
                print(f"✅ Scene complete after decision: {current_word_count}/{target_word_count} words")
                state.next_node = "End"
            else:
                print(f"🔄 Continuing scene after decision: {current_word_count}/{target_word_count} words")
                state.next_node = "Continue"
            
            state.scene_memory = scene_memory
            
        except Exception as e:
            print(f"❌ ERROR in DecisionHandler: {e}")
            import traceback
            traceback.print_exc()
            scene_memory.UserInput = "default_choice"
            scene_memory.scene_so_far += "\n(No user input; continuing with default choice)\n"
            scene_memory.ai_question = ""
            scene_memory.number_of_options = 0
            state.decision_in_progress = False
            state.next_node = "End"
            state.scene_memory = scene_memory
        
        return state

    async def _scene_writer_agent(self, state: SceneState) -> SceneState:
        print(f"🔍 DEBUG: SceneWriter node for user_context_id={state.user_context_id}")
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
            urgency_note = f"\n\nCRITICAL: Only {remaining_words} words remaining! This MUST be the FINAL paragraph. Set next_action: 'End'. Do NOT add decision points."
        elif completion_pct >= 90:
            urgency_note = f"\n\nScene at {completion_pct:.1f}% completion. Consider setting next_action: 'End' soon."

# Scene so far (continue from here, DO NOT REPEAT):
# {scene_memory.scene_so_far if scene_memory.scene_so_far else "[Scene starting now]"}
        if scene_memory.scene_so_far:
            # Add a clear marker showing where new content should start
            scene_context = f"""SCENE SO FAR (CONTINUE FROM END):
{scene_memory.scene_so_far}

        [ WRITE NEW CONTENT FROM THIS POINT - Everything above has already been written, including questions - DO NOT REPEAT ]
        """
        else:
            scene_context = "[Scene starting now]"
        human_prompt = f"""
Director's Instructions (Blueprint):
{scene_memory.DirectorInstructions}


{scene_context}

╔════════════════════════════════════════════════════════════╗
║ WORD COUNT STATUS (YOU DECIDE WHEN TO END)                 ║
╠════════════════════════════════════════════════════════════╣
║ Target Word Count:  {target_word_count:>5} words           ║
║ Current Word Count: {current_word_count:>5} words          ║
║ Remaining:          {remaining_words:>5} words             ║
║ Progress:           {completion_pct:>5.1f}%                ║
║ Iteration:          {state.iteration_count:>5}             ║
╚════════════════════════════════════════════════════════════╝
{urgency_note}

IMPORTANT: Do NOT repeat scene texts or decision points that have already been presented. Check the "Scene so far" section above - if you see "(The scene writer asked: ...)" followed by "(The user chose: ...)", that decision has already been handled. Move the story forward with NEW content.

{self.writer_schema}

Write the NEXT paragraph (80-130 words) and decide if scene should continue or end.
Output ONLY valid JSON, no extra text or markdown.
"""

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
                question_text = parsed.question
                number_of_options = parsed.number_of_options
                next_action = parsed.next_action
            except Exception as e:
                print(f"❌ SceneWriter parsing failed: {e}")
                fixed_json = await StoryHelpers._json_fixer(clean_resp)
                try:
                    parsed = self.scene_writer_parser.parse(fixed_json)
                    scene_text = parsed.scene
                    question_text = parsed.question
                    number_of_options = parsed.number_of_options
                    next_action = parsed.next_action
                except Exception as repair_e:
                    print(f"❌ JSON fixer failed: {repair_e}")
                    match = re.search(r'"scene"\s*:\s*"([^"]*)"', clean_resp)
                    scene_text = match.group(1) if match else ""
                    question_text = ""
                    number_of_options = 0
                    next_action = "End"

            # Check for duplicate scene text
            if self._is_duplicate_text(scene_text, scene_memory.scene_texts_written):
                print(f"🚫 REJECTED: LLM produced duplicate scene text")
                print(f"   Duplicate text preview: {scene_text[:100]}...")
                
                # Don't add to scene_cluster or scene_so_far
                # Don't send to frontend
                # Don't update word count
                
                # Force the writer to try again if we haven't exceeded iteration limit
                if state.iteration_count < 15:
                    print("🔄 Forcing retry with Continue action")
                    state.next_node = "Continue"
                else:
                    print("🛑 Max iterations reached, ending scene")
                    state.next_node = "End"
                
                state.iteration_count += 1
                del llm_response, clean_resp, tokens
                gc.collect()
                return state

            # Update word count and scene content
            new_word_count = StoryHelpers._count_words_split(scene_text)
            scene_memory.word_count += new_word_count
            
            # Add to tracking list BEFORE adding to scene_cluster
            scene_memory.scene_texts_written.append(scene_text)
            
            # Add to scene_cluster
            scene_memory.scene_cluster.append({
                "type": "text",
                "scene_text": scene_text,
                "word_count": new_word_count
            })
            scene_memory.scene_so_far += scene_text + "\n"
            
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

            # Handle decision points - verify it's not a duplicate
            if question_text.strip() and scene_memory.word_count < target_word_count * 0.9:
                # Check if this exact question was already asked
                if question_text.strip() not in scene_memory.questions_asked:
                    scene_memory.scene_so_far += f"\n(The scene writer asked: {question_text.strip()})\n"
                    scene_memory.ai_question = question_text.strip()
                    scene_memory.questions_asked.append(question_text.strip())
                    scene_memory.number_of_options = number_of_options if isinstance(number_of_options, int) and 2 <= number_of_options <= 4 else 2
                    print(f"❓ Decision point added: {number_of_options} options")
                else:
                    print(f"🚫 REJECTED: LLM tried to repeat a previously asked question")
                    scene_memory.ai_question = ""
                    scene_memory.number_of_options = 0
            else:
                if question_text.strip() and scene_memory.word_count >= target_word_count * 0.9:
                    print(f"⚠️ Skipping decision point - too close to word count target")
                scene_memory.ai_question = ""
                scene_memory.number_of_options = 0

            state.scene_memory = scene_memory
            state.iteration_count += 1
            state.next_node = next_action

            # Send text to frontend (only if not duplicate)
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
                    import traceback
                    traceback.print_exc()

            del llm_response, clean_resp, tokens
            gc.collect()
            return state

        except asyncio.TimeoutError as e:
            return self._handle_fatal_error(state, e, "SceneWriter Timeout")
        except Exception as e:
            return self._handle_fatal_error(state, e, "SceneWriter")