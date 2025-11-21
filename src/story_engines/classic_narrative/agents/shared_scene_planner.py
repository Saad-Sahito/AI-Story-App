import json
import asyncio
import re
from typing import List, Optional, Callable, Tuple, Dict, Any

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import PydanticOutputParser
from src.llm_client.llm_client import writer_client
from src.utilities.story_helpers import StoryHelpers
from dataclasses import dataclass

import textstat


class AgeAppropriateReport(BaseModel):
    is_appropriate: bool
    readability_grade: float
    target_grade: float
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class AgeAppropriateValidator:
    """Ensures scene meets age-appropriateness requirements"""
    
    AGE_TO_GRADE = {
        8: 3.0, 9: 4.0, 10: 5.0, 11: 6.0, 12: 7.0, 13: 8.0,
        14: 9.0, 15: 10.0, 16: 11.0, 17: 12.0, 18: 13.0
    }
    
    SENSITIVE_KEYWORDS = {
        "violence": ["blood", "gore", "mutilated", "dismember", "torture", "slaughter"],
        "mature_themes": ["sexual", "seductive", "arousal", "explicit"],
        "strong_language": ["damn", "hell", "bastard", "bitch"],
        "dark_themes": ["suicide", "self-harm", "overdose", "rape"]
    }
    
    AGE_RESTRICTIONS = {
        8: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
        9: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
        10: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
        11: {"block": ["mature_themes", "strong_language", "dark_themes"], "warn": ["violence"]},
        12: {"block": ["mature_themes", "dark_themes"], "warn": ["violence", "strong_language"]},
        13: {"block": ["mature_themes", "dark_themes"], "warn": ["violence", "strong_language"]},
        14: {"block": ["mature_themes"], "warn": ["dark_themes", "violence"]},
        15: {"warn": ["mature_themes", "dark_themes"]},
        16: {"warn": ["mature_themes", "dark_themes"]},
        17: {"warn": ["mature_themes"]},
        18: {}
    }

    @staticmethod
    def get_target_grade(age: int) -> float:
        return AgeAppropriateValidator.AGE_TO_GRADE.get(age, 8.0)

    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        if len(text.strip()) < 100:
            return {"flesch_kincaid_grade": 5.0, "flesch_reading_ease": 80.0, "gunning_fog": 8.0}
        try:
            return {
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "gunning_fog": textstat.gunning_fog(text)
            }
        except:
            return {"flesch_kincaid_grade": 8.0, "flesch_reading_ease": 70.0, "gunning_fog": 10.0}

    @staticmethod
    def detect_sensitive_content(text: str, age: int) -> Dict[str, List[str]]:
        text_lower = text.lower()
        detected = {"blocked": [], "warnings": []}
        restrictions = AgeAppropriateValidator.AGE_RESTRICTIONS.get(age, {})
        blocked = restrictions.get("block", [])
        warn = restrictions.get("warn", [])
        
        for category, keywords in AgeAppropriateValidator.SENSITIVE_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                if category in blocked:
                    detected["blocked"].append(f"{category}: {', '.join(found)}")
                elif category in warn:
                    detected["warnings"].append(f"{category}: {', '.join(found)}")
        return detected

    @staticmethod
    def validate_content(text: str, target_age: int) -> AgeAppropriateReport:
        target_grade = AgeAppropriateValidator.get_target_grade(target_age)
        readability = AgeAppropriateValidator.calculate_readability(text)
        sensitive = AgeAppropriateValidator.detect_sensitive_content(text, target_age)
        actual_grade = readability["flesch_kincaid_grade"]

        issues, warnings, recommendations = [], [], []
        
        grade_diff = actual_grade - target_grade
        if grade_diff > 2.0:
            issues.append(f"Too complex: grade {actual_grade:.1f} (target {target_grade:.1f})")
            recommendations.append("Simplify vocabulary and sentences")
        elif grade_diff > 1.0:
            warnings.append(f"Slightly complex: grade {actual_grade:.1f}")
        
        if sensitive["blocked"]:
            issues.extend(sensitive["blocked"])
            recommendations.append("Remove flagged content")
        
        if sensitive["warnings"]:
            warnings.extend(sensitive["warnings"])

        if target_age <= 12:
            if textstat.avg_sentence_length(text) > 20:
                warnings.append("Long sentences for young readers")
                recommendations.append("Break up long sentences")

        is_appropriate = len(issues) == 0 and len(sensitive["blocked"]) == 0
        
        return AgeAppropriateReport(
            is_appropriate=is_appropriate,
            readability_grade=actual_grade,
            target_grade=target_grade,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )


# ==================== OUTPUT MODELS ====================

class SceneOutput(BaseModel):
    scene_text: str = Field(description="Complete scene with \\n\\n between paragraphs and dialogues")
    #events_executed: List[str] = Field(description="Which required events appeared in scene")


class ValidationResult(BaseModel):
    """Simplified validation - just tracks what was found"""
    missing_events: List[str] = Field(default_factory=list)
    vague_events: List[str] = Field(default_factory=list)
    word_count_percent_of_target: int
    repeated_images_or_phrases: List[str] = Field(default_factory=list)
    tone_drift: str
    needs_fix: bool = False


class SceneMemory(BaseModel):
    DirectorInstructions: Any
    scene_text: str = ""
    scene_cluster: List[Dict] = Field(default_factory=list)
    word_count: int = 0
    final_scene: Optional[str] = None


class SceneState(BaseModel):
    pov: str
    tense: str
    tone: str
    voice: str
    scene_memory: SceneMemory
    user_age: int = 13
    scene_target_length: int = 500
    token_usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    fatal: bool = False
    llm_temp: float = 0.7
    scene_chunk_callback: Optional[Callable] = None


@dataclass
class UserSceneContext:
    """Context passed to scene writer"""
    user_id: str
    story_id: str
    director_instructions: Any
    scene_state: SceneState
    scene_memory: SceneMemory
    scene_chunk_callback: Callable

    @classmethod
    def create_for_user(
        cls,
        user_id: str,
        story_id: str,
        director_instructions: Any,
        scene_chunk_callback: Callable,
        pov: str,
        tone: str,
        tense: str,
        voice: str,
        llm_temp: float = 0.7,
        token_usage: dict = None,
        scene_target_length: int = 500,
        user_age: int = 13,
    ):
        token_usage = token_usage or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        scene_memory = SceneMemory(DirectorInstructions=director_instructions.strip())
        scene_state = SceneState(
            pov=pov,
            tense=tense,
            tone=tone,
            voice=voice,
            scene_memory=scene_memory,
            user_age=user_age,
            scene_chunk_callback=scene_chunk_callback,
            llm_temp=llm_temp,
            token_usage=token_usage,
            scene_target_length=scene_target_length
        )
        return cls(user_id, story_id, director_instructions, scene_state, scene_memory, scene_chunk_callback)


# ==================== SIMPLIFIED SCENE PLANNER ====================

class ScenePlannerService:
    def __init__(self):
        self.writer_parser = PydanticOutputParser(pydantic_object=SceneOutput)
        self.validation_parser = PydanticOutputParser(pydantic_object=ValidationResult)
        # Simple linear graph: Write → ValidateAndFix → AgeCheck → END
        graph = StateGraph(SceneState)
        graph.add_node("Write", self._write_scene)
        graph.add_node("ValidateAndFix", self._validate_and_fix)
        graph.add_node("AgeCheck", self._age_check_and_fix)

        graph.set_entry_point("Write")
        graph.add_edge("Write", "ValidateAndFix")
        graph.add_edge("ValidateAndFix", "AgeCheck")
        graph.add_edge("AgeCheck", END)

        self.app = graph.compile()

    async def _write_scene(self, state: SceneState) -> SceneState:
        """Initial scene generation"""
        mem = state.scene_memory
        directive_text = mem.DirectorInstructions if isinstance(mem.DirectorInstructions, str) else json.dumps(mem.DirectorInstructions, indent=2)
        #print(directive_text)
        events = self._extract_events(directive_text)

        system_prompt = f"""You are an elite, chameleon scene writer. Your only job is to execute the directive perfectly in whatever genre, age, length, and style it demands.

OBEY THESE RULES IN THIS EXACT ORDER:

1. POV, tense, and prose_style in the directive are law (first-person present, third limited past, etc.).
2. Hit every single story_event explicitly through action or spoken dialogue. Never summarise, never imply.
3. Word-count target ±15 % is sacred. Never go under 75 % or over 125 %.
4. Sensory images / metaphors: maximum one strong, genre-appropriate image per paragraph. Never repeat the same image.
5. Internal thought is allowed only when it is short, distinctive, and moves the plot. No mood journaling.
6. Dialogue must sound like that specific character in that specific genre and age bracket.
7. Tone and emotional_arc fields are absolute. Match them exactly.
8. If the directive says “lyrical,” you may be poetic. If it says “sparse,” you write like Cormac McCarthy. If it says “cinematic,” you write like a shooting script. Never mix unless explicitly told.

Output ONLY the scene. No titles, no notes, no explanations, no “here is the scene” preamble.

Paragraphs separated by exactly two newlines. Dialogue on its own line when genre expects it.

{self.writer_parser.get_format_instructions()}

That’s it. Now write.
"""

        human_prompt = f"""EXECUTE THIS DIRECTIVE:
POV: {state.pov}
Tone: {state.tone}
Tense: {state.tense}
Prose Style: {state.voice}

{mem.DirectorInstructions}

INSTRUCTIONS:
1. You MUST include all {len(events)} events EXPLICITLY (shown via action/dialogue, not vaguely implied).
2. Target {state.scene_target_length} words. Cut descriptions first—events are non-negotiable.
3. Structure your scene to hit events in sequence.

NOW WRITE THE COMPLETE SCENE WITH ALL {len(events)} EVENTS.
Return ONLY the corrected scene, perfectly formatted production ready version of story book. No draft notes like event markers, etc."""

        print(f"Scene Target Word Count: {state.scene_target_length}")
        resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.4)
        self._add_tokens(state, tokens)

        clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp))
        result: SceneOutput = await StoryHelpers.load_json_with_retry(clean, self.writer_parser)

        mem.scene_text = result.scene_text
        mem.word_count = StoryHelpers._count_words_split(result.scene_text)
        print(f"[WRITE] {mem.word_count}w")
        return state

    async def _validate_and_fix(self, state: SceneState) -> SceneState:
        """Validate scene and fix inline if needed - single pass"""
        mem = state.scene_memory
        directive_text = mem.DirectorInstructions if isinstance(mem.DirectorInstructions, str) else json.dumps(mem.DirectorInstructions, indent=2)
        events = self._extract_events(directive_text)

        # Quick validation via LLM
        validation = await self._check_scene(mem.scene_text, events, state)
        
        if not validation.needs_fix:
            print(f"[VALIDATE] Passed - all {len(events)} events present, no bloat")
            return state

        # Fix issues in single pass
        print(f"[VALIDATE] Issues found - Missing: {validation.missing_events}, Vague: {validation.vague_events}")
        
        system_prompt = f"""You are a ruthless, genre-savvy editor.

Fix everything flagged in the analysis while preserving:
- Every required story_event shown explicitly
- Exact target word count ±15 %
- The exact tone, prose_style, and emotional_arc demanded
- Character voices and genre conventions

Cut repetition, purple prose, summary, and any imagery that appears more than once.
Add missing events naturally in sequence.
Return ONLY the corrected scene, perfectly formatted production ready version of story book. No draft notes like event markers, etc.

{self.writer_parser.get_format_instructions()}"""

        issues = []
        if validation.missing_events:
            issues.append(f"MISSING EVENTS (must add): {validation.missing_events}")
        if validation.vague_events:
            issues.append(f"VAGUE  EVENTS to fix: {validation.vague_events}")

        human_prompt = f"""
POV: {state.pov}
Tone: {state.tone}
Tense: {state.tense}
Prose Style: {state.voice}

ORIGINAL SCENE:
{mem.scene_text}

ISSUES TO FIX:
{chr(10).join(issues)}

REQUIRED EVENTS (all must appear):
{chr(10).join(f'- {e}' for e in events)}

Output the FIXED scene."""

        resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.3)
        self._add_tokens(state, tokens)

        clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp))
        result: SceneOutput = await StoryHelpers.load_json_with_retry(clean, self.writer_parser)

        mem.scene_text = result.scene_text
        mem.word_count = StoryHelpers._count_words_split(result.scene_text)
        print(f"[FIX] Revised to {mem.word_count}w")
        return state

    async def _check_scene(self, scene_text: str, events: List[str], state: SceneState) -> ValidationResult:
        """Quick LLM check for missing events and bloat"""
        events_str = "\n".join(f"- {e}" for e in events)
        
        system_prompt = f"""Analyze the scene. Return JSON only:
dont include numbers in 'missing_events', 'vague_events', 'repeated_images_or_phrases' or 'tone_drift'; only text.
e.g:
"missing_events": ["exact wording of any story_event not shown via action/dialogue"],
"vague_events": ["events that are only implied or summarised"],
"word_count_percent_of_target": 94,
"repeated_images_or_phrases": ["pendant warmed", "vines pulsed"],
"tone_drift": "scene feels wistful when directive demanded urgent",
"needs_fix": true/false

{self.validation_parser.get_format_instructions()}
"""

        human_prompt = f"""
POV: {state.pov}
Tone: {state.tone}
Tense: {state.tense}
Prose Style: {state.voice}

REQUIRED EVENTS:
{events_str}

SCENE:
{scene_text}

Analyze and return JSON."""

        resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.1)
        self._add_tokens(state, tokens)

        clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp))
        data: ValidationResult = await StoryHelpers.load_json_with_retry(clean, self.validation_parser)
        return data

    async def _age_check_and_fix(self, state: SceneState) -> SceneState:
        """Check age appropriateness and fix inline if needed"""
        mem = state.scene_memory
        report = AgeAppropriateValidator.validate_content(mem.scene_text, state.user_age)

        if report.is_appropriate:
            print(f"[AGE CHECK] Passed for age {state.user_age}")
            mem.final_scene = mem.scene_text
            if state.scene_chunk_callback:
                state.scene_chunk_callback({"type": "text", "scene_text": mem.scene_text})
                mem.scene_cluster.append({"type": "text", "scene_text": mem.scene_text})
                state.scene_chunk_callback({"type": "status", "word_count": mem.word_count, "completion_percentage": 100})

            print(f"[SUCCESS] Final scene: {mem.word_count}w")
            return state
        else:
            print(f"[AGE CHECK] Issues: {report.issues} - Fixing...")
            
            system_prompt = f"""You are an editor making content age-appropriate for {state.user_age} year olds.

RULES:
- Simplify vocabulary if readability grade is too high
- Remove or soften any flagged content
- Keep the story intact - just make it appropriate
- Target readability grade: {report.target_grade}

Return ONLY the corrected scene, perfectly formatted production ready version of story book.

{self.writer_parser.get_format_instructions()}"""

            human_prompt = f"""SCENE TO FIX:
{mem.scene_text}

ISSUES:
{chr(10).join(report.issues)}

RECOMMENDATIONS:
{chr(10).join(report.recommendations)}

Output the age-appropriate version."""

            resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.3)
            self._add_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp))
            result: SceneOutput = await StoryHelpers.load_json_with_retry(clean, self.writer_parser)

            mem.scene_text = result.scene_text
            mem.word_count = StoryHelpers._count_words_split(result.scene_text)
            print(f"[AGE FIX] Revised to {mem.word_count}w")

        # Finalize
        mem.final_scene = mem.scene_text
        if state.scene_chunk_callback:
            state.scene_chunk_callback({"type": "text", "scene_text": mem.scene_text})
            mem.scene_cluster.append({"type": "text", "scene_text": mem.scene_text})
            state.scene_chunk_callback({"type": "status", "word_count": mem.word_count, "completion_percentage": 100})

        print(f"[SUCCESS] Final scene: {mem.word_count}w")
        return state

    def _extract_events(self, directive_text: str) -> List[str]:
        """Extract story_events from directive"""
        try:
            if "story_events" in directive_text:
                start = directive_text.find('"story_events"')
                if start > 0:
                    start = directive_text.find('[', start)
                    if start > 0:
                        depth = 0
                        end = start
                        for i in range(start, len(directive_text)):
                            if directive_text[i] == '[':
                                depth += 1
                            elif directive_text[i] == ']':
                                depth -= 1
                                if depth == 0:
                                    end = i + 1
                                    break
                        events = json.loads(directive_text[start:end])
                        return events if isinstance(events, list) else []
        except:
            pass
        
        events = []
        for line in directive_text.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                line = line.split('.', 1)[-1].strip() if line[0].isdigit() else line[1:].strip()
                if line:
                    events.append(line)
        return events or ["Scene progresses"]

    def _add_tokens(self, state: SceneState, tokens: dict):
        for k in state.token_usage:
            state.token_usage[k] += tokens.get(k, 0)

    async def run_scene(
        self,
        ctx: UserSceneContext,
        stop_event: asyncio.Event | None = None
    ) -> Tuple[str, list, str, dict]:
        task = asyncio.create_task(self.app.ainvoke(ctx.scene_state))
        
        while not task.done():
            if stop_event and stop_event.is_set():
                task.cancel()
                return "", [], "CANCELLED", ctx.scene_state.token_usage
            await asyncio.sleep(0.1)
        
        result = await task
        mem: SceneMemory = result['scene_memory']
        
        if result['fatal']:
            print(f"[FINAL] FAILED")
            return "", mem.scene_cluster, "FAILED_VALIDATION", result['token_usage']

        return mem.scene_text, mem.scene_cluster, "SUCCESS", result['token_usage']


# ============================================================================
# GLOBAL SERVICE INITIALIZATION
# ============================================================================

CLASSIC_SCENE_PLANNER_SERVICE = None


def initialize_classic_scene_planner():
    global CLASSIC_SCENE_PLANNER_SERVICE
    if CLASSIC_SCENE_PLANNER_SERVICE is None:
        CLASSIC_SCENE_PLANNER_SERVICE = ScenePlannerService()
        print("✓ Scene Planner initialized (simplified single-pass)")
    return CLASSIC_SCENE_PLANNER_SERVICE


CLASSIC_SCENE_PLANNER_SERVICE = initialize_classic_scene_planner()