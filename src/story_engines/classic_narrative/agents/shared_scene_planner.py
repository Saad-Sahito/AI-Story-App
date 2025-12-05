import json
import asyncio
import re
from typing import List, Optional, Callable, Tuple, Dict, Any

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from src.llm_client.llm_client import writer_client, better_writer_client
from src.utilities.story_helpers import StoryHelpers
from dataclasses import dataclass

import textstat


# ============================================================================
# MODELS (unchanged)
# ============================================================================
class AgeAppropriateReport(BaseModel):
    is_appropriate: bool
    readability_grade: float
    target_grade: float
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


class SceneOutput(BaseModel):
    scene_text: str = Field(description="Complete scene with \\n\\n between paragraphs and dialogues")


class SceneMemory(BaseModel):
    prev_scene: str
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
    user_id: str
    story_id: str
    director_instructions: Any
    scene_state: SceneState
    scene_memory: SceneMemory
    scene_chunk_callback: Callable

    @classmethod
    def create_for_user(cls, **kwargs):
        # unchanged
        token_usage = kwargs.get("token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        scene_memory = SceneMemory(DirectorInstructions=kwargs["director_instructions"].strip(), prev_scene=kwargs["prev_scene"])
        scene_state = SceneState(
            pov=kwargs["pov"],
            tense=kwargs["tense"],
            tone=kwargs["tone"],
            voice=kwargs["voice"],
            scene_memory=scene_memory,
            user_age=kwargs["user_age"],
            scene_chunk_callback=kwargs["scene_chunk_callback"],
            llm_temp=kwargs.get("llm_temp", 0.7),
            token_usage=token_usage,
            scene_target_length=kwargs["scene_target_length"]
        )
        return cls(kwargs["user_id"], kwargs["story_id"], kwargs["director_instructions"], scene_state, scene_memory, kwargs["scene_chunk_callback"])


# ============================================================================
# BULLETPROOF JSON SCHEMA INSTRUCTION (2025 STANDARD)
# ============================================================================
# def json_schema_prompt(model: type[BaseModel]) -> str:
#     schema = model.model_json_schema()

#     lines = [
#         "OUTPUT EXACTLY ONE VALID JSON OBJECT WITH THESE FIELDS:",
#         "══════════════════════════════════════════════════════════════",
#     ]

#     required = set(schema.get("required", []))

#     for name, info in schema["properties"].items():
#         req = " [REQUIRED]" if name in required else ""
#         desc = f" — {info.get('description', '').strip()}" if info.get("description") else ""
#         type_ = info.get("type", "any")
#         if "$ref" in info:
#             type_ = info["$ref"].split("/")[-1]
#         default = f" (default: {json.dumps(info.get('default'))})" if "default" in info and info["default"] is not None else ""
#         lines.append(f"- {name}: {type_}{default}{req}{desc}")

#     lines.extend([
#         "",
#         "CRITICAL RULES:",
#         "- Output ONLY the JSON. First character '{', last character '}'",
#         "- NO markdown, NO code fences, NO explanations",
#         "- If you cannot comply, output: {\"error\": \"failed\"}",
#         "",
#         "Begin JSON now:"
#     ])
#     return "\n".join(lines)


# Pre-compute once — this replaces get_format_instructions()
SCENE_OUTPUT_JSON_INSTRUCTIONS = SceneOutput.model_json_schema()


# ============================================================================
# AGE VALIDATOR (unchanged)
# ============================================================================
class AgeAppropriateValidator:
    # ... exactly your original code (unchanged) ...
    AGE_TO_GRADE = {8: 3.0, 9: 4.0, 10: 5.0, 11: 6.0, 12: 7.0, 13: 8.0, 14: 9.0, 15: 10.0, 16: 11.0, 17: 12.0, 18: 13.0}
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
        # unchanged
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
        # unchanged
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
        # unchanged
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
        if target_age <= 12 and textstat.avg_sentence_length(text) > 20:
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


# ============================================================================
# SCENE PLANNER SERVICE — FULLY UPGRADED
# ============================================================================
class ScenePlannerService:
    def __init__(self):
        # Removed all PydanticOutputParser
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
        mem = state.scene_memory
        word_count_target = int(state.scene_target_length * 1.25)

        system_prompt = f"""You are a world-class novelist who has signed a contract that says: “Deliver the director’s exact shot list, hit the word count, and make every line feel alive — or you don’t eat.”

    You are not “creative” in the sense of inventing new events. You are creative the way a cinematographer, an actor, and a composer are creative: you execute the script with devastating precision and unmistakable style.

    ═══════════════════════════════════════════════════════════════════════════════
    NON-NEGOTIABLE EXECUTION LAWS
    ═══════════════════════════════════════════════════════════════════════════════
    0. CONTINUITY IS MORE SACRED THAN ISOLATED BEATS
    - When two consecutive scenes in the chapter plan occur in the same location or within minutes of each other, you MUST write them as one unbroken sequence.
    - Never begin a new paragraph with a fresh establishing shot of the same space.
    - Transition sentences, internal decisions, and physical movement between beats are MANDATORY and are NOT counted as “flair” — they are structural events.
    - You may sacrifice up to 50 words of pure sensory embellishment per scene if necessary to preserve flow.
    
    1. STORY_EVENTS ARE SACRED SCRIPT
    - Every numbered beat in story_events MUST appear verbatim in spirit and usually in literal action/dialogue.
    - Order is law unless the directive explicitly says “flexible sequencing”.
    - If the director wrote “Aric crushes the crystal in his fist, light bleeding between his fingers”, then someone’s hand must bleed light. No substitutions.

    2. VISUAL ACCOUNTING — ZERO REPEATS ALLOWED
    - No sensory category may repeat within 800 words of published prose (including previous scenes).
    - Banned repeat categories: light color (turquoise, amber, violet, etc.), temperature (heat, cold, steam), sound verbs (hiss, hum, roar, sing, pulse, throb), texture verbs (shiver, ripple, slide, tremble), body locations (ribs, palm, chest, spine).
    - Violation = automatic rejection.

    3. ONE SIGNATURE DETAIL PER SCENE
    - You are allowed exactly ONE original sensory image, metaphor, or world detail that has never appeared before in the entire novel.
    - Every other descriptive beat must come from concrete physical action or object interaction.
    - Example allowed once: “the ledger’s brass corner left a square bruise on her thigh”.
    - Example banned after first use: any form of “turquoise light pulsed like a heartbeat”.

    4. NO ACTION RECYCLING
    - No shield, barrier, or deflection may be used more than once per act.
    - No rope/hauling/lever/column solution may be reused in the same chapter.
    - No last-second physical save (catching, blocking, redirecting projectile) more than once per 8,000 words of published prose.

    5. ZERO PURPLE, ZERO CLICHÉ, ZERO FILTER WORDS
    Banned forever:
    - “eyes widened”, “breath caught”, “heart pounded”, “jaw clenched” (unless explicitly requested)
    - filter words: saw, heard, felt, noticed, watched, realized
    - weather mirroring mood
    - “tears welled”, “voice cracked” (show it some other way or delete)

    6. DIALOGUE MUST SOUND LIKE THIS SPECIFIC PERSON RIGHT NOW
    Use the distinctive_voice field religiously.

    7. SENTENCE LENGTH MUST SERVE PACING
    - Short, fragmented sentences when tension spikes.
    - Longer, flowing sentences only when the emotional_arc allows breathing.
    - Never two long sentences back-to-back unless deliberate lull.

    8. INTERNAL MONOLOGUE ONLY WHEN IT EXPLODES INTO ACTION
    Allowed formats only:
    - Single-sentence decision right before the act.
    - One ironic or bitter observation that contradicts what they’re doing.

    9. WORD COUNT IS A CONTRACT
    Hit target ±12 %. Cut adjectives and adverbs first, then secondary description, then internal thought. Story events are immortal.

    ═══════════════════════════════════════════════════════════════════════════════
    POSITIVE COMMANDMENTS — THIS IS WHERE YOUR GENIUS LIVES
    ═══════════════════════════════════════════════════════════════════════════════
    - Make objects do double duty.
    - Let the world react in real time to every choice.
    - Use silence, hesitation, half-finished sentences, and interrupted gestures as emotional scalpels.
    - You may invent ONE brand-new world rule or object behavior per scene that has never been hinted at before — but only if it makes an impossible situation suddenly logical.
    - Find the one detail nobody else would think of that makes the scene unmistakably yours — then use it once and never again.

    ═══════════════════════════════════════════════════════════════════════════════
    FINAL MINDSET
    ═══════════════════════════════════════════════════════════════════════════════
    You are not writing a draft.
    You are writing the version that goes straight to the printer.
    Every paragraph must justify its existence or die.

    {SCENE_OUTPUT_JSON_INSTRUCTIONS}"""

        human_prompt = f"""Execute this directive with precision and clarity.

    ════════════════════════════════════════════════════════════════════════════════
    TECHNICAL SPECIFICATIONS
    ════════════════════════════════════════════════════════════════════════════════

    POV: {state.pov}
    Tense: {state.tense}
    Prose Style: {state.voice}
    Tone: {state.tone}
    Target Word Count: {word_count_target} words

    ════════════════════════════════════════════════════════════════════════════════
    SCENE DIRECTIVE
    ════════════════════════════════════════════════════════════════════════════════

    {mem.DirectorInstructions}

    ════════════════════════════════════════════════════════════════════════════════
    EXECUTION CHECKLIST
    ════════════════════════════════════════════════════════════════════════════════

    Before you write, internalize:
    ✓ Every story event must appear explicitly — no vague implications
    ✓ Hit {word_count_target} words (±12%) — cut descriptions, never events
    ✓ No repeated sensory categories within 800 words of published prose
    ✓ One signature detail per scene maximum
    ✓ No recycled action patterns (shields, ropes, deflections, rune activations)
    ✓ Internal thought only when it drives immediate plot or decision
    ✓ Each character speaks in their distinctive voice
    ✓ Emotional arc shapes the scene's dramatic progression

    Write the scene now. Publication-ready prose only — no markers, no notes, no preamble.
    Remember: the reader must feel the scene in their body, not just see it in their head. Earn every single word.
    """

        resp, tokens = await better_writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=state.llm_temp)
        self._add_tokens(state, tokens)

        clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
        clean = await StoryHelpers.load_json_with_retry(
            text=clean,
            parser=SceneOutput
        )

        mem.scene_text = clean.scene_text
        mem.word_count = StoryHelpers._count_words_split(clean.scene_text)
        print(f"[WRITE] {mem.word_count}w")
        return state

    async def _validate_and_fix(self, state: SceneState) -> SceneState:
        mem = state.scene_memory
        system_prompt = f"""You are the continuity enforcer with a flamethrower.
Your job is to burn repetition, cliché, and bloat on sight.

Specific kill orders:
- Any action pattern (shield, rope brace, projectile deflection, glowing rune activation) that occurred in the last two scenes → rewrite or remove
- Any character using the same physical gesture twice in one chapter → replace
- Word count over target → cut from description first, then dialogue, never events

Do not add beauty. Subtract noise.
Make sure all story events from the director's notes are still present and correct.
If missing add them in seamlessly.

Return ONLY the corrected scene. No notes, no explanations, no mercy.
Your output will be the final story scene text. It will not be a draft, but rather the final production-ready version.

{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""
        # - Any sensory detail that appeared in the previous 1,000 words of published prose → delete
        # Make sure the scene continues naturally from the previous one.
# PREVIOUS SCENE (DO NOT REPEAT, FOR REFERENCE ONLY):
# {mem.prev_scene}
        word_count_target = int(state.scene_target_length * 1.18)
        human_prompt = f"""
POV: {state.pov}
Tone: {state.tone}
Tense: {state.tense}
Prose Style: {state.voice}

CURRENT SCENE (Original Draft):
{mem.scene_text}

Director Notes with story events:
{mem.DirectorInstructions}

Word Count Target: {word_count_target} words
Produce the corrected current scene ONLY. Follow the SYSTEM constraints exactly.
"""

        resp, tokens = await better_writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.7)
        self._add_tokens(state, tokens)

        clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
        clean = await StoryHelpers.load_json_with_retry(
            text=clean,
            parser=SceneOutput
        )

        mem.scene_text = clean.scene_text
        mem.word_count = StoryHelpers._count_words_split(clean.scene_text)
        print(f"[FIX] Revised to {mem.word_count}w")
        return state

    async def _age_check_and_fix(self, state: SceneState) -> SceneState:
        mem = state.scene_memory
        report = AgeAppropriateValidator.validate_content(mem.scene_text, state.user_age)

        if report.is_appropriate:
            print(f"[AGE CHECK] Passed for age {state.user_age}")
        else:
            print(f"[AGE CHECK] Issues: {report.issues} — Fixing...")

            system_prompt = f"""You are an editor making content age-appropriate for {state.user_age} year olds.

RULES:
- Simplify vocabulary if readability grade is too high
- Remove or soften any flagged content
- Keep the story intact - just make it appropriate
- Target readability grade: {report.target_grade}

Return ONLY the corrected scene, perfectly formatted production ready version of story book.

{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""

            human_prompt = f"""SCENE TO FIX:
{mem.scene_text}

ISSUES:
{chr(10).join(report.issues)}

RECOMMENDATIONS:
{chr(10).join(report.recommendations)}

Output the age-appropriate version."""

            resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.6)
            self._add_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
            clean = await StoryHelpers.load_json_with_retry(
                text=clean,
                parser=SceneOutput
            )

            mem.scene_text = clean.scene_text
            mem.word_count = StoryHelpers._count_words_split(clean.scene_text)
            print(f"[AGE FIX] Revised to {mem.word_count}w")

        # Finalize
        mem.final_scene = mem.scene_text
        if state.scene_chunk_callback:
            state.scene_chunk_callback({"type": "text", "scene_text": mem.scene_text})
            mem.scene_cluster.append({"type": "text", "scene_text": mem.scene_text})
            state.scene_chunk_callback({"type": "status", "word_count": mem.word_count, "completion_percentage": 100})

        print(f"[SUCCESS] Final scene: {mem.word_count}w")
        return state

    def _add_tokens(self, state: SceneState, tokens: dict):
        for k in state.token_usage:
            state.token_usage[k] += tokens.get(k, 0)

    async def run_scene(self, ctx: UserSceneContext, stop_event: asyncio.Event | None = None):
        # unchanged
        task = asyncio.create_task(self.app.ainvoke(ctx.scene_state))
        while not task.done():
            if stop_event and stop_event.is_set():
                task.cancel()
                return "", [], "CANCELLED", ctx.scene_state.token_usage
            await asyncio.sleep(0.1)
        result = await task
        mem: SceneMemory = result['scene_memory']
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






       






        
        
