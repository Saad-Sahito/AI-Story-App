import asyncio
from typing import List, Optional, Callable, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from src.llm_client.llm_client import writer_client, better_writer_client
from src.utilities.story_helpers import StoryHelpers
from dataclasses import dataclass
import json
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
    genre: list[str]
    scene_memory: SceneMemory
    user_age: int = 13
    scene_target_length: int = 500
    writer_token_usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    utility_token_usage: dict = Field(default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
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
        writer_token_usage = kwargs.get("writer_token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        utility_token_usage = kwargs.get("utility_token_usage") or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        scene_memory = SceneMemory(DirectorInstructions=kwargs["director_instructions"].strip(), prev_scene=kwargs["prev_scene"])
        scene_state = SceneState(
            pov=kwargs["pov"],
            tense=kwargs["tense"],
            tone=kwargs["tone"],
            voice=kwargs["voice"],
            genre=kwargs["genre"],
            scene_memory=scene_memory,
            user_age=kwargs["user_age"],
            scene_chunk_callback=kwargs["scene_chunk_callback"],
            llm_temp=kwargs.get("llm_temp", 0.7),
            writer_token_usage=writer_token_usage,
            utility_token_usage=utility_token_usage,
            scene_target_length=kwargs["scene_target_length"]
        )
        return cls(kwargs["user_id"], kwargs["story_id"], kwargs["director_instructions"], scene_state, scene_memory, kwargs["scene_chunk_callback"])


SCENE_OUTPUT_JSON_INSTRUCTIONS = SceneOutput.model_json_schema()

# ======================
#   AGE VALIDATOR
# ======================
class AgeAppropriateValidator:
    # ... exactly your original code (unchanged) ...
    AGE_TO_GRADE = {8: 3.0, 9: 4.0, 10: 5.0, 11: 6.0, 12: 7.0, 13: 8.0, 14: 9.0, 15: 10.0, 16: 11.0, 17: 12.0, 18: 13.0}
    # SENSITIVE_KEYWORDS = {
    #     "violence": ["blood", "gore", "mutilated", "dismember", "torture", "slaughter"],
    #     "mature_themes": ["sexual", "seductive", "arousal", "explicit"],
    #     "strong_language": ["damn", "hell", "bastard", "bitch"],
    #     "dark_themes": ["suicide", "self-harm", "overdose", "rape"]
    # }
    # AGE_RESTRICTIONS = {
    #     8: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
    #     9: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
    #     10: {"block": ["violence", "mature_themes", "strong_language", "dark_themes"]},
    #     11: {"block": ["mature_themes", "strong_language", "dark_themes"], "warn": ["violence"]},
    #     12: {"block": ["mature_themes", "dark_themes"], "warn": ["violence", "strong_language"]},
    #     13: {"block": ["mature_themes", "dark_themes"], "warn": ["violence", "strong_language"]},
    #     14: {"block": ["mature_themes"], "warn": ["dark_themes", "violence"]},
    #     15: {"warn": ["mature_themes", "dark_themes"]},
    #     16: {"warn": ["mature_themes", "dark_themes"]},
    #     17: {"warn": ["mature_themes"]},
    #     18: {}
    # }

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

    # @staticmethod
    # def detect_sensitive_content(text: str, age: int) -> Dict[str, List[str]]:
    #     # unchanged
    #     text_lower = text.lower()
    #     detected = {"blocked": [], "warnings": []}
    #     restrictions = AgeAppropriateValidator.AGE_RESTRICTIONS.get(age, {})
    #     blocked = restrictions.get("block", [])
    #     warn = restrictions.get("warn", [])
    #     for category, keywords in AgeAppropriateValidator.SENSITIVE_KEYWORDS.items():
    #         found = [kw for kw in keywords if kw in text_lower]
    #         if found:
    #             if category in blocked:
    #                 detected["blocked"].append(f"{category}: {', '.join(found)}")
    #             elif category in warn:
    #                 detected["warnings"].append(f"{category}: {', '.join(found)}")
    #     return detected

    @staticmethod
    def validate_content(text: str, target_age: int) -> AgeAppropriateReport:
        # unchanged
        target_grade = AgeAppropriateValidator.get_target_grade(target_age)
        readability = AgeAppropriateValidator.calculate_readability(text)
        # sensitive = AgeAppropriateValidator.detect_sensitive_content(text, target_age)
        actual_grade = readability["flesch_kincaid_grade"]

        issues, warnings, recommendations = [], [], []
        grade_diff = actual_grade - target_grade
        if grade_diff > 2.0:
            issues.append(f"Too complex: grade {actual_grade:.1f} (target {target_grade:.1f})")
            recommendations.append("Simplify vocabulary and sentences")
        elif grade_diff > 1.0:
            warnings.append(f"Slightly complex: grade {actual_grade:.1f}")
        # if sensitive["blocked"]:
        #     issues.extend(sensitive["blocked"])
        #     recommendations.append("Remove flagged content")
        # if sensitive["warnings"]:
        #     warnings.extend(sensitive["warnings"])
        if target_age <= 12 and textstat.avg_sentence_length(text) > 20:
            warnings.append("Long sentences for young readers")
            recommendations.append("Break up long sentences")

        is_appropriate = len(issues) == 0
        return AgeAppropriateReport(
            is_appropriate=is_appropriate,
            readability_grade=actual_grade,
            target_grade=target_grade,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations
        )


# ================================
#   SCENE PLANNER SERVICE
# ================================
class ScenePlannerService:
    def __init__(self):
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
        
        directive_dict = mem.DirectorInstructions
        
        agents_psychology = {}
        relationship_tensions = {}
        world_rules = []
        thematic_beat = ""
        
        if 'agents_in_scene' in directive_dict:
            for name, agent_data in directive_dict['agents_in_scene'].items():
                # Extract core motivation
                if 'core_motivation' in agent_data:
                    agents_psychology[name] = agent_data['core_motivation']
                
                # Extract relationship tensions
                if 'relationships' in agent_data:
                    relationship_tensions[name] = agent_data['relationships']
        
        if 'context' in directive_dict:
            world_rules = directive_dict['context'].get('world_rules_active', [])
            thematic_beat = directive_dict['context'].get('thematic_beat', '')

        # Build psychology context string (token-efficient)
        psychology_context = ""
        if agents_psychology:
            psychology_context = "\nCHARACTER MOTIVATIONS:\n" + "\n".join(
                f"• {name}: {motive}" for name, motive in agents_psychology.items()
            )
        
        tension_context = ""
        if relationship_tensions:
            tension_context = "\nRELATIONSHIP DYNAMICS:\n" + "\n".join(
                f"• {name} tension with: {', '.join(f'{other} ({t})' for other, t in tensions.items())}"
                for name, tensions in relationship_tensions.items()
            )
        
        rules_context = ""
        if world_rules:
            rules_context = "\nWORLD CONSTRAINTS:\n" + "\n".join(f"• {rule}" for rule in world_rules[:3])

        system_prompt = f"""
You are a novelist executing the director’s scene with absolute character truth.

PSYCHOLOGICAL DRIVES (obey implicitly):
{psychology_context if psychology_context else "(none specified)"}
{tension_context if tension_context else ""}
{rules_context if rules_context else ""}

THEMATIC PURPOSE:
{thematic_beat if thematic_beat else "Advance plot naturally"}

CREATIVE COMMANDMENTS (this is where your skill lives):
- Make physical objects carry emotional or narrative weight.
- Let the environment respond immediately to character choices.
- Use silence, interruption, and restraint as emotional tools.
- You may invent ONE new world rule or object behavior if it resolves an otherwise impossible situation.
- Find one precise, unexpected detail that defines this scene — use it once, then abandon it.
- Favor specificity over atmosphere. Cut anything familiar.

STYLE CONSTRAINTS:
POV: {state.pov}
Tense: {state.tense}
Prose Style: {state.voice}
Tone: {state.tone}
Genre: {state.genre}
Target Length: {word_count_target} words

FINAL MINDSET:
This is not a draft.
Every paragraph must justify its existence.

ANTI-REPETITION LAW:
- Treat the previous scene as radioactive: no reused imagery, metaphors, sensory domains, or gestures.
- Within this scene, avoid repeating descriptive verbs or adjectives, like a typical generative AI text.
- Dialogue, action, and emotional expression must feel new, not iterated.

OUTPUT FORMAT:
{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""


        human_prompt = f"""
DIRECTOR INSTRUCTIONS:
{mem.DirectorInstructions}

Execute the scene as directed.


Character behavior MUST reflect core motivations.
Relationship tension MUST appear through action and dialogue.
Everything must feel freshly invented relative to the previous scene.

Write the scene in {state.voice} prose style.
"""

    

        while True:
            resp, tokens = await better_writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=state.llm_temp)
            self._add_writer_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
            clean, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean,
                parser=SceneOutput
            )
            if isinstance(clean, tuple):
                clean = clean[0]
            self._add_utility_tokens(state, utility_tokens)

            mem.scene_text = clean.scene_text
            mem.word_count = StoryHelpers._count_words_split(clean.scene_text)
            if mem.word_count <= int(word_count_target * 0.25):
                continue
            print(f"[Target] {word_count_target}w")
            print(f"[WRITE] {mem.word_count}w")
            return state

    async def _validate_and_fix(self, state: SceneState) -> SceneState:
        mem = state.scene_memory
        
        # NEW: Extract expected character behaviors
        directive_dict = mem.DirectorInstructions
        
        expected_motivations = {}
        if 'agents_in_scene' in directive_dict:
            for name, agent_data in directive_dict['agents_in_scene'].items():
                if 'core_motivation' in agent_data:
                    expected_motivations[name] = agent_data['core_motivation']
        
        motivation_check = ""
        if expected_motivations:
            motivation_check = f"""
CHARACTER MOTIVATION CHECK:
Ensure all actions align with:
{chr(10).join(f"- {name}: {motive}" for name, motive in expected_motivations.items())}
Revise any contradiction.
"""


        system_prompt = f"""
You are the final editor. Your job is correction, not commentary.

{motivation_check}

PRIMARY OBJECTIVES (in order):
1. Preserve all director-mandated story events.
2. Enforce continuity with the previous scene.
3. Eliminate repetition, cliché, excess, or any text that feels AI generted.
4. Hit word count ±15%.

HARD LAWS (violations MUST be fixed):

STORY EVENTS
- Every director-specified event must appear clearly in action or dialogue.
- Order is fixed unless explicitly stated otherwise.

CONTINUITY
- If scenes occur in the same place or minutes apart, write as a continuous sequence.
- Do not re-establish locations already active.
- Transitions are structural, not optional.

REPETITION BAN
- No sensory domain, metaphor, gesture, or descriptive language may repeat from the previous scene.
- Within this scene, avoid repeating content words tied to sensation, emotion, or environment.
- If anything feels echoed, replace or remove it.
- The text should feel human-edited not AI generated.

STYLE ENFORCEMENT
- Strip purple prose. Prefer action over description.
- Remove filter words (saw, felt, heard, noticed, realized).
- Remove clichés (eyes widening, breath catching, heart pounding, etc.).
- Dialogue must sound like the specific character in this moment.

PACING
- Vary sentence and paragraph length.
- Break monotony aggressively if detected.

WORD COUNT
- Target is binding.
- Cut adjectives first, then secondary description.
- If under target, add concrete action or dialogue — not atmosphere.

OUTPUT RULES:
- Return ONLY the corrected current scene.
- Do not explain changes.
- Do not modify prior scenes.
- Preserve the writer’s voice unless it violates the laws.

FORMAT:
{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""

        
        word_count_target = int(state.scene_target_length * 1.25)
        human_prompt = f"""
DIRECTOR NOTES:
{mem.DirectorInstructions}

Draft Word Count: {mem.word_count}
Target Word Count: {word_count_target}

Fix violations. Preserve story events. Enforce freshness.
Ensure writing prose style is {state.voice}.
Return ONLY the final corrected scene.


PREVIOUS SCENE (reference only):
{mem.prev_scene}


CURRENT SCENE DRAFT:
{mem.scene_text}
"""

        while True:
            resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.4)
            self._add_writer_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
            clean, utility_tokens = await StoryHelpers.load_json_with_retry(
                text=clean,
                parser=SceneOutput
            )
            if isinstance(clean, tuple):
                clean = clean[0]
            self._add_utility_tokens(state, utility_tokens)

            mem.scene_text = clean.scene_text
            word_count = StoryHelpers._count_words_split(clean.scene_text)
            if word_count == 0:
                continue
            mem.word_count = word_count
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
- Soften any flagged content
- Keep the story intact - just make it appropriate
- Target readability grade: {report.target_grade}

Return ONLY the corrected scene, perfectly formatted production ready version of story book.

OUTPUT FORMAT INSTRUCTIONS:
{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""

            human_prompt = f"""ISSUES:
{chr(10).join(report.issues)}

RECOMMENDATIONS:
{chr(10).join(report.recommendations)}

Output the age-appropriate version.

SCENE TO FIX:
{mem.scene_text}
"""
            while True:
                resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.6)
                self._add_writer_tokens(state, tokens)

                clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
                clean, utility_tokens = await StoryHelpers.load_json_with_retry(
                    text=clean,
                    parser=SceneOutput
                )
                if isinstance(clean, tuple):
                    clean = clean[0]
                self._add_utility_tokens(state, utility_tokens)

                mem.scene_text = clean.scene_text
                word_count = StoryHelpers._count_words_split(clean.scene_text)
                if word_count == 0:
                    continue
                else:
                    break
                # print(f"[AGE FIX] Revised to {mem.word_count}w")

        # Finalize
        mem.final_scene = mem.scene_text
        if state.scene_chunk_callback:
            state.scene_chunk_callback({"type": "text", "scene_text": mem.scene_text})
            mem.scene_cluster.append({"type": "text", "scene_text": mem.scene_text})
            state.scene_chunk_callback({"type": "status", "word_count": mem.word_count, "completion_percentage": 100})

        print(f"[SUCCESS] Final scene: {mem.word_count}w")
        return state

    def _add_writer_tokens(self, state: SceneState, tokens: dict):
        for k in state.writer_token_usage:
            state.writer_token_usage[k] += tokens.get(k, 0)
    
    def _add_utility_tokens(self, state: SceneState, tokens: dict):
        if tokens:
            for k in state.utility_token_usage:
                state.utility_token_usage[k] += tokens.get(k, 0)

    async def run_scene(self, ctx: UserSceneContext, stop_event: asyncio.Event | None = None):
        # unchanged
        task = asyncio.create_task(self.app.ainvoke(ctx.scene_state))
        while not task.done():
            if stop_event and stop_event.is_set():
                task.cancel()
                return "", [], "CANCELLED", ctx.scene_state.writer_token_usage, ctx.scene_state.utility_token_usage
            await asyncio.sleep(0.1)
        result = await task
        mem: SceneMemory = result['scene_memory']
        return mem.scene_text, mem.scene_cluster, "SUCCESS", result['writer_token_usage'], result['utility_token_usage']

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