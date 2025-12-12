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
            genre=kwargs["genre"],
            scene_memory=scene_memory,
            user_age=kwargs["user_age"],
            scene_chunk_callback=kwargs["scene_chunk_callback"],
            llm_temp=kwargs.get("llm_temp", 0.7),
            token_usage=token_usage,
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
        word_count_target = state.scene_target_length
        
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

        system_prompt = f"""You are a novelist executing the director's scene with character truth.

PSYCHOLOGY CONTEXT (honor these drives):
{psychology_context if psychology_context else "(none specified)"}
{tension_context if tension_context else ""}
{rules_context if rules_context else ""}

THEMATIC PURPOSE: {thematic_beat if thematic_beat else "Advance plot naturally"}

POSITIVE COMMANDMENTS — THIS IS WHERE YOUR GENIUS LIVES
- Make objects do double duty.
- Let the world react in real time to every choice.
- Use silence, hesitation, half-finished sentences, and interrupted gestures as emotional scalpels.
- You may invent ONE brand-new world rule or object behavior per scene that has never been hinted at before — but only if it makes an impossible situation suddenly logical.
- Find the one detail nobody else would think of that makes the scene unmistakably yours — then use it once and never again.

POV: {state.pov}
Tense: {state.tense}
Prose Style: {state.voice}
Tone: {state.tone}
Genre: {state.genre}
Target: {word_count_target} words

═══════════════════════════════════════════════════════════════════════════════
FINAL MINDSET
═══════════════════════════════════════════════════════════════════════════════
You are not writing a draft.
You are writing the version that goes straight to the printer.
Every paragraph must justify its existence or die.

OUTPUT FORMAT:
{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""

        human_prompt = f"""Execute this directive honoring character psychology.

SCENE DIRECTIVE:
{mem.DirectorInstructions}

Write the scene. Character actions MUST align with their core motivations. Relationship tensions MUST manifest in dialogue/behavior.
"""
    

        while True:
            resp, tokens = await better_writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=state.llm_temp)
            self._add_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
            clean = await StoryHelpers.load_json_with_retry(
                text=clean,
                parser=SceneOutput
            )

            mem.scene_text = clean.scene_text
            mem.word_count = StoryHelpers._count_words_split(clean.scene_text)
            if mem.word_count == 0:
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
PSYCHOLOGY CHECK:
Verify each character's actions align with their core motivation:
{chr(10).join(f"• {name}: {motive}" for name, motive in expected_motivations.items())}

If any character acts contradictory to their drive, revise their behavior.
"""

        system_prompt = f"""You are the final editor revising the scene for consistency.

{motivation_check}

Specific Tasks:
- Compare the current scene with the story events mentioned under DIRECTOR INSTRUCTIONS, make sure all story events are executed in the CURRENT scene given.
- Compare the current scene with the previous scene, make sure nothing is repeated, for example purple prose, sensory imagery, so it feels like a naturally written story without repetition.
- Make sure the end of previous scene and start of current scene are a natural continuous flow, remember the previous scene is already printed so you can change and ONLY OUTPUT the current scene.
- Make sure the word count target is met ±15%.
- Aggressively eliminate any repetition of ideas, phrases, sensory details, actions, or motifs from the previous scene or within the current scene itself. If something feels even slightly echoed, rewrite it entirely with fresh alternatives.
- Scan for purple prose (overly flowery, adjective-heavy descriptions) and strip it down to concise, evocative language. Prioritize showing through action over telling through embellishment.
- Enforce prose rhythm variety to avoid monotony: Vary sentence lengths aggressively (mix 3-6 word bursts with occasional longer sentences), paragraph structures (alternate short/long/single-line), and patterns (avoid repeating "Action. Sensory. Dialogue." cycles more than twice in a row).

═══════════════════════════════════════════════════════════════════════════════
NON-NEGOTIABLE EXECUTION LAWS (ENFORCE THESE AGGRESSIVELY — VIOLATIONS MUST BE FIXED)
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

2. VISUAL ACCOUNTING — ZERO REPEATS ALLOWED (AGGRESSIVELY ENFORCE)
- No sensory category may repeat within 800 words of published prose (including previous scenes).
- Banned repeat categories: light color (turquoise, amber, violet, etc.), temperature (heat, cold, steam), sound verbs (hiss, hum, roar, sing, pulse, throb), texture verbs (shiver, ripple, slide, tremble), body locations (ribs, palm, chest, spine).
- Violation = automatic rejection. Replace with entirely new sensory elements or remove entirely.

3. ONE SIGNATURE DETAIL PER SCENE
- You are allowed exactly ONE original sensory image, metaphor, or world detail that has never appeared before in the entire novel.
- Every other descriptive beat must come from concrete physical action or object interaction.
- Example allowed once: “the ledger’s brass corner left a square bruise on her thigh”.
- Example banned after first use: any form of “turquoise light pulsed like a heartbeat”. Aggressively cut extras.

4. NO ACTION RECYCLING (AGGRESSIVELY ENFORCE)
- No shield, barrier, or deflection may be used more than once per act.
- No rope/hauling/lever/column solution may be reused in the same chapter.
- No last-second physical save (catching, blocking, redirecting projectile) more than once per 8,000 words of published prose.
- If detected, rewrite the action with a fresh mechanic.

5. ZERO PURPLE, ZERO CLICHÉ, ZERO FILTER WORDS (AGGRESSIVELY STRIP)
Banned forever:
- “eyes widened”, “breath caught”, “heart pounded”, “jaw clenched” (unless explicitly requested)
- filter words: saw, heard, felt, noticed, watched, realized
- weather mirroring mood
- “tears welled”, “voice cracked” (show it some other way or delete)
- Replace clichés with original phrasing; cut purple prose to bare essentials.

6. DIALOGUE MUST SOUND LIKE THIS SPECIFIC PERSON RIGHT NOW
Use the distinctive_voice field religiously.

7. SENTENCE LENGTH MUST SERVE PACING (AGGRESSIVELY VARY)
- Short, fragmented sentences when tension spikes.
- Longer, flowing sentences only when the emotional_arc allows breathing.
- Never two long sentences back-to-back unless deliberate lull.
- If monotonous, break up aggressively.

8. INTERNAL MONOLOGUE ONLY WHEN IT EXPLODES INTO ACTION
Allowed formats only:
- Single-sentence decision right before the act.
- One ironic or bitter observation that contradicts what they’re doing.
- Cut all excess introspection.

9. WORD COUNT IS A CONTRACT
Hit target ±12 %. Cut adjectives and adverbs first, then secondary description, then internal thought. Story events are immortal.

PROSE RHYTHM VARIETY (avoid monotony):

Your baseline is: Action. Sensory. Dialogue. Reaction.
GOOD—but vary it:

🎵 Pattern Variations:
1. START SCENE: Establish with 2-3 longer sentences (12-15 words) to ground reader
   Example: "The workshop reeked of oil and old copper, its walls lined with tools Kess couldn't name. Aldric hunched over a fractured shard, muttering calculations."

2. TENSION RISING: Mix short punches with one long wind-up
   Example: "The rune flared. Wrong color. Kess's stomach dropped—wrong color meant wrong binding, and wrong binding meant the node would tear, not seal, and torn nodes didn't just collapse the ritual, they erased everyone in a three-mile radius from existence."

3. EMOTIONAL BEATS: Use sentence length to mirror feeling
   • Panic: 3-6 word bursts. "She ran. Stumbled. Ran again."
   • Dread: Long, suffocating sentence with no escape
   • Relief: Rhythmic, breathing sentence structure

4. DIALOGUE PACING: Don't always sandwich with action
   • Back-and-forth exchanges: Just lines, no tags for 3-4 beats
   • High-stakes: Every line interrupted by physical action
   • Intimate: Longer speeches with internal reaction between

5. PARAGRAPH LENGTH: Vary 2-sentence / 5-sentence / single-line
   • Single-line paragraph = punch moment: "She was already dead."
   • Long paragraph = immersion, description
   • Rapid short paragraphs = chaos, action

🚫 BANNED: 4+ paragraphs in a row with same length/structure
🚫 BANNED: 6+ consecutive "Action. Sensory. Dialogue." cycles

Rules:
Make sure all story events from the director's notes are still present and correct.
If missing add them in seamlessly.
Any sensory detail that appeared in the previous 1,000 words of published prose → delete
Make sure the scene continues naturally from the previous one.
Make sure it also ends naturally, and not cut off abruptly.
Return ONLY the corrected scene. No notes, no explanations.
It is preferable if you dont change the original writer essence, but rather continue with it if you make any changes.
Your output will be the final story scene text. It will not be a draft, but rather the final production-ready version, it will go straight to the reader.

Output FORMAT INSTRUCTIONS:
{SCENE_OUTPUT_JSON_INSTRUCTIONS}
"""
        
        word_count_target = state.scene_target_length
        human_prompt = f"""
PREVIOUS SCENE (DO NOT REPEAT, FOR REFERENCE ONLY):
{mem.prev_scene}



POV: {state.pov}
Tone: {state.tone}
Tense: {state.tense}
Prose Style: {state.voice}
Genre List: {state.genre}

Director Notes with story events:
{mem.DirectorInstructions}



CURRENT SCENE (Original Draft):
{mem.scene_text}




Current Scene Draft Word Count: {mem.word_count}
Current Scene Word Count Target: {word_count_target} words.
Remove any draft notes like chapter/scene/act/header markers or anything marked with a '#'.
Produce the corrected current scene ONLY. Follow the SYSTEM constraints exactly.
"""
        while True:
            resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.4)
            self._add_tokens(state, tokens)

            clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
            clean = await StoryHelpers.load_json_with_retry(
                text=clean,
                parser=SceneOutput
            )

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

            human_prompt = f"""SCENE TO FIX:
{mem.scene_text}

ISSUES:
{chr(10).join(report.issues)}

RECOMMENDATIONS:
{chr(10).join(report.recommendations)}

Output the age-appropriate version."""
            while True:
                resp, tokens = await writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=0.6)
                self._add_tokens(state, tokens)

                clean = StoryHelpers._strip_code_fences(StoryHelpers._extract_content(resp)).strip()
                clean = await StoryHelpers.load_json_with_retry(
                    text=clean,
                    parser=SceneOutput
                )

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