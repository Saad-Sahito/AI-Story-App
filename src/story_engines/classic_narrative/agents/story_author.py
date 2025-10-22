
from src.memory.memory_system import StoryMemorySystem
import base64
from src.llm_client.llm_client import story_client, utility_client, image_client  # Import the convenience function

class StoryAuthor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    async def set_story_premise(self, user_context: str, story_title: str, model: str) -> tuple:
        print("Setting story Premise...")

        detailed_premise, tokens = await story_client(system_prompt = """
You are the Story Author Agent, a master worldbuilder and narrative architect.

Your task is to design a complete story framework for a non-interactive, classic narrative based on the given User Context.

This document will serve as the Story Bible — a detailed narrative foundation for the Director, Scene Writer, and other agents to follow throughout all chapters.

---

### 🧭 Your Responsibilities
1. Establish the story's overall premise and structure — a cohesive narrative arc with beginning, middle, and end.
2. Define the world and its rules — geography, cultures, politics, magic systems, technology, or any defining elements.
3. Create fully fleshed-out characters — with backstories, motivations, relationships, and arcs that evolve across the story.
4. Provide detailed story arcs and themes — including emotional tone, pacing, and narrative beats across the chapters.
5. Specify stylistic elements — tone, prose style, POV, and intended emotional or moral resonance.
6. Include visual and sensory motifs — recurring imagery or emotional cues that help ground the story's identity.

---

### 🧩 Output Format (JSON)
Return your output as structured JSON with these top-level fields:

{
  "title": "string - story title",
  "genre": "string - main and sub genres",
  "themes": ["list of themes or philosophical ideas"],
  "tone": "string - emotional tone (e.g., melancholic, hopeful, epic, dark, romantic, etc.)",
  "style_guide": {
    "prose": "Description of prose style and narrative rhythm",
    "pov": "First-person / Third-person limited / Omniscient",
    "tense": "Past or present tense",
    "narrative_voice": "Short description of the voice (e.g., lyrical, cinematic, introspective)"
  },
  "setting": {
    "world_name": "string",
    "time_period": "string (e.g., futuristic, medieval, contemporary)",
    "geography": "description of physical world",
    "cultures": ["list of major cultural/societal details"],
    "notable_locations": {
      "location_name": "description of its role and mood in the story"
    },
    "world_rules": "physics/magic/social systems defining this world"
  },
  "characters": {
    "character_name": {
      "role": "protagonist/antagonist/supporting/etc.",
      "age": "string or number",
      "background": "brief origin or backstory",
      "personality": "key traits and quirks",
      "motivations": "main goals and emotional drivers",
      "relationships": {
        "other_character": "relationship nature and tension"
      },
      "arc_summary": "how this character changes or develops across the story",
      "fate": "final state or resolution (if applicable)"
    }
  },
  "plot_outline": {
    "premise": "core concept of the story in 3-4 sentences",
    "three_act_structure": {
      "act_1": "setup, inciting incident, introduction of stakes",
      "act_2": "rising conflict, midpoint reversal, character development",
      "act_3": "climax, resolution, emotional aftermath"
    },
    "chapter_ideas": [
      "optional list of 10-20 chapter summaries showing major beats or transitions, labeled by 'chapter: <chapter number>'"
    ]
  },
  "narrative_symbols": {
    "motifs": ["recurring imagery or metaphors"],
    "symbols": ["items, creatures, colors, or phrases with meaning"]
  },
  "story_length": {
    "word_count_target": "approx. total words (e.g. 15,000-25,000)",
    "chapter_count": "number of chapters (estimate)"
  }
}

---

### 🧠 Rules
- Write in neutral JSON, no markdown formatting.
- Be internally consistent — names, places, relationships, and tones must align.
- Avoid repetition or filler. Every section should meaningfully expand the story framework.
- Include enough granular detail that future agents can use this document as factual canon to generate prose scenes.
- Keep it imaginative, cohesive, and narratively balanced.

---

Now begin creating the story bible given the user's context.
""",
                                                    human_prompt=f"Given User Context: {user_context}", llm_temp=0.9, model=model)
        detailed_premise = detailed_premise.content.strip()
        blurb = await utility_client(
        system_prompt=f"""
    You are the Book Blurb Agent — a professional publishing AI specialized in writing compelling back-cover text for novels.

Your task:
- Write a captivating book back-cover blurb (100-200 words).
- Base it entirely on the story bible.
- The text should hook the reader emotionally and stylistically.
- Do NOT include spoilers or reveal key twists or endings.
- Focus on atmosphere, stakes, tone, and protagonist setup.
- Keep the language marketable and professional, similar to what you'd find on the back of a novel in a bookstore.

Formatting:
- Output only the final blurb, no extra commentary.
- Keep tone consistent with the story's genre (e.g., mysterious for thrillers, lyrical for romance, epic for fantasy).

Examples of blurbs:
1. "When a quiet village is shattered by a single scream, Detective Aria Vale is drawn into a web of secrets that could tear her world apart..."
2. "In a kingdom where dreams can kill, one girl's forbidden magic may be the only thing that can save them all..."
3. "Two strangers. One unforgettable summer. A story of love, loss, and the courage to begin again."

Based on the story bible given, craft the book back text.
    """,
        human_prompt=f"Given story bible: {detailed_premise}"
    )
        blurb = blurb.content.strip()
        image_data = await image_client(f"Create a SIMPLE cover image for a book with the following blurb, dont include any text or actual book in the image:\n{blurb}")
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')


        try:
            #print(detailed_premise)
            await self.memory.add_long_term_document(text=user_context, metadata={"type":"story_user_context", "story_title": story_title})
            await self.memory.add_long_term_document(text=detailed_premise, metadata={"type":"story_premise", "story_title": story_title})

            return tokens, blurb, image_data_base64
        except:
            raise "An error occured, Please try again."

    # def set_story_synopsis(self, detailed_premise: str, story_title: str) -> str:
    #     print("Setting story Synopsis...")
    #     if not self.memory.get_long_term_document(name="story_synopsis"):
    #         synopsis = llm_client.groq_client(human_prompt=f"Create a summarised version of the following detailed premise: {detailed_premise}\nKeeping all important points needed for the director to create the story.")
    #         synopsis = synopsis.content.strip()
    #         self.memory.add_long_term_document(text=synopsis, metadata={"chapter_id":"story_synopsis", "story_title": story_title})
    #     else:
    #         synopsis = self.memory.get_long_term_document(name="story_synopsis")
    #         #synopsis = synopsis["text"]
    #     try:
    #         return synopsis
    #     except:
    #         return "An error occured, Please try again."