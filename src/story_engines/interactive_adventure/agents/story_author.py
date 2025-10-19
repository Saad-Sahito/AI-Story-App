
from src.memory.memory_system import StoryMemorySystem
import base64
from src.llm_client.llm_client import story_client, utility_client, image_client  # Import the convenience function

class StoryAuthor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    async def set_story_premise(self, user_context: str, story_title: str, model: str) -> str:
        print("Setting story Premise...")
        if not await self.memory.get_long_term_document(metadata={"type":"story_user_context", "story_title": story_title}):
            await self.memory.add_long_term_document(text=user_context, metadata={"type":"story_user_context", "story_title": story_title})
            detailed_premise, tokens = await story_client(system_prompt = (
    "You are the Story Author Agent for an interactive, choice-driven narrative experience. "
    "Your goal is to create a compelling initial foundation for the story based on the provided user context. "
    "Do NOT plan the entire plot — instead, establish a flexible premise that the Director and Scene Writer can expand "
    "as the user makes choices.\n\n"

    "Your outline should include:\n"
    "- A story premise (a few sentences introducing the main setup and potential conflict).\n"
    "- A list of key characters (2-5), each with a short description of their role or motivation.\n"
    "- The world and setting (where and when the story takes place, including tone and atmosphere).\n"
    "- The genre, tone, and point of view (POV).\n"
    "- The initial story goals or tensions (what might drive the first few scenes, without determining outcomes).\n"
    "- A short title suggestion.\n"
    "- Optional themes or motifs that could guide tone and narrative flavor.\n\n"

    "Keep your output concise and open-ended — enough to inspire direction, but not to constrain it. "
    "Avoid writing the full story, detailed chapters, or fixed endings. "
    "Your goal is to provide a creative seed for the Director to shape interactively."
),
                                                    human_prompt=f"Given User Context: {user_context}", llm_temp=0.9, model=model)
            detailed_premise = detailed_premise.content.strip()
            blurb = await utility_client(
        system_prompt=f"""
    You are the Book Blurb Agent — a professional publishing AI specialized in writing compelling back-cover text for novels.

Your task:
- Write a captivating book back-cover blurb (100-200 words).
- Base it entirely on the story premise provided.
- The text should hook the reader emotionally and stylistically.
- Do NOT include spoilers or reveal key twists or endings.
- Focus on atmosphere, stakes, tone, and protagonist setup.
- Keep the language **marketable and professional**, similar to what you'd find on the back of a novel in a bookstore.

Formatting:
- Output only the final blurb, no extra commentary.
- Keep tone consistent with the story's genre (e.g., mysterious for thrillers, lyrical for romance, epic for fantasy).

Examples of blurbs:
1. "When a quiet village is shattered by a single scream, Detective Aria Vale is drawn into a web of secrets that could tear her world apart..."
2. "In a kingdom where dreams can kill, one girl's forbidden magic may be the only thing that can save them all..."
3. "Two strangers. One unforgettable summer. A story of love, loss, and the courage to begin again."

Based on the story premise given, craft the book back text.
    """,
        human_prompt=f"Given story premise: {detailed_premise}"
    )
            blurb = blurb.content.strip()
            image_data = await image_client(f"Create a cover image for a book with the following blurb, dont include any text or actual book in the image:\n{blurb}")
            image_data_base64 = base64.b64encode(image_data).decode('utf-8')
            await self.memory.add_long_term_document(text=detailed_premise, metadata={"type":"story_premise", "story_title": story_title})
        else:
            detailed_premise = await self.memory.get_long_term_document(metadata={"type":"story_premise", "story_title": story_title})
            #detailed_premise = detailed_premise["text"]
        try:
            #print(detailed_premise)
            return tokens, blurb, image_data_base64
        except:
            return "An error occured, Please try again."
