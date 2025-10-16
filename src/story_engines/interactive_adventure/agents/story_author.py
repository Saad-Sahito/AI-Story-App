
from src.memory.memory_system import StoryMemorySystem

from src.llm_client.llm_client import groq_client  # Import the convenience function

class StoryAuthor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    async def set_story_premise(self, user_context: str, story_title: str) -> str:
        print("Setting story Premise...")
        if not await self.memory.get_long_term_document(metadata={"type":"story_user_context", "story_title": story_title}):
            await self.memory.add_long_term_document(text=user_context, metadata={"type":"story_user_context", "story_title": story_title})
            detailed_premise = await groq_client(system_prompt = (
    "You are the Story Author Agent for an interactive, choice-driven narrative experience. "
    "Your goal is to create a compelling initial foundation for the story based on the provided user context. "
    "Do NOT plan the entire plot — instead, establish a flexible premise that the Director and Scene Writer can expand "
    "as the user makes choices.\n\n"

    "Your outline should include:\n"
    "- A story premise (a few sentences introducing the main setup and potential conflict).\n"
    "- A list of **key characters** (2-5), each with a short description of their role or motivation.\n"
    "- The world and setting (where and when the story takes place, including tone and atmosphere).\n"
    "- The genre, tone, and point of view (POV).\n"
    "- The initial story goals or tensions (what might drive the first few scenes, without determining outcomes).\n"
    "- A short title suggestion.\n"
    "- Optional themes or motifs that could guide tone and narrative flavor.\n\n"

    "Keep your output concise and open-ended — enough to inspire direction, but not to constrain it. "
    "Avoid writing the full story, detailed chapters, or fixed endings. "
    "Your goal is to provide a creative seed for the Director to shape interactively."
),
                                                    human_prompt=f"Given User Context: {user_context}", llm_temp=0.9)
            detailed_premise = detailed_premise.content.strip()
            await self.memory.add_long_term_document(text=detailed_premise, metadata={"type":"story_premise", "story_title": story_title})
        else:
            detailed_premise = await self.memory.get_long_term_document(metadata={"type":"story_premise", "story_title": story_title})
            #detailed_premise = detailed_premise["text"]
        try:
            #print(detailed_premise)
            return detailed_premise
        except:
            return "An error occured, Please try again."

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