
from src.memory.memory_system import StoryMemorySystem

from src.llm_client.llm_client import groq_client  # Import the convenience function

class StoryAuthor:
    def __init__(self, memory_system: StoryMemorySystem):
        #self.llm_client: LLMClient = llm_client
        self.memory = memory_system

    async def set_story_premise(self, user_context: str, story_title: str) -> str:
        print("Setting story Premise...")
        if not self.memory.get_long_term_document(name="story_user_context"):
            self.memory.add_long_term_document(text=user_context, metadata={"chapter_id":"story_user_context"})
            detailed_premise = await groq_client(system_prompt="You are a interactive text based Story author, Create an initial outline/premise for the story for " \
                                                    "the director to follow, given the user context. It should be the basis for the director to follow respective of user choices." \
                                                    "Include potential characters, world contexts, etc, which are relevant to the story. " \
                                                    "Include these details: guide prose, tone, genre, setting, POV, length, title and additional themes. For the director " \
                                                    "to follow.",
                                                    human_prompt=f"Given User Context: {user_context}")
            detailed_premise = detailed_premise.content.strip()
            self.memory.add_long_term_document(text=detailed_premise, metadata={"chapter_id":"story_premise", "story_title": story_title})
        else:
            detailed_premise = self.memory.get_long_term_document(name="story_premise")
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