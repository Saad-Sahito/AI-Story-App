from src.llm_client.llm_client import LLMClient
from src.memory.memory_system import StoryMemorySystem

class StoryAuthor:
    def __init__(self, llm_client: LLMClient, memory_system: StoryMemorySystem):
        self.llm = llm_client
        self.memory = memory_system

    def set_story_premise(self, user_context: str, story_title: str) -> str:
        print("Setting story Premise...")
        if not self.memory.get_long_term_document(name="story_user_context"):
            self.memory.add_long_term_document(text=user_context, metadata={"chapter_id":"story_user_context"})
            detailed_premise = self.llm.groq_client(system_prompt="You are a text based Novel author, Create a detailed outline/premise for the entire story for " \
                                                    "the director to follow, given the user context. " \
                                                    "Include potential characters, world contexts, etc, which are relevant to the story. " \
                                                    "Include these details: guide prose, tone, genre, setting, POV, length, title and additional themes. For the director " \
                                                    "to follow.",
                                                    human_prompt=f"User Context: {user_context}")
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

    def set_story_synopsis(self, detailed_premise: str, story_title: str) -> str:
        print("Setting story Synopsis...")
        if not self.memory.get_long_term_document(name="story_synopsis"):
            synopsis = self.llm.groq_client(human_prompt=f"Create a summarised version of the following detailed premise: {detailed_premise}\nKeeping all important points needed for the director to create the story.")
            synopsis = synopsis.content.strip()
            self.memory.add_long_term_document(text=synopsis, metadata={"chapter_id":"story_synopsis", "story_title": story_title})
        else:
            synopsis = self.memory.get_long_term_document(name="story_synopsis")
            #synopsis = synopsis["text"]
        try:
            return synopsis
        except:
            return "An error occured, Please try again."