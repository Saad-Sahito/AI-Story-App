from src.story_engines.classic_narrative.agents.story_author import StorySeed
from src.llm_client.llm_client import utility_client

async def generate_blurb(story_seed: StorySeed) -> str:
        """
        Create compelling back-cover blurb from story seed.
        
        Args:
            story_seed: The StorySeed object
            
        Returns:
            Marketing blurb as string
        """
        print("📖 Generating book blurb...")
        
        system_prompt = """You are a professional book marketer specializing in back-cover copy.

Write a captivating blurb (<50 words) that:
- Hooks readers emotionally
- Establishes atmosphere and stakes
- Teases the protagonist's journey
- Matches the story's genre and tone
- Avoids spoilers or major plot reveals

Style: Professional, marketable, similar to what you'd find in a bookstore.

Output only the blurb—no commentary or formatting."""
        genres_str = ", ".join(story_seed.genre)  # Updated to handle list of genres
        # Format seed info for context
        seed_summary = f"""
Title: {story_seed.title}
Genre: {genres_str}
Tone: {story_seed.tone}
Premise: {story_seed.premise}
Protagonist: {story_seed.protagonist}
Central Conflict: {story_seed.central_conflict}
Themes: {', '.join(story_seed.themes)}
Setting: {story_seed.world_essentials.get('setting', 'Unknown')}
"""
        
        response = await utility_client(
            system_prompt=system_prompt,
            human_prompt=f"Create a blurb for:\n\n{seed_summary}"
        )
        
        blurb = response.content.strip()
        print("✅ Blurb generated")
        return blurb