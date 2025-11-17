#from src.story_engines.classic_narrative.agents.story_author import StorySeed
from src.llm_client.llm_client import utility_client

async def generate_blurb(minimal_plot) -> str:
        """
        Create compelling back-cover blurb from story seed.
        
        Args:
            minimal_plot: The MinimalPlot object
            
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
        

        plot_summary = f"""
Premise: {minimal_plot.premise}
Central Question: {minimal_plot.central_question}
Narrative Arc: {minimal_plot.narrative_arc}
"""
        
        response = await utility_client(
            system_prompt=system_prompt,
            human_prompt=f"Create a blurb for:\n\n{plot_summary}"
        )
        
        blurb = response.content.strip()
        print("✅ Blurb generated")
        return blurb