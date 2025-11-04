from src.llm_client.llm_client import image_client
import base64

async def generate_cover_image(blurb: str) -> str:
        """
        Generate cover image based on blurb.
        
        Args:
            blurb: Book blurb text
            
        Returns:
            Base64-encoded image data
        """
        print("🎨 Generating cover image...")
        
        image_prompt = (
    f"Create a visually striking, professional book cover illustration inspired by the following blurb. "
    f"Focus on mood, setting, color palette, and atmosphere. "
    f"Do NOT include any text, titles, signatures, or images of an actual book. "
    f"The image should feel like a real book cover concept, artistic and marketable, with strong composition and emotional tone. "
    f"Output should be in 1024x1024 resolution.\n\n{blurb}"
)

        
        image_data = await image_client(image_prompt)
        image_data_base64 = base64.b64encode(image_data).decode('utf-8')
        
        print("✅ Cover image generated")
        return image_data_base64
