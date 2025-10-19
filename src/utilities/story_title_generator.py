from src.llm_client.llm_client import utility_client
from src.utilities.story_helpers import StoryHelpers
import json
import gc
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class StoryTitle(BaseModel):
    title: str = Field(
        description="The Story Title for the story."
    )
title_parser = PydanticOutputParser(pydantic_object=StoryTitle)

async def _title_suggestor_ai(user_context: str) -> str:
    format_instructions = title_parser.get_format_instructions()
    resp = await utility_client(
        system_prompt=f"""
    You are an AI story title generator for an AI Storytelling application.
    Based on the user context provided, suggest the most suitable story title based on user context.

    Use the following format:
    {format_instructions}
    """,
        human_prompt=f"Given User Context: {user_context}"
    )
    
    raw_text = StoryHelpers._extract_content(resp)
    clean_resp = StoryHelpers._strip_code_fences(raw_text)
    del raw_text, resp
    gc.collect()
    if isinstance(clean_resp, dict):
        clean_resp = json.dumps(clean_resp)

    # Try validate/parse and get dict back
    success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, StoryTitle, title_parser)
    if success:
        return(result.get("title"))
    else:
        return("")

async def user_context_extractor_title(user_context: dict) -> dict:
    print("Generating Title....")
    tone_dict = {
            0: "Playful", 20: "Lighthearted", 40: "Adventurous",
            60: "Dramatic", 80: "Serious", 100: "Intense"
        }
    length_dict = {
        0: "Short Long Story (7,500 - 15,000 words)",
        20: "Novelette (15,000 - 25,000 words)",
        40: "Novella (25,000 - 40,000 words)",
        60: "Novel Chapter (40,000 - 60,000 words)",
        80: "Full Novel (60,000 - 90,000 words)",
        100: "Epic / Series (90,000 - 150,000+ words)"
    }

    if "Tone" in user_context:
        tone_value = int(user_context["Tone"])
        mapped_tone = tone_dict.get(tone_value)
        if mapped_tone is None:
            mapped_tone = tone_dict[min(tone_dict.keys(), key=lambda k: abs(k - tone_value))]
        user_context["Tone"] = mapped_tone

    if "Length" in user_context:
        length_value = int(user_context["Length"])
        mapped_length = length_dict.get(length_value)
        if mapped_length is None:
            mapped_length = length_dict[min(length_dict.keys(), key=lambda k: abs(k - length_value))]
        user_context["Length"] = mapped_length

    filtered_data = {k: v for k, v in user_context.items() if k not in ["story_id", "user_id"]}
    form_string = "\n".join([f"{k.capitalize()}: {v}" for k, v in filtered_data.items()])

    try:
        return {"status": "success", "data": await _title_suggestor_ai(user_context=form_string) }
    except Exception as e:
        return {"status": "error", "message": str(e)}