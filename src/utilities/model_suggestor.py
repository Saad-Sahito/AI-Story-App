from src.llm_client.llm_client import utility_client
from src.utilities.story_helpers import StoryHelpers
import json
import gc
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

tier_1 = {
    "gpt-5-nano-2025-08-07": "Lightweight yet capable for compact narratives (500-3,000 words) in genres like mystery, romance, or slice-of-life. " \
    "Shines in first-person POV with intimate settings (e.g., small towns, personal journeys). Ideal for classic stories with emotional " \
    "or reflective tones. Handles prose styles like \"lyrical\" or \"introspective\" well, especially for themes like personal growth or loss.",
    
    "openai/gpt-oss-120b": "Versatile for medium-length stories (2,000-7,000 words) in genres like thriller, epic fantasy, or horror. " \
    "Strong in third-person POV with complex settings (e.g., sprawling cities, haunted forests). Supports classic stories with darker " \
    "or suspenseful tones. Fits prose styles like \"dark\" or \"evocative\" and themes like betrayal or survival.",
    "llama-3.3-70b-versatile": "Powerful for longer narratives (5,000-15,000 words) in genres like high fantasy, historical epics, or intricate mysteries. " \
    "Excels in omniscient POV with richly detailed worlds (e.g., fantasy realms, ancient civilizations). Ideal for classic stories with grand " \
    "or dramatic tones. Handles elaborate prose styles (\"cinematic,\" \"epic\") and themes like heroism or destiny."
}
# "gemini-2.5-flash-lite": "Multimodal support enhances stories with vivid settings (e.g., alien planets, historical eras) in genres like sci-fi, " \
#     "historical fiction, or fantasy. Great for short to medium stories (1,000-5,000 words) in third-person or omniscient POV. Suits interactive stories " \
#     "with dynamic tones. Excels in descriptive prose (\"cinematic,\" \"immersive\") and themes like exploration or cultural identity.",
tier_2 = {
    "gpt-5-mini-2025-08-07": "Enhanced for mid-length narratives (2,000-8,000 words) in genres like mystery, fantasy, or psychological thriller. " \
    "Strong in first-person POV with intricate settings (e.g., detective agencies, magical realms). Fits interactive stories with shifting tones. " \
    "Handles prose like \"engaging\" or \"suspenseful\" and themes like deception or redemption.",
    "gpt-4o-mini-2024-07-18": "Efficient for character-driven stories (1,000-5,000 words) in genres like romance, drama, or coming-of-age. "
    "Excels in first- or second-person POV with relatable settings (e.g., urban life, schools). Ideal for classic stories with heartfelt or" \
    " nuanced tones. Supports prose like \"conversational\" or \"emotional\" and themes like love or self-discovery.",
    "gemini-2.5-flash": "Dynamic for medium-length interactive stories (3,000-10,000 words) in genres like sci-fi, adventure, or alternate history. " \
    "Excels in third-person POV with vivid, multimodal-inspired settings (e.g., futuristic cities, war-torn lands). Suits bold or epic tones. " \
    "Strong for prose like \"action-packed\" or \"world-building\" and themes like heroism or innovation.",
    "claude-haiku-4-5-20251001": "Fast and human-like for concise stories (1,000-5,000 words) in genres like literary fiction, fantasy, or satire. Shines in " \
    "first- or third-person POV with evocative settings (e.g., mythical forests, quiet villages). Ideal for classic stories with witty or poignant tones. " \
    "Fits prose like \"poetic\" or \"witty\" and themes like morality or identity."
}

tier_3 = {
    "claude-sonnet-4-5-20250929": "Superior for intricate stories (5,000-20,000 words) in genres like literary fiction, fantasy, or psychological drama. S" \
    "hines in any POV with rich settings (e.g., enchanted realms, introspective minds). Suits classic or interactive stories with nuanced tones. " \
    "Excels in prose like \"lyrical\" or \"philosophical\" and themes like love, loss, or existentialism.",
    "gpt-5-2025-08-07": "Advanced for long-form stories (5,000-20,000 words) in genres like epic fantasy, sci-fi, or historical drama. Excels in any POV with " \
    "complex settings (e.g., galactic empires, medieval courts). Suits interactive or classic stories with varied tones. Handles prose like \"epic\" or " \
    "\"nuanced\" and themes like power, legacy, or sacrifice.",
    "gpt-4o-2024-08-06": "Multimodal for immersive stories (5,000-15,000 words) in genres like sci-fi, horror, or adventure. Excels in second- or third-person POV " \
    "with vivid settings (e.g., alien worlds, haunted mansions). Ideal for interactive stories with cinematic tones. Fits prose like \"vivid\" or " \
    "\"atmospheric\" and themes like fear or discovery.",
    "gemini-2.5-pro": "Advanced for multimodal, long-form stories (8,000-30,000 words) in genres like sci-fi, adventure, or historical fiction. Strong in " \
    "third-person or omniscient POV with vivid settings (e.g., futuristic dystopias, ancient civilizations). Ideal for interactive stories with bold tones. " \
    "Fits prose like \"cinematic\" or \"descriptive\" and themes like exploration or revolution.",
    "gpt-4.1-2025-04-14": "Reliable for detailed narratives (4,000-15,000 words) in genres like thriller, fantasy, or dystopian. Strong in third-person POV with " \
    "gritty settings (e.g., post-apocalyptic wastelands, crime-ridden cities). Fits classic stories with intense or dramatic tones. Supports prose like " \
    "\"gritty\" or \"intense\" and themes like justice or survival."
}

tier_4 = {
    "claude-opus-4-1-20250805": "Exceptional for epic, multi-layered stories (10,000–50,000 words) in genres like high fantasy, historical epic, or literary fiction. " \
    "Masterful in any POV with expansive settings (e.g., ancient empires, cosmic voids). Ideal for classic stories with profound tones. Fits prose like " \
    "\"eloquent\" or \"intricate\" and themes like destiny or morality.",
    "gpt-5-pro-2025-10-06": "Flagship for professional-grade narratives (10,000–50,000 words) in genres like epic sci-fi, fantasy, or thriller. Excels in any POV with " \
    "complex, world-building settings (e.g., interstellar wars, mythical continents). Suits interactive or classic stories with any tone. Handles prose like " \
    "\"epic\" or \"immersive\" and themes like power, betrayal, or transcendence."
}


class SelectedModel(BaseModel):
    model: str = Field(
        description="The model selected for the story."
    )
model_parser = PydanticOutputParser(pydantic_object=SelectedModel)

async def _model_suggestor_ai(user_context: str, models: dict) -> str:
    format_instructions = model_parser.get_format_instructions()
    resp = await utility_client(
        system_prompt=f"""
    You are an AI model suggestor for an AI Storytelling application.
    Based on the user context provided, suggest the most suitable AI model to use for generating the story.
    AI models available:
    {', '.join([f'- {model_name}: {desc}' for model_name, desc in models.items()])}
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
    success, result, exc = StoryHelpers._try_validate_with_model_then_parser(clean_resp, SelectedModel, model_parser)
    if success:
        return(result.get("model"))
    else:
        return("")

async def user_context_extractor_model(user_context: dict, tier: int) -> dict:
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
    if tier == 1:
        models = tier_1
    elif tier == 2:
        models = tier_2
    elif tier == 3:
        models = tier_3
    elif tier == 4:
        models = tier_4
    try:
        for _ in range(2):
            resp = await _model_suggestor_ai(user_context=form_string, models=models)
            if resp in models.keys():
                print(resp)
                return {"status": "success", "data": resp }
        return {"status": "error", "message": "model_parser failed"}
    except Exception as e:
        return {"status": "error", "message": str(e)}