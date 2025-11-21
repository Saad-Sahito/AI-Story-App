# src/utils/story_helpers.py

import re
import json
from typing import Any, Dict, Tuple, Optional
from langchain_core.messages import AIMessage
from src.llm_client.llm_client import ingestor_client
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

class StoryHelpers:
    @staticmethod
    def _count_words_split(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if not isinstance(text, str):
            return text
        if text.startswith("```"):
            # remove leading/trailing ```json ... ```
            return re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
        return text

    @staticmethod
    def _extract_content(resp: Any) -> str:
        """
        Normalize LLM responses to plain string content.
        Handles LangChain AIMessage, dicts, objects with .content,
        and plain strings.
        """
        # LangChain AIMessage
        try:
            if isinstance(resp, AIMessage):
                return resp.content
        except Exception:
            # If AIMessage isn't available or isinstance throws, continue
            pass

        # dict response (some SDKs return {"content": "..."} or {"text": "..."})
        if isinstance(resp, dict):
            if "content" in resp and isinstance(resp["content"], str):
                return resp["content"]
            if "text" in resp and isinstance(resp["text"], str):
                return resp["text"]
            # fallback: dump dict
            return json.dumps(resp)

        # object with .content attribute
        if hasattr(resp, "content") and isinstance(getattr(resp, "content"), str):
            return resp.content

        # already a string
        if isinstance(resp, str):
            return resp

        # fallback: stringify anything else
        return str(resp)

    @staticmethod
    def _coerce_character_world_field(val: Any) -> Dict[str, Any]:
        """
        Convert a variety of shapes into Dict[str,str]:
        - dict with non-str values => stringify values
        - list of "Name - summary" strings => convert to dict
        - list of dicts => merge into single dict
        - other => wrap as {"unknown": str(val)}
        """
        if isinstance(val, dict):
            out: Dict[str, str] = {}
            for k, v in val.items():
                out[str(k)] = v if isinstance(v, str) else json.dumps(v)
            return out

        if isinstance(val, list):
            out: Dict[str, str] = {}
            for item in val:
                if isinstance(item, str):
                    # try to split "Name – summary" or "Name: summary"
                    if "–" in item:
                        name, summary = item.split("–", 1)
                        out[name.strip()] = summary.strip()
                    elif ":" in item:
                        name, summary = item.split(":", 1)
                        out[name.strip()] = summary.strip()
                    else:
                        out[f"entry_{len(out)+1}"] = item
                elif isinstance(item, dict):
                    for k, v in item.items():
                        out[str(k)] = v if isinstance(v, str) else json.dumps(v)
                else:
                    out[f"entry_{len(out)+1}"] = json.dumps(item)
            return out

        # fallback: stringify
        return {"unknown": str(val)}

    @staticmethod
    def _normalize_payload_for_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalizes top-level fields commonly used in Chapter/Scene bundles:
        - ensure 'summary' is a string
        - ensure 'character_summary' and 'world_summary' are Dict[str,str]
        This mutates in-place (to avoid extra large copies) and returns the payload.
        """
        if not isinstance(payload, dict):
            return payload

        # operate in-place to avoid extra memory copies
        if "summary" in payload and not isinstance(payload["summary"], str):
            payload["summary"] = json.dumps(payload["summary"])

        for field in ("character_summary", "world_summary"):
            if field in payload:
                payload[field] = StoryHelpers._coerce_character_world_field(payload[field])

        return payload

    @staticmethod
    def _model_validate_or_parse_obj(bundle_cls, data: Dict[str, Any]):
        """Try pydantic v2 model_validate, otherwise parse_obj (v1)."""
        if hasattr(bundle_cls, "model_validate"):
            # pydantic v2
            return bundle_cls.model_validate(data)
        elif hasattr(bundle_cls, "parse_obj"):
            return bundle_cls.parse_obj(data)
        else:
            raise RuntimeError("No pydantic validation method found on class")

    @staticmethod
    def _try_validate_with_model_then_parser(
        raw_str: str, bundle_cls, parser
    ) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Exception]]:
        """
        Attempts:
        1) json.loads(raw_str) -> normalize -> model_validate/parse_obj
        2) parser.parse(raw_str) (LangChain PydanticOutputParser)
        Returns (success, result_dict_or_None, last_exception)
        """
        last_exc: Optional[Exception] = None

        # 1) try to load JSON -> validate with pydantic model
        try:
            data = json.loads(raw_str)
            data = StoryHelpers._normalize_payload_for_bundle(data)
            model = StoryHelpers._model_validate_or_parse_obj(bundle_cls, data)
            # model_dump may not exist for all versions, guard it
            if hasattr(model, "model_dump"):
                return True, model.model_dump(), None
            try:
                return True, dict(model), None
            except Exception:
                return True, json.loads(json.dumps(model)), None
        except Exception as e:
            last_exc = e
            # fall through to step 2

        # 2) try langchain parser
        try:
            parsed = parser.parse(raw_str)
            try:
                # parser.parse often returns a pydantic-like object with `.model_dump()`
                return True, parsed.model_dump(), None
            except Exception:
                # if not, try to coerce to dict
                try:
                    return True, dict(parsed), None
                except Exception:
                    return True, json.loads(json.dumps(parsed)), None
        except Exception as e:
            last_exc = e

        return False, None, last_exc
    
    @staticmethod
    async def load_json_with_retry(text: str, parser, max_attempts: int = 10):
        """Generic JSON loader with self-healing retry logic"""

        attempt = 0
        current_text = text

        while attempt < max_attempts:
            try:
                data = parser.parse(current_text)
                return data
            
            except Exception as e:
                print("attempting _json_fixer...")
                current_text = await StoryHelpers._json_fixer(
                    parser=parser,
                    exc=e
                )
                attempt += 1

        raise ValueError("❌ JSON parsing failed after maximum attempts")
    
    @staticmethod
    async def _json_fixer(parser, exc = None) -> str:
        """
        Uses the provided llm client to repair malformed JSON strings.
        """
        # Safely get format instructions – they are very useful for repair quality,
        # but we must not crash if the parser doesn't have the method (e.g. JsonOutputParser,
        # custom parser, or someone accidentally passed the pydantic class itself).
        format_instructions = ""
        if hasattr(parser, "get_format_instructions"):
            try:
                format_instructions = parser.get_format_instructions()
            except Exception:
                # Some parsers have the method but it can still raise – be defensive
                format_instructions = ""

        system_prompt = f"""
You are an expert JSON repair agent. Your only job is to output valid JSON.

Rules (follow exactly):
- Output ONLY the repaired JSON. No explanations, no markdown, no wrappers, no extra text.
- Never wrap the JSON in a string or add extra nesting.
- Preserve the exact same top-level keys as the original.
- If the input is already valid JSON, return it unchanged.
- Convert any null/None in string fields to empty string "".
- Fix trailing commas, missing quotes, unescaped characters, etc.
- Do not add, remove, or rename any fields.

Expected format (strictly adhere to this):
{format_instructions}
""".strip()

        prompt = f"""
Repair the following text into valid JSON.

{"Error that occurred during previous parsing attempt: " + str(exc) if exc else ""}
Output the fixed JSON now. Remember: ONLY the JSON, nothing else.
""".strip()

        resp = await ingestor_client(system_prompt=system_prompt, human_prompt=prompt)
        raw_text = StoryHelpers._extract_content(resp)
        clean_resp = StoryHelpers._strip_code_fences(raw_text)
        
        # Clean up big variables immediately – important when dealing with long stories
        del resp, raw_text, prompt, system_prompt 
        return clean_resp