# src/utils/story_helpers.py

import re
import json
from typing import Any, Dict, Tuple, Optional
from src.llm_client.llm_client import ingestor_client, enhanced_ingestor_client
from json_repair import repair_json
from pydantic import BaseModel


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
        if hasattr(resp, "content"):
            return str(getattr(resp, "content", "") or "")
        if isinstance(resp, dict):
            return resp.get("content") or resp.get("text") or json.dumps(resp)
        return str(resp)
    
    # @staticmethod
    # def _extract_content(resp: Any) -> str:
    #     """
    #     Normalize LLM responses to plain string content.
    #     Handles LangChain AIMessage, dicts, objects with .content,
    #     and plain strings.
    #     """
    #     # LangChain AIMessage
    #     try:
    #         if isinstance(resp, AIMessage):
    #             return resp.content
    #     except Exception:
    #         # If AIMessage isn't available or isinstance throws, continue
    #         pass

    #     # dict response (some SDKs return {"content": "..."} or {"text": "..."})
    #     if isinstance(resp, dict):
    #         if "content" in resp and isinstance(resp["content"], str):
    #             return resp["content"]
    #         if "text" in resp and isinstance(resp["text"], str):
    #             return resp["text"]
    #         # fallback: dump dict
    #         return json.dumps(resp)

    #     # object with .content attribute
    #     if hasattr(resp, "content") and isinstance(getattr(resp, "content"), str):
    #         return resp.content

    #     # already a string
    #     if isinstance(resp, str):
    #         return resp

    #     # fallback: stringify anything else
    #     return str(resp)

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
    async def load_json_with_retry(text: str, parser: Any, max_attempts: int = 6) -> Any:
        """
        Robust JSON + Pydantic parser with smart retries:
        1. Try direct parse using best available parser method
        2. Try json-repair
        3. Use LLM fix (regular → enhanced)
        """
        current_text = (text or "").strip()

        async def try_parse(text_or_obj, parser):
            last_exc = None

            # Handle LangChain's PydanticOutputParser specifically
            if hasattr(parser, "parse") and callable(getattr(parser, "parse")):
                try:
                    # PydanticOutputParser expects a JSON *string*, not a dict
                    if isinstance(text_or_obj, (dict, list)):
                        text_or_obj = json.dumps(text_or_obj)
                    return parser.parse(text_or_obj)
                except Exception as e:
                    last_exc = e

            # If parser has pydantic_model attribute (common in PydanticOutputParser), use that
            actual_model = getattr(parser, "pydantic_model", None) or getattr(parser, "_pydantic_model", None)
            if actual_model:
                parser = actual_model  # redirect to real model

            # Now proceed with standard Pydantic model parsing methods
            try:
                if hasattr(parser, "model_validate_json") and isinstance(text_or_obj, str):
                    return parser.model_validate_json(text_or_obj)
            except Exception as e:
                last_exc = e

            try:
                if hasattr(parser, "parse_raw") and isinstance(text_or_obj, str):
                    return parser.parse_raw(text_or_obj)
            except Exception as e:
                last_exc = e

            try:
                if not isinstance(text_or_obj, str):
                    if hasattr(parser, "model_validate"):
                        return parser.model_validate(text_or_obj)
                    if hasattr(parser, "parse_obj"):
                        return parser.parse_obj(text_or_obj)
            except Exception as e:
                last_exc = e

            # Remove this line entirely — dangerous and caused the crash!
            # try:
            #     return parser(text_or_obj)
            # except Exception as e:
            #     last_exc = e

            raise last_exc or ValueError("No parser method succeeded")
        # main retry loop
        for attempt in range(max_attempts):
            # 1) Try direct parse attempt
            try:
                if isinstance(current_text, str) and current_text.strip().startswith(('{', '[')):
                    try:
                        # First try as raw JSON string (many parsers accept this)
                        result = await try_parse(current_text, parser)
                        return result
                    except Exception:
                        pass

                # Fall back to loading JSON first then parsing object
                try:
                    obj = json.loads(current_text)
                    return await try_parse(obj, parser)
                except Exception as e:
                    last_error = e

            except Exception as e:
                # move on to repair attempts
                last_error = e

            # 2) Attempts 0-1: try json-repair
            if attempt < 2:
                try:
                    repaired = repair_json(current_text, return_objects=False)
                    # if repair_json returned a new string, validate it
                    if repaired and repaired != current_text:
                        current_text = repaired.strip()
                        try:
                            obj = json.loads(current_text)
                            return await try_parse(obj, parser)
                        except Exception as e:
                            # keep going to further repairs if parse fails
                            last_error = e
                except Exception:
                    # ignore repair_json internal errors, continue
                    pass

            # 3) Attempts >= 2: use LLM fix (regular first, enhanced for later attempts)
            if attempt >= 2:
                enhanced = attempt >= 4
                try:
                    current_text = await StoryHelpers._repair_json_with_llm(
                        current_text, parser, enhanced=enhanced, exc=last_error
                    )
                    # validate produced JSON before handing to parser
                    obj = json.loads(current_text)
                    return await try_parse(obj, parser)
                except Exception as e:
                    # on failure, continue to next loop iteration (retry)
                    last_error = e
                    continue

            # loop continues automatically; no manual attempt increment

        # After exhausting attempts, raise informative error
        raise ValueError(f"Failed to parse JSON after {max_attempts} attempts. Last error: {last_error}")
    
    # @staticmethod
    # def _generate_json_schema_for_llm(model: type[BaseModel]) -> str:
    #     """Generate clean, LLM-friendly schema with field names, types, and descriptions."""
    #     schema = model.model_json_schema()
    #     lines = ["REQUIRED JSON STRUCTURE (exact field names):", ""]

    #     required = set(schema.get("required", []))

    #     for name, info in schema["properties"].items():
    #         req = " [REQUIRED]" if name in required else ""
    #         desc = f" — {info.get('description', '').strip()}" if info.get("description") else ""
    #         type_ = info.get("type", "any")
    #         if "$ref" in info:
    #             type_ = info["$ref"].split("/")[-1]
    #         elif "anyOf" in info:
    #             refs = [r["$ref"].split("/")[-1] for r in info["anyOf"] if "$ref" in r]
    #             type_ = " | ".join(refs) if refs else "union"
            
    #         default = ""
    #         if "default" in info and info["default"] is not None:
    #             default = f" (default: {json.dumps(info['default'])})"

    #         lines.append(f"- {name}: {type_}{default}{req}{desc}")

    #         # Show nested structure for complex fields
    #         if info.get("type") == "object" and "properties" in info:
    #             lines.append("  Contains:")
    #             for sub_name, sub_info in info["properties"].items():
    #                 sub_req = f" [REQUIRED]" if sub_name in info.get("required", []) else ""
    #                 sub_desc = f" — {sub_info.get('description', '')}" if sub_info.get("description") else ""
    #                 lines.append(f"    - {sub_name}: {sub_info.get('type', 'any')}{sub_req}{sub_desc}")
    #         elif info.get("type") == "array" and "items" in info and "$ref" in info["items"]:
    #             ref_name = info["items"]["$ref"].split("/")[-1]
    #             lines.append(f"  → List of {ref_name} objects")

    #     lines.extend([
    #         "",
    #         "Output ONLY valid JSON matching this exact structure.",
    #         "No extra fields. No renamed fields. No missing required fields."
    #     ])
    #     return "\n".join(lines)
    
    @staticmethod
    async def _repair_json_with_llm(text: str, parser, enhanced: bool = False, exc: Exception = None) -> str:
        """
        Unified JSON repair using LLM — now works perfectly with raw Pydantic models.
        """
        print("Attempting LLM JSON repair" + (" (enhanced)" if enhanced else ""))

        # if hasattr(parser, "get_format_instructions") and callable(getattr(parser, "get_format_instructions")):
        #     format_instructions = parser.get_format_instructions()
        # else:
        #     # Raw Pydantic model → use our beautiful schema printer
        format_instructions = parser.model_json_schema()

        system_prompt = f"""You are an expert JSON repair agent. Your job is to fix broken JSON and make it 100% valid.

CRITICAL RULES:
- Output ONLY valid JSON. No explanations, no markdown, no extra text.
- Fix syntax: missing commas, quotes, brackets, trailing commas
- Preserve ALL existing data
- Do NOT add, remove, or rename any fields
- Convert null/None/"null" in strings → ""
- If a field is missing but required → you may NOT guess it
- The JSON must exactly match this schema:

{format_instructions}

Fix the broken JSON below and return ONLY the corrected version.
"""

        human_prompt = f"""Broken JSON to fix:
{text}

Last error (for context):
{str(exc) if exc else "Unknown parsing error"}

Return only the fixed JSON."""
        
        client = enhanced_ingestor_client if enhanced else ingestor_client
        resp, _ = await client(system_prompt=system_prompt, human_prompt=human_prompt)

        content = StoryHelpers._extract_content(resp)
        cleaned = StoryHelpers._strip_code_fences(content.strip())

        # Final sanity check
        try:
            json.loads(cleaned)
        except Exception as e:
            raise ValueError("LLM repair failed — returned invalid JSON") from e

        del resp, content
        return cleaned