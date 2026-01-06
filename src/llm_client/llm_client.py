#from urllib import response
#from langchain_ollama import ChatOllama  # for local deployment only not render
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import replicate
import os
import requests
import asyncio
from asyncio import Semaphore
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# from langchain_community.callbacks import get_openai_callback
from openai import AsyncOpenAI
# from langchain_openai import OpenAI  # Optional, for prompt refinement

SHARED_LLM_CLIENT = None

# GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
# CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805"]
# GPT_MODELS = ["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-4o-mini-2024-07-18", "gpt-5-2025-08-07", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "gpt-5-pro-2025-10-06"]
# GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
# GROK_MODELS = ["grok-4-fast-reasoning"]

# --- Provider Wrappers ---

class BaseWrapper:
    """Base class for LLM provider wrappers."""
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.client = None  # To be initialized in subclasses

    async def ainvoke(self, messages, temperature: float, model: str):
        """Async invoke method to be implemented by subclasses."""
        raise NotImplementedError("ainvoke must be implemented by subclasses")

    def get_token_usage(self, response) -> dict:
        """Extract token usage from response; implemented by subclasses."""
        raise NotImplementedError("get_token_usage must be implemented by subclasses")


class AnthropicWrapper(BaseWrapper):
    """Wrapper for Anthropic (Claude) provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Token keys specific to Anthropic
        self.token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": None}
        self.usage_key = "usage"

    async def ainvoke(self, messages, temperature: float, model: str):
        self.client = ChatAnthropic(model_name=model, temperature=temperature, api_key=self.api_key, max_tokens_to_sample=12000)
        return await self.client.ainvoke(messages)

    def get_token_usage(self, response) -> dict:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        meta = getattr(response, "response_metadata", {})
        usage = meta.get(self.usage_key, {})
        if usage:
            token_usage["prompt_tokens"] = usage.get(self.token_keys["prompt"], 0)
            token_usage["completion_tokens"] = usage.get(self.token_keys["completion"], 0)
            if self.token_keys["total"] is None:
                token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
            else:
                token_usage["total_tokens"] = usage.get(self.token_keys["total"], 0)
        return token_usage


class OpenAIWrapper(BaseWrapper):
    """Wrapper for OpenAI (GPT) provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Token keys specific to OpenAI
        self.token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
        self.usage_key = "token_usage"

    async def ainvoke(self, messages, temperature: float, model: str):
        self.client = ChatOpenAI(model=model, temperature=temperature, openai_api_key=self.api_key)
        return await self.client.ainvoke(messages)

    def get_token_usage(self, response) -> dict:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        meta = getattr(response, "response_metadata", {})
        usage = meta.get(self.usage_key, {})
        if usage:
            token_usage["prompt_tokens"] = usage.get(self.token_keys["prompt"], 0)
            token_usage["completion_tokens"] = usage.get(self.token_keys["completion"], 0)
            token_usage["total_tokens"] = usage.get(self.token_keys["total"], 0)
        return token_usage


class GoogleWrapper(BaseWrapper):
    """Wrapper for Google (Gemini) provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Token keys specific to Google Gemini (assuming langchain_google_genai)
        self.token_keys = {"prompt": "prompt_token_count", "completion": "candidates_token_count", "total": "total_token_count"}
        self.usage_key = "usage_metadata"  # Gemini stores it directly on response sometimes

    async def ainvoke(self, messages, temperature: float, model: str):
        from langchain_google_genai import ChatGoogleGenerativeAI  # Lazy import
        self.client = ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=self.api_key)
        return await self.client.ainvoke(messages)

    def get_token_usage(self, response) -> dict:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        # Gemini might have usage_metadata directly on response
        usage = getattr(response, self.usage_key, {})
        if not usage:
            meta = getattr(response, "response_metadata", {})
            usage = meta.get(self.usage_key, {})
        if usage:
            token_usage["prompt_tokens"] = usage.get(self.token_keys["prompt"], 0)
            token_usage["completion_tokens"] = usage.get(self.token_keys["completion"], 0)
            token_usage["total_tokens"] = usage.get(self.token_keys["total"], 0)
        return token_usage


class GroqWrapper(BaseWrapper):
    """Wrapper for Groq provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Token keys specific to Groq (similar to OpenAI)
        self.token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
        self.usage_key = "token_usage"

    async def ainvoke(self, messages, temperature: float, model: str):
        self.client = ChatGroq(model=model, temperature=temperature, groq_api_key=self.api_key)
        return await self.client.ainvoke(messages)

    def get_token_usage(self, response) -> dict:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        meta = getattr(response, "response_metadata", {})
        usage = meta.get(self.usage_key, {})
        if usage:
            token_usage["prompt_tokens"] = usage.get(self.token_keys["prompt"], 0)
            token_usage["completion_tokens"] = usage.get(self.token_keys["completion"], 0)
            token_usage["total_tokens"] = usage.get(self.token_keys["total"], 0)
        return token_usage


class XAIWrapper(BaseWrapper):
    """Wrapper for xAI (Grok) provider."""
    def __init__(self, api_key: str):
        super().__init__(api_key)
        # Token keys specific to xAI (similar to OpenAI)
        self.token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
        self.usage_key = "token_usage"

    async def ainvoke(self, messages, temperature: float, model: str):
        from langchain_xai import ChatXAI
        self.client = ChatXAI(xai_api_key=self.api_key, temperature=temperature, model=model)
        return await self.client.ainvoke(messages)

    def get_token_usage(self, response) -> dict:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        meta = getattr(response, "response_metadata", {})
        usage = meta.get(self.usage_key, {})
        if usage:
            token_usage["prompt_tokens"] = usage.get(self.token_keys["prompt"], 0)
            token_usage["completion_tokens"] = usage.get(self.token_keys["completion"], 0)
            token_usage["total_tokens"] = usage.get(self.token_keys["total"], 0)
        return token_usage


# class OllamaWrapper(BaseWrapper):
#     """Wrapper for Ollama (local) provider."""
#     def __init__(self, api_key: str = None):  # Ollama doesn't need API key
#         super().__init__(api_key)  # api_key is ignored
#         # Token keys for Ollama (may not have detailed usage; stubbed)
#         self.token_keys = {"prompt": "prompt_eval_count", "completion": "eval_count", "total": None}
#         self.usage_key = "response_metadata"  # Ollama has basic metrics in metadata

#     async def ainvoke(self, messages, temperature: float, model: str):
#         from langchain_ollama import ChatOllama  # Lazy import
#         self.client = ChatOllama(model=model, temperature=temperature)
#         return await self.client.ainvoke(messages)  # Note: Ollama may not support async natively; adjust if needed

#     def get_token_usage(self, response) -> dict:
#         token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
#         meta = getattr(response, "response_metadata", {})
#         if meta:
#             token_usage["prompt_tokens"] = meta.get(self.token_keys["prompt"], 0)
#             token_usage["completion_tokens"] = meta.get(self.token_keys["completion"], 0)
#             if self.token_keys["total"] is None:
#                 token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
#             else:
#                 token_usage["total_tokens"] = meta.get(self.token_keys["total"], 0)
#         return token_usage


class LLMClient:
    def __init__(self):
        self.sem = Semaphore(5)  # Limit concurrent LLM calls
        # Load environment variables from .env file
        load_dotenv()
        try:
            # Get API keys from the environment
            self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.claude_api_key = os.environ.get("CLAUDE_API_KEY")
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            self.google_api_key = os.environ.get("GOOGLE_API_KEY")
            self.xai_api_key = os.environ.get("XAI_API_KEY")
            # No API key for Ollama
        except Exception as e:
            print(f"Failed to initialize LLMClient: {e}")
            raise

    # def llama3_1_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    #     """Blocking call — returns the full response."""
    #     return self.llm_ollama.invoke([
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content=human_prompt)
    #     ])


    async def _image_client(self, prompt: str = ""):
        async with self.sem:  # Use the same semaphore as other methods
            try:
                # Run Replicate API call in a thread to keep it async-compatible
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    lambda: replicate.run(
                        "stability-ai/stable-diffusion-3.5-medium",
                        input={
                            "prompt": prompt,
                            #"num_outputs": 1,  # Match DALL-E's n=1

                            # "height": 1024,    # Match DALL-E's 1024x1024
                            # "width": 1024,
                            #"num_inference_steps": 4,  # Fast inference for schnell
                            #"guidance_scale": 7.5,    # Default for FLUX
                            "negative_prompt": "text, typography, letters, title, logo, signature, watermark, words, book object, frame, collage",

                            "output_format": "png"    # Ensure PNG output
                        }
                    )
                )
                # Replicate returns a list of URLs; get the first one
                image_url = output[0] if isinstance(output, list) else output
                print(f"Generated Image URL: {image_url}")
            except Exception as e:
                print(f"Error generating image: {str(e)}")
                raise
            await asyncio.sleep(1)  # Add a small delay to avoid rate limits

            # Download image bytes (same as your original code)
            try:
                response = await loop.run_in_executor(
                    None,
                    lambda: requests.get(image_url, stream=True)
                )
                response.raise_for_status()
                return response.content  # Return image bytes, matching original
            except requests.RequestException as e:
                print(f"Error downloading image: {str(e)}")
                raise

    async def _utility_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Suggests model for user context."""
        # Use GroqWrapper as an example; replace with desired wrapper
        wrapper = GroqWrapper(self.groq_api_key)
        #llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=0.8, model="grok-4-fast-reasoning")
        #llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8, groq_api_key=self.groq_api_key)

        #llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=self.openai_api_key)
        #with get_openai_callback() as cb:
        async with self.sem:
            response = await wrapper.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ], temperature=0.1, model="openai/gpt-oss-120b")
        token_usage = wrapper.get_token_usage(response)
                # Validate token usage
        if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
            print(f"Warning: Invalid token usage values for model: {token_usage}")
        return response, token_usage

    async def _enhanced_utility_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to LLM – returns the full response."""
        # Use GroqWrapper as example; replace with GoogleWrapper if needed
        #wrapper = GoogleWrapper(self.google_api_key)
        wrapper = GroqWrapper(self.groq_api_key)
        #wrapper = XAIWrapper(self.xai_api_key)
        # wrapper = OpenAIWrapper(self.openai_api_key)
        async with self.sem:
            response = await wrapper.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ], temperature=0.1, model="llama-3.3-70b-versatile")
        token_usage = wrapper.get_token_usage(response)
        # Validate token usage
        if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
            print(f"Warning: Invalid token usage values for model: {token_usage}")
        await asyncio.sleep(1)
        return response, token_usage
    
    async def _better_author_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "gpt-5-mini-2025-08-07"
            # wrapper = OpenAIWrapper(self.openai_api_key)
            # model = "grok-4-0709"
            # wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-sonnet-4-5-20250929"
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            model = "openai/gpt-oss-120b"
            # model = "llama-3.3-70b-versatile"
            wrapper = GroqWrapper(self.groq_api_key)
            # model = "gemini-3-pro-preview"
            # wrapper = GoogleWrapper(self.google_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)

            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _author_client for model {model}: {str(e)}")
            raise

    async def _author_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "gpt-5-mini-2025-08-07"
            # wrapper = OpenAIWrapper(self.openai_api_key)
            # llm_temp = 1
            # model = "grok-4-fast-reasoning"
            # wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-sonnet-4-5-20250929"
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            model = "openai/gpt-oss-120b"
            # model = "llama-3.3-70b-versatile"
            wrapper = GroqWrapper(self.groq_api_key)
            # model = "gemini-3-pro-preview"
            # wrapper = GoogleWrapper(self.google_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)
            # print(f"Author Client Token Usage: {token_usage}")
            # Validate token usage
            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _author_client for model {model}: {str(e)}")
            raise
    
    async def _author_fast_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "grok-4-fast-reasoning"
            # wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-sonnet-4-5-20250929"
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            model = "openai/gpt-oss-120b"
            wrapper = GroqWrapper(self.groq_api_key)
            # model = "gemini-3-pro-preview"
            # wrapper = GoogleWrapper(self.google_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)
            # print(f"Author Client Token Usage: {token_usage}")
            # Validate token usage
            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _author_client for model {model}: {str(e)}")
            raise

    async def _director_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "gpt-5-mini-2025-08-07"
            # wrapper = OpenAIWrapper(self.openai_api_key)
            # llm_temp = 1
            # model = "grok-4-fast-reasoning"
            # wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            # model = "llama-3.3-70b-versatile"
            model = "openai/gpt-oss-120b"
            wrapper = GroqWrapper(self.groq_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)

            # Validate token usage
            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _director_client for model {model}: {str(e)}")
            raise

    async def _better_writer_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "gpt-5-mini-2025-08-07"
            # wrapper = OpenAIWrapper(self.openai_api_key)
            # llm_temp = 1
            #model = "claude-3-haiku-20240307"
            #wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            model = "openai/gpt-oss-120b"
            # model = "llama-3.3-70b-versatile"
            wrapper = GroqWrapper(self.groq_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)

            # Validate token usage
            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)
            # print("WRTIER RESPONSE: ", response.content)
            return response, token_usage
        
        except Exception as e:
            print(f"Error in _writer_client for model {model}: {str(e)}")
            raise

    async def _writer_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            #model = "claude-3-haiku-20240307"
            
            #wrapper = XAIWrapper(self.xai_api_key)
            # model = "claude-haiku-4-5-20251001"
            # wrapper = AnthropicWrapper(self.claude_api_key)
            model = "openai/gpt-oss-120b"
            # model = "llama-3.3-70b-versatile"
            wrapper = GroqWrapper(self.groq_api_key)
            async with self.sem:
                    response = await wrapper.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ], temperature=llm_temp, model=model)
            token_usage = wrapper.get_token_usage(response)

            # Validate token usage
            if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # Delay for rate limiting safety
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _writer_client for model {model}: {str(e)}")
            raise

    async def _ingestor_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to LLM – returns the full response."""
        # Use GroqWrapper as example; replace with GoogleWrapper if needed
        #wrapper = GoogleWrapper(self.google_api_key)
        wrapper = GroqWrapper(self.groq_api_key)
        #wrapper = XAIWrapper(self.xai_api_key)
        #wrapper = OpenAIWrapper(self.openai_api_key)
        async with self.sem:
            response = await wrapper.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ], temperature=0.4, model="openai/gpt-oss-120b")
        token_usage = wrapper.get_token_usage(response)
        # Validate token usage
        if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
            print(f"Warning: Invalid token usage values for model: {token_usage}")
        await asyncio.sleep(1)
        # print("INGESTOR RESPONSE: ", response.content)
        return response, token_usage


    


init_lock = asyncio.Lock()
# Convenience functions to access the shared instance
async def get_shared_client():
    async with init_lock:
        global SHARED_LLM_CLIENT
        if SHARED_LLM_CLIENT is None:
            SHARED_LLM_CLIENT = LLMClient()
        return SHARED_LLM_CLIENT

# Convenience wrapper functions for easy access
async def better_author_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._better_author_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def author_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._author_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def author_fast_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._author_fast_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def director_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._director_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def writer_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def better_writer_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7) -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._better_writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def ingestor_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    """Convenience function to access gemini_client through shared instance."""
    client = await get_shared_client()
    return await client._ingestor_client(system_prompt, human_prompt)

async def enhanced_utility_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    """Convenience function to access gemini_client through shared instance."""
    client = await get_shared_client()
    return await client._enhanced_utility_client(system_prompt, human_prompt)

async def utility_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    """Convenience function to access model through shared instance."""
    client = await get_shared_client()
    return await client._utility_client(system_prompt, human_prompt)

async def image_client(prompt: str = "") -> AIMessage:
    """Convenience function to access model through shared instance."""
    client = await get_shared_client()
    return await client._image_client(prompt)