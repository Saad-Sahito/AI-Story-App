#from urllib import response
#from langchain_ollama import ChatOllama  # for local deployment only not render
from langchain_openai import ChatOpenAI
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_xai import ChatXAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import replicate
import os
import requests
import asyncio
from asyncio import Semaphore
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.callbacks import get_openai_callback
from openai import AsyncOpenAI
from langchain_openai import OpenAI  # Optional, for prompt refinement

SHARED_LLM_CLIENT = None

GROQ_MODELS = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"]
CLAUDE_MODELS = ["claude-haiku-4-5-20251001", "claude-sonnet-4-5-20250929", "claude-opus-4-1-20250805"]
GPT_MODELS = ["gpt-5-nano-2025-08-07", "gpt-5-mini-2025-08-07", "gpt-4o-mini-2024-07-18", "gpt-5-2025-08-07", "gpt-4o-2024-08-06", "gpt-4.1-2025-04-14", "gpt-5-pro-2025-10-06"]
GEMINI_MODELS = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
GROK_MODELS = ["grok-4-fast-reasoning"]

class LLMClient:
    def __init__(self):
        self.sem = Semaphore(5)  # Limit concurrent LLM calls
        # Load environment variables from .env file
        load_dotenv()
        try:
            # Get the Groq API key from the environment
            self.openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.claude_api_key = os.environ.get("CLAUDE_API_KEY")
            self.groq_api_key = os.environ.get("GROQ_API_KEY")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
            #self.google_api_key = os.environ.get("GOOGLE_API_KEY")
            self.xai_api_key = os.environ.get("XAI_API_KEY")
            #self.llm_ollama = ChatOllama(model="llama3.1", temperature=0.5)
           # self.llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=self.openai_api_key)
            
            #self.llm_for_scene_planner = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct", temperature=0.2, groq_api_key=self.groq_api_key)
            #self.llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=self.google_api_key)
        except Exception as e:
            print(f"Failed to initialize LLMClient: {e}")
            raise
    # def llama3_1_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    #     """Blocking call — returns the full response."""
    #     return self.llm_ollama.invoke([
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content=human_prompt)
    #     ])

    # def llama3_1_stream(self, system_prompt: str = "", human_prompt: str = ""):
    #     """Streaming call — yields text chunks as they are generated."""
    #     stream = self.llm_ollama.stream([
    #         SystemMessage(content=system_prompt),
    #         HumanMessage(content=human_prompt)
    #     ])
    #     for chunk in stream:
    #         # Each chunk is a ChatMessage — only yield new text
    #         if hasattr(chunk, "content") and chunk.content:
    #             yield chunk.content


    # async def _image_client(self, prompt: str = ""):
    #     async with self.sem:  # Use the same semaphore as other methods
    #         try:
    #             response = await self.openai_client.images.generate(
    #                 model="dall-e-3",
    #                 prompt=prompt,
    #                 size="1024x1024",
    #                 quality="standard",
    #                 style="vivid",
    #                 n=1
    #             )
    #             image_url = response.data[0].url
    #             print(f"Generated Image URL: {image_url}")
    #         except Exception as e:
    #             print(f"Error generating image: {str(e)}")
    #             raise
    #         await asyncio.sleep(1)  # Add a small delay to avoid rate limits

    #     # Download image bytes
    #     try:
    #         loop = asyncio.get_event_loop()
    #         response = await loop.run_in_executor(None, lambda: requests.get(image_url, stream=True))
    #         response.raise_for_status()
    #         return response.content
    #     except requests.RequestException as e:
    #         print(f"Error downloading image: {str(e)}")
    #         raise

    async def _image_client(self, prompt: str = ""):
        async with self.sem:  # Use the same semaphore as other methods
            try:
                # Run Replicate API call in a thread to keep it async-compatible
                loop = asyncio.get_event_loop()
                output = await loop.run_in_executor(
                    None,
                    lambda: replicate.run(
                        "ideogram-ai/ideogram-v3-turbo",
                        input={
                            "prompt": prompt,
                            #"num_outputs": 1,  # Match DALL-E's n=1

                            # "height": 1024,    # Match DALL-E's 1024x1024
                            # "width": 1024,
                            #"num_inference_steps": 4,  # Fast inference for schnell
                            #"guidance_scale": 7.5,    # Default for FLUX
                            #"negative_prompt": "text, typography, letters, title, logo, signature, watermark, words, book object, frame, collage"

                            #"output_format": "png"    # Ensure PNG output
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
        #llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=0.8, model="grok-4-fast-reasoning")
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8, groq_api_key=self.groq_api_key)

        #llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=self.openai_api_key)
        #with get_openai_callback() as cb:
        async with self.sem:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])

        return response

    async def _author_client(
        self,
        system_prompt: str = "",
        human_prompt: str = "",
        llm_temp: float = 0.7,
        #model: str = "None"
        ):
        try:
            # model = "grok-4-fast-reasoning"
            # llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=llm_temp, model=model)
            # token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
            # usage_key = "token_usage"
            model = "claude-haiku-4-5-20251001"
            llm = ChatAnthropic(model_name=model, temperature=llm_temp, api_key=self.claude_api_key)
            token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": None}
            usage_key = "usage"
            async with self.sem:
                    response = await llm.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ])
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            meta = getattr(response, "response_metadata", {})
            if not meta:
                print(f"Warning: No response_metadata for model {model}")
            usage = meta.get(usage_key, {})
            if usage:
                token_usage["prompt_tokens"] = usage.get(token_keys["prompt"], 0)
                token_usage["completion_tokens"] = usage.get(token_keys["completion"], 0)
                if token_keys["total"] is None:
                    token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                else:
                    token_usage["total_tokens"] = usage.get(token_keys["total"], 0)

                # --- Validate token usage ---
                if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                    print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # --- Delay for rate limiting safety ---
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
            # model = "grok-4-fast-reasoning"
            # llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=llm_temp, model=model)
            # token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
            # usage_key = "token_usage"
            model = "claude-haiku-4-5-20251001"
            llm = ChatAnthropic(model_name=model, temperature=llm_temp, api_key=self.claude_api_key)
            token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": None}
            usage_key = "usage"
            async with self.sem:
                    response = await llm.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ])
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            meta = getattr(response, "response_metadata", {})
            if not meta:
                print(f"Warning: No response_metadata for model {model}")
            usage = meta.get(usage_key, {})
            if usage:
                token_usage["prompt_tokens"] = usage.get(token_keys["prompt"], 0)
                token_usage["completion_tokens"] = usage.get(token_keys["completion"], 0)
                if token_keys["total"] is None:
                    token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                else:
                    token_usage["total_tokens"] = usage.get(token_keys["total"], 0)

                # --- Validate token usage ---
                if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                    print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # --- Delay for rate limiting safety ---
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _director_client for model {model}: {str(e)}")
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
            model = "openai/gpt-oss-120b"
            #llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=llm_temp, model=model)
            #token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
            #usage_key = "token_usage"
            # llm = ChatAnthropic(model_name=model, temperature=llm_temp, api_key=self.claude_api_key)
            # token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": None}
            # usage_key = "usage"
            llm = ChatGroq(model=model, temperature=llm_temp, groq_api_key=self.groq_api_key)
            token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
            usage_key = "token_usage"
            async with self.sem:
                    response = await llm.ainvoke([
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=human_prompt)
                    ])
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            meta = getattr(response, "response_metadata", {})
            if not meta:
                print(f"Warning: No response_metadata for model {model}")
            usage = meta.get(usage_key, {})
            if usage:
                token_usage["prompt_tokens"] = usage.get(token_keys["prompt"], 0)
                token_usage["completion_tokens"] = usage.get(token_keys["completion"], 0)
                if token_keys["total"] is None:
                    token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
                else:
                    token_usage["total_tokens"] = usage.get(token_keys["total"], 0)

                # --- Validate token usage ---
                if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
                    print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

            # --- Delay for rate limiting safety ---
            await asyncio.sleep(2)

            return response, token_usage
        
        except Exception as e:
            print(f"Error in _writer_client for model {model}: {str(e)}")
            raise

    async def _ingestor_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to Gemini LLM – returns the full response."""
        #llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1, google_api_key=self.google_api_key)
        #llm = ChatOpenAI(model_name="gpt-5-nano", temperature=0.1, openai_api_key=self.openai_api_key)
        llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.1, groq_api_key=self.groq_api_key)
        #llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=0.1, model="grok-4-fast-reasoning")
        #llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, openai_api_key=self.openai_api_key)
        async with self.sem:
            response = await llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])
        time.sleep(1)  # delay to avoid rate limits
        return response

    # async def _story_client(
    #     self,
    #     system_prompt: str = "",
    #     human_prompt: str = "",
    #     llm_temp: float = 0.7,
    #     model: str = "None"
    # ) -> tuple[AIMessage, dict]:
    #     try:
    #         model = "grok-4-fast-reasoning"
    #         # --- Select appropriate client ---
    #         # if model in GROQ_MODELS:
    #         #     llm = ChatGroq(model=model, temperature=llm_temp, groq_api_key=self.groq_api_key)
    #         #     token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
    #         #     usage_key = "token_usage"
    #         if model in GPT_MODELS:
    #             if model == "gpt-5-nano-2025-08-07":
    #                 llm = ChatOpenAI(model=model, temperature=1, openai_api_key=self.openai_api_key)
    #             else:
    #                 llm = ChatOpenAI(model=model, temperature=llm_temp, openai_api_key=self.openai_api_key)

    #             token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
    #             usage_key = "token_usage"
    #         elif model in CLAUDE_MODELS:
    #             llm = ChatAnthropic(model_name=model, temperature=llm_temp, api_key=self.claude_api_key)
    #             token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": None}
    #             usage_key = "usage"
    #         # elif model in GEMINI_MODELS:
    #         #     llm = ChatGoogleGenerativeAI(model=model, temperature=llm_temp, google_api_key=self.google_api_key)
    #         #     token_keys = {"prompt": "input_tokens", "completion": "output_tokens", "total": "total_tokens"}
    #         #     usage_key = "usage_metadata"
    #         elif model in GROK_MODELS:
    #             llm = ChatXAI(xai_api_key=self.xai_api_key, temperature=llm_temp, model=model)
    #             token_keys = {"prompt": "prompt_tokens", "completion": "completion_tokens", "total": "total_tokens"}
    #             usage_key = "token_usage"
    #         else:
    #             raise ValueError(f"Unknown model: {model}")

    #         # --- Invoke the model ---
    #         async with self.sem:
    #             response = await llm.ainvoke([
    #                 SystemMessage(content=system_prompt),
    #                 HumanMessage(content=human_prompt)
    #             ])

    #         # --- Extract token usage ---
    #         token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            
    #         # Gemini stores usage_metadata directly on response, not in response_metadata
    #         if model in GEMINI_MODELS:
    #             usage = getattr(response, usage_key, {})
    #         else:
    #             meta = getattr(response, "response_metadata", {})
    #             if not meta:
    #                 print(f"Warning: No response_metadata for model {model}")
    #             usage = meta.get(usage_key, {})
            
    #         if usage:
    #             token_usage["prompt_tokens"] = usage.get(token_keys["prompt"], 0)
    #             token_usage["completion_tokens"] = usage.get(token_keys["completion"], 0)
    #             if token_keys["total"] is None:
    #                 token_usage["total_tokens"] = token_usage["prompt_tokens"] + token_usage["completion_tokens"]
    #             else:
    #                 token_usage["total_tokens"] = usage.get(token_keys["total"], 0)

    #             # --- Validate token usage ---
    #             if not all(isinstance(v, int) and v >= 0 for v in token_usage.values()):
    #                 print(f"Warning: Invalid token usage values for model {model}: {token_usage}")

    #         # --- Delay for rate limiting safety ---
    #         await asyncio.sleep(2)

    #         return response, token_usage

    #     except Exception as e:
    #         print(f"Error in _story_client for model {model}: {str(e)}")
    #         raise
    
    # async def _llm_for_scene_planner_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    #     """Blocking call to Groq LLM – returns the full response and token counts."""
    #     # The .invoke() method returns an object that contains the response metadata
    #     #llm_for_scene_planner = ChatOpenAI(model="gpt-4o-mini", temperature="0.1", openai_api_key=self.openai_api_key)
    #     #llm_for_scene_planner = ChatGroq(model="openai/gpt-oss-20b", temperature=0.1, groq_api_key=self.groq_api_key)
    #     llm_for_scene_planner = ChatAnthropic(model_name="claude-3-haiku-20240307", temperature=0.1, api_key=self.claude_api_key)
    #     async with self.sem:
    #         response = await llm_for_scene_planner.ainvoke([
    #             SystemMessage(content=system_prompt),
    #             HumanMessage(content=human_prompt)
    #         ])
        
        # Access the token usage from the response's metadata
        #token_usage = response.response_metadata.get('token_usage', {})

        # Extract the prompt and completion token counts
        # prompt_tokens = token_usage.get('prompt_tokens', 0)
        # completion_tokens = token_usage.get('completion_tokens', 0)
        # total_tokens = token_usage.get('total_tokens', 0)
        
        # print("--- Token Usage ---")
        # print(f"Prompt Tokens (Input): {prompt_tokens}")
        # print(f"Completion Tokens (Output): {completion_tokens}")
        # print(f"Total Tokens: {total_tokens}")
        # print("-------------------")
        #time.sleep(4)  # delay to avoid rate limits
        #return response


init_lock = asyncio.Lock()
# Convenience functions to access the shared instance
async def get_shared_client():
    async with init_lock:
        global SHARED_LLM_CLIENT
        if SHARED_LLM_CLIENT is None:
            SHARED_LLM_CLIENT = LLMClient()
        return SHARED_LLM_CLIENT

# Convenience wrapper functions for easy access
# async def story_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7, model: str = "No model given") -> AIMessage:
#     """Convenience function to access groq_client through shared instance."""
#     client = await get_shared_client()
#     return await client._story_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp, model=model)


async def author_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7, model: str = "No model given") -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._author_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def director_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7, model: str = "No model given") -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._director_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

async def writer_client(system_prompt: str = "", human_prompt: str = "", llm_temp: float = 0.7, model: str = "No model given") -> AIMessage:
    """Convenience function to access groq_client through shared instance."""
    client = await get_shared_client()
    return await client._writer_client(system_prompt=system_prompt, human_prompt=human_prompt, llm_temp=llm_temp)

# async def llm_for_scene_planner_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
#     """Convenience function to access groq_client through shared instance."""
#     client = await get_shared_client()
#     return await client._llm_for_scene_planner_client(system_prompt, human_prompt)

async def ingestor_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    """Convenience function to access gemini_client through shared instance."""
    client = await get_shared_client()
    return await client._ingestor_client(system_prompt, human_prompt)

async def utility_client(system_prompt: str = "", human_prompt: str = "") -> AIMessage:
    """Convenience function to access model through shared instance."""
    client = await get_shared_client()
    return await client._utility_client(system_prompt, human_prompt)

async def image_client(prompt: str = "") -> AIMessage:
    """Convenience function to access model through shared instance."""
    client = await get_shared_client()
    return await client._image_client(prompt)