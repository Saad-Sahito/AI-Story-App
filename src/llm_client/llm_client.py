from urllib import response
from langchain_ollama import ChatOllama  # for local deployment only not render
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.callbacks import get_openai_callback

class LLMClient:
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        # Get the Groq API key from the environment
        groq_api_key = os.environ.get("GROQ_API_KEY")
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.llm_ollama = ChatOllama(model="llama3.1", temperature=0.5)
        self.llm_openai = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
        self.llm_groq = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0.5, groq_api_key=groq_api_key)
        self.llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5, google_api_key=google_api_key)

    def llama3_1_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call — returns the full response."""
        return self.llm_ollama.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])

    def llama3_1_stream(self, system_prompt: str = "", human_prompt: str = ""):
        """Streaming call — yields text chunks as they are generated."""
        stream = self.llm_ollama.stream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        for chunk in stream:
            # Each chunk is a ChatMessage — only yield new text
            if hasattr(chunk, "content") and chunk.content:
                yield chunk.content

    def groq_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to Groq LLM – returns the full response and token counts."""
        # The .invoke() method returns an object that contains the response metadata
        response = self.llm_groq.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        
        # Access the token usage from the response's metadata
        token_usage = response.response_metadata.get('token_usage', {})

        # Extract the prompt and completion token counts
        prompt_tokens = token_usage.get('prompt_tokens', 0)
        completion_tokens = token_usage.get('completion_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        print("--- Token Usage ---")
        print(f"Prompt Tokens (Input): {prompt_tokens}")
        print(f"Completion Tokens (Output): {completion_tokens}")
        print(f"Total Tokens: {total_tokens}")
        print("-------------------")

        return response
    

    def openai_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to OpenAI LLM – returns the full response and token counts."""

        with get_openai_callback() as cb:
            response = self.llm_openai.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ])

            print("--- Token Usage ---")
            print(f"Prompt Tokens (Input): {cb.prompt_tokens}")
            print(f"Completion Tokens (Output): {cb.completion_tokens}")
            print(f"Total Tokens: {cb.total_tokens}")
            print(f"Cost (USD): ${cb.total_cost:.6f}")
            print("-------------------")

        return response

    def gemini_client(self, system_prompt: str = "", human_prompt: str = "") -> AIMessage:
        """Blocking call to Gemini LLM – returns the full response."""
        
        response = self.llm_gemini.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ])
        # Access the token usage from the response's metadata
        token_usage = response.usage_metadata
        
        # Extract the prompt and completion token counts
        prompt_tokens = token_usage.get('input_tokens', 0)
        completion_tokens = token_usage.get('output_tokens', 0)
        total_tokens = token_usage.get('total_tokens', 0)
        
        # print("--- Token Usage ---")
        # print(f"Prompt Tokens (Input): {prompt_tokens}")
        # print(f"Completion Tokens (Output): {completion_tokens}")
        # print(f"Total Tokens: {total_tokens}")
        # print("-------------------")
        time.sleep(5)  # delay to avoid rate limits
        return response