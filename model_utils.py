from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import os
from retry import retry

class Messages:
    def __init__(self):
        self.messages: List[Dict[str, str]] = []
    
    def add_system(self, content: str) -> None:
        """Add a system message."""
        self.messages.append({"role": "system", "content": content})
    
    def add_user(self, content: str) -> None:
        """Add a user message."""
        self.messages.append({"role": "user", "content": content})
    
    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        self.messages.append({"role": "assistant", "content": content})
    
    def __iter__(self):
        return iter(self.messages)

@retry(tries=3, backoff=2)
def call_model(model: str, messages: Messages) -> str:
    """
    Call a model through OpenRouter API with the given messages.
    
    Args:
        model: The model identifier to use
        messages: A Messages instance containing the conversation
    
    Returns:
        The model's response content as a string
    """
    load_dotenv()  # Load environment variables
    
    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )

    # Handle special cases for model quantization
    extra_body = {}
    if model == "meta-llama/llama-3.1-405b-instruct:bf16":
        model = "meta-llama/llama-3.1-405b-instruct"
        extra_body = {
            "provider": {
                "quantizations": ["bf16"]
            }
        }

    response = client.chat.completions.create(
        model=model,
        messages=list(messages),  # Convert Messages instance to list
        temperature=0.7,
        extra_body=extra_body
    )
    
    return response.choices[0].message.content