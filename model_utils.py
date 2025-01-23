from typing import List, Iterator
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from dotenv import load_dotenv
import os
from retry import retry


class Messages:
    def __init__(self):
        self.messages: List[ChatCompletionMessageParam] = []

    def add_system(self, content: str) -> None:
        """Add a system message."""
        message: ChatCompletionSystemMessageParam = {
            "role": "system",
            "content": content,
        }
        self.messages.append(message)

    def add_user(self, content: str) -> None:
        """Add a user message."""
        message: ChatCompletionUserMessageParam = {"role": "user", "content": content}
        self.messages.append(message)

    def add_assistant(self, content: str) -> None:
        """Add an assistant message."""
        message: ChatCompletionAssistantMessageParam = {
            "role": "assistant",
            "content": content,
        }
        self.messages.append(message)

    def __iter__(self) -> Iterator[ChatCompletionMessageParam]:
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
        api_key=os.getenv("OPEN_ROUTER_KEY"), base_url="https://openrouter.ai/api/v1"
    )

    response = client.chat.completions.create(
        model=model,
        messages=list(messages),  # Convert Messages instance to list
        temperature=0,
    )

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model response content was None")

    return content
