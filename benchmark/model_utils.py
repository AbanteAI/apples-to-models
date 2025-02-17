from typing import List, Iterator, Optional
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
import time
import json
from datetime import datetime
from pathlib import Path


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


class ModelLogger:
    """Class to handle logging of model calls."""

    def __init__(self, log_dir: str = "benchmark/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log_model_call(
        self,
        model: str,
        messages: Messages,
        response: str,
        start_time: float,
        end_time: float,
        cost: Optional[float] = None,
    ) -> None:
        """
        Log a model call to a file.

        Args:
            model: The model identifier used
            messages: The Messages instance containing the conversation
            response: The model's response
            start_time: Timestamp when the call started
            end_time: Timestamp when the call ended
            cost: The cost of the call (if available)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"model_call_{timestamp}.log"

        log_data = {
            "timestamp": timestamp,
            "model": model,
            "duration_seconds": end_time - start_time,
            "cost": cost,
            "messages": [
                {"role": msg["role"], "content": msg.get("content", "") or ""}
                for msg in messages.messages
            ],
            "response": response,
        }

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


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
    logger = ModelLogger()

    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"), base_url="https://openrouter.ai/api/v1"
    )

    start_time = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=list(messages),  # Convert Messages instance to list
        temperature=0,
    )
    end_time = time.time()

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model response content was None")

    # Log the model call
    logger.log_model_call(
        model=model,
        messages=messages,
        response=content,
        start_time=start_time,
        end_time=end_time,
        # Cost calculation could be added here if the API provides it
        cost=None,
    )

    return content
