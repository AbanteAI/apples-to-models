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
    """Class to handle logging of model calls using a context manager."""

    def __init__(self, log_dir: str = "benchmark/logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time: float = 0
        self.end_time: float = 0
        self.model: str = ""
        self.messages: Optional[Messages] = None
        self.response: Optional[str] = None
        self.cost: Optional[float] = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.model and self.messages and self.response is not None:
            self._write_log()

    def set_model_info(self, model: str, messages: Messages):
        """Set the model and messages information."""
        self.model = model
        self.messages = messages

    def set_response(self, response: str):
        """Set the model's response."""
        self.response = response

    def set_cost(self, cost: float):
        """Set the cost of the model call (optional)."""
        self.cost = cost

    def _write_log(self) -> None:
        """Write the log file with all collected information."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"model_call_{timestamp}.log"

        if not self.messages:
            raise ValueError("Messages not set before writing log")

        log_data = {
            "timestamp": timestamp,
            "model": self.model,
            "duration_seconds": self.end_time - self.start_time,
            "cost": self.cost,
            "messages": [
                {"role": msg["role"], "content": msg.get("content", "") or ""}
                for msg in self.messages.messages
            ],
            "response": self.response,
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

    client = OpenAI(
        api_key=os.getenv("OPEN_ROUTER_KEY"), base_url="https://openrouter.ai/api/v1"
    )

    with ModelLogger() as logger:
        logger.set_model_info(model, messages)

        response = client.chat.completions.create(
            model=model,
            messages=list(messages),  # Convert Messages instance to list
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model response content was None")

        logger.set_response(content)
        # Cost calculation could be added here if the API provides it
        # logger.set_cost(cost)

        return content
