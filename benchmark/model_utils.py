from typing import List, Iterator, Optional
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam,
)
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from retry import retry
import time
from datetime import datetime
from pathlib import Path
import requests


class ModelResponse(BaseModel):
    """Response data from a model call including content and usage statistics."""

    content: str
    tokens_prompt: int
    tokens_completion: int
    total_cost: float
    generation_id: str


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
        """Write the log file with all collected information in a human-readable format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = self.log_dir / f"model_call_{timestamp.replace(':', '-')}.log"

        if not self.messages:
            raise ValueError("Messages not set before writing log")

        with open(log_file, "w", encoding="utf-8") as f:
            # Write header with all metadata
            duration = round(self.end_time - self.start_time, 3)
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Duration: {duration} seconds\n")
            if self.cost is not None:
                f.write(f"Cost: ${self.cost:.4f}\n")
            f.write("-" * 80 + "\n\n")

            # Write messages in a conversation format
            f.write("=== Input Messages ===\n")
            for msg in self.messages.messages:
                role = msg["role"].upper()
                content = msg.get("content", "") or ""
                f.write(f"\n[{role}]\n{content}\n")
            f.write("\n" + "-" * 80 + "\n")

            # Write model response
            f.write("\n=== Model Response ===\n")
            f.write(f"{self.response}\n")
            f.write("\n" + "-" * 80)


@retry(tries=3, backoff=2)
def call_model(model: str, messages: Messages) -> ModelResponse:
    """
    Call a model through OpenRouter API with the given messages.

    Args:
        model: The model identifier to use
        messages: A Messages instance containing the conversation

    Returns:
        A ModelResponse object containing the response content and usage statistics
    """
    load_dotenv()  # Load environment variables
    api_key = os.getenv("OPEN_ROUTER_KEY")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    with ModelLogger() as logger:
        logger.set_model_info(model, messages)

        # Make the initial completion request
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),  # Convert Messages instance to list
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model response content was None")

        # Get the generation ID from the response
        generation_id = response.id

        # Fetch generation stats
        headers = {"Authorization": f"Bearer {api_key}"}
        stats_response = requests.get(
            f"https://openrouter.ai/api/v1/generation?id={generation_id}",
            headers=headers,
        )

        # Debug logging
        print(f"Generation stats response: {stats_response.status_code}")
        print(f"Response content: {stats_response.text}")

        stats_data = stats_response.json()
        if "data" not in stats_data:
            # If stats are not available, return with default values
            model_response = ModelResponse(
                content=content,
                tokens_prompt=0,  # Will update when stats are available
                tokens_completion=0,  # Will update when stats are available
                total_cost=0.0,  # Will update when stats are available
                generation_id=generation_id,
            )
        else:
            stats = stats_data["data"]
            model_response = ModelResponse(
                content=content,
                tokens_prompt=stats["tokens_prompt"],
                tokens_completion=stats["tokens_completion"],
                total_cost=stats["total_cost"],
                generation_id=generation_id,
            )

        logger.set_response(content)
        if "data" in stats_data:
            logger.set_cost(stats_data["data"]["total_cost"])
        else:
            logger.set_cost(0.0)

        return model_response
