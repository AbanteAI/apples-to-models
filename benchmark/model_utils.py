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


def get_completion_cost(client: OpenAI, completion_id: str) -> float:
    """
    Get the cost of a completion from OpenRouter API.

    Args:
        client: The OpenAI client instance
        completion_id: The ID of the completion to get the cost for

    Returns:
        The total cost of the completion
    """
    response = client.with_options(timeout=10.0).chat.completions.retrieve(
        completion_id=completion_id
    )
    return float(response.usage.get("total_cost", 0.0))


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

        # Get the cost of the completion
        try:
            cost = get_completion_cost(client, response.id)
            logger.set_cost(cost)
        except Exception as e:
            print(f"Warning: Failed to get completion cost: {e}")

        return content
