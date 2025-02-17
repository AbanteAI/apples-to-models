from typing import List, Iterator, Optional
from dataclasses import dataclass
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


@dataclass
class ModelResponse:
    """Data class to hold model response data."""

    content: str
    cost: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


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
        self.response: Optional[ModelResponse] = None

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

    def set_response(self, response: ModelResponse):
        """Set the model's response."""
        self.response = response

    def _write_log(self) -> None:
        """Write the log file with all collected information in a human-readable format."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file = self.log_dir / f"model_call_{timestamp.replace(':', '-')}.log"

        if not self.messages or not self.response:
            raise ValueError("Messages and response must be set before writing log")

        with open(log_file, "w", encoding="utf-8") as f:
            # Write header with all metadata
            duration = round(self.end_time - self.start_time, 3)
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Model: {self.model}\n")
            f.write(f"Duration: {duration} seconds\n")
            f.write(f"Cost: ${self.response.cost:.4f}\n")
            f.write(f"Tokens: {self.response.total_tokens} ")
            f.write(f"(prompt: {self.response.prompt_tokens}, ")
            f.write(f"completion: {self.response.completion_tokens})\n")
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
            f.write(f"{self.response.content}\n")
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
    import httpx

    # OpenRouter's cost endpoint for specific completion
    cost_url = f"https://openrouter.ai/api/v1/costs/{completion_id}"
    headers = {"Authorization": f"Bearer {client.api_key}"}

    try:
        with httpx.Client(timeout=10.0) as http_client:
            response = http_client.get(cost_url, headers=headers)
            response.raise_for_status()
            cost_data = response.json()

            # The response should directly contain the cost information for this completion
            return float(cost_data.get("total_cost", 0.0))
    except Exception as e:
        print(f"Warning: Failed to get cost data: {e}")

    return 0.0


@retry(tries=3, backoff=2)
def call_model(model: str, messages: Messages) -> ModelResponse:
    """
    Call a model through OpenRouter API with the given messages.

    Args:
        model: The model identifier to use
        messages: A Messages instance containing the conversation

    Returns:
        A ModelResponse object containing the response content, cost, and token counts
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

        # Get the cost and token counts
        try:
            cost = get_completion_cost(client, response.id)
        except Exception as e:
            print(f"Warning: Failed to get completion cost: {e}")
            cost = 0.0

        # Handle the case where usage might be None
        usage = response.usage
        if (
            usage is None
            or not hasattr(usage, "prompt_tokens")
            or not hasattr(usage, "completion_tokens")
            or not hasattr(usage, "total_tokens")
        ):
            # If usage data is missing, use default values
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
        else:
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens

        model_response = ModelResponse(
            content=content,
            cost=cost,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        # Set the complete response object in the logger
        logger.set_response(model_response)

        return model_response
