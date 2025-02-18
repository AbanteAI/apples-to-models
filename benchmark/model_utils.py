from typing import List, Iterator, Optional, Dict, Any, Protocol
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
import json


class Messages:
    """Container for chat messages with helper methods for adding different message types."""

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


class ModelResponse(BaseModel):
    """Response data from a model call including content and usage statistics."""

    content: str
    tokens_prompt: int
    tokens_completion: int
    total_cost: float
    generation_id: str

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"ModelResponse(content='{self.content}', "
            f"tokens_prompt={self.tokens_prompt}, "
            f"tokens_completion={self.tokens_completion}, "
            f"total_cost=${self.total_cost:.6f}, "
            f"generation_id='{self.generation_id}')"
        )


class ModelCall:
    """Represents a single interaction with a model, including input, output, and metadata."""

    def __init__(self, model: str, messages: Messages):
        self.model = model
        self.messages = messages
        self.response: Optional[ModelResponse] = None
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        self.error: Optional[Exception] = None

    def complete(self, response: ModelResponse) -> None:
        """Mark the model call as complete with the given response."""
        self.response = response
        self.end_time = time.time()

    def fail(self, error: Exception) -> None:
        """Mark the model call as failed with the given error."""
        self.error = error
        self.end_time = time.time()

    @property
    def duration(self) -> float:
        """Return the duration of the model call in seconds."""
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert the model call to a dictionary representation."""
        data = {
            "model": self.model,
            "messages": [dict(m) for m in self.messages],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
        }
        if self.response:
            data["response"] = self.response.model_dump()
        if self.error:
            data["error"] = str(self.error)
        return data


class LogFormatter(Protocol):
    """Protocol for formatting model calls into logs."""

    def format(self, model_call: ModelCall) -> str:
        """Format a model call into a log string."""
        ...


class HumanReadableFormatter:
    """Formats model calls into human-readable text."""

    def format(self, model_call: ModelCall) -> str:
        lines = []

        # Header
        lines.append(f"Timestamp: {datetime.fromtimestamp(model_call.start_time)}")
        lines.append(f"Model: {model_call.model}")
        lines.append(f"Duration: {model_call.duration:.3f} seconds")
        if model_call.response:
            lines.append(f"Cost: ${model_call.response.total_cost:.4f}")
        lines.append("-" * 80 + "\n")

        # Messages
        lines.append("=== Input Messages ===")
        for msg in model_call.messages:
            role = msg["role"].upper()
            content = msg.get("content", "") or ""
            lines.append(f"\n[{role}]\n{content}")
        lines.append("\n" + "-" * 80 + "\n")

        # Response or Error
        if model_call.response:
            lines.append("=== Model Response ===")
            lines.append(model_call.response.content)
        elif model_call.error:
            lines.append("=== Error ===")
            lines.append(str(model_call.error))
        lines.append("\n" + "-" * 80)

        return "\n".join(lines)


class JsonFormatter:
    """Formats model calls into JSON."""

    def format(self, model_call: ModelCall) -> str:
        return json.dumps(model_call.to_dict(), indent=2)


class ModelLogger:
    """Handles logging of model calls with configurable output format and destination."""

    def __init__(
        self,
        log_dir: str = "benchmark/logs",
        formatter: Optional[LogFormatter] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.formatter = formatter or HumanReadableFormatter()

    def log(self, model_call: ModelCall) -> None:
        """Log a model call using the configured formatter."""
        timestamp = datetime.fromtimestamp(model_call.start_time).strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        extension = ".json" if isinstance(self.formatter, JsonFormatter) else ".log"
        log_file = self.log_dir / f"model_call_{timestamp}{extension}"

        log_content = self.formatter.format(model_call)
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(log_content)


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
    load_dotenv()
    api_key = os.getenv("OPEN_ROUTER_KEY")
    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    # Create a ModelCall instance to track this interaction
    model_call = ModelCall(model, messages)
    logger = ModelLogger()

    try:
        # Make the initial completion request
        response = client.chat.completions.create(
            model=model,
            messages=list(messages),
            temperature=0,
        )

        content = response.choices[0].message.content
        if content is None:
            raise ValueError("Model response content was None")

        # Get the generation ID and stats
        generation_id = response.id
        headers = {"Authorization": f"Bearer {api_key}"}
        stats_response = requests.get(
            f"https://openrouter.ai/api/v1/generation?id={generation_id}",
            headers=headers,
        )

        stats_data = stats_response.json()
        if "data" not in stats_data:
            raise ValueError(
                f"Stats data not available in response: {stats_response.text}"
            )

        stats = stats_data["data"]
        model_response = ModelResponse(
            content=content,
            tokens_prompt=stats["tokens_prompt"],
            tokens_completion=stats["tokens_completion"],
            total_cost=stats["total_cost"],
            generation_id=generation_id,
        )

        # Complete the model call and log it
        model_call.complete(model_response)
        logger.log(model_call)

        return model_response

    except Exception as e:
        # Log failed calls as well
        model_call.fail(e)
        logger.log(model_call)
        raise
