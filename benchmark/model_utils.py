import asyncio
import os
import time
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Iterator, List, Optional

import aiohttp
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel


def async_retry(tries=8, delay=0.1, backoff=2):
    """Retry decorator for async functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    _tries -= 1
                    if _tries == 0:
                        raise
                    await asyncio.sleep(_delay)
                    _delay *= backoff
            return None

        return wrapper

    return decorator


class ModelResponse(BaseModel):
    """Response data from a model call including content and usage statistics."""

    content: str
    tokens_prompt: int
    tokens_completion: int
    total_cost: float
    generation_id: str
    log_path: Optional[Path] = None

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"ModelResponse(content='{self.content}', "
            f"tokens_prompt={self.tokens_prompt}, "
            f"tokens_completion={self.tokens_completion}, "
            f"total_cost=${self.total_cost:.6f}, "
            f"generation_id='{self.generation_id}', "
            f"log_path='{self.log_path}')"
        )


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

    def __init__(self, log_dir: Optional[str] = None):
        # Use game-specific log directory if set in environment, otherwise use default
        self.log_dir = Path(log_dir or os.getenv("GAME_LOG_DIR", "benchmark/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.start_time: float = 0
        self.end_time: float = 0
        self.model: str = ""
        self.messages: Optional[Messages] = None
        self.response: Optional[str] = None
        self.cost: Optional[float] = None
        self._log_counter: int = 0

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        if self.model and self.messages and self.response is not None:
            return self._write_log()

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

    def _write_log(self) -> Path:
        """Write the log file with all collected information in a human-readable format.

        Returns:
            Path: The path to the written log file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_counter += 1
        log_file = self.log_dir / f"model_call_{self._log_counter:03d}_{timestamp}.log"

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

        return log_file


@async_retry(tries=8, delay=0.1, backoff=2)
async def get_generation_stats(generation_id: str, api_key: str) -> dict:
    """
    Fetch generation statistics from OpenRouter API with retry logic.

    Args:
        generation_id: The ID of the generation to fetch stats for
        api_key: OpenRouter API key

    Returns:
        Dictionary containing the generation statistics
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession() as session:
        response = await session.get(
            f"https://openrouter.ai/api/v1/generation?id={generation_id}",
            headers=headers,
        )
        async with response:
            stats_data = await response.json()
            if "data" not in stats_data:
                raise ValueError(
                    f"Stats data not available in response: {await response.text()}"
                )

            return stats_data["data"]


@async_retry(tries=5, delay=0.1, backoff=2)
async def call_model(model: str, messages: Messages) -> ModelResponse:
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
    if not api_key:
        raise ValueError("OPEN_ROUTER_KEY environment variable is not set")

    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")

    with ModelLogger() as logger:
        logger.set_model_info(model, messages)

        # Make the initial completion request
        response = await client.chat.completions.create(
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
        stats = await get_generation_stats(generation_id, api_key)
        model_response = ModelResponse(
            content=content,
            tokens_prompt=stats["tokens_prompt"],
            tokens_completion=stats["tokens_completion"],
            total_cost=stats["total_cost"],
            generation_id=generation_id,
        )

        logger.set_response(content)
        logger.set_cost(stats["total_cost"])

        # Get the log path from the logger
        log_path = logger._write_log()
        model_response.log_path = log_path

        return model_response
