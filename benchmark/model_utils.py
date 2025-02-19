import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional

import aiohttp  # type: ignore
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)
from pydantic import BaseModel

from benchmark.utils import async_retry


class ModelResponse(BaseModel):
    """Response data from a model call including content and usage statistics."""

    content: str
    model: str = "test-model"  # Default for tests
    tokens_prompt: int
    tokens_completion: int
    total_cost: float
    duration: float = 0.5  # Default for tests
    generation_id: str
    log_path: Optional[Path] = None

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"ModelResponse(content='{self.content}', "
            f"model='{self.model}', "
            f"tokens_prompt={self.tokens_prompt}, "
            f"tokens_completion={self.tokens_completion}, "
            f"total_cost=${self.total_cost:.6f}, "
            f"duration={self.duration:.2f}s, "
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


def write_model_log(
    model: str,
    messages: Messages,
    response: str,
    cost: float,
    duration: float,
    log_dir: Optional[str] = None,
) -> Path:
    """Write a log file with model call information.

    Args:
        model: The model identifier used
        messages: The Messages instance containing the conversation
        response: The model's response content
        cost: The cost of the model call
        duration: The duration of the model call in seconds
        log_dir: Optional directory to write logs to (defaults to env var or benchmark/logs)

    Returns:
        Path: The path to the written log file
    """
    # Use game-specific log directory if set in environment, otherwise use default
    log_dir_path = Path(log_dir or os.getenv("GAME_LOG_DIR", "benchmark/logs"))
    log_dir_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Generate a short random key for uniqueness
    random_key = str(uuid.uuid4())[:8]
    filename = f"model_call_{timestamp}_{random_key}.log"
    log_file = log_dir_path / filename

    with open(log_file, "w", encoding="utf-8") as f:
        # Write header with all metadata
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Duration: {duration:.3f} seconds\n")
        f.write(f"Cost: ${cost:.4f}\n")
        f.write("-" * 80 + "\n\n")

        # Write messages in a conversation format
        f.write("=== Input Messages ===\n")
        for msg in messages.messages:
            role = msg["role"].upper()
            content = msg.get("content", "") or ""
            f.write(f"\n[{role}]\n{content}\n")
        f.write("\n" + "-" * 80 + "\n")

        # Write model response
        f.write("\n=== Model Response ===\n")
        f.write(f"{response}\n")
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
    start_time = time.time()

    # Make the initial completion request
    response = await client.chat.completions.create(
        model=model,
        messages=list(messages),  # Convert Messages instance to list
        temperature=0,
    )
    duration = time.time() - start_time  # Calculate duration right after completion

    content = response.choices[0].message.content
    if content is None:
        raise ValueError("Model response content was None")

    # Get the generation ID from the response
    generation_id = response.id

    # Fetch generation stats
    stats = await get_generation_stats(generation_id, api_key)

    # Write log file
    log_path = write_model_log(
        model=model,
        messages=messages,
        response=content,
        cost=stats["total_cost"],
        duration=duration,
    )

    return ModelResponse(
        content=content,
        model=model,
        tokens_prompt=stats["tokens_prompt"],
        tokens_completion=stats["tokens_completion"],
        total_cost=stats["total_cost"],
        duration=duration,
        generation_id=generation_id,
        log_path=log_path,
    )
