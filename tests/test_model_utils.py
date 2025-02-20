from pathlib import Path

import pytest

from benchmark.model_utils import Messages, call_model, write_model_log


def test_messages_creation():
    messages = Messages()
    assert len(messages.messages) == 0


def test_messages_add_user():
    messages = Messages()
    test_message = "Test message"
    messages.add_user(test_message)
    assert len(messages.messages) == 1
    assert messages.messages[0]["role"] == "user"
    assert messages.messages[0]["content"] == test_message


@pytest.mark.skip(reason="Requires API key and network access")
@pytest.mark.asyncio
async def test_call_model():
    messages = Messages()
    messages.add_user("What is 2 + 2?")
    response = await call_model("openai/gpt-4o-mini-2024-07-18", messages)

    # Check response structure
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert isinstance(response.tokens_prompt, int)
    assert isinstance(response.tokens_completion, int)
    assert isinstance(response.total_cost, float)
    assert isinstance(response.generation_id, str)
    assert isinstance(response.log_path, (Path, type(None)))

    # Basic validation of values
    assert response.tokens_prompt > 0
    assert response.tokens_completion > 0
    assert response.total_cost > 0


def test_write_model_log(tmp_path):
    messages = Messages()
    messages.add_system("Test system message")
    messages.add_user("Test user message")

    log_path = write_model_log(
        model="test-model",
        messages=messages,
        response="Test response",
        cost=0.001,
        duration=1.0,
        log_dir=str(tmp_path),
    )

    assert log_path.exists()
    assert log_path.is_file()

    with open(log_path) as f:
        content = f.read()
        assert "Test system message" in content
        assert "Test user message" in content
        assert "Test response" in content
        assert "Cost: $0.0010" in content


def test_write_model_log_uniqueness(tmp_path):
    messages = Messages()
    messages.add_system("Test system message")

    # Write multiple logs in quick succession
    log_path1 = write_model_log(
        model="test-model",
        messages=messages,
        response="First response",
        cost=0.001,
        duration=1.0,
        log_dir=str(tmp_path),
    )

    log_path2 = write_model_log(
        model="test-model",
        messages=messages,
        response="Second response",
        cost=0.001,
        duration=1.0,
        log_dir=str(tmp_path),
    )

    # Verify logs exist and have different paths
    assert log_path1.exists() and log_path2.exists()
    assert log_path1 != log_path2

    # Verify content is correct for each log
    with open(log_path1) as f:
        content1 = f.read()
        assert "First response" in content1

    with open(log_path2) as f:
        content2 = f.read()
        assert "Second response" in content2
