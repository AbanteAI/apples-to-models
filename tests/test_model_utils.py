import pytest
import json
import time
from pathlib import Path
from benchmark.model_utils import (
    Messages,
    call_model,
    ModelCall,
    ModelResponse,
    ModelLogger,
    HumanReadableFormatter,
    JsonFormatter,
)


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


def test_messages_add_system():
    messages = Messages()
    test_message = "System instruction"
    messages.add_system(test_message)
    assert len(messages.messages) == 1
    assert messages.messages[0]["role"] == "system"
    assert messages.messages[0]["content"] == test_message


def test_messages_add_assistant():
    messages = Messages()
    test_message = "Assistant response"
    messages.add_assistant(test_message)
    assert len(messages.messages) == 1
    assert messages.messages[0]["role"] == "assistant"
    assert messages.messages[0]["content"] == test_message


def test_model_call_creation():
    messages = Messages()
    messages.add_user("Test message")
    model_call = ModelCall("test-model", messages)

    assert model_call.model == "test-model"
    assert model_call.messages == messages
    assert model_call.response is None
    assert model_call.error is None
    assert model_call.end_time is None
    assert isinstance(model_call.start_time, float)


def test_model_call_complete():
    messages = Messages()
    model_call = ModelCall("test-model", messages)

    response = ModelResponse(
        content="Test response",
        tokens_prompt=10,
        tokens_completion=20,
        total_cost=0.001,
        generation_id="test-id",
    )

    model_call.complete(response)
    assert model_call.response == response
    assert model_call.end_time is not None
    assert model_call.error is None


def test_model_call_fail():
    messages = Messages()
    model_call = ModelCall("test-model", messages)

    error = ValueError("Test error")
    model_call.fail(error)
    assert model_call.error == error
    assert model_call.end_time is not None
    assert model_call.response is None


def test_model_call_duration():
    messages = Messages()
    model_call = ModelCall("test-model", messages)

    time.sleep(0.1)  # Small delay to ensure measurable duration
    assert model_call.duration > 0

    model_call.complete(
        ModelResponse(
            content="Test",
            tokens_prompt=1,
            tokens_completion=1,
            total_cost=0.001,
            generation_id="test",
        )
    )

    duration = model_call.duration
    time.sleep(0.1)
    assert model_call.duration == duration  # Duration should be fixed after completion


def test_human_readable_formatter(tmp_path):
    messages = Messages()
    messages.add_user("Test message")
    model_call = ModelCall("test-model", messages)

    response = ModelResponse(
        content="Test response",
        tokens_prompt=10,
        tokens_completion=20,
        total_cost=0.001,
        generation_id="test-id",
    )
    model_call.complete(response)

    formatter = HumanReadableFormatter()
    log_content = formatter.format(model_call)

    assert "Test message" in log_content
    assert "Test response" in log_content
    assert "test-model" in log_content
    assert "$0.0010" in log_content


def test_json_formatter(tmp_path):
    messages = Messages()
    messages.add_user("Test message")
    model_call = ModelCall("test-model", messages)

    response = ModelResponse(
        content="Test response",
        tokens_prompt=10,
        tokens_completion=20,
        total_cost=0.001,
        generation_id="test-id",
    )
    model_call.complete(response)

    formatter = JsonFormatter()
    log_content = formatter.format(model_call)

    # Verify it's valid JSON
    data = json.loads(log_content)
    assert data["model"] == "test-model"
    assert data["messages"][0]["content"] == "Test message"
    assert data["response"]["content"] == "Test response"
    assert data["duration"] > 0


def test_model_logger(tmp_path):
    log_dir = tmp_path / "logs"
    logger = ModelLogger(log_dir=str(log_dir))

    messages = Messages()
    messages.add_user("Test message")
    model_call = ModelCall("test-model", messages)

    response = ModelResponse(
        content="Test response",
        tokens_prompt=10,
        tokens_completion=20,
        total_cost=0.001,
        generation_id="test-id",
    )
    model_call.complete(response)

    logger.log(model_call)

    # Verify log file was created
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) == 1

    # Verify log content
    with open(log_files[0], "r") as f:
        content = f.read()
        assert "Test message" in content
        assert "Test response" in content


def test_model_logger_with_json_formatter(tmp_path):
    log_dir = tmp_path / "logs"
    logger = ModelLogger(log_dir=str(log_dir), formatter=JsonFormatter())

    messages = Messages()
    messages.add_user("Test message")
    model_call = ModelCall("test-model", messages)
    model_call.complete(
        ModelResponse(
            content="Test response",
            tokens_prompt=10,
            tokens_completion=20,
            total_cost=0.001,
            generation_id="test-id",
        )
    )

    logger.log(model_call)

    # Verify JSON log file was created
    log_files = list(log_dir.glob("*.json"))
    assert len(log_files) == 1

    # Verify it's valid JSON
    with open(log_files[0], "r") as f:
        data = json.load(f)
        assert data["model"] == "test-model"
        assert data["messages"][0]["content"] == "Test message"
        assert data["response"]["content"] == "Test response"


@pytest.mark.skip(reason="Requires API key and network access")
def test_call_model():
    messages = Messages()
    messages.add_user("What is 2 + 2?")
    response = call_model("openai/gpt-4o-mini-2024-07-18", messages)

    # Check response structure
    assert isinstance(response.content, str)
    assert len(response.content) > 0
    assert isinstance(response.tokens_prompt, int)
    assert isinstance(response.tokens_completion, int)
    assert isinstance(response.total_cost, float)
    assert isinstance(response.generation_id, str)

    # Basic validation of values
    assert response.tokens_prompt > 0
    assert response.tokens_completion > 0
    assert response.total_cost > 0

    # Verify log file was created
    log_dir = Path("benchmark/logs")
    log_files = list(log_dir.glob("*.log"))
    assert len(log_files) > 0

    # Verify log content
    with open(log_files[-1], "r") as f:
        content = f.read()
        assert "What is 2 + 2?" in content
        assert response.content in content
