import pytest
from benchmark.model_utils import Messages, call_model


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
