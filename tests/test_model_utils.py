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
    assert isinstance(response, str)
    assert len(response) > 0
