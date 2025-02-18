from benchmark.model_utils import (
    Messages,
    call_model,
    ModelLogger,
    ModelCall,
    JsonFormatter,
)
from pathlib import Path


def demonstrate_basic_usage():
    """Demonstrate basic usage of the model utilities."""
    print("\n=== Basic Usage ===")
    messages = Messages()
    messages.add_user("What is 2 + 2?")

    try:
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)
        print("\nModel Response Details:")
        print("-" * 40)
        print(f"Content: {response.content}")
        print(f"Prompt Tokens: {response.tokens_prompt}")
        print(f"Completion Tokens: {response.tokens_completion}")
        print(f"Total Cost: ${response.total_cost:.6f}")
        print(f"Generation ID: {response.generation_id}")
    except Exception as e:
        print("Error calling model:", e)


def demonstrate_conversation():
    """Demonstrate a multi-turn conversation with the model."""
    print("\n=== Conversation Example ===")
    messages = Messages()

    # Set up a math tutor conversation
    messages.add_system(
        "You are a math tutor. Keep your responses brief and focused on mathematics."
    )

    try:
        # First turn
        messages.add_user("What is the formula for the area of a circle?")
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)
        print("\nStudent: What is the formula for the area of a circle?")
        print(f"Tutor: {response.content}")
        messages.add_assistant(response.content)

        # Second turn
        messages.add_user("Can you explain why we square the radius?")
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)
        print("\nStudent: Can you explain why we square the radius?")
        print(f"Tutor: {response.content}")
    except Exception as e:
        print("Error in conversation:", e)


def demonstrate_different_formatters():
    """Demonstrate using different log formatters."""
    print("\n=== Different Log Formats ===")

    # Create separate directories for different log formats
    log_base = Path("example_logs")
    text_dir = log_base / "text"
    json_dir = log_base / "json"

    messages = Messages()
    messages.add_user("What is the capital of France?")

    try:
        # First call with text formatter
        print("\nMaking API call with text formatter...")
        ModelLogger(log_dir=str(text_dir)).log(
            ModelCall("openai/gpt-4o-mini-2024-07-18", messages)
        )

        # Second call with JSON formatter
        print("Making API call with JSON formatter...")
        ModelLogger(log_dir=str(json_dir), formatter=JsonFormatter()).log(
            ModelCall("openai/gpt-4o-mini-2024-07-18", messages)
        )

        print("\nLogs have been written to:")
        print(f"Text format: {text_dir}")
        print(f"JSON format: {json_dir}")
    except Exception as e:
        print("Error demonstrating formatters:", e)


def main():
    """Run all demonstrations."""
    print("Model Utilities Examples")
    print("=" * 40)

    demonstrate_basic_usage()
    demonstrate_conversation()
    demonstrate_different_formatters()


if __name__ == "__main__":
    main()
