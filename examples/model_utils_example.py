import os
from benchmark.model_utils import Messages, call_model


def main():
    print("Starting model call example...")

    # Check if API key is set
    if not os.getenv("OPEN_ROUTER_KEY"):
        print("\nError: OPEN_ROUTER_KEY environment variable is not set.")
        print("Please set it with your OpenRouter API key:")
        print("export OPEN_ROUTER_KEY='your-api-key-here'")
        return

    # Create a Messages instance
    messages = Messages()

    # Add a simple question
    messages.add_user("What is 2 + 2?")
    print("\nSending question to model: 'What is 2 + 2?'")

    # Call the model
    try:
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)

        # Display all response information
        print("\nModel Response:")
        print("-" * 40)
        print(f"Content: {response.content}")
        print(f"Cost: ${response.cost:.4f}")
        print(f"Total Tokens: {response.total_tokens}")
        print(f"  - Prompt Tokens: {response.prompt_tokens}")
        print(f"  - Completion Tokens: {response.completion_tokens}")

        print(
            "\nNote: A detailed log file has also been created in the benchmark/logs directory."
        )

    except Exception as e:
        print("\nError calling model:", e)


if __name__ == "__main__":
    main()
