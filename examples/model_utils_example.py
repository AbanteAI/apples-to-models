from benchmark.model_utils import Messages, call_model


def main():
    # Create a Messages instance
    messages = Messages()

    # Add a simple question
    messages.add_user("What is 2 + 2?")

    # Call the model
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


if __name__ == "__main__":
    main()
