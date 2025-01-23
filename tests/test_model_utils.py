from src.model_utils import Messages, call_model


def main():
    # Create a Messages instance
    messages = Messages()

    # Add a simple question
    messages.add_user("What is 2 + 2?")

    # Call the model
    try:
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)
        print("Model response:", response)
    except Exception as e:
        print("Error calling model:", e)


if __name__ == "__main__":
    main()
