from benchmark.model_utils import Messages, call_model
from pathlib import Path


def main():
    print("Starting model call example...")

    # Create a Messages instance
    messages = Messages()

    # Add a simple question
    messages.add_user("What is 2 + 2?")
    print("\nSending question to model: 'What is 2 + 2?'")

    # Call the model
    try:
        response = call_model("openai/gpt-4o-mini-2024-07-18", messages)
        print("\nModel response:", response)

        # Get the most recent log file to show the cost
        log_dir = Path("benchmark/logs")
        log_files = sorted(log_dir.glob("*.log"))
        if log_files:
            latest_log = log_files[-1]
            with open(latest_log, "r") as f:
                log_content = f.read()
                # Extract and display the cost line if it exists
                for line in log_content.split("\n"):
                    if line.startswith("Cost: $"):
                        print("\nGeneration cost:", line)
                        break

    except Exception as e:
        print("Error calling model:", e)


if __name__ == "__main__":
    main()
