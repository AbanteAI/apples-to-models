# Apples to Models

This is a work in progress LLM benchmark being written entirely using Mentat (the GitHub bot). The project aims to provide a framework for comparing and evaluating different language models.

## How It Works

In each round of the game:
1. A red card (adjective) is drawn
2. Players are dealt green cards (nouns)
3. Players choose a green card from their hand that best matches the red card
4. A judge selects the best match among the played cards

The benchmark supports both real language models and random players, allowing for:
- Evaluation of model performance in understanding word relationships
- Comparison between different models
- Testing and development using random players
- Mixed games with both real models and random players

## Running the Benchmark

To run a game, use the `benchmark.run` module with the following arguments:
- `--rounds`: Number of rounds to play
- `--players`: Number of players in the game
- `--models`: Model type for each player (one per player)

Example commands:
```bash
# Run a game with all real models
python -m benchmark.run --rounds 5 --players 3 --models gpt-4 claude-2 gpt-3.5-turbo

# Mix random and real models
python -m benchmark.run --rounds 5 --players 3 --models random gpt-4 random

# Test with all random players
python -m benchmark.run --rounds 5 --players 3 --models random random random
```

### Available Model Types
- `random`: Makes random selections (useful for testing and baselines)
- Real models (via OpenRouter API):
  - `gpt-4`
  - `gpt-3.5-turbo`
  - `claude-2`
  - And other models supported by OpenRouter

### Configuration
The benchmark uses the OpenRouter API for model access. Set up your environment:
1. Create a `.env` file in the project root
2. Add your OpenRouter API key:
   ```
   OPEN_ROUTER_KEY=your_api_key_here
   ```

## Development Status

🚧 **Work in Progress** 🚧

This project is in its early stages of development. Stay tuned for updates!

## About Mentat

This project is being developed using [Mentat](https://mentat.ai), an AI-powered coding assistant. The entire codebase is being written through interactions with the Mentat GitHub bot.