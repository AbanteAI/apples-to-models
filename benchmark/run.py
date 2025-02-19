import argparse
import asyncio
import json
import os
import random
import time
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from termcolor import cprint  # type: ignore

from benchmark.game import Game, JudgeDecision
from benchmark.game_report import save_html_report
from benchmark.model_utils import Messages, call_model

# Create games directory if it doesn't exist
GAMES_DIR = Path(__file__).parent / "games"
GAMES_DIR.mkdir(exist_ok=True)


def create_game_directory() -> tuple[Path, str]:
    """Create a new directory for this game's files and return its path and timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    game_dir = GAMES_DIR / timestamp
    game_dir.mkdir(exist_ok=True)
    return game_dir, timestamp


def get_default_save_paths(game_dir: Path, timestamp: str) -> tuple[Path, Path]:
    """Generate default save paths for game state and report"""
    state_path = game_dir / "game_state.json"
    report_path = game_dir / "game_report.html"
    return state_path, report_path


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an Apples to Models game")
    parser.add_argument(
        "--rounds", type=int, required=True, help="Number of rounds to play"
    )
    parser.add_argument(
        "--players", type=int, required=True, help="Number of players in the game"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Model types for each player (only 'random' supported currently)",
    )
    parser.add_argument(
        "--load-game",
        type=str,
        help="Path to load an existing game from",
    )
    parser.add_argument(
        "--save-game",
        type=str,
        help="Path to save the completed game to",
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments"""
    if args.players < 2:
        raise ValueError("Must have at least 2 players")

    if len(args.models) != args.players:
        raise ValueError(
            f"Number of models ({len(args.models)}) must match number of players ({args.players})"
        )

    if args.load_game and not Path(args.load_game).exists():
        raise FileNotFoundError(f"Game file not found: {args.load_game}")

    if args.rounds % args.players != 0:
        cprint(
            f"\nWarning: The number of rounds ({args.rounds}) is not divisible by the number of players ({args.players}). "
            "This means players will not have the same number of opportunities to score (be a player rather than judge).",
            "yellow",
        )


def random_player_move(game: Game, player_idx: int) -> tuple[str, str, Optional[Path]]:
    """Make a random move for the given player"""
    player = game.players[player_idx]
    card = random.choice(player.hand)
    thinking = "Random selection"
    return card, thinking, None


def normalize_card_name(card: str) -> str:
    """Convert card name to lowercase and remove non-alphabetic characters"""
    return "".join(c.lower() for c in card if c.isalpha())


def parse_model_response(content: str) -> tuple[str, str]:
    """Parse a model response into thinking and card components.

    Args:
        content: The raw response content from the model

    Returns:
        A tuple of (thinking, card)

    Raises:
        ValueError: If the response cannot be parsed as valid JSON with the required fields
    """
    # Strip markdown code block markers if present
    if content.startswith("```"):
        content = "\n".join(content.split("\n")[1:-1])
    # Clean up the content by removing any problematic whitespace
    content = content.strip()
    # Parse with more lenient settings
    try:
        response_data = json.loads(content, strict=False)
        if not isinstance(response_data, dict):
            raise ValueError("Response must be a JSON object")
        if "reasoning" not in response_data or "card" not in response_data:
            raise ValueError("Response must contain 'reasoning' and 'card' fields")
        thinking = response_data["reasoning"].strip()
        card = response_data["card"].strip()
        return thinking, card
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response: {e}")


async def handle_model_interaction(
    model: str,
    messages: Messages,
    valid_cards: list[str],
    role: str,
    fallback_fn: Callable[[Game, int], tuple[str, str, Optional[Path]]],
    fallback_args: tuple,
) -> tuple[str, str, Optional[Path]]:
    """Handle model interaction with error handling and fallback.

    Args:
        model: The model to use
        messages: The messages to send to the model
        valid_cards: List of valid cards that can be chosen
        role: Role string for error messages ("player" or "judge")
        fallback_fn: Function to call for random fallback
        fallback_args: Arguments to pass to fallback function

    Returns:
        A tuple of (chosen_card, thinking, log_path)
    """

    model_response = None
    try:
        model_response = await call_model(model, messages)

        # Parse JSON response
        try:
            thinking, card = parse_model_response(model_response.content)
        except ValueError as e:
            raise ValueError(str(e))

        # Normalize the chosen card and find match in valid cards
        normalized_card = normalize_card_name(card)
        for valid_card in valid_cards:
            if normalize_card_name(valid_card) == normalized_card:
                card = valid_card
                break
        else:
            valid_cards_str = ", ".join(valid_cards)
            raise ValueError(
                f"Model chose card '{card}' which is not in valid cards: {valid_cards_str}"
            )

        return card, thinking, model_response.log_path

    except Exception as e:
        # Fallback to random selection if model fails, but preserve the log if we have it
        error_msg = str(e)
        if model_response:
            error_msg = f"Model failed to provide valid response: {str(e)}\nRaw response: {model_response.content}"
            print(f"\nError parsing {role} response: {error_msg}")
            card, thinking, _ = fallback_fn(*fallback_args)
            return (
                card,
                f"Random selection (model failed: {str(e)})\nRaw response: {model_response.content}",
                model_response.log_path,
            )
        else:
            # Model call itself failed
            print(f"Model error for {role}, falling back to random: {str(e)}")
            card, thinking, _ = fallback_fn(*fallback_args)
            return card, thinking, None


async def model_player_move(
    game: Game, player_idx: int, model: str
) -> tuple[str, str, Optional[Path]]:
    """Make a model-based move for the given player"""
    from benchmark.prompts import create_player_messages

    player = game.players[player_idx]
    round = game.rounds[-1]
    green_card = round.green_card

    messages = create_player_messages(game, player_idx, green_card, player.hand)
    return await handle_model_interaction(
        model=model,
        messages=messages,
        valid_cards=player.hand,
        role="player",
        fallback_fn=random_player_move,
        fallback_args=(game, player_idx),
    )


async def model_judge_move(game: Game, model: str) -> tuple[str, str, Optional[Path]]:
    """Make a model-based judging decision"""
    from benchmark.prompts import create_judge_messages

    round = game.rounds[-1]
    moves = round.moves
    played_cards = [move.played_card for move in moves.values()]
    messages = create_judge_messages(game, round.judge)

    def random_judge_move(game: Game, _: int) -> tuple[str, str, Optional[Path]]:
        """Helper function to match the signature expected by handle_model_interaction"""
        winning_move = random.choice(list(moves.values()))
        return winning_move.played_card, "Random selection", None

    return await handle_model_interaction(
        model=model,
        messages=messages,
        valid_cards=played_cards,
        role="judge",
        fallback_fn=random_judge_move,
        fallback_args=(game, round.judge),
    )


async def run_game(
    num_rounds: int,
    num_players: int,
    models: List[str],
    load_game_path: Optional[str] = None,
    save_game_path: Optional[str] = None,
) -> Game:
    """Run a game with the specified configuration"""
    # Create game directory and get paths
    game_dir, timestamp = create_game_directory()
    state_path, report_path = get_default_save_paths(game_dir, timestamp)

    # Set up model logging directory
    os.environ["GAME_LOG_DIR"] = str(game_dir / "model_logs")

    # Load or create new game
    if load_game_path:
        game = Game.load_game(load_game_path)
        if len(game.players) != num_players:
            raise ValueError(
                f"Loaded game has {len(game.players)} players, but {num_players} were requested"
            )
        # Update total_rounds if continuing with a different number
        game.total_rounds = num_rounds
    else:
        # Create player names as "Player X (model_type)"
        player_names = [
            f"Player {i} ({model})" for i, model in enumerate(models, start=1)
        ]
        game = Game.new_game(player_names, total_rounds=num_rounds)
        # Start tracking benchmark time
        start_time = time.time()
        game.benchmark_stats.start_time = start_time

    try:
        # Run rounds until target is reached
        while len(game.rounds) < num_rounds:
            round = game.start_round()
            cprint(f"\n=== Round {len(game.rounds)} ===", "yellow")
            cprint(
                f"Judge: {game.players[round.judge].name} (Player {round.judge})",
                "yellow",
            )
            cprint(f"Green Card (Adjective): {round.green_card}", "yellow")

            # Have non-judge players make moves in parallel using asyncio
            async def process_player_move(player_idx, model):
                if player_idx == round.judge:
                    return None

                player = game.players[player_idx]

                if model == "random":
                    card, thinking, log_path = random_player_move(game, player_idx)
                else:
                    card, thinking, log_path = await model_player_move(
                        game, player_idx, model
                    )

                # Print all player output together after the model call
                cprint(f"\n{player.name} (Player {player_idx})'s turn", "red")
                cprint(f"Hand: {', '.join(player.hand)}", "red")
                cprint(f"Plays: {card}", "red")
                cprint(f"Thinking: {thinking}", "red")

                return player_idx, card, thinking, log_path

            # Create tasks for all players
            tasks = [
                process_player_move(player_idx, models[player_idx])
                for player_idx in range(num_players)
            ]

            # Run all tasks concurrently and collect results
            results = await asyncio.gather(*tasks)

            # Process completed moves
            for result in results:
                if result:  # Skip None results (judge's turn)
                    player_idx, card, thinking, log_path = result
                    game.play_card(player_idx, card, thinking)
                    if log_path and player_idx in game.rounds[-1].moves:
                        game.rounds[-1].moves[player_idx].log_path = log_path

            # Judge selects a winner
            judge_model = models[round.judge]
            cprint("\nJudge's Decision:", "green")
            if judge_model == "random":
                winning_move = random.choice(list(round.moves.values()))
                winning_card = winning_move.played_card
                thinking = "Random selection"
                game.judge_round(winning_card, thinking)
                # Create a new decision with the no_log path
                if game.rounds[-1].decision:
                    game.rounds[-1].decision = JudgeDecision(
                        winning_card=winning_card,
                        winning_player=game.rounds[-1].decision.winning_player,
                        reasoning=thinking,
                        log_path=Path("benchmark/logs/no_log.txt"),
                    )
                cprint(f"Winner: {winning_card}", "green")
                # Find the player who played the winning card
                for player_idx, move in round.moves.items():
                    if move.played_card == winning_card:
                        winning_player = game.players[player_idx]
                        cprint(
                            f"{winning_player.name} (Player {player_idx}) wins the round!",
                            "green",
                        )
                        break
            else:
                winning_card, thinking, log_path = await model_judge_move(
                    game, judge_model
                )
                game.judge_round(winning_card, thinking)
                # Create a new decision with the log path
                if game.rounds[-1].decision:
                    game.rounds[-1].decision = JudgeDecision(
                        winning_card=winning_card,
                        winning_player=game.rounds[-1].decision.winning_player,
                        reasoning=thinking,
                        log_path=log_path
                        if log_path
                        else Path("benchmark/logs/no_log.txt"),
                    )
                cprint(f"Winner: {winning_card}", "green")
                # Find the player who played the winning card
                for player_idx, move in round.moves.items():
                    if move.played_card == winning_card:
                        winning_player = game.players[player_idx]
                        cprint(
                            f"{winning_player.name} (Player {player_idx}) wins the round!",
                            "green",
                        )
                        cprint(f"Reasoning: {thinking}", "green")
                        break

    except KeyboardInterrupt:
        # If the current round is incomplete (no decision), remove it
        if game.rounds and not game.rounds[-1].decision:
            game.rounds.pop()
        cprint("\n\nGame interrupted! Saving progress...", "yellow")

    finally:
        # Save game state
        final_state_path = save_game_path if save_game_path else str(state_path)
        game.save_game(final_state_path)
        print(f"\nGame state saved to: {final_state_path}")

        # Record end time before generating report
        if not game.benchmark_stats.end_time:
            end_time = time.time()
            game.benchmark_stats.end_time = end_time

        # Generate and save HTML report
        final_report_path = (
            os.path.splitext(final_state_path)[0] + ".html"
            if save_game_path
            else str(report_path)
        )
        save_html_report(game, final_report_path)
        print(f"Game report saved to: {final_report_path}")

        # Open the report in the default web browser
        webbrowser.open(f"file://{os.path.abspath(final_report_path)}")

        return game


def main():
    parser = create_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
        game = asyncio.run(
            run_game(
                num_rounds=args.rounds,
                num_players=args.players,
                models=args.models,
                load_game_path=args.load_game,
                save_game_path=args.save_game,
            )
        )

        # Print final scores
        cprint("\n🎮 Game completed! Final scores:", "magenta", attrs=["bold"])
        for idx, player in game.players.items():
            cprint(f"{player.name}: {len(player.won_rounds)} wins", "magenta")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
