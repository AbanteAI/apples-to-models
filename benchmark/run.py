import argparse
import os
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from termcolor import cprint  # type: ignore

from benchmark.game import Game

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


def random_player_move(game: Game, player_idx: int) -> tuple[str, str]:
    """Make a random move for the given player"""
    player = game.players[player_idx]
    card = random.choice(player.hand)
    thinking = "Random selection"
    return card, thinking


def normalize_card_name(card: str) -> str:
    """Convert card name to lowercase and remove non-alphabetic characters"""
    return "".join(c.lower() for c in card if c.isalpha())


def model_player_move(game: Game, player_idx: int, model: str) -> tuple[str, str]:
    """Make a model-based move for the given player"""
    from benchmark.model_utils import call_model
    from benchmark.prompts import create_player_messages

    player = game.players[player_idx]
    round = game.rounds[-1]
    green_card = round.green_card

    try:
        messages = create_player_messages(game, player_idx, green_card, player.hand)
        response = call_model(model, messages)
        thinking, card = response.content.split("|", 1)
        thinking = thinking.strip()

        # Normalize the chosen card and player's hand
        normalized_card = normalize_card_name(card)
        # Find the matching card from the hand using normalized comparison
        for original_card in player.hand:
            if normalize_card_name(original_card) == normalized_card:
                card = original_card
                break
        else:
            raise ValueError(f"Model chose card '{card}' which is not in player's hand")

        return card, thinking
    except Exception as e:
        # Fallback to random selection if model fails
        print(
            f"Model error for player {player_idx + 1}, falling back to random: {str(e)}"
        )
        return random_player_move(game, player_idx)


def model_judge_move(game: Game, model: str) -> tuple[str, str]:
    """Make a model-based judging decision"""
    from benchmark.model_utils import call_model
    from benchmark.prompts import create_judge_messages

    round = game.rounds[-1]
    moves = round.moves
    played_cards = [move.played_card for move in moves.values()]
    messages = create_judge_messages(game, round.judge)

    response = None
    try:
        response = call_model(model, messages)
        try:
            # Require exactly one separator
            if response.content.count("|") != 1:
                raise ValueError(
                    f"Response must contain exactly one '|' separator: {response.content}"
                )

            thinking, card = response.content.split("|", 1)
            thinking = thinking.strip()
            card = card.strip()

            # Normalize card name and find match
            normalized_card = normalize_card_name(card)
            for played_card in played_cards:
                if normalize_card_name(played_card) == normalized_card:
                    card = played_card
                    break
            else:
                raise ValueError(
                    f"Could not find matching card '{card}' among played cards: {played_cards}"
                )

            return card, thinking
        except Exception as e:
            # Print raw response only when there's an error parsing it
            print(f"\nError parsing judge response. Raw response was: {response}")
            raise e
    except Exception as e:
        # Fallback to random selection if model fails
        print(f"Model error for judge, falling back to random: {str(e)}")
        winning_move = random.choice(list(moves.values()))
        return winning_move.played_card, "Random selection (model failed)"


def run_game(
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
    else:
        # Create player names as "Player X (model_type)"
        player_names = [
            f"Player {i} ({model})" for i, model in enumerate(models, start=1)
        ]
        game = Game.new_game(player_names)

    # Run rounds until target is reached
    while len(game.rounds) < num_rounds:
        round = game.start_round()
        cprint(f"\n=== Round {len(game.rounds)} ===", "yellow")
        cprint(
            f"Judge: {game.players[round.judge].name} (Player {round.judge})", "yellow"
        )
        cprint(f"Green Card (Adjective): {round.green_card}", "yellow")

        # Have non-judge players make moves
        for player_idx in range(num_players):
            if player_idx != round.judge:
                model = models[player_idx]
                player = game.players[player_idx]
                cprint(f"\n{player.name} (Player {player_idx})'s turn", "red")
                cprint(f"Hand: {', '.join(player.hand)}", "red")

                if model == "random":
                    card, thinking = random_player_move(game, player_idx)
                else:
                    card, thinking = model_player_move(game, player_idx, model)
                game.play_card(player_idx, card, thinking)
                cprint(f"Plays: {card}", "red")
                cprint(f"Thinking: {thinking}", "red")

        # Judge selects a winner
        judge_model = models[round.judge]
        cprint("\nJudge's Decision:", "green")
        if judge_model == "random":
            moves = round.moves
            winning_move = random.choice(list(moves.values()))
            game.judge_round(
                winning_move.played_card,
                "Random selection",
            )
            cprint(f"Winner: {winning_move.played_card}", "green")
            # Find the player who played the winning card
            for player_idx, move in moves.items():
                if move.played_card == winning_move.played_card:
                    winning_player = game.players[player_idx]
                    cprint(
                        f"{winning_player.name} (Player {player_idx}) wins the round!",
                        "green",
                    )
                    break
        else:
            winning_card, thinking = model_judge_move(game, judge_model)
            game.judge_round(winning_card, thinking)
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

    # Save game state
    final_state_path = save_game_path if save_game_path else str(state_path)
    game.save_game(final_state_path)
    print(f"\nGame state saved to: {final_state_path}")

    # Generate and save HTML report
    import os
    import webbrowser

    from benchmark.game_report import save_html_report

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
        game = run_game(
            num_rounds=args.rounds,
            num_players=args.players,
            models=args.models,
            load_game_path=args.load_game,
            save_game_path=args.save_game,
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
