import argparse
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from benchmark.game import Game

# Create games directory if it doesn't exist
GAMES_DIR = Path(__file__).parent / "games"
GAMES_DIR.mkdir(exist_ok=True)


def get_default_save_path() -> Path:
    """Generate a default save path with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return GAMES_DIR / f"game_{timestamp}.json"


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

    for model in args.models:
        if model != "random":
            raise NotImplementedError(
                f"Model type '{model}' not supported. Only 'random' is currently available."
            )

    if args.load_game and not Path(args.load_game).exists():
        raise FileNotFoundError(f"Game file not found: {args.load_game}")


def random_player_move(game: Game, player_idx: int) -> tuple[str, str]:
    """Make a random move for the given player"""
    player = game.players[player_idx]
    card = random.choice(player.hand)
    thinking = "Random selection"
    return card, thinking


def run_game(
    num_rounds: int,
    num_players: int,
    models: List[str],
    load_game_path: Optional[str] = None,
    save_game_path: Optional[str] = None,
) -> Game:
    """Run a game with the specified configuration"""
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

        # Have non-judge players make moves
        for player_idx in range(num_players):
            if player_idx != round.judge:
                card, thinking = random_player_move(game, player_idx)
                game.play_card(player_idx, card, thinking)

        # Judge randomly selects a winner
        moves = round.moves
        winning_move = random.choice(list(moves.values()))
        game.judge_round(
            winning_move.played_card,
            "Random selection",
        )

    # Save game (use default path if none specified)
    save_path = save_game_path if save_game_path else str(get_default_save_path())
    game.save_game(save_path)
    print(f"\nGame saved to: {save_path}")

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
        print("\nGame completed! Final scores:")
        for idx, player in game.players.items():
            print(f"{player.name}: {len(player.won_rounds)} wins")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
