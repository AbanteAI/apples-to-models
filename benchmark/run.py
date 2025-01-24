import argparse
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime
from termcolor import cprint  # type: ignore

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

    if args.load_game and not Path(args.load_game).exists():
        raise FileNotFoundError(f"Game file not found: {args.load_game}")


def random_player_move(game: Game, player_idx: int) -> tuple[str, str]:
    """Make a random move for the given player"""
    player = game.players[player_idx]
    card = random.choice(player.hand)
    thinking = "Random selection"
    return card, thinking


def model_player_move(game: Game, player_idx: int, model: str) -> tuple[str, str]:
    """Make a model-based move for the given player"""
    from benchmark.model_utils import Messages, call_model

    player = game.players[player_idx]
    round = game.rounds[-1]
    adjective = (
        round.green_card
    )  # Note: Despite the name, this is the red (adjective) card

    # Create conversation for the model
    messages = Messages()
    messages.add_system(
        "You are playing Apples to Apples, a word association game. "
        "In each round, there is a red card (adjective) and players play green cards (nouns) "
        "that they think best match the red card. The judge picks the best match."
    )

    # Provide context about the current round
    messages.add_user(
        f"You are Player {player_idx + 1}. The red card (adjective) is: {adjective}\n"
        f"Your hand (green cards) contains: {', '.join(player.hand)}\n"
        "Which green card from your hand best matches this red card? "
        "Respond with just the card name and a brief explanation separated by '|'. "
        "For example: 'Dinosaurs|They were absolutely enormous!'"
    )

    try:
        response = call_model(model, messages)
        card, thinking = response.split("|", 1)
        card = card.strip()
        thinking = thinking.strip()

        # Validate that the chosen card is in the player's hand
        if card not in player.hand:
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
    from benchmark.model_utils import Messages, call_model

    round = game.rounds[-1]
    adjective = (
        round.green_card
    )  # Note: Despite the name, this is the red (adjective) card
    moves = round.moves

    # Create conversation for the model
    messages = Messages()
    messages.add_system(
        "You are the judge in Apples to Apples, a word association game. "
        "In each round, there is a red card (adjective) and players play green cards (nouns) "
        "that they think best match the red card. As the judge, you need to pick the best match."
    )

    # Provide context about the current round
    cards_list = "\n".join(f"- {move.played_card}" for move in moves.values())
    messages.add_user(
        f"The red card (adjective) is: {adjective}\n"
        f"The played green cards (nouns) are:\n{cards_list}\n"
        "Which green card best matches the red card? "
        "Respond with just the card name and a brief explanation separated by '|'. "
        "For example: 'Dinosaurs|They perfectly represent something enormous!'"
    )

    try:
        response = call_model(model, messages)
        card, thinking = response.split("|", 1)
        card = card.strip()
        thinking = thinking.strip()

        # Validate that the chosen card was actually played
        if not any(move.played_card == card for move in moves.values()):
            raise ValueError(
                f"Model chose card '{card}' which was not among the played cards"
            )

        return card, thinking
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
        cprint(f"Judge: {game.players[round.judge].name}", "yellow")
        cprint(
            f"Red Card: {round.green_card}", "yellow"
        )  # Note: green_card field contains the red (adjective) card

        # Have non-judge players make moves
        for player_idx in range(num_players):
            if player_idx != round.judge:
                model = models[player_idx]
                if model == "random":
                    card, thinking = random_player_move(game, player_idx)
                else:
                    card, thinking = model_player_move(game, player_idx, model)
                game.play_card(player_idx, card, thinking)
                cprint(f"{game.players[player_idx].name} plays: {card}", "green")
                cprint(f"Thinking: {thinking}", "green")

        # Judge selects a winner
        judge_model = models[round.judge]
        cprint("\nJudge's Decision:", "blue")
        if judge_model == "random":
            moves = round.moves
            winning_move = random.choice(list(moves.values()))
            game.judge_round(
                winning_move.played_card,
                "Random selection",
            )
            cprint(f"Winner: {winning_move.played_card}", "blue")
            # Find the player who played the winning card
            for player_idx, move in moves.items():
                if move.played_card == winning_move.played_card:
                    winning_player = game.players[player_idx]
                    cprint(f"Player {winning_player.name} wins the round!", "blue")
                    break
        else:
            winning_card, thinking = model_judge_move(game, judge_model)
            game.judge_round(winning_card, thinking)
            cprint(f"Winner: {winning_card}", "blue")
            # Find the player who played the winning card
            for player_idx, move in round.moves.items():
                if move.played_card == winning_card:
                    winning_player = game.players[player_idx]
                    cprint(f"Player {winning_player.name} wins the round!", "blue")
                    cprint(f"Reasoning: {thinking}", "blue")
                    break

    # Save game (use default path if none specified)
    save_path = save_game_path if save_game_path else str(get_default_save_path())
    game.save_game(save_path)
    print(f"\nGame saved to: {save_path}")

    # Generate and open HTML report
    from benchmark.game_report import save_html_report
    import webbrowser
    import os

    report_path = os.path.splitext(save_path)[0] + ".html"
    save_html_report(game, report_path)
    print(f"Game report saved to: {report_path}")

    # Open the report in the default web browser
    webbrowser.open(f"file://{os.path.abspath(report_path)}")

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
