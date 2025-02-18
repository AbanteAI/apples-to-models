from typing import List

from benchmark.game import Game
from benchmark.model_utils import Messages

SYSTEM_MESSAGE = (
    "You are playing Apples to Apples, a word association game. "
    "In each round, there is a green card (an adjective) and players play red cards (nouns) "
    "that they think best match the green card. The judge picks the best match."
)


def format_cards_list(cards: List[str]) -> str:
    """Format a list of cards with bullet points."""
    return "\n".join(f"- {card}" for card in cards)


def get_player_prompt_template() -> str:
    """Get the template for player prompts."""
    return (
        "Which card from your hand best matches this green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'Looking at my options, Dinosaurs would be perfect because they represent something truly enormous. "
        "While Mountains are also big, Dinosaurs have a more impressive and awe-inspiring scale | Dinosaurs'"
    )


def get_judge_prompt_template() -> str:
    """Get the template for judge prompts."""
    return (
        "Which red card best matches the green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'After comparing all options, Dinosaurs stands out the most. While both Mountains and Whales "
        "are impressively large, Dinosaurs capture the essence of enormity in a way that sparks imagination | Dinosaurs'"
    )


def create_player_prompt(player_idx: int, green_card: str, hand: List[str]) -> str:
    """Create the prompt for a player to select a card."""
    return (
        f"You are Player {player_idx + 1}. The green card is: {green_card}\n"
        f"Your hand (red cards) contains: {', '.join(hand)}\n"
        f"{get_player_prompt_template()}"
    )


def create_judge_prompt(
    round_num: int, green_card: str, played_cards: List[str]
) -> str:
    """Create the prompt for a judge to select a winning card."""
    return (
        f"Current Round {round_num + 1}\n"
        f"You are the judge. The green card is: {green_card}\n"
        f"The played red cards are:\n{format_cards_list(played_cards)}\n"
        f"{get_judge_prompt_template()}"
    )


def create_game_history(game: "Game", player_idx: int, is_judge: bool) -> Messages:
    """Create messages containing the game history from a player's perspective.

    Args:
        game: The current game state
        player_idx: The index of the player (0-based)
        is_judge: Whether the player is currently the judge

    Returns:
        Messages object containing the system and historical messages
    """
    messages = Messages()
    messages.add_system(SYSTEM_MESSAGE)

    # Add game history for all completed rounds
    for round in game.rounds[:-1]:
        messages.add_user(
            f"Round {round.round_number + 1} - Green Card: {round.green_card}"
        )

        # Show played cards and thinking
        for pid, move in round.moves.items():
            if pid == player_idx:
                # Show the player's own move
                messages.add_user(
                    create_player_prompt(pid, round.green_card, game.players[pid].hand)
                )
                messages.add_assistant(f"{move.thinking} | {move.played_card}")
            else:
                # Show other players' moves
                if is_judge:
                    messages.add_user(
                        f"Player {pid + 1} played: {move.played_card}\n"
                        f"Their thinking: {move.thinking}"
                    )
                else:
                    messages.add_user(f"Player {pid + 1} played: {move.played_card}")

        # Show judge's decision
        if round.decision:
            if round.judge == player_idx:
                if is_judge:
                    # Show the cards and prompt before showing judge's decision
                    played_cards = [move.played_card for move in round.moves.values()]
                    messages.add_user(
                        create_judge_prompt(
                            round.round_number, round.green_card, played_cards
                        )
                    )
                    messages.add_assistant(
                        f"{round.decision.reasoning} | {round.decision.winning_card}"
                    )
                else:
                    messages.add_user(
                        f"You (as judge) selected '{round.decision.winning_card}' as the winner.\n"
                        f"Your reasoning: {round.decision.reasoning}"
                    )
            else:
                messages.add_user(
                    f"Player {round.judge + 1} (judge) selected '{round.decision.winning_card}' as the winner.\n"
                    f"Their reasoning: {round.decision.reasoning}"
                )

    return messages


def create_player_messages(
    game: "Game", player_idx: int, green_card: str, hand: List[str]
) -> Messages:
    """Create messages for a player to select a card to play.

    Args:
        game: The current game state
        player_idx: The index of the player (0-based)
        green_card: The green card (adjective) for this round
        hand: List of red cards in the player's hand

    Returns:
        Messages object containing the system and user messages
    """
    messages = create_game_history(game, player_idx, is_judge=False)
    messages.add_user(create_player_prompt(player_idx, green_card, hand))
    return messages


def create_judge_messages(game: "Game", judge_idx: int) -> Messages:
    """Create messages for the judge to select a winning card, including game history.

    Args:
        game: The current game state
        judge_idx: Index of the judging player

    Returns:
        Messages object containing the system and user messages
    """
    messages = create_game_history(game, judge_idx, is_judge=True)

    # Add current round prompt
    current_round = game.rounds[-1]
    played_cards = [move.played_card for move in current_round.moves.values()]
    messages.add_user(
        create_judge_prompt(
            current_round.round_number, current_round.green_card, played_cards
        )
    )

    return messages
