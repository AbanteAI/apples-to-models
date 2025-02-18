from typing import List

from benchmark.game import Game
from benchmark.model_utils import Messages

PLAYER_PROMPT = (
    "Which card from your hand best matches this green card? "
    "Respond with your reasoning followed by the card name, separated by ' | '. "
    "For example: 'Looking at my options, Dinosaurs would be perfect because they represent something truly enormous. "
    "While Mountains are also big, Dinosaurs have a more impressive and awe-inspiring scale | Dinosaurs'"
)

JUDGE_PROMPT = (
    "Which red card best matches the green card? "
    "Respond with your reasoning followed by the card name, separated by ' | '. "
    "For example: 'After comparing all options, Dinosaurs stands out the most. While both Mountains and Whales "
    "are impressively large, Dinosaurs capture the essence of enormity in a way that sparks imagination | Dinosaurs'"
)


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
    messages = Messages()
    messages.add_system(
        "You are playing Apples to Apples, a word association game. "
        "In each round, there is a green card and players play red cards "
        "that they think best match the green card. The judge picks the best match."
    )

    # Add game history
    for round in game.rounds[:-1]:  # All rounds except current
        messages.add_user(
            f"Round {round.round_number + 1} - Green Card: {round.green_card}"
        )

        # Show played cards and thinking
        for pid, move in round.moves.items():
            if pid == player_idx:
                # Show the hand and prompt before showing player's thinking
                messages.add_user(
                    f"You are Player {pid + 1}. The green card is: {round.green_card}\n"
                    f"Your hand (red cards) contains: {', '.join(game.players[pid].hand)}\n"
                    f"{PLAYER_PROMPT}"
                )
                # Show player their own thinking in the reasoning | card format
                messages.add_assistant(f"{move.thinking} | {move.played_card}")
            else:
                # Only show the card played by others, not their thinking
                messages.add_user(f"Player {pid + 1} played: {move.played_card}")

        # Show judge's decision and reasoning (visible to all)
        if round.decision:
            if round.judge == player_idx:
                messages.add_user(
                    f"You (as judge) selected '{round.decision.winning_card}' as the winner.\n"
                    f"Your reasoning: {round.decision.reasoning}"
                )
            else:
                messages.add_user(
                    f"Player {round.judge + 1} (judge) selected '{round.decision.winning_card}' as the winner.\n"
                    f"Their reasoning: {round.decision.reasoning}"
                )

    messages.add_user(
        f"You are Player {player_idx + 1}. The green card is: {green_card}\n"
        f"Your hand (red cards) contains: {', '.join(hand)}\n"
        f"{PLAYER_PROMPT}"
    )

    return messages


def create_judge_messages(game: "Game", judge_idx: int) -> Messages:
    """Create messages for the judge to select a winning card, including game history.

    Args:
        game: The current game state
        judge_idx: Index of the judging player

    Returns:
        Messages object containing the system and user messages
    """
    messages = Messages()
    messages.add_system(
        "You are playing Apples to Apples, a word association game. "
        "In each round, there is a green card (an adjective) and players play red cards (nouns) "
        "that they think best match the green card. The judge picks the best match."
    )

    # Add game history
    for round in game.rounds[:-1]:  # All rounds except current
        messages.add_user(
            f"Round {round.round_number + 1} - Green Card: {round.green_card}"
        )

        # Show played cards and thinking
        for player_idx, move in round.moves.items():
            if player_idx == judge_idx:
                # Show the hand and prompt before showing judge's thinking
                messages.add_user(
                    f"You are Player {player_idx + 1}. The green card is: {round.green_card}\n"
                    f"Your hand (red cards) contains: {', '.join(game.players[player_idx].hand)}\n"
                    f"{PLAYER_PROMPT}"
                )
                # Show judge's own thinking in the reasoning | card format
                messages.add_assistant(f"{move.thinking} | {move.played_card}")
            else:
                messages.add_user(
                    f"Player {player_idx + 1} played: {move.played_card}\n"
                    f"Their thinking: {move.thinking}"
                )

        # Show judge's decision
        if round.decision:
            if round.judge == judge_idx:
                # Show the cards to judge and prompt before showing judge's decision
                played_cards = [move.played_card for move in round.moves.values()]
                cards_list = "\n".join(f"- {card}" for card in played_cards)
                messages.add_user(
                    f"Current Round {round.round_number + 1}\n"
                    f"You are the judge. The green card is: {round.green_card}\n"
                    f"The played red cards are:\n{cards_list}\n"
                    f"{JUDGE_PROMPT}"
                )
                messages.add_assistant(
                    f"{round.decision.reasoning} | {round.decision.winning_card}"
                )
            else:
                messages.add_user(
                    f"Player {round.judge + 1} (judge) selected '{round.decision.winning_card}' as the winner.\n"
                    f"Their reasoning: {round.decision.reasoning}"
                )

    # Current round
    current_round = game.rounds[-1]
    played_cards = [move.played_card for move in current_round.moves.values()]
    cards_list = "\n".join(f"- {card}" for card in played_cards)

    messages.add_user(
        f"Current Round {current_round.round_number + 1}\n"
        f"You are the judge. The green card is: {current_round.green_card}\n"
        f"The played red cards are:\n{cards_list}\n"
        f"{JUDGE_PROMPT}"
    )

    return messages
