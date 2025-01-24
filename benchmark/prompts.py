from typing import List
from benchmark.model_utils import Messages


def create_player_messages(
    player_idx: int, green_card: str, hand: List[str]
) -> Messages:
    """Create messages for a player to select a card to play.

    Args:
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

    messages.add_user(
        f"You are Player {player_idx + 1}. The green card is: {green_card}\n"
        f"Your hand (red cards) contains: {', '.join(hand)}\n"
        "Which card from your hand best matches this green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'Looking at my options, Dinosaurs would be perfect because they represent something truly enormous. "
        "While Mountains are also big, Dinosaurs have a more impressive and awe-inspiring scale | Dinosaurs'"
    )

    return messages


def create_judge_messages(green_card: str, played_cards: List[str]) -> Messages:
    """Create messages for the judge to select a winning card.

    Args:
        green_card: The green card (adjective) for this round
        played_cards: List of red cards that were played

    Returns:
        Messages object containing the system and user messages
    """
    messages = Messages()
    messages.add_system(
        "You are the judge in Apples to Apples, a word association game. "
        "In each round, there is a green card and players play red cards "
        "that they think best match the green card. As the judge, you need to pick the best match. "
        "IMPORTANT: Your response must be in the format: 'reasoning | card_name' where card_name "
        "must exactly match one of the played cards."
    )

    cards_list = "\n".join(f"- {card}" for card in played_cards)
    messages.add_user(
        f"The green card is: {green_card}\n"
        f"The played red cards are:\n{cards_list}\n"
        "Which red card best matches the green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'After comparing all options, Dinosaurs stands out the most. While both Mountains and Whales "
        "are impressively large, Dinosaurs capture the essence of enormity in a way that sparks imagination | Dinosaurs'"
    )

    return messages
