from typing import List, Dict, Optional
from benchmark.model_utils import Messages
from benchmark.game import Game, Round, PlayerMove, JudgeDecision


def format_judge_decision(decision: JudgeDecision, green_card: str, played_cards: List[str]) -> str:
    """Format the judge's decision and reasoning for display in the conversation."""
    cards_list = "
".join(f"- {card}" for card in played_cards)
    return (
        f"As the judge for the green card '{green_card}', looking at:
{cards_list}
"
        f"I choose '{decision.winning_card}' because {decision.reasoning}"
    )


def format_player_move(move: PlayerMove, green_card: str, for_player: bool = False) -> str:
    """Format a player's move for display in the conversation.
    
    Args:
        move: The player's move
        green_card: The green card for the round
        for_player: If True, includes the private thinking
    """
    if for_player:
        return f"For the green card '{green_card}', {move.thinking} | {move.played_card}"
    return f"For the green card '{green_card}', I play: {move.played_card}"


def create_game_history_messages(game: Game, current_round: Round, player_idx: Optional[int] = None) -> Messages:
    """Create messages containing the game history up to the current round.
    
    Args:
        game: The current game state
        current_round: The current round
        player_idx: If provided, includes private thinking for this player's moves
    
    Returns:
        Messages containing the complete game history
    """
    messages = Messages()
    messages.add_system(
        "You are playing Apples to Apples, a word association game. "
        "In each round, there is a green card (an adjective) and players play red cards (nouns) "
        "that they think best match the green card. The judge then picks the best match."
    )

    # Add history of completed rounds
    for past_round in game.rounds:
        if past_round.round_number >= current_round.round_number:
            break

        messages.add_user(f"
Round {past_round.round_number + 1}:")
        messages.add_user(f"The green card is: {past_round.green_card}")
        messages.add_user(f"Player {past_round.judge + 1} is the judge.")

        # Add player moves
        for move_player_idx, move in past_round.moves.items():
            if move_player_idx == player_idx:
                messages.add_assistant(format_player_move(move, past_round.green_card, True))
            else:
                messages.add_user(
                    f"Player {move_player_idx + 1}: " + 
                    format_player_move(move, past_round.green_card, False)
                )

        # Add judge's decision
        if past_round.decision:
            messages.add_user(
                f"Judge (Player {past_round.judge + 1}): " +
                format_judge_decision(
                    past_round.decision,
                    past_round.green_card,
                    [move.played_card for move in past_round.moves.values()]
                )
            )

    return messages


def create_player_messages(game: Game, player_idx: int) -> Messages:
    """Create messages for a player to select a card to play.

    Args:
        game: The current game state
        player_idx: The index of the player (0-based)

    Returns:
        Messages object containing the complete conversation history and current instructions
    """
    current_round = game.rounds[game.current_round]
    messages = create_game_history_messages(game, current_round, player_idx)

    # Add current round context
    messages.add_user(f"
Round {current_round.round_number + 1}:")
    messages.add_user(f"The green card is: {current_round.green_card}")
    messages.add_user(f"Player {current_round.judge + 1} is the judge.")

    # Add any moves already made this round
    for move_player_idx, move in current_round.moves.items():
        if move_player_idx == player_idx:
            messages.add_assistant(format_player_move(move, current_round.green_card, True))
        else:
            messages.add_user(
                f"Player {move_player_idx + 1}: " +
                format_player_move(move, current_round.green_card, False)
            )

    # Add instructions for current move
    player = game.players[player_idx]
    messages.add_user(
        f"You are Player {player_idx + 1}. Your hand contains: {', '.join(player.hand)}
"
        "Which card from your hand best matches this green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'Looking at my options, Dinosaurs would be perfect because they represent something truly enormous. "
        "While Mountains are also big, Dinosaurs have a more impressive and awe-inspiring scale | Dinosaurs'"
    )

    return messages


def create_judge_messages(game: Game) -> Messages:
    """Create messages for the judge to select a winning card.

    Args:
        game: The current game state

    Returns:
        Messages object containing the complete conversation history and current instructions
    """
    current_round = game.rounds[game.current_round]
    messages = create_game_history_messages(game, current_round)

    # Add current round context
    messages.add_user(f"
Round {current_round.round_number + 1}:")
    messages.add_user(f"The green card is: {current_round.green_card}")
    messages.add_user(f"You are Player {current_round.judge + 1} and the judge for this round.")

    # Add all moves for this round
    played_cards = []
    for move_player_idx, move in current_round.moves.items():
        messages.add_user(
            f"Player {move_player_idx + 1}: " +
            format_player_move(move, current_round.green_card, False)
        )
        played_cards.append(move.played_card)

    # Add judging instructions
    cards_list = "
".join(f"- {card}" for card in played_cards)
    messages.add_user(
        f"The played red cards are:
{cards_list}
"
        "Which red card best matches the green card? "
        "Respond with your reasoning followed by the card name, separated by ' | '. "
        "For example: 'After comparing all options, Dinosaurs stands out the most. While both Mountains and Whales "
        "are impressively large, Dinosaurs capture the essence of enormity in a way that sparks imagination | Dinosaurs'"
    )

    return messages