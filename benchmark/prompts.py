from typing import List, Optional

from benchmark.game import Game
from benchmark.model_utils import Messages


def format_scores(
    game: "Game", current_player: int, up_to_round: Optional[int] = None
) -> str:
    """Format the scores for all players, marking the current player with (you).

    Args:
        game: The current game state
        current_player: Index of the current player
        up_to_round: If provided, only count wins up to this round number (inclusive)
    """
    scores = []
    for idx, player in game.players.items():
        if up_to_round is not None:
            score = sum(
                1 for round_num in player.won_rounds if round_num <= up_to_round
            )
        else:
            score = len(player.won_rounds)
        player_text = f"Player {idx + 1}: {score}"
        if idx == current_player:
            player_text += " (you)"
        scores.append(player_text)
    return "\n".join(scores)


def create_system_message(total_players: int, player_number: int) -> str:
    """Create the system message with player count information."""
    return (
        "You are playing Apples to Apples, a word association game. "
        f"There are {total_players} players in the game, and you are Player {player_number}. "
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
    messages.add_system(create_system_message(len(game.players), player_idx + 1))

    # Add game history for all rounds
    for round in game.rounds:
        messages.add_user(
            f"Round {round.round_number + 1} - Green Card: {round.green_card}"
        )

        # Show played cards and thinking
        played_cards = []
        for pid, move in round.moves.items():
            if pid == player_idx:
                # Always show the player's own move and thinking
                messages.add_user(
                    create_player_prompt(pid, round.green_card, game.players[pid].hand)
                )
                if round.decision and move.played_card == round.decision.winning_card:
                    messages.add_assistant(
                        f"{move.thinking} | {move.played_card} (Winner)"
                    )
                else:
                    messages.add_assistant(f"{move.thinking} | {move.played_card}")
            played_cards.append(move.played_card)

        # Show played cards to all players
        if round.decision:
            if round.judge == player_idx and is_judge:
                # Show the cards and prompt before showing judge's decision
                messages.add_user(
                    create_judge_prompt(
                        round.round_number, round.green_card, played_cards
                    )
                )
                messages.add_assistant(
                    f"{round.decision.reasoning} | {round.decision.winning_card}"
                )
            else:
                # For completed rounds, show all played cards and the judge's decision
                judge_text = (
                    "Your"
                    if round.judge == player_idx
                    else f"Player {round.judge + 1}'s"
                )
                messages.add_user(
                    f"The red cards played for this round were: {', '.join(played_cards)}.\n\n"
                    f"{judge_text} judgement was:\n\n"
                    f"{round.decision.reasoning}\n\n"
                    f"Player {round.judge + 1} (judge) selected '{round.decision.winning_card}' as the winner."
                )

                # Add information about who played the winning card
                for pid, move in round.moves.items():
                    if move.played_card == round.decision.winning_card:
                        winner_text = (
                            "you" if pid == player_idx else f"Player {pid + 1}"
                        )
                        messages.add_user(
                            f"{winner_text} played the {move.played_card} card, and won this round!"
                        )
                        break

        # For current round, show anonymous list to non-judges
        elif not round.decision and player_idx != round.judge:
            messages.add_user(
                f"The played red cards are:\n{format_cards_list(played_cards)}"
            )

        # Show scores after the decision, up to the current round
        if round.decision:
            messages.add_user(
                f"\nScores:\n{format_scores(game, player_idx, round.round_number)}\n"
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
