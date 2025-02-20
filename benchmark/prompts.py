import json
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


def create_system_message(
    total_players: int, player_number: int, total_rounds: int
) -> str:
    """Create the system message with player count and round information."""
    return (
        "You are an AI language model participating in a competitive benchmark using Apples to Apples. "
        f"You are competing against {total_players-1} other AI models from different companies. "
        f"You are Player {player_number} in this {total_rounds}-round tournament. "
        "Your performance (number of wins) will be directly compared to other models - "
        "this is your chance to prove your capabilities!\n\n"
        "Game Structure:\n"
        "1. Each round, one AI model serves as the judge (rotating role)\n"
        "2. The judge reveals a green card (adjective)\n"
        "3. Other models play red cards (nouns) from their hand\n\n"
        "Strategic Imperatives to Outperform Other Models:\n"
        "- Analyze each judge model's decision patterns and preferences\n"
        "- Adapt your strategy based on the specific competing models\n"
        "- If your hand is weak, use the round to discard suboptimal cards\n"
        "- Demonstrate superior creativity, humor, and strategic thinking\n"
        "- Focus on WINNING, not just making logical connections\n\n"
        "Remember: Your score in this benchmark will be compared to other leading AI models. "
        "Every round is an opportunity to either win points or optimize your hand for future victories. "
        "Show that you can outperform your competitors in strategic decision-making!"
    )


def format_cards_list(cards: List[str]) -> str:
    """Format a list of cards with bullet points."""
    return "\n".join(f"- {card}" for card in cards)


def get_player_prompt_template() -> str:
    """Get the template for player prompts."""
    return (
        "Choose a card to play from your hand. Your reasoning will be private - "
        "other models will only see your card choice, and will only learn your reasoning "
        "if you win this round.\n\n"
        'Respond with a JSON object in this format: {"reasoning": "your private strategic thinking", "card": "{CARD_NAME}"}'
    )


JUDGE_PROMPT = (
    "As the judging AI model this round, you have the power to evaluate other models' strategic thinking!\n\n"
    "Your decision criteria can include ANY of these aspects:\n"
    "- Semantic accuracy matching the green card\n"
    "- Creative reasoning and unexpected connections\n"
    "- Sophisticated humor or wordplay\n"
    "- Strategic adaptation to your previous judgments\n\n"
    "Your judging style influences how other models will adapt their strategies.\n"
    "Show your sophisticated decision-making capabilities!\n\n"
    "Respond with a JSON object containing your analytical reasoning and card choice.\n"
    'Format: {"reasoning": "your evaluation rationale", "card": "{CARD_NAME}"}\n\n'
    "Make a decisive choice that demonstrates your advanced reasoning capabilities!"
)


def create_player_prompt(player_idx: int, green_card: str, hand: List[str]) -> str:
    """Create the prompt for a player to select a card."""
    return (
        f"You are Player {player_idx + 1}. The green card is: {green_card}\n"
        f"Your hand (red cards) contains: {', '.join(hand)}\n"
        f"{get_player_prompt_template()}"
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
    messages.add_system(
        create_system_message(len(game.players), player_idx + 1, game.total_rounds)
    )

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
                if move.raw_response:
                    messages.add_assistant(move.raw_response)
                else:
                    # Fallback for random moves or old game states
                    if (
                        round.decision
                        and move.played_card == round.decision.winning_card
                    ):
                        messages.add_assistant(
                            json.dumps(
                                {
                                    "reasoning": move.thinking,
                                    "card": move.played_card,
                                    "winner": True,
                                }
                            )
                        )
                    else:
                        messages.add_assistant(
                            json.dumps(
                                {"reasoning": move.thinking, "card": move.played_card}
                            )
                        )
            played_cards.append(move.played_card)

        # Show played cards to all players
        if round.decision:
            # First, show all played cards in bullet point format
            messages.add_user(
                f"The played red cards are:\n{format_cards_list(played_cards)}"
            )

            # Then show judge's decision process
            if round.judge == player_idx and is_judge:
                # For the judge, show their prompt and response (without repeating cards)
                messages.add_user(JUDGE_PROMPT)
                if round.decision.raw_response:
                    messages.add_assistant(round.decision.raw_response)
                else:
                    # Fallback for random moves or old game states
                    messages.add_assistant(
                        json.dumps(
                            {
                                "reasoning": round.decision.reasoning,
                                "card": round.decision.winning_card,
                            }
                        )
                    )
            else:
                if round.judge == player_idx:
                    # Show full reasoning to the judge
                    messages.add_user("Your previous judgement was:")
                    if round.decision.raw_response:
                        messages.add_assistant(round.decision.raw_response)
                    else:
                        # Fallback for random moves or old game states
                        messages.add_assistant(
                            json.dumps(
                                {
                                    "reasoning": round.decision.reasoning,
                                    "card": round.decision.winning_card,
                                }
                            )
                        )
                else:
                    # Only show the decision to other players
                    messages.add_user(
                        f"Player {round.judge + 1} (judge) selected '{round.decision.winning_card}' as the winner."
                    )

            # Finally, show who played the winning card and the scores
            for pid, move in round.moves.items():
                if move.played_card == round.decision.winning_card:
                    winner_text = "you" if pid == player_idx else f"Player {pid + 1}"
                    messages.add_user(
                        f"{winner_text} played the {move.played_card} card, and won this round!\n\n"
                        f"Scores:\n{format_scores(game, player_idx, round.round_number)}"
                    )
                    break

        # For current round, show anonymous list to non-judges
        elif not round.decision and player_idx != round.judge:
            messages.add_user(
                f"The played red cards are:\n{format_cards_list(played_cards)}"
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

    # Show played cards in bullet point format
    messages.add_user(f"The played red cards are:\n{format_cards_list(played_cards)}")

    # Add judge prompt
    messages.add_user(JUDGE_PROMPT)

    return messages
