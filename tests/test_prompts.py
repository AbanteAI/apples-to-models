import pytest
from typing import TypedDict, cast
from benchmark.prompts import (
    create_player_messages,
    create_judge_messages,
    create_game_history_messages,
)
from benchmark.game import Game


class MessageDict(TypedDict):
    role: str
    content: str


@pytest.fixture
def sample_game():
    game = Game.new_game(["Alice", "Bob", "Charlie"])
    game.start_round()

    # Player 1 plays a card
    game.play_card(1, game.players[1].hand[0], "This card is perfect because...")

    return game


def test_game_history_messages(sample_game):
    """Test that game history messages are constructed correctly"""
    current_round = sample_game.rounds[sample_game.current_round]
    messages = create_game_history_messages(sample_game, current_round, player_idx=1)

    # Check system message
    assert any(
        cast(MessageDict, msg)["content"]
        and "You are playing Apples to Apples" in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )

    # Check that player 1's move is shown as assistant message (their own move)
    move = current_round.moves[1]
    assert any(
        cast(MessageDict, msg)["role"] == "assistant"
        and cast(MessageDict, msg)["content"]
        and f"For the green card '{current_round.green_card}', {move.thinking}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )


def test_player_messages(sample_game):
    """Test that player messages include game history and current instructions"""
    messages = create_player_messages(sample_game, player_idx=2)

    # Check that current round info is included
    current_round = sample_game.rounds[sample_game.current_round]
    assert any(
        cast(MessageDict, msg)["content"]
        and f"Round {current_round.round_number + 1}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )
    assert any(
        cast(MessageDict, msg)["content"]
        and f"The green card is: {current_round.green_card}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )

    # Check that player's hand is included in instructions
    player = sample_game.players[2]
    assert any(
        cast(MessageDict, msg)["content"]
        and f"Your hand contains: {', '.join(player.hand)}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )


def test_judge_messages(sample_game):
    """Test that judge messages include game history and judging instructions"""
    messages = create_judge_messages(sample_game)

    # Check that current round info is included
    current_round = sample_game.rounds[sample_game.current_round]
    assert any(
        cast(MessageDict, msg)["content"]
        and f"Round {current_round.round_number + 1}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )

    # Check that played cards are listed
    move = current_round.moves[1]
    assert any(
        cast(MessageDict, msg)["content"]
        and f"Player 2: For the green card '{current_round.green_card}', I play: {move.played_card}"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )

    # Check that judging instructions are included
    assert any(
        cast(MessageDict, msg)["content"]
        and "Which red card best matches the green card?"
        in cast(MessageDict, msg)["content"]
        for msg in messages.messages
    )
