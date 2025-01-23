import pytest
from game import Game
import tempfile
from pathlib import Path


def test_game_initialization():
    players = ["Alice", "Bob", "Charlie"]
    game = Game.new_game(players)

    assert len(game.players) == 3
    assert all(len(player.hand) == 7 for player in game.players.values())
    assert game.current_round == 0
    assert len(game.rounds) == 0


def test_round_gameplay():
    players = ["Alice", "Bob", "Charlie"]
    game = Game.new_game(players)

    # Start round
    round = game.start_round()
    judge_idx = round.judge
    non_judge_players = [i for i in range(len(players)) if i != judge_idx]

    # Players play cards
    for player_idx in non_judge_players:
        card_to_play = game.players[player_idx].hand[0]
        game.play_card(player_idx, card_to_play, "My thinking process")
        assert card_to_play not in game.players[player_idx].hand
        assert len(game.players[player_idx].hand) == 7  # Should draw new card

    # Judge decides
    played_cards = [move.played_card for move in round.moves.values()]
    game.judge_round(played_cards[0], "My judging reasoning")

    assert round.decision is not None
    assert round.decision.winning_card == played_cards[0]
    assert len(game.players[round.decision.winning_player].won_rounds) == 1


def test_game_serialization():
    players = ["Alice", "Bob", "Charlie"]
    game = Game.new_game(players)

    # Play a round
    round = game.start_round()
    judge_idx = round.judge
    non_judge_players = [i for i in range(len(players)) if i != judge_idx]

    for player_idx in non_judge_players:
        card_to_play = game.players[player_idx].hand[0]
        game.play_card(player_idx, card_to_play, "My thinking process")

    played_cards = [move.played_card for move in round.moves.values()]
    game.judge_round(played_cards[0], "My judging reasoning")

    # Save and load game
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        game.save_game(f.name)
        loaded_game = Game.load_game(f.name)

    # Verify game state was preserved
    assert loaded_game.version == "1.0"
    assert len(loaded_game.players) == len(game.players)
    assert len(loaded_game.rounds) == len(game.rounds)
    assert loaded_game.current_round == game.current_round

    # Clean up
    Path(f.name).unlink()


def test_error_handling():
    players = ["Alice", "Bob"]
    game = Game.new_game(players)

    # Test playing before round starts
    with pytest.raises(ValueError):
        game.play_card(0, "Some Card", "thinking")

    # Start round and test invalid plays
    round = game.start_round()
    judge_idx = round.judge
    player_idx = 1 - judge_idx  # Other player

    # Test judge trying to play
    with pytest.raises(ValueError):
        game.play_card(judge_idx, "Some Card", "thinking")

    # Test playing invalid card
    with pytest.raises(ValueError):
        game.play_card(player_idx, "Invalid Card", "thinking")

    # Test playing twice in same round
    valid_card = game.players[player_idx].hand[0]
    game.play_card(player_idx, valid_card, "first play")
    with pytest.raises(ValueError, match="Player has already played a card this round"):
        game.play_card(player_idx, game.players[player_idx].hand[0], "second play")

    # Test judging invalid card
    with pytest.raises(ValueError):
        game.judge_round("Invalid Card", "reasoning")
