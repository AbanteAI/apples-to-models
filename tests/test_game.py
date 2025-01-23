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


def test_html_game_report():
    from benchmark.game_report import generate_html_report, save_html_report

    players = ["Alice", "Bob", "Charlie"]
    game = Game.new_game(players)

    # Play a complete round
    round = game.start_round()
    judge_idx = round.judge
    non_judge_players = [i for i in range(len(players)) if i != judge_idx]

    # Players play cards
    for player_idx in non_judge_players:
        card_to_play = game.players[player_idx].hand[0]
        game.play_card(player_idx, card_to_play, "Test thinking")

    # Judge decides
    played_cards = [move.played_card for move in round.moves.values()]
    game.judge_round(played_cards[0], "Test reasoning")

    # Generate report
    report = generate_html_report(game)

    # Verify report contains key HTML elements and game information
    assert "<!DOCTYPE html>" in report
    assert "<html>" in report
    assert "<head>" in report
    assert "<body>" in report

    # Check for interactive elements
    assert "function toggleThinking" in report
    assert "onclick=" in report
    assert 'class="thinking"' in report

    # Check game content
    assert "Total Rounds: 1" in report
    assert "Green Card:" in report
    assert "Judge:" in report
    assert "Test thinking" in report
    assert "Test reasoning" in report

    # Check styling
    assert "<style>" in report
    assert "display: none" in report  # For hidden thinking sections
    assert "background-color" in report  # For winner highlighting

    # Test saving report
    with tempfile.NamedTemporaryFile(mode="w", suffix=".html", delete=False) as f:
        save_html_report(game, f.name)
        with open(f.name) as saved:
            saved_report = saved.read()
        assert saved_report == report
        assert "<!DOCTYPE html>" in saved_report

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

    # Test judging before all players have played
    game.start_round()
    with pytest.raises(ValueError, match="Not all players have played their cards yet"):
        game.judge_round(game.players[0].hand[0], "too early")

    # Test judging invalid card
    with pytest.raises(ValueError):
        game.judge_round("Invalid Card", "reasoning")
