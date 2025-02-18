import asyncio
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.game import Game, PlayerMove, Round
from benchmark.model_utils import ModelResponse
from benchmark.run import create_parser, main, model_judge_move, run_game, validate_args

pytestmark = pytest.mark.asyncio


def test_argument_validation():
    parser = create_parser()

    # Test valid arguments
    args = parser.parse_args(
        ["--rounds", "5", "--players", "3", "--models", "random", "random", "random"]
    )
    validate_args(args)  # Should not raise

    # Test too few players
    args = parser.parse_args(["--rounds", "5", "--players", "1", "--models", "random"])
    with pytest.raises(ValueError, match="Must have at least 2 players"):
        validate_args(args)

    # Test mismatched models and players
    args = parser.parse_args(
        ["--rounds", "5", "--players", "3", "--models", "random", "random"]
    )
    with pytest.raises(
        ValueError, match="Number of models .* must match number of players"
    ):
        validate_args(args)

    # Test nonexistent load file
    args = parser.parse_args(
        [
            "--rounds",
            "5",
            "--players",
            "2",
            "--models",
            "random",
            "random",
            "--load-game",
            "nonexistent.json",
        ]
    )
    with pytest.raises(FileNotFoundError):
        validate_args(args)


@patch("benchmark.model_utils.call_model")
async def test_run_game(mock_call_model):
    # Mock model responses
    mock_response = ModelResponse(
        content="Test Card|Because it matches",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-run",
    )
    mock_call_model.return_value = mock_response

    # Test new game with random models
    game = await run_game(num_rounds=3, num_players=2, models=["random", "random"])
    assert len(game.rounds) == 3
    assert len(game.players) == 2
    assert all(len(player.won_rounds) > 0 for player in game.players.values())
    assert mock_call_model.call_count == 0  # No model calls for random players

    # Test game with real model
    mock_call_model.reset_mock()
    game = await run_game(num_rounds=2, num_players=2, models=["random", "gpt-4"])
    assert len(game.rounds) == 2
    assert len(game.players) == 2
    # Should have model calls for the gpt-4 player's moves and when they judge
    assert mock_call_model.call_count > 0

    # Test save and load game
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        game = await run_game(
            num_rounds=2, num_players=3, models=["random"] * 3, save_game_path=f.name
        )

        # Verify save worked
        assert Path(f.name).exists()
        with open(f.name) as saved:
            saved_data = json.load(saved)
            assert saved_data["version"] == "1.0"
            assert len(saved_data["rounds"]) == 2

        # Test loading and continuing
        continued_game = await run_game(
            num_rounds=4,  # Continue for 2 more rounds
            num_players=3,
            models=["random"] * 3,
            load_game_path=f.name,
        )
        assert len(continued_game.rounds) == 4

        # Clean up
        Path(f.name).unlink()


async def test_main_success(capsys):
    # Test successful run with output capture
    test_args = [
        "benchmark.run",
        "--rounds",
        "2",
        "--players",
        "2",
        "--models",
        "random",
        "random",
    ]
    with patch.object(sys, "argv", test_args):
        await asyncio.get_event_loop().run_in_executor(None, main)
        captured = capsys.readouterr()
        assert "Game completed!" in captured.out
        assert "wins" in captured.out


def test_main_error(capsys):
    # Test error handling in main
    test_args = [
        "benchmark.run",
        "--rounds",
        "2",
        "--players",
        "2",
        "--models",
        "random",  # One model missing
    ]
    with patch.object(sys, "argv", test_args):
        with pytest.raises(SystemExit):
            main()
        captured = capsys.readouterr()
        assert "Error:" in captured.out


async def test_benchmark_command(capsys):
    """Test that the benchmark runs successfully with the example command"""
    test_args = [
        "benchmark.run",
        "--rounds",
        "5",
        "--players",
        "3",
        "--models",
        "random",
        "random",
        "random",
    ]
    with patch.object(sys, "argv", test_args):
        await asyncio.get_event_loop().run_in_executor(None, main)
        captured = capsys.readouterr()
        assert "Game completed!" in captured.out
        assert "Final scores:" in captured.out


def test_normalize_card_name():
    """Test the card name normalization function"""
    from benchmark.run import normalize_card_name

    # Test basic normalization
    assert normalize_card_name("Queen Elizabeth") == "queenelizabeth"

    # Test punctuation removal
    assert normalize_card_name("Queen Elizabeth.") == "queenelizabeth"
    assert normalize_card_name("Queen Elizabeth!") == "queenelizabeth"
    assert normalize_card_name("Queen Elizabeth?") == "queenelizabeth"
    assert normalize_card_name("Queen-Elizabeth") == "queenelizabeth"

    # Test case normalization
    assert normalize_card_name("QUEEN ELIZABETH") == "queenelizabeth"
    assert normalize_card_name("queen elizabeth") == "queenelizabeth"
    assert normalize_card_name("QuEeN eLiZaBeTh") == "queenelizabeth"


def test_judge_move_with_exact_cards():
    """Test the judge's move with the exact cards from issue #24"""
    # Create a mock game state
    game = Game.new_game(["Player 1", "Player 2", "Player 3"])
    round = Round(round_number=0, green_card="Graceful", judge=0)

    # Add the exact moves from the issue
    round.moves = {
        1: PlayerMove(
            played_card="Queen Elizabeth",
            thinking="Some thinking",
            drawn_card="New Card",
        ),
        2: PlayerMove(
            played_card="Dreams", thinking="Some thinking", drawn_card="New Card"
        ),
    }
    game.rounds = [round]

    # Test case 1: Model responds with proper format
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="After careful consideration | Queen Elizabeth",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card == "Queen Elizabeth"
        assert thinking == "After careful consideration"

    # Test case 2: Model responds with proper format and punctuation
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="She's very graceful! | Queen Elizabeth.",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-2",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card == "Queen Elizabeth"
        assert thinking == "She's very graceful!"

    # Test case 3: Model responds with proper format and different case
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="Most graceful choice | QUEEN ELIZABETH",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-3",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card == "Queen Elizabeth"
        assert thinking == "Most graceful choice"

    # Test case 4: Model responds without separator
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="Queen Elizabeth is the most graceful choice",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-4",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
        assert thinking == "Random selection (model failed)"

    # Test case 5: Model responds with multiple separators
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="First | Second | Third",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-5",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
        assert thinking == "Random selection (model failed)"

    # Test case 6: Model responds with invalid card
    with patch("benchmark.model_utils.call_model") as mock_call:
        mock_response = ModelResponse(
            content="This is graceful | The Moon",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-6",
        )
        mock_call.return_value = mock_response
        card, thinking = model_judge_move(game, "test-model")
        assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
        assert thinking == "Random selection (model failed)"
