import pytest
from pathlib import Path
import tempfile
import json
from unittest.mock import patch
import sys

from benchmark.run import create_parser, validate_args, run_game, main


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

    # Test unsupported model type
    args = parser.parse_args(
        ["--rounds", "5", "--players", "2", "--models", "random", "gpt4"]
    )
    with pytest.raises(NotImplementedError, match="Model type 'gpt4' not supported"):
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


def test_run_game():
    # Test new game
    game = run_game(num_rounds=3, num_players=2, models=["random", "random"])
    assert len(game.rounds) == 3
    assert len(game.players) == 2
    assert all(len(player.won_rounds) > 0 for player in game.players.values())

    # Test save and load game
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        game = run_game(
            num_rounds=2, num_players=3, models=["random"] * 3, save_game_path=f.name
        )

        # Verify save worked
        assert Path(f.name).exists()
        with open(f.name) as saved:
            saved_data = json.load(saved)
            assert saved_data["version"] == "1.0"
            assert len(saved_data["rounds"]) == 2

        # Test loading and continuing
        continued_game = run_game(
            num_rounds=4,  # Continue for 2 more rounds
            num_players=3,
            models=["random"] * 3,
            load_game_path=f.name,
        )
        assert len(continued_game.rounds) == 4

        # Clean up
        Path(f.name).unlink()


def test_main_success(capsys):
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
        main()
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
