import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from benchmark.game import Game, PlayerMove, Round
from benchmark.model_utils import ModelResponse
from benchmark.prompts import create_judge_messages
from benchmark.run import create_parser, main, model_move, run_game, validate_args


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


@pytest.mark.asyncio
@patch("benchmark.run.call_model")  # Patch before it's imported in run.py
async def test_run_game(mock_call_model):
    # Mock model responses
    mock_response = ModelResponse(
        content='{"reasoning": "Good thinking", "card": "Test Card"}',
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-run",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.return_value = mock_response

    # Test new game with random models
    game_coro = run_game(num_rounds=3, num_players=2, models=["random", "random"])
    game = await game_coro
    assert len(game.rounds) == 3
    assert len(game.players) == 2
    assert game.total_rounds == 3
    assert all(len(player.won_rounds) > 0 for player in game.players.values())
    assert mock_call_model.call_count == 0  # No model calls for random players

    # Test game with real model
    mock_call_model.reset_mock()
    game_coro = run_game(num_rounds=2, num_players=2, models=["random", "gpt-4"])
    game = await game_coro
    assert len(game.rounds) == 2
    assert len(game.players) == 2
    assert game.total_rounds == 2
    # Should have model calls for the gpt-4 player's moves and when they judge
    assert mock_call_model.call_count > 0

    # Test save and load game
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        game_coro = run_game(
            num_rounds=2, num_players=3, models=["random"] * 3, save_game_path=f.name
        )
        game = await game_coro

        # Verify save worked
        assert Path(f.name).exists()
        with open(f.name) as saved:
            saved_data = json.load(saved)
            assert saved_data["version"] == "1.0"
            assert len(saved_data["rounds"]) == 2
            assert saved_data["total_rounds"] == 2

        # Test loading and continuing
        continued_game_coro = run_game(
            num_rounds=4,  # Continue for 2 more rounds
            num_players=3,
            models=["random"] * 3,
            load_game_path=f.name,
        )
        continued_game = await continued_game_coro
        assert len(continued_game.rounds) == 4
        assert continued_game.total_rounds == 4

        # Clean up
        Path(f.name).unlink()


@pytest.mark.asyncio
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


@pytest.mark.asyncio
async def test_main_error(capsys):
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
            await asyncio.get_event_loop().run_in_executor(None, main)
        captured = capsys.readouterr()
        assert "Error:" in captured.out


@pytest.mark.asyncio
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


def test_parse_model_response():
    """Test the model response parsing function"""
    import pytest

    from benchmark.run import parse_model_response

    # Test valid JSON response
    response = '{"reasoning": "Good thinking", "card": "Test Card"}'
    thinking, card = parse_model_response(response)
    assert thinking == "Good thinking"
    assert card == "Test Card"

    # Test response with markdown code block
    response = """```json
    {
        "reasoning": "Good thinking",
        "card": "Test Card"
    }
    ```"""
    thinking, card = parse_model_response(response)
    assert thinking == "Good thinking"
    assert card == "Test Card"

    # Test response with extra whitespace
    response = """
    {
        "reasoning": "  Good thinking  ",
        "card": "  Test Card  "
    }
    """
    thinking, card = parse_model_response(response)
    assert thinking == "Good thinking"
    assert card == "Test Card"

    # Test response with newlines in reasoning
    response = """
    {
        "reasoning": "Line 1\\nLine 2\\nLine 3",
        "card": "Test Card"
    }
    """
    thinking, card = parse_model_response(response)
    assert "Line 1" in thinking
    assert "Line 2" in thinking
    assert "Line 3" in thinking
    assert card == "Test Card"

    # Test invalid JSON
    with pytest.raises(ValueError, match="Invalid JSON response"):
        parse_model_response("not json")

    # Test missing required fields
    with pytest.raises(ValueError, match="must contain 'reasoning' and 'card' fields"):
        parse_model_response('{"thinking": "Good thinking"}')

    # Test non-object response
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_model_response('"string response"')

    # Test array response
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_model_response('["thinking", "card"]')

    # Test response with wrong field names
    with pytest.raises(ValueError, match="must contain 'reasoning' and 'card' fields"):
        parse_model_response('{"thought": "Good thinking", "choice": "Test Card"}')


@pytest.mark.asyncio
@patch("benchmark.run.call_model")  # Patch before it's imported in run.py
async def test_model_log_preservation(mock_call_model):
    """Test that model logs are preserved when model responses are invalid"""
    # Create a temporary directory for the game
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up the game directory as environment variable
        os.environ["GAME_LOG_DIR"] = os.path.join(temp_dir, "model_logs")

        # Create mock responses for different scenarios
        player_valid_response = ModelResponse(
            content='{"reasoning": "Good thinking", "card": "Silent Movies"}',  # Will be in player's hand
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-1",
            log_path=Path(os.path.join(temp_dir, "model_logs", "valid_response.log")),
        )

        player_invalid_response = ModelResponse(
            content="Invalid JSON response",  # Invalid JSON
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-2",
            log_path=Path(os.path.join(temp_dir, "model_logs", "invalid_response.log")),
        )

        judge_response = ModelResponse(
            content='{"reasoning": "Good choice", "card": "Silent Movies"}',  # Valid judge response
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-3",
            log_path=Path(os.path.join(temp_dir, "model_logs", "judge_response.log")),
        )

        # Create a mock that cycles through responses in a specific order:
        # 1. Player 1's valid move
        # 2. Judge's response
        # 3. Player 1's invalid move
        # 4. Judge's response
        responses = [
            player_valid_response,
            judge_response,
            player_invalid_response,
            judge_response,
        ]
        response_index = 0

        # Set up the mock to cycle through responses
        async def cycle_responses(*args, **kwargs):
            nonlocal response_index
            response = responses[response_index]
            response_index = (response_index + 1) % len(responses)
            return response

        mock_call_model.side_effect = cycle_responses
        game = await run_game(
            num_rounds=2,
            num_players=2,
            models=["gpt-4", "gpt-4"],  # Make both players use the model
            save_game_path=os.path.join(temp_dir, "game_state.json"),
        )

        # Check that both rounds were completed
        assert len(game.rounds) == 2

        # Check first round (valid response)
        round1 = game.rounds[0]
        if 0 in round1.moves:  # If player 1 (gpt-4) wasn't judge
            move = round1.moves[0]
            assert move.log_path == player_valid_response.log_path
            assert "Silent Movies" in move.played_card
            assert "Good thinking" in move.thinking
        elif round1.decision:  # If player 1 was judge
            assert round1.decision.log_path == judge_response.log_path
            assert "Good choice" in round1.decision.reasoning

        # Check second round (invalid response)
        round2 = game.rounds[1]
        if 0 in round2.moves:  # If player 1 (gpt-4) wasn't judge
            move = round2.moves[0]
            assert move.log_path == player_invalid_response.log_path
            assert "Random selection" in move.thinking
            assert "Invalid JSON response" in move.thinking
        elif round2.decision:  # If player 1 was judge
            assert round2.decision.log_path == judge_response.log_path
            assert "Good choice" in round2.decision.reasoning

        # Verify that the HTML report contains links to both logs
        state_path = os.path.join(temp_dir, "game_state.json")
        report_path = os.path.splitext(state_path)[0] + ".html"
        with open(report_path) as f:
            report_content = f.read()
            assert str(player_valid_response.log_path) in report_content
            assert str(player_invalid_response.log_path) in report_content
            assert (
                "no_log.txt" not in report_content
            )  # Should not use no_log.txt for model failures


@pytest.mark.asyncio
@patch("benchmark.run.call_model")  # Patch before it's imported in run.py
async def test_judge_move_with_exact_cards(mock_call_model):
    """Test the judge's move with the exact cards from issue #24"""
    # Create a mock game state
    game = Game.new_game(["Player 1", "Player 2", "Player 3"], total_rounds=6)

    # Start a round and modify it with our test data
    game.start_round()
    game.rounds[0] = Round(round_number=0, green_card="Graceful", judge=0)
    round = game.rounds[0]

    # Add the exact moves from the issue
    round.moves = {
        1: PlayerMove(
            played_card="Queen Elizabeth",
            thinking="Some thinking",
            drawn_card="New Card",
            log_path=Path("tests/test.log"),
        ),
        2: PlayerMove(
            played_card="Dreams",
            thinking="Some thinking",
            drawn_card="New Card",
            log_path=Path("tests/test.log"),
        ),
    }
    game.rounds = [round]

    # Test case 1: Model responds with proper JSON format
    mock_response = ModelResponse(
        content='{"reasoning": "After careful consideration", "card": "Queen Elizabeth"}',
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.return_value = mock_response
    messages = create_judge_messages(game, round.judge)
    played_cards = [move.played_card for move in round.moves.values()]
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card == "Queen Elizabeth"
    assert thinking == "After careful consideration"
    assert log_path == Path("tests/test.log")
    mock_call_model.assert_called_once()

    # Test case 2: Model responds with proper JSON format and punctuation
    mock_response = ModelResponse(
        content='{"reasoning": "She\'s very graceful!", "card": "Queen Elizabeth."}',
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-2",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.reset_mock()
    mock_call_model.return_value = mock_response
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card == "Queen Elizabeth"
    assert thinking == "She's very graceful!"
    assert log_path == Path("tests/test.log")
    mock_call_model.assert_called_once()

    # Test case 3: Model responds with proper JSON format and different case
    mock_response = ModelResponse(
        content='{"reasoning": "Most graceful choice", "card": "QUEEN ELIZABETH"}',
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-3",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.reset_mock()
    mock_call_model.return_value = mock_response
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card == "Queen Elizabeth"
    assert thinking == "Most graceful choice"
    assert log_path == Path("tests/test.log")
    mock_call_model.assert_called_once()

    # Test case 4: Model responds with invalid JSON
    mock_response = ModelResponse(
        content="Queen Elizabeth is the most graceful choice",
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-4",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.reset_mock()
    mock_call_model.return_value = mock_response
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed:" in thinking
    assert "Invalid JSON response" in thinking
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    mock_call_model.assert_called_once()

    # Test case 5: Model responds with JSON missing required fields
    mock_response = ModelResponse(
        content='{"thinking": "Most graceful choice"}',  # Missing "card" field
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-5",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.reset_mock()
    mock_call_model.return_value = mock_response
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed:" in thinking
    assert "Response must contain 'reasoning' and 'card' fields" in thinking
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    mock_call_model.assert_called_once()

    # Test case 6: Model responds with invalid card in JSON
    mock_response = ModelResponse(
        content='{"reasoning": "This is graceful", "card": "The Moon"}',
        model="test-model",
        tokens_prompt=10,
        tokens_completion=5,
        total_cost=0.0001,
        generation_id="test-id-6",
        log_path=Path("tests/test.log"),
    )
    mock_call_model.reset_mock()
    mock_call_model.return_value = mock_response
    card, thinking, log_path = await model_move(
        game, "test-model", played_cards, messages, "judge"
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed:" in thinking
    assert "which is not in played cards" in thinking
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    mock_call_model.assert_called_once()
