import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Union
from unittest.mock import patch

import pytest
from openai.types.chat import ChatCompletionMessageParam

from benchmark.game import Game, PlayerMove, Round
from benchmark.model_utils import Messages, ModelResponse
from benchmark.prompts import create_judge_messages
from benchmark.run import create_parser, main, model_move, run_game, validate_args


def get_message_content(msg: Union[dict[str, Any], ChatCompletionMessageParam]) -> str:
    """Safely extract message content as string."""
    if isinstance(msg, dict):
        content = msg.get("content")
    else:
        content = msg.get("content", None)
    if content is None:
        return ""
    return str(content)


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
    with pytest.raises(ValueError, match="Your entire response must be valid JSON"):
        parse_model_response("not json")

    # Test missing required fields
    with pytest.raises(
        ValueError,
        match="Your JSON response must contain both 'reasoning' and 'card' fields",
    ):
        parse_model_response('{"thinking": "Good thinking"}')

    # Test non-object response
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_model_response('"string response"')

    # Test array response
    with pytest.raises(ValueError, match="must be a JSON object"):
        parse_model_response('["thinking", "card"]')

    # Test response with wrong field names
    with pytest.raises(
        ValueError,
        match="Your JSON response must contain both 'reasoning' and 'card' fields",
    ):
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
        player_invalid_response = ModelResponse(
            content="Invalid JSON response",  # Invalid JSON
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-1",
            log_path=Path(os.path.join(temp_dir, "model_logs", "invalid_response.log")),
        )

        player_invalid_response2 = ModelResponse(
            content='{"reasoning": "Bad choice", "card": "InvalidCard"}',  # Invalid card
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-2",
            log_path=Path(
                os.path.join(temp_dir, "model_logs", "invalid_response2.log")
            ),
        )

        player_valid_response = ModelResponse(
            content='{"reasoning": "Good thinking", "card": "Card1"}',  # Valid response
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-3",
            log_path=Path(os.path.join(temp_dir, "model_logs", "valid_response.log")),
        )

        # Set up test data
        valid_cards = ["Card1", "Card2"]
        messages = Messages()
        messages.add_user("Initial prompt")

        # Test case: Success after two retries
        mock_call_model.side_effect = [
            player_invalid_response,
            player_invalid_response2,
            player_valid_response,
        ]

        card, thinking, log_path = await model_move(
            "test-model", valid_cards, messages, "player"
        )

        # Verify final result
        assert card == "Card1"
        assert thinking == "Good thinking"
        assert log_path == player_valid_response.log_path

        # Verify error messages were added
        error_messages = [get_message_content(msg) for msg in messages.messages]
        assert any(
            "Your entire response must be valid JSON" in msg for msg in error_messages
        )
        assert any("not in player's hand" in msg for msg in error_messages)

        # Verify all model calls were made
        assert mock_call_model.call_count == 3

        # Verify that error responses were preserved in messages
        assert any(
            "Your entire response must be valid JSON" in msg for msg in error_messages
        )
        assert any("InvalidCard" in msg for msg in error_messages)

        # Verify that the final log path is from the successful response
        assert log_path == player_valid_response.log_path


@pytest.mark.asyncio
@patch("benchmark.run.call_model")  # Patch before it's imported in run.py
async def test_model_move_retries(mock_call_model):
    """Test that model_move retries on failures and provides proper guidance"""
    # Create test data
    messages = Messages()
    messages.add_user("Initial prompt")

    # Test case 1: Success after first retry (JSON parsing error)
    test1_cards = ["Card2", "Card3"]  # Different set of valid cards
    responses = [
        ModelResponse(
            content="Invalid JSON",
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-1",
            log_path=Path("tests/test1.log"),
        ),
        ModelResponse(
            content='{"reasoning": "Good choice", "card": "Card2"}',  # Changed to Card2 to match test expectations
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-2",
            log_path=Path("tests/test2.log"),
        ),
    ]
    mock_call_model.side_effect = responses.copy()  # Use copy to preserve list
    card, thinking, log_path = await model_move(
        "test-model", test1_cards, messages, "player"
    )
    assert card == "Card2"  # Updated to match the expected card
    assert thinking == "Good choice"
    assert log_path == Path("tests/test2.log")
    assert mock_call_model.call_count == 2
    # Verify error guidance was added
    assert any(
        "Your entire response must be valid JSON" in get_message_content(msg)
        for msg in messages.messages
    )

    # Test case 2: Success after second retry (invalid card then JSON error)
    mock_call_model.reset_mock()
    messages = Messages()
    messages.add_user("Initial prompt")
    test2_cards = ["Card2", "Card4"]  # Different set of valid cards
    responses = [
        ModelResponse(
            content='{"reasoning": "Bad choice", "card": "InvalidCard"}',
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-3",
            log_path=Path("tests/test3.log"),
        ),
        ModelResponse(
            content="More invalid JSON",
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-4",
            log_path=Path("tests/test4.log"),
        ),
        ModelResponse(
            content='{"reasoning": "Finally good", "card": "Card2"}',
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-5",
            log_path=Path("tests/test5.log"),
        ),
    ]
    mock_call_model.side_effect = responses.copy()  # Use copy to preserve list
    card, thinking, log_path = await model_move(
        "test-model", test2_cards, messages, "player"
    )
    assert card == "Card2"  # Should get Card2 from the final successful response
    assert thinking == "Finally good"
    assert log_path == Path("tests/test5.log")
    assert mock_call_model.call_count == 3
    # Verify error guidances were added
    error_messages = [get_message_content(msg) for msg in messages.messages]
    assert any("not in player's hand" in msg for msg in error_messages)
    assert any(
        "Your entire response must be valid JSON" in msg for msg in error_messages
    )

    # Test case 3: Fallback to random after all retries fail
    mock_call_model.reset_mock()
    messages = Messages()
    messages.add_user("Initial prompt")
    responses = [
        ModelResponse(
            content='{"reasoning": "Bad1", "card": "Invalid1"}',
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-6",
            log_path=Path("tests/test6.log"),
        ),
        ModelResponse(
            content='{"reasoning": "Bad2", "card": "Invalid2"}',
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-7",
            log_path=Path("tests/test7.log"),
        ),
        ModelResponse(
            content='{"reasoning": "Bad3", "card": "Invalid3"}',
            model="test-model",
            tokens_prompt=10,
            tokens_completion=5,
            total_cost=0.0001,
            generation_id="test-id-8",
            log_path=Path("tests/test8.log"),
        ),
    ]
    mock_call_model.side_effect = responses
    test3_cards = ["Card5", "Card6"]  # Different set of valid cards
    card, thinking, log_path = await model_move(
        "test-model", test3_cards, messages, "player"
    )
    assert card in test3_cards  # Should be random choice from valid cards
    assert "Random selection" in thinking
    assert "after 3 attempts" in thinking
    assert "Invalid3" in thinking  # Should include last raw response
    assert log_path == Path("tests/test8.log")
    assert mock_call_model.call_count == 3
    # Verify all error guidances were added
    error_messages = [get_message_content(msg) for msg in messages.messages]
    error_count = sum(1 for msg in error_messages if "not in player's hand" in msg)
    assert error_count == 2  # Two error guidances


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
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
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
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
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
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
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
    mock_call_model.side_effect = [
        mock_response
    ] * 3  # Will be called 3 times for retries
    card, thinking, log_path = await model_move(
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed after 3 attempts:" in thinking
    assert "Your entire response must be valid JSON" in thinking
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    assert mock_call_model.call_count == 3  # Should be called 3 times for retries

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
    mock_call_model.side_effect = [
        mock_response
    ] * 3  # Will be called 3 times for retries
    card, thinking, log_path = await model_move(
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed after 3 attempts:" in thinking
    assert (
        "Your JSON response must contain both 'reasoning' and 'card' fields" in thinking
    )
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    assert mock_call_model.call_count == 3  # Should be called 3 times for retries
    # Verify error guidance was added
    error_messages = [get_message_content(msg) for msg in messages.messages]
    error_count = sum(
        1
        for msg in error_messages
        if "Your JSON response must contain both 'reasoning' and 'card' fields" in msg
    )
    assert error_count == 2  # Two retries

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
    mock_call_model.side_effect = [
        mock_response
    ] * 3  # Will be called 3 times for retries
    card, thinking, log_path = await model_move(
        model="test-model",
        valid_cards=played_cards,
        messages=messages,
        role="judge",
    )
    assert card in ["Queen Elizabeth", "Dreams"]  # Should fall back to random
    assert "Random selection (model failed after 3 attempts:" in thinking
    assert "which is not in played cards" in thinking
    assert mock_response.content in thinking  # Raw response should be included
    assert log_path == Path("tests/test.log")  # Log path should be preserved
    assert mock_call_model.call_count == 3  # Should be called 3 times for retries
    # Verify error guidance was added
    error_messages = [get_message_content(msg) for msg in messages.messages]
    error_count = sum(
        1 for msg in error_messages if "which is not in played cards" in msg
    )
    assert error_count == 2  # Two retries
