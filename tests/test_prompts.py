from pathlib import Path

from openai.types.chat import (
    ChatCompletionMessageParam,
)

from benchmark.game import Game, JudgeDecision, PlayerMove, Round
from benchmark.prompts import (
    create_judge_messages,
    create_player_messages,
    get_judge_prompt_template,
    get_player_prompt_template,
)


def get_message_content(msg: ChatCompletionMessageParam) -> str:
    """Helper function to safely get message content as string."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return " ".join(str(part) for part in content)
    return str(content)


def assert_content_contains(content: str, substring: str) -> None:
    """Helper function to assert string containment with proper type handling."""
    assert substring in content, f"Expected '{substring}' in '{content}'"


def test_create_player_messages_basic():
    """Test basic player message creation without game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])
    game.start_round()

    messages = create_player_messages(
        game, 0, "Colorful", ["Rainbow", "Sunset", "Paint"]
    )

    # Check system message
    system_content = get_message_content(messages.messages[0])
    assert_content_contains(system_content, "You are playing Apples to Apples")

    # Check current round prompt
    last_content = get_message_content(messages.messages[-1])
    assert_content_contains(last_content, "You are Player 1")
    assert_content_contains(last_content, "The green card is: Colorful")
    assert_content_contains(last_content, "Rainbow, Sunset, Paint")
    assert_content_contains(last_content, get_player_prompt_template())


def test_create_player_messages_with_history():
    """Test player message creation with game history"""
    game = Game.new_game(["Alice", "Bob"])

    # Set up a completed round
    round1 = Round(round_number=0, green_card="Fast", judge=1)
    round1.moves[0] = PlayerMove(
        played_card="Cheetah",
        thinking="Cheetahs are the fastest land animal",
        drawn_card="Car",
        log_path=Path("tests/test.log"),
    )
    round1.decision = JudgeDecision(
        winning_card="Cheetah",
        winning_player=0,
        reasoning="Cheetahs are indeed the fastest",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round1)

    # Start new round
    game.start_round()

    messages = create_player_messages(game, 0, "Strong", ["Elephant", "Bodybuilder"])

    # Check history is included
    history = " ".join(get_message_content(msg) for msg in messages.messages)
    assert_content_contains(history, "Round 1")
    assert_content_contains(history, "Fast")
    assert_content_contains(history, "Cheetah")
    assert_content_contains(history, "Cheetahs are the fastest land animal")
    assert_content_contains(history, "Cheetahs are indeed the fastest")

    # Check current round prompt
    last_content = get_message_content(messages.messages[-1])
    assert_content_contains(last_content, "Strong")
    assert_content_contains(last_content, "Elephant, Bodybuilder")


def test_create_judge_messages_basic():
    """Test basic judge message creation without game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])
    round1 = game.start_round()

    # Players make moves
    round1.moves[0] = PlayerMove(
        played_card="Mountain",
        thinking="Mountains are tall",
        drawn_card="River",
        log_path=Path("tests/test.log"),
    )
    round1.moves[2] = PlayerMove(
        played_card="Skyscraper",
        thinking="Skyscrapers reach high",
        drawn_card="House",
        log_path=Path("tests/test.log"),
    )

    messages = create_judge_messages(game, 1)  # Player 2 is judge

    # Check system message
    system_content = get_message_content(messages.messages[0])
    assert_content_contains(system_content, "You are playing Apples to Apples")

    # Check current round prompt
    last_content = get_message_content(messages.messages[-1])
    assert_content_contains(last_content, "You are the judge")
    assert_content_contains(last_content, "Mountain")
    assert_content_contains(last_content, "Skyscraper")
    assert_content_contains(last_content, get_judge_prompt_template())


def test_create_judge_messages_with_history():
    """Test judge message creation with game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Set up a completed round
    round1 = Round(round_number=0, green_card="Scary", judge=0)
    round1.moves[1] = PlayerMove(
        played_card="Ghost",
        thinking="Ghosts are spooky",
        drawn_card="Zombie",
        log_path=Path("tests/test.log"),
    )
    round1.moves[2] = PlayerMove(
        played_card="Spider",
        thinking="Spiders frighten many",
        drawn_card="Snake",
        log_path=Path("tests/test.log"),
    )
    round1.decision = JudgeDecision(
        winning_card="Ghost",
        winning_player=1,
        reasoning="Ghosts are the scariest",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round1)

    # Start new round with different judge
    round2 = Round(round_number=1, green_card="Happy", judge=1)
    round2.moves[0] = PlayerMove(
        played_card="Puppy",
        thinking="Puppies bring joy",
        drawn_card="Kitten",
        log_path=Path("tests/test.log"),
    )
    round2.moves[2] = PlayerMove(
        played_card="Birthday",
        thinking="Birthdays are celebrations",
        drawn_card="Party",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round2)

    messages = create_judge_messages(game, 1)

    # Check history is included
    history = " ".join(get_message_content(msg) for msg in messages.messages)
    assert_content_contains(history, "Round 1")
    assert_content_contains(history, "Scary")
    assert_content_contains(history, "Ghost")
    assert_content_contains(history, "Spider")
    assert_content_contains(history, "Ghosts are the scariest")

    # Check current round
    last_content = get_message_content(messages.messages[-1])
    assert_content_contains(last_content, "Round 2")
    assert_content_contains(last_content, "Happy")
    assert_content_contains(last_content, "Puppy")
    assert_content_contains(last_content, "Birthday")


def test_player_perspective_in_history():
    """Test that players only see their own thinking in history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Complete first round
    round1 = Round(round_number=0, green_card="Loud", judge=2)
    round1.moves[0] = PlayerMove(
        played_card="Thunder",
        thinking="Thunder is deafening",
        drawn_card="Lightning",
        log_path=Path("tests/test.log"),
    )
    round1.moves[1] = PlayerMove(
        played_card="Explosion",
        thinking="Explosions are very loud",
        drawn_card="Bomb",
        log_path=Path("tests/test.log"),
    )
    round1.decision = JudgeDecision(
        winning_card="Thunder",
        winning_player=0,
        reasoning="Thunder is naturally loud",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round1)

    # Start new round
    round2 = Round(round_number=1, green_card="Bright", judge=0)
    game.rounds.append(round2)

    # Get messages for player 0
    player0_messages = create_player_messages(game, 0, "Bright", ["Sun", "Star"])
    player0_history = " ".join(
        get_message_content(msg) for msg in player0_messages.messages
    )

    # Player 0 should see their own move and the judge's decision
    assert_content_contains(player0_history, "Thunder")  # Their played card
    assert_content_contains(player0_history, "Thunder is deafening")  # Their thinking
    assert_content_contains(
        player0_history, "Thunder is naturally loud"
    )  # Judge's reasoning
    assert "Explosions are very loud" not in player0_history  # Other player's thinking

    # Get messages for player 1
    player1_messages = create_player_messages(game, 1, "Bright", ["Moon", "Fire"])
    player1_history = " ".join(
        get_message_content(msg) for msg in player1_messages.messages
    )

    # Player 1 should see their own move and the judge's decision
    assert_content_contains(player1_history, "Explosion")  # Their played card
    assert_content_contains(
        player1_history, "Explosions are very loud"
    )  # Their thinking
    assert_content_contains(
        player1_history, "Thunder is naturally loud"
    )  # Judge's reasoning
    assert "Thunder is deafening" not in player1_history  # Other player's thinking


def test_game_history_visibility():
    """Test that game history shows the right information to each player"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Set up a completed round
    round1 = Round(round_number=0, green_card="Cold", judge=0)
    round1.moves[1] = PlayerMove(
        played_card="Ice",
        thinking="Ice is frozen water",
        drawn_card="Snow",
        log_path=Path("tests/test.log"),
    )
    round1.moves[2] = PlayerMove(
        played_card="Winter",
        thinking="Winter is the coldest season",
        drawn_card="Fall",
        log_path=Path("tests/test.log"),
    )
    round1.decision = JudgeDecision(
        winning_card="Ice",
        winning_player=1,
        reasoning="Ice is literally frozen and therefore the coldest",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round1)

    # Start new round
    round2 = Round(round_number=1, green_card="Hot", judge=1)
    round2.moves[0] = PlayerMove(
        played_card="Fire",
        thinking="Fire is extremely hot",
        drawn_card="Sun",
        log_path=Path("tests/test.log"),
    )
    round2.moves[2] = PlayerMove(
        played_card="Desert",
        thinking="Deserts are very hot places",
        drawn_card="Beach",
        log_path=Path("tests/test.log"),
    )
    game.rounds.append(round2)

    # Test Player 1's view (judge in first round)
    messages = create_player_messages(game, 0, "Hot", ["Sun", "Star"])
    player1_view = " ".join(get_message_content(msg) for msg in messages.messages)

    # Should see their own thinking from judging
    assert_content_contains(
        player1_view, "Ice is literally frozen and therefore the coldest"
    )
    # Should see winning player's identity
    assert_content_contains(player1_view, "Player 2 played: Ice (Winner)")
    # Should see other cards anonymously
    assert_content_contains(player1_view, "Someone played: Winter")

    # Test Player 2's view (winner of first round, judge in second)
    messages = create_judge_messages(game, 1)
    player2_view = " ".join(get_message_content(msg) for msg in messages.messages)

    # Should see their own thinking from winning move
    assert_content_contains(player2_view, "Ice is frozen water")
    # Should see their own card marked as winner
    assert_content_contains(player2_view, "Ice (Winner)")
    # Should see other cards anonymously
    assert_content_contains(player2_view, "Someone played: Winter")
    # Should see current round cards anonymously
    assert_content_contains(player2_view, "Fire")
    assert_content_contains(player2_view, "Desert")

    # Test Player 3's view (regular player)
    messages = create_player_messages(game, 2, "Hot", ["Desert"])
    player3_view = " ".join(get_message_content(msg) for msg in messages.messages)

    # Should see their own thinking
    assert_content_contains(player3_view, "Winter is the coldest season")
    # Should see winning player's identity
    assert_content_contains(player3_view, "Player 2 played: Ice (Winner)")
    # Should see current round cards anonymously
    assert_content_contains(player3_view, "The played red cards are:")
    # Should not see other players' thinking
    assert "Fire is extremely hot" not in player3_view
    assert "Ice is frozen water" not in player3_view
