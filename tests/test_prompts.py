
from benchmark.game import Game, JudgeDecision, PlayerMove, Round
from benchmark.prompts import (
    JUDGE_PROMPT,
    PLAYER_PROMPT,
    create_judge_messages,
    create_player_messages,
)


def test_create_player_messages_basic():
    """Test basic player message creation without game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])
    game.start_round()

    messages = create_player_messages(
        game, 0, "Colorful", ["Rainbow", "Sunset", "Paint"]
    )

    # Check system message
    assert "You are playing Apples to Apples" in messages.messages[0]["content"]

    # Check current round prompt
    last_message = messages.messages[-1]["content"]
    assert "You are Player 1" in last_message
    assert "The green card is: Colorful" in last_message
    assert "Rainbow, Sunset, Paint" in last_message
    assert PLAYER_PROMPT in last_message


def test_create_player_messages_with_history():
    """Test player message creation with game history"""
    game = Game.new_game(["Alice", "Bob"])

    # Set up a completed round
    round1 = Round(round_number=0, green_card="Fast", judge=1)
    round1.moves[0] = PlayerMove(
        played_card="Cheetah",
        thinking="Cheetahs are the fastest land animal",
        drawn_card="Car",
    )
    round1.decision = JudgeDecision(
        winning_card="Cheetah",
        winning_player=0,
        reasoning="Cheetahs are indeed the fastest",
    )
    game.rounds.append(round1)

    # Start new round
    game.start_round()

    messages = create_player_messages(game, 0, "Strong", ["Elephant", "Bodybuilder"])

    # Check history is included
    history = " ".join(msg["content"] for msg in messages.messages)
    assert "Round 1" in history
    assert "Fast" in history
    assert "Cheetah" in history
    assert "Cheetahs are the fastest land animal" in history
    assert "Cheetahs are indeed the fastest" in history

    # Check current round prompt
    last_message = messages.messages[-1]["content"]
    assert "Strong" in last_message
    assert "Elephant, Bodybuilder" in last_message


def test_create_judge_messages_basic():
    """Test basic judge message creation without game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])
    round1 = game.start_round()

    # Players make moves
    round1.moves[0] = PlayerMove(
        played_card="Mountain", thinking="Mountains are tall", drawn_card="River"
    )
    round1.moves[2] = PlayerMove(
        played_card="Skyscraper", thinking="Skyscrapers reach high", drawn_card="House"
    )

    messages = create_judge_messages(game, 1)  # Player 2 is judge

    # Check system message
    assert "You are playing Apples to Apples" in messages.messages[0]["content"]

    # Check current round prompt
    last_message = messages.messages[-1]["content"]
    assert "You are the judge" in last_message
    assert "Mountain" in last_message
    assert "Skyscraper" in last_message
    assert JUDGE_PROMPT in last_message


def test_create_judge_messages_with_history():
    """Test judge message creation with game history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Set up a completed round
    round1 = Round(round_number=0, green_card="Scary", judge=0)
    round1.moves[1] = PlayerMove(
        played_card="Ghost", thinking="Ghosts are spooky", drawn_card="Zombie"
    )
    round1.moves[2] = PlayerMove(
        played_card="Spider", thinking="Spiders frighten many", drawn_card="Snake"
    )
    round1.decision = JudgeDecision(
        winning_card="Ghost", winning_player=1, reasoning="Ghosts are the scariest"
    )
    game.rounds.append(round1)

    # Start new round with different judge
    round2 = Round(round_number=1, green_card="Happy", judge=1)
    round2.moves[0] = PlayerMove(
        played_card="Puppy", thinking="Puppies bring joy", drawn_card="Kitten"
    )
    round2.moves[2] = PlayerMove(
        played_card="Birthday",
        thinking="Birthdays are celebrations",
        drawn_card="Party",
    )
    game.rounds.append(round2)

    messages = create_judge_messages(game, 1)

    # Check history is included
    history = " ".join(msg["content"] for msg in messages.messages)
    assert "Round 1" in history
    assert "Scary" in history
    assert "Ghost" in history
    assert "Spider" in history
    assert "Ghosts are the scariest" in history

    # Check current round
    last_message = messages.messages[-1]["content"]
    assert "Round 2" in last_message
    assert "Happy" in last_message
    assert "Puppy" in last_message
    assert "Birthday" in last_message


def test_player_perspective_in_history():
    """Test that players only see their own thinking in history"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Complete first round
    round1 = Round(round_number=0, green_card="Loud", judge=2)
    round1.moves[0] = PlayerMove(
        played_card="Thunder", thinking="Thunder is deafening", drawn_card="Lightning"
    )
    round1.moves[1] = PlayerMove(
        played_card="Explosion", thinking="Explosions are very loud", drawn_card="Bomb"
    )
    round1.decision = JudgeDecision(
        winning_card="Thunder", winning_player=0, reasoning="Thunder is naturally loud"
    )
    game.rounds.append(round1)

    # Get messages for player 0
    player0_messages = create_player_messages(game, 0, "Bright", ["Sun", "Star"])
    player0_history = " ".join(msg["content"] for msg in player0_messages.messages)

    # Player 0 should see their own thinking but not player 1's
    assert "Thunder is deafening" in player0_history
    assert "Explosions are very loud" not in player0_history

    # Get messages for player 1
    player1_messages = create_player_messages(game, 1, "Bright", ["Moon", "Fire"])
    player1_history = " ".join(msg["content"] for msg in player1_messages.messages)

    # Player 1 should see their own thinking but not player 0's
    assert "Thunder is deafening" not in player1_history
    assert "Explosions are very loud" in player1_history


def test_judge_sees_all_thinking():
    """Test that judges see all players' thinking"""
    game = Game.new_game(["Alice", "Bob", "Charlie"])

    # Set up current round with moves
    round1 = Round(round_number=0, green_card="Cold", judge=0)
    round1.moves[1] = PlayerMove(
        played_card="Ice", thinking="Ice is frozen water", drawn_card="Snow"
    )
    round1.moves[2] = PlayerMove(
        played_card="Winter", thinking="Winter is the coldest season", drawn_card="Fall"
    )
    game.rounds.append(round1)

    messages = create_judge_messages(game, 0)
    judge_view = " ".join(msg["content"] for msg in messages.messages)

    # Judge should see all players' thinking
    assert "Ice is frozen water" in judge_view
    assert "Winter is the coldest season" in judge_view
