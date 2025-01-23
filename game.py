from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import random
import json
from pathlib import Path


class PlayerMove(BaseModel):
    """Represents a player's move in a round"""

    played_card: str
    thinking: str  # Private reasoning for the move
    drawn_card: str  # Card drawn to replace the played card


class JudgeDecision(BaseModel):
    """Represents the judge's decision for a round"""

    winning_card: str
    winning_player: int
    reasoning: str  # Public reasoning for the decision


class Round(BaseModel):
    """Represents a single round in the game"""

    round_number: int
    green_card: str
    judge: int  # Index of the judging player
    moves: Dict[int, PlayerMove] = Field(default_factory=dict)  # player_index -> move
    decision: Optional[JudgeDecision] = None


class Player(BaseModel):
    """Represents a player in the game"""

    name: str
    hand: List[str] = Field(default_factory=list)
    won_rounds: List[int] = Field(default_factory=list)


class Game(BaseModel):
    """Represents the full game state"""

    players: Dict[int, Player]  # player_index -> Player
    rounds: List[Round] = Field(default_factory=list)
    current_round: Optional[int] = None
    red_deck: List[str] = Field(default_factory=list)
    green_deck: List[str] = Field(default_factory=list)

    @classmethod
    def new_game(cls, player_names: List[str], cards_per_hand: int = 7) -> "Game":
        """Initialize a new game with the given players"""
        # Load card decks
        cards_dir = Path("cards")
        with open(cards_dir / "red_cards.txt") as f:
            red_cards = [line.strip() for line in f if line.strip()]
        with open(cards_dir / "green_cards.txt") as f:
            green_cards = [line.strip() for line in f if line.strip()]

        # Shuffle decks
        random.shuffle(red_cards)
        random.shuffle(green_cards)

        # Create players with initial hands
        players = {}
        for i, name in enumerate(player_names):
            hand = red_cards[:cards_per_hand]
            red_cards = red_cards[cards_per_hand:]
            players[i] = Player(name=name, hand=hand)

        return cls(
            players=players, red_deck=red_cards, green_deck=green_cards, current_round=0
        )

    def start_round(self) -> Round:
        """Start a new round, selecting the next judge and green card"""
        if not self.green_deck:
            raise ValueError("No more green cards in deck")

        round_num = len(self.rounds)
        judge = round_num % len(self.players)
        green_card = self.green_deck.pop()

        new_round = Round(round_number=round_num, green_card=green_card, judge=judge)
        self.rounds.append(new_round)
        self.current_round = round_num
        return new_round

    def play_card(self, player_index: int, card: str, thinking: str) -> None:
        """Play a card for the given player in the current round"""
        if self.current_round is None:
            raise ValueError("No active round")

        current_round = self.rounds[self.current_round]
        if player_index == current_round.judge:
            raise ValueError("Judge cannot play a card")

        player = self.players[player_index]
        if card not in player.hand:
            raise ValueError("Card not in player's hand")

        # Draw replacement card
        if not self.red_deck:
            raise ValueError("No more red cards in deck")
        new_card = self.red_deck.pop()

        # Update player's hand
        player.hand.remove(card)
        player.hand.append(new_card)

        # Record the move
        current_round.moves[player_index] = PlayerMove(
            played_card=card, thinking=thinking, drawn_card=new_card
        )

    def judge_round(self, winning_card: str, reasoning: str) -> None:
        """Judge the current round, selecting a winning card"""
        if self.current_round is None:
            raise ValueError("No active round")

        current_round = self.rounds[self.current_round]

        # Find the player who played the winning card
        winning_player = None
        for player_idx, move in current_round.moves.items():
            if move.played_card == winning_card:
                winning_player = player_idx
                break

        if winning_player is None:
            raise ValueError("Winning card was not played this round")

        # Record the decision
        current_round.decision = JudgeDecision(
            winning_card=winning_card,
            winning_player=winning_player,
            reasoning=reasoning,
        )

        # Update winner's score
        self.players[winning_player].won_rounds.append(self.current_round)

    def save_game(self, filepath: str) -> None:
        """Save the game state to a JSON file"""
        with open(filepath, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    @classmethod
    def load_game(cls, filepath: str) -> "Game":
        """Load a game state from a JSON file"""
        with open(filepath) as f:
            data = json.load(f)
        return cls.model_validate(data)
