from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import random
import json
from pathlib import Path


class Deck(BaseModel):
    """Represents a deck of cards with draw and discard piles"""

    draw_pile: List[str] = Field(default_factory=list)
    discard_pile: List[str] = Field(default_factory=list)

    @classmethod
    def from_file(cls, filepath: Path) -> "Deck":
        """Create a new deck from a file of card names"""
        with open(filepath) as f:
            cards = [line.strip() for line in f if line.strip()]
        random.shuffle(cards)
        return cls(draw_pile=cards)

    def draw(self) -> str:
        """Draw a card from the deck, reshuffling discards if needed"""
        if not self.draw_pile and not self.discard_pile:
            raise ValueError("No cards left in deck")

        if not self.draw_pile:
            # Reshuffle discards into draw pile
            self.draw_pile = self.discard_pile
            random.shuffle(self.draw_pile)
            self.discard_pile = []

        return self.draw_pile.pop()

    def discard(self, card: str) -> None:
        """Add a card to the discard pile"""
        self.discard_pile.append(card)


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

    version: str = "1.0"  # Version of the game state format
    players: Dict[int, Player]  # player_index -> Player
    rounds: List[Round] = Field(default_factory=list)
    current_round: Optional[int] = None
    red_deck: Deck = Field(default_factory=Deck)
    green_deck: Deck = Field(default_factory=Deck)

    @classmethod
    def new_game(cls, player_names: List[str], cards_per_hand: int = 7) -> "Game":
        """Initialize a new game with the given players"""
        # Load card decks
        cards_dir = Path("cards")
        red_deck = Deck.from_file(cards_dir / "red_cards.txt")
        green_deck = Deck.from_file(cards_dir / "green_cards.txt")

        # Create players with initial hands
        players = {}
        for i, name in enumerate(player_names):
            hand = [red_deck.draw() for _ in range(cards_per_hand)]
            players[i] = Player(name=name, hand=hand)

        return cls(
            players=players, red_deck=red_deck, green_deck=green_deck, current_round=0
        )

    def start_round(self) -> Round:
        """Start a new round, selecting the next judge and green card"""
        try:
            green_card = self.green_deck.draw()
        except ValueError:
            raise ValueError("No more green cards in deck")

        round_num = len(self.rounds)
        judge = round_num % len(self.players)

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
        if player_index in current_round.moves:
            raise ValueError("Player has already played a card this round")

        player = self.players[player_index]
        if card not in player.hand:
            raise ValueError("Card not in player's hand")

        # Draw replacement card and update player's hand
        try:
            new_card = self.red_deck.draw()
            player.hand.remove(card)
            player.hand.append(new_card)
            self.red_deck.discard(card)
        except ValueError:
            raise ValueError("No more red cards in deck")

        # Record the move
        current_round.moves[player_index] = PlayerMove(
            played_card=card, thinking=thinking, drawn_card=new_card
        )

    def judge_round(self, winning_card: str, reasoning: str) -> None:
        """Judge the current round, selecting a winning card"""
        if self.current_round is None:
            raise ValueError("No active round")

        current_round = self.rounds[self.current_round]

        # Check that all non-judge players have played
        non_judge_players = set(range(len(self.players))) - {current_round.judge}
        if set(current_round.moves.keys()) != non_judge_players:
            raise ValueError("Not all players have played their cards yet")

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

    def generate_report(self) -> str:
        """Generate a detailed report of the game"""
        report = []

        # Game Overview
        report.append("=== GAME REPORT ===")
        report.append(f"\nPlayers ({len(self.players)}):")
        for idx, player in self.players.items():
            wins = len(player.won_rounds)
            report.append(f"- {player.name}: {wins} win{'s' if wins != 1 else ''}")

        # Round-by-round breakdown
        report.append(f"\nTotal Rounds: {len(self.rounds)}")
        for round in self.rounds:
            report.append(f"\n=== Round {round.round_number + 1} ===")
            report.append(f"Green Card: {round.green_card}")
            report.append(f"Judge: {self.players[round.judge].name}")

            # Player moves
            report.append("\nMoves:")
            for player_idx, move in round.moves.items():
                player_name = self.players[player_idx].name
                report.append(f"\n{player_name}:")
                report.append(f"  Played: {move.played_card}")
                report.append(f"  Thinking: {move.thinking}")

            # Round decision
            if round.decision:
                winner_name = self.players[round.decision.winning_player].name
                report.append(f"\nWinner: {winner_name}")
                report.append(f"Winning Card: {round.decision.winning_card}")
                report.append(f"Judge's Reasoning: {round.decision.reasoning}")

        # Final Standings
        report.append("\n=== Final Standings ===")
        standings = [
            (player.name, len(player.won_rounds)) for player in self.players.values()
        ]
        standings.sort(key=lambda x: x[1], reverse=True)
        for name, wins in standings:
            report.append(f"{name}: {wins} win{'s' if wins != 1 else ''}")

        return "\n".join(report)

    def save_report(self, filepath: str) -> None:
        """Save the game report to a file"""
        report = self.generate_report()
        with open(filepath, "w") as f:
            f.write(report)
