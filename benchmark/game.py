import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field

from benchmark.model_utils import ModelResponse


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
    log_path: Union[str, Path]  # Path to the model call log
    raw_response: Optional[str] = None  # Raw model response if available


class JudgeDecision(BaseModel):
    """Represents the judge's decision for a round"""

    winning_card: str
    winning_player: int
    reasoning: str  # Public reasoning for the decision
    log_path: Union[str, Path]  # Path to the model call log
    raw_response: Optional[str] = None  # Raw model response if available


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


class BenchmarkStats(BaseModel):
    """Tracks benchmark statistics including timing and model usage"""

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    total_cost: float = 0.0
    model_stats: Dict[str, Dict[str, float]] = Field(default_factory=lambda: {})

    def add_response(self, response: ModelResponse) -> None:
        """Add a model response to the stats"""
        self.total_cost += response.total_cost

        if response.model not in self.model_stats:
            self.model_stats[response.model] = {"cost": 0.0, "calls": 0.0}

        self.model_stats[response.model]["cost"] += response.total_cost
        self.model_stats[response.model]["calls"] += 1

    @property
    def total_time(self) -> Optional[float]:
        """Get total time in seconds for the benchmark run"""
        if self.start_time is None or self.end_time is None:
            return None
        return self.end_time - self.start_time


class Game(BaseModel):
    """Represents the full game state"""

    version: str = "1.0"  # Version of the game state format
    players: Dict[int, Player]  # player_index -> Player
    rounds: List[Round] = Field(default_factory=list)
    current_round: Optional[int] = None
    total_rounds: int  # Total number of rounds to be played
    red_deck: Deck = Field(default_factory=Deck)
    green_deck: Deck = Field(default_factory=Deck)
    benchmark_stats: BenchmarkStats = Field(default_factory=BenchmarkStats)

    @classmethod
    def new_game(
        cls, player_names: List[str], total_rounds: int, cards_per_hand: int = 7
    ) -> "Game":
        """Initialize a new game with the given players"""
        # Load card decks from benchmark directory
        cards_dir = Path(__file__).parent / "cards"
        red_deck = Deck.from_file(cards_dir / "red_cards.txt")
        green_deck = Deck.from_file(cards_dir / "green_cards.txt")

        # Create players with initial hands
        players = {}
        for i, name in enumerate(player_names):
            hand = [red_deck.draw() for _ in range(cards_per_hand)]
            players[i] = Player(name=name, hand=hand)

        return cls(
            players=players,
            red_deck=red_deck,
            green_deck=green_deck,
            current_round=0,
            total_rounds=total_rounds,
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

    def play_card(
        self,
        player_index: int,
        card: str,
        thinking: str,
        model_response: Optional[ModelResponse] = None,
    ) -> None:
        """Play a card for the given player in the current round"""
        if self.current_round is None or len(self.rounds) <= self.current_round:
            raise ValueError("No active round")

        current_round = self.rounds[self.current_round]
        player = self.players[player_index]

        if player_index == current_round.judge:
            raise ValueError(f"{player.name} is the judge and cannot play a card")
        if player_index in current_round.moves:
            raise ValueError(f"{player.name} has already played a card this round")

        if card not in player.hand:
            raise ValueError(f"Card '{card}' is not in {player.name}'s hand")

        # Draw replacement card and update player's hand
        try:
            new_card = self.red_deck.draw()
            player.hand.remove(card)
            player.hand.append(new_card)
            self.red_deck.discard(card)
        except ValueError:
            raise ValueError("No more red cards in deck")

        # Update benchmark stats if response provided
        log_path = Path("benchmark/logs/no_log.txt")  # Default path for random moves
        if model_response:
            self.benchmark_stats.add_response(model_response)
            log_path = model_response.log_path or log_path

        # Record the move
        current_round.moves[player_index] = PlayerMove(
            played_card=card,
            thinking=thinking,
            drawn_card=new_card,
            log_path=log_path,
            raw_response=model_response.content if model_response else None,
        )

    def judge_round(
        self,
        winning_card: str,
        reasoning: str,
        model_response: Optional[ModelResponse] = None,
    ) -> None:
        """Judge the current round, selecting a winning card"""
        if self.current_round is None:
            raise ValueError("No active round")

        current_round = self.rounds[self.current_round]

        # Check that all non-judge players have played
        non_judge_players = set(range(len(self.players))) - {current_round.judge}
        if set(current_round.moves.keys()) != non_judge_players:
            # Find who hasn't played yet
            missing_players = [
                self.players[i].name
                for i in non_judge_players
                if i not in current_round.moves
            ]
            raise ValueError(
                f"Waiting for players to play: {', '.join(missing_players)}"
            )

        # Find the player who played the winning card
        winning_player = None
        for player_idx, move in current_round.moves.items():
            if move.played_card == winning_card:
                winning_player = player_idx
                break

        if winning_player is None:
            raise ValueError(f"Card '{winning_card}' was not played this round")

        # Update benchmark stats if response provided
        log_path = Path("benchmark/logs/no_log.txt")  # Default path for random moves
        if model_response:
            self.benchmark_stats.add_response(model_response)
            log_path = model_response.log_path or log_path

        # Record the decision
        current_round.decision = JudgeDecision(
            winning_card=winning_card,
            winning_player=winning_player,
            reasoning=reasoning,
            log_path=log_path,
            raw_response=model_response.content if model_response else None,
        )

        # Update winner's score
        self.players[winning_player].won_rounds.append(self.current_round)

    def save_game(self, filepath: str) -> None:
        """Save the game state to a JSON file"""

        def path_serializer(obj):
            if isinstance(obj, Path):
                return str(obj)
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )

        with open(filepath, "w") as f:
            data = self.model_dump()
            json.dump(data, f, indent=2, default=path_serializer)

    @classmethod
    def load_game(cls, filepath: str) -> "Game":
        """Load a game state from a JSON file"""
        with open(filepath) as f:
            data = json.load(f)
            # Convert string paths back to Path objects
            for round in data.get("rounds", []):
                for move in round.get("moves", {}).values():
                    if "log_path" in move:
                        move["log_path"] = Path(move["log_path"])
                if round.get("decision") and "log_path" in round["decision"]:
                    round["decision"]["log_path"] = Path(round["decision"]["log_path"])
        return cls.model_validate(data)
