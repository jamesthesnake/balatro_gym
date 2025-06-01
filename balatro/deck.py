from enum import Enum
from dataclasses import dataclass
from typing import Optional

class Rank(Enum):
    """Card ranks"""
    Two = "2"
    Three = "3"
    Four = "4"
    Five = "5"
    Six = "6"
    Seven = "7"
    Eight = "8"
    Nine = "9"
    Ten = "10"
    Jack = "Jack"
    Queen = "Queen"
    King = "King"
    Ace = "Ace"

class Suit(Enum):
    """Card suits"""
    Spades = "Spades"
    Hearts = "Hearts"
    Clubs = "Clubs"
    Diamonds = "Diamonds"

@dataclass
class Card:
    """Playing card"""
    rank: Rank
    suit: Suit
    
    def chip_value(self) -> int:
        """Get base chip value for this card"""
        chip_map = {
            Rank.Two: 2, Rank.Three: 3, Rank.Four: 4, Rank.Five: 5,
            Rank.Six: 6, Rank.Seven: 7, Rank.Eight: 8, Rank.Nine: 9,
            Rank.Ten: 10, Rank.Jack: 10, Rank.Queen: 10, Rank.King: 10,
            Rank.Ace: 11
        }
        return chip_map[self.rank]

