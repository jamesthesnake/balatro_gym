# balatro/play.py
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

# Import Card, Rank, and Suit from deck module
from .deck import Card, Rank, Suit

@dataclass
class HandCard:
    """Card in hand with selection state"""
    card: Optional[Card]
    selected: bool

# Result classes that match how they're used in smart_analyzer.py
class PlayResult:
    """Result of playing a hand"""
    AGAIN = "Again"
    ROUND_OVER = "RoundOver"
    GAME_OVER = "GameOver"
    
    def __init__(self, type_str: str, data=None, connection=None):
        self.type = type_str
        if type_str == self.AGAIN:
            self.play = Play(data, connection) if data else None
        elif type_str == self.ROUND_OVER:
            from .overview import RoundOverview
            self.overview = RoundOverview(data, connection) if data else None
        elif type_str == self.GAME_OVER:
            from .overview import GameOverview
            self.game_over = GameOverview(connection) if connection else None

class DiscardResult:
    """Result of discarding cards"""
    AGAIN = "Again"
    GAME_OVER = "GameOver"
    
    def __init__(self, type_str: str, data=None, connection=None):
        self.type = type_str
        if type_str == self.AGAIN:
            self.play = Play(data, connection) if data else None
        elif type_str == self.GAME_OVER:
            from .overview import GameOverview
            self.game_over = GameOverview(connection) if connection else None

class Play:
    """Play screen handler"""
    def __init__(self, play_info: dict, connection):
        self.info = play_info
        self.connection = connection
    
    def hand(self) -> List[HandCard]:
        """Get current hand"""
        hand = []
        for card_data in self.info["hand"]:
            if card_data["card"]:
                rank = Rank(card_data["card"]["rank"])
                suit = Suit(card_data["card"]["suit"])
                card = Card(rank=rank, suit=suit)
                hand.append(HandCard(card=card, selected=card_data["selected"]))
            else:
                hand.append(HandCard(card=None, selected=False))
        return hand
    
    def score(self) -> float:
        return self.info["score"]
    
    def hands(self) -> int:
        return self.info["hands"]
    
    def discards(self) -> int:
        return self.info["discards"]
    
    def money(self) -> int:
        return self.info["money"]
    
    def blind(self) -> dict:
        return self.info["current_blind"]
    
    async def click(self, indices: List[int]):
        """Click cards at specified indices"""
        response = await self.connection.send_request("play/click", {"indices": indices})
        
        if "Ok" in response["body"]:
            self.info = response["body"]["Ok"]
            return self
        else:
            raise Exception(f"Click failed: {response['body']['Err']}")
    
    async def play(self):
        """Play the selected cards"""
        response = await self.connection.send_request("play/play")
        
        if "Ok" in response["body"]:
            result_data = response["body"]["Ok"]
            
            if "Again" in result_data:
                return PlayResult(PlayResult.AGAIN, result_data["Again"], self.connection)
            elif "RoundOver" in result_data:
                return PlayResult(PlayResult.ROUND_OVER, result_data["RoundOver"], self.connection)
            elif "GameOver" in result_data:
                return PlayResult(PlayResult.GAME_OVER, None, self.connection)
        else:
            raise Exception(f"Play failed: {response['body']['Err']}")
    
    async def discard(self):
        """Discard the selected cards"""
        response = await self.connection.send_request("play/discard")
        
        if "Ok" in response["body"]:
            result_data = response["body"]["Ok"]
            
            if "Again" in result_data:
                return DiscardResult(DiscardResult.AGAIN, result_data["Again"], self.connection)
            elif "GameOver" in result_data:
                return DiscardResult(DiscardResult.GAME_OVER, None, self.connection)
        else:
            raise Exception(f"Discard failed: {response['body']['Err']}")
