from enum import Enum
from typing import Optional
from dataclasses import dataclass

class Deck(Enum):
    """Deck types in Balatro"""
    Red = "b_red"
    Blue = "b_blue"
    Yellow = "b_yellow"
    Green = "b_green"
    Black = "b_black"
    Magic = "b_magic"
    Nebula = "b_nebula"
    Ghost = "b_ghost"
    Abandoned = "b_abandoned"
    Checkered = "b_checkered"
    Zodiac = "b_zodiac"
    Painted = "b_painted"
    Anaglyph = "b_anaglyph"
    Plasma = "b_plasma"
    Erratic = "b_erratic"

class Stake(Enum):
    """Stake levels in Balatro"""
    White = 1
    Red = 2
    Green = 3
    Black = 4
    Blue = 5
    Purple = 6
    Orange = 7
    Gold = 8

@dataclass
class Seed:
    """Game seed for reproducible runs"""
    value: str

class Menu:
    """Main menu screen handler"""
    def __init__(self, connection):
        self.connection = connection
    
    async def new_run(self, deck: Deck, stake: Stake, seed: Optional[Seed] = None):
        """Start a new run with specified parameters"""
        from .blinds import SelectBlind
        
        request_body = {
            "back": deck.value,
            "stake": stake.value,
            "seed": seed.value if seed else None
        }
        
        response = await self.connection.send_request("main_menu/start_run", request_body)
        
        if "Ok" in response["body"]:
            blind_info = response["body"]["Ok"]
            return SelectBlind(blind_info, self.connection)
        else:
            raise Exception(f"Failed to start run: {response['body']['Err']}")

