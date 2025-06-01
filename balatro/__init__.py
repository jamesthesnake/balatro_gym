from .menu import Menu, Deck, Stake, Seed
from .deck import Card, Rank, Suit
from .play import Play, HandCard, PlayResult, DiscardResult
from .blinds import SelectBlind, CurrentBlind
from .shop import Shop
from .overview import RoundOverview, GameOverview
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from .net.connection import Connection
__all__ = [
    'Menu', 'Deck', 'Stake', 'Seed',
    'Card', 'Rank', 'Suit',
    'Play', 'HandCard', 'PlayResult', 'DiscardResult',
    'SelectBlind', 'CurrentBlind',
    'Shop',
    'RoundOverview', 'GameOverview'
]
class ScreenType(Enum):
    """Screen types in Balatro"""
    MENU = "Menu"
    SELECT_BLIND = "SelectBlind"
    PLAY = "Play"
    SHOP = "Shop"

@dataclass
class Screen:
    """Current screen with type and content"""
    type: ScreenType
    menu: Optional[Menu] = None
    select_blind: Optional[SelectBlind] = None
    play: Optional[Play] = None
    shop: Optional[Shop] = None

class Balatro:
    """Main Balatro game interface matching Rust structure"""
    
    def __init__(self, connection: Connection):
        self.connection = connection
    
    async def screen(self) -> Screen:
        """Get current game screen"""
        response = await self.connection.send_request("screen/get")
        
        if "Err" in response.get("body", {}):
            raise Exception(f"Failed to get screen: {response['body']['Err']}")
        
        screen_info = response["body"]["Ok"]
        
        # Parse screen type and create appropriate screen object
        if "Menu" in screen_info:
            return Screen(
                type=ScreenType.MENU,
                menu=Menu(self.connection)
            )
        elif "SelectBlind" in screen_info:
            return Screen(
                type=ScreenType.SELECT_BLIND,
                select_blind=SelectBlind(screen_info["SelectBlind"], self.connection)
            )
        elif "Play" in screen_info:
            return Screen(
                type=ScreenType.PLAY,
                play=Play(screen_info["Play"], self.connection)
            )
        elif "Shop" in screen_info:
            return Screen(
                type=ScreenType.SHOP,
                shop=Shop(screen_info["Shop"], self.connection)
            )
        else:
            raise Exception(f"Unknown screen type: {screen_info}")

