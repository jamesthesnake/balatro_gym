#!/usr/bin/env python3
"""
Smart Analyzer with RunPod API Integration
Handles game navigation locally, queries RunPod API for card decisions
NOW USING THE LLM-POWERED ENDPOINT!
"""

import asyncio
import logging
import sys
import httpx
import uuid
from pathlib import Path
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Add the parent directory to path to import the balatro library
sys.path.insert(0, str(Path(__file__).parent.parent / "lib" / "src"))

# Import from balatro library
from balatro import Balatro, Screen, ScreenType
from balatro.menu import Menu, Deck, Stake
from balatro.play import Play, PlayResult, DiscardResult, HandCard
from balatro.deck import Card, Rank, Suit
from balatro.blinds import SelectBlind
from balatro.shop import Shop
from balatro.overview import RoundOverview, GameOverview
from balatro.net import Connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('smart_analyzer_api')

# ===== Configuration =====
HOST = "127.0.0.1"
PORT = 34143
RUNPOD_API_URL = "https://kq6aplijt4qgmz-8000.proxy.runpod.net"  # Your RunPod URL

# ===== API Client =====
class BalatroAPIClient:
    """Client for communicating with RunPod Balatro API"""
    
    def __init__(self, api_url: str):
        self.api_url = api_url
        self.session_id = None
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def get_decision(self, hand: List[HandCard], game_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision from RunPod API using the LLM-powered endpoint"""
        # Convert ALL cards in hand to API format
        hand_strings = []
        for hc in hand:
            if hc.card:
                # Convert rank and suit to string format
                rank_str = hc.card.rank.value
                suit_map = {
                    Suit.Clubs: "C",
                    Suit.Diamonds: "D", 
                    Suit.Hearts: "H",
                    Suit.Spades: "S"
                }
                suit_str = suit_map.get(hc.card.suit, "?")
                hand_strings.append(f"{rank_str}{suit_str}")
        
        # Create session ID if needed
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        
        # Build request with ALL game information
        request_data = {
            "game_state": {
                "hand": hand_strings,  # ALL cards, not limited to 8
                "points_needed": game_state.get("points_needed", 300),
                "hands_remaining": game_state.get("hands_left", 1),
                "current_score": game_state.get("current_score", 0),
                "round": game_state.get("round", 1),
                "discards_remaining": game_state.get("discards_left", 0)
            },
            "session_id": self.session_id,
            "timeout": 45.0
        }
        
        logger.info(f"🤖 Sending to LLM API - Hand: {len(hand_strings)} cards")
        logger.info(f"Game state: Score {request_data['game_state']['current_score']}, "
                   f"Need {request_data['game_state']['points_needed']} more points, "
                   f"{request_data['game_state']['hands_remaining']} hands left, "
                   f"{request_data['game_state']['discards_remaining']} discards left")
        
        try:
            # USE THE ONE-WORD ENDPOINT!
            response = await self.client.post(
                f"{self.api_url}/decide_one_word",  # Changed to one-word endpoint
                json=request_data
            )
            response.raise_for_status()
            
            decision = response.json()
            
            # Log the decision with fancy formatting
            logger.info("━" * 60)
            logger.info(f"🎯 LLM DECISION: {decision['action'].upper()}")
            logger.info("━" * 60)
            
            # Log thoughts if available
            if decision.get('thoughts'):
                logger.info("💭 LLM Thinking:")
                for thought in decision['thoughts']:
                    logger.info(f"   • {thought}")
            
            # Log display message if available
            if decision.get('display_message'):
                logger.info(f"📺 Display: {decision['display_message']}")
            
            logger.info(f"🎲 Confidence: {decision.get('confidence', 'N/A')}")
            logger.info("━" * 60)
            
            return decision
            
        except httpx.HTTPStatusError as e:
            logger.error(f"API error {e.response.status_code}: {e.response.text}")
            # Return fallback decision
            return {
                "action": "keep",
                "data": {"indices": [0, 1, 2, 3, 4]},
                "thoughts": ["API error, using fallback"],
                "confidence": 0.1
            }
        except Exception as e:
            logger.error(f"API request failed: {e}")
            # Return fallback decision
            return {
                "action": "keep", 
                "data": {"indices": [0, 1, 2, 3, 4]},
                "thoughts": ["API unreachable, using fallback"],
                "confidence": 0.1
            }
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()

# ===== Main Smart Analyzer =====
class SmartAnalyzerAPI:
    """Smart Analyzer that uses RunPod API for decisions"""
    
    def __init__(self):
        self.logger = logger
        self.api_client = BalatroAPIClient(RUNPOD_API_URL)
        self.current_round = 1
    
    async def run(self, host: str = HOST, port: int = PORT):
        """Run the smart analyzer server"""
        server = await asyncio.start_server(
            self.handle_connection, host, port
        )
        
        addr = server.sockets[0].getsockname()
        self.logger.info(f'🚀 Smart Analyzer API hosted on {addr[0]}:{addr[1]} - waiting for Balatro...')
        self.logger.info(f'🤖 Using ONE-WORD LLM API at: {RUNPOD_API_URL}/decide_one_word')
        self.logger.info(f'⚡ Ultra-fast decisions with no examples!')
        
        async with server:
            await server.serve_forever()
    
    async def handle_connection(self, reader, writer):
        """Handle a single Balatro connection"""
        connection = Connection(reader, writer)
        addr = writer.get_extra_info('peername')
        self.logger.info(f"✓ Connected to Balatro from {addr}")
        
        try:
            # Create Balatro instance with this connection
            balatro = Balatro(connection)
            
            # Run the game loop
            await self.game_loop(balatro)
            
        except Exception as e:
            self.logger.error(f"Game error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            writer.close()
            await writer.wait_closed()
            self.logger.info("Connection lost, waiting for new connection...")
    
    async def game_loop(self, balatro: Balatro):
        """Main game loop using Balatro instance"""
        self.logger.info("Starting game loop")
        while True:
            try:
                # Get current screen
                screen = await balatro.screen()
                self.logger.info(f"Got screen type: {screen.type}")
                
                # Handle based on screen type
                if screen.type == ScreenType.MENU:
                    await self.handle_menu(screen.menu)
                
                elif screen.type == ScreenType.SELECT_BLIND:
                    await self.handle_select_blind(screen.select_blind)
                
                elif screen.type == ScreenType.PLAY:
                    await self.handle_play(screen.play)
                
                elif screen.type == ScreenType.SHOP:
                    await self.handle_shop(screen.shop)
                
                else:
                    self.logger.warning(f"Unknown screen type: {screen.type}")
                    await asyncio.sleep(2)
                
                # Small delay between actions
                await asyncio.sleep(0.1)
                
            except ConnectionError:
                self.logger.info("Connection closed by Balatro")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error in game loop: {e}")
                import traceback
                traceback.print_exc()
                break
    
    async def handle_menu(self, menu: Menu):
        """Handle main menu - same as original"""
        self.logger.info("▶ At Main Menu ◀")
        self.logger.info("🎮 Starting new run automatically...")
        
        # Reset round counter
        self.current_round = 1
        
        # Auto-start with Red deck and White stake
        select_blind = await menu.new_run(Deck.Red, Stake.White)
        self.logger.info("Started new run with Red deck and White stake")
    
    async def handle_select_blind(self, select_blind: SelectBlind):
        """Handle blind selection - same as original"""
        self.logger.info("▶ At Select Blind screen ◀")
        self.logger.info(f"Small blind: {select_blind.small()}")
        self.logger.info(f"Big blind: {select_blind.big()}")
        self.logger.info(f"Boss blind: {select_blind.boss()}")
        
        await asyncio.sleep(0.5)
        
        try:
            play = await select_blind.select()
            self.logger.info("Selected blind")
        except Exception as e:
            self.logger.warning(f"Blind selection failed: {e}, retrying...")
            await asyncio.sleep(1.0)
    
    async def handle_play(self, play: Play):
        """Handle play screen - queries API for decisions"""
        self.logger.info("\n▶ At Play screen - Querying LLM-powered RunPod API ◀")
        
        # Get ALL game state information
        hand = play.hand()
        hands_left = play.hands()
        discards_left = play.discards()
        current_score = play.score()
        money = play.money()
        blind_info = play.blind()
        blind_target = self.get_blind_target(blind_info)
        
        # Calculate points needed to win
        points_needed = blind_target - current_score
        
        # Display current state
        self.logger.info(f"Current hand size: {len(hand)}")
        self.logger.info(f"Hands left: {hands_left}")
        self.logger.info(f"Discards left: {discards_left}")
        self.logger.info(f"Current score: {current_score}/{blind_target}")
        self.logger.info(f"Points needed: {points_needed}")
        self.logger.info(f"Money: ${money}")
        
        # Print current hand (ALL cards)
        self.logger.info("Current hand:")
        for i, hc in enumerate(hand):
            if hc.card:
                self.logger.info(f"  [{i}] {hc.card.rank.value} of {hc.card.suit.value}")
        
        # Build complete game state for API
        game_state = {
            "hands_left": hands_left,
            "discards_left": discards_left,
            "current_score": current_score,
            "blind_target": blind_target,
            "points_needed": points_needed,
            "money": money,
            "round": self.current_round
        }
        
        # Get decision from API - let it decide whether to discard or play
        self.logger.info("🤖 Consulting LLM for strategic decision...")
        decision = await self.api_client.get_decision(hand, game_state)
        
        # API decides everything - just execute what it says
        if decision["action"] == "discard":
            # API wants to discard
            if discards_left > 0:
                await self.execute_discard(play, decision["data"]["indices"])
            else:
                self.logger.warning("API wanted to discard but no discards left! Playing instead.")
                # Fallback to playing the indices it would have kept
                play_indices = decision["data"].get("indices", [0, 1, 2, 3, 4])
                await self.execute_play(play, play_indices)
        elif decision["action"] == "keep":
            # API wants to play these specific cards
            indices = decision["data"].get("indices", [0, 1, 2, 3, 4])
            await self.execute_play(play, indices)
        else:
            self.logger.error(f"Unknown action from API: {decision['action']}")
            # Fallback
            await self.execute_play(play, [0, 1, 2, 3, 4])
    
    def get_blind_target(self, blind_info: dict) -> float:
        """Extract blind target score from blind info"""
        if "Small" in blind_info:
            return blind_info["Small"]["chips"]
        elif "Big" in blind_info:
            return blind_info["Big"]["chips"]
        elif "Boss" in blind_info:
            return blind_info["Boss"]["chips"]
        return 300  # Default fallback
    
    async def execute_discard(self, play: Play, discard_indices: List[int]):
        """Execute discard action"""
        self.logger.info(f"📤 Executing discard: {discard_indices}")
        
        # Validate indices
        hand = play.hand()
        valid_indices = [i for i in discard_indices if 0 <= i < len(hand)]
        
        if not valid_indices:
            self.logger.warning("No valid discard indices, playing instead")
            await self.execute_play(play, [0, 1, 2, 3, 4])
            return
        
        # Log cards being discarded
        for idx in valid_indices:
            if hand[idx].card:
                card = hand[idx].card
                self.logger.info(f"  Discarding [{idx}]: {card.rank.value} of {card.suit.value}")
        
        # Click cards to discard
        play = await play.click(valid_indices)
        
        # Execute discard
        result = await play.discard()
        
        if result.type == DiscardResult.AGAIN:
            self.logger.info("✅ Discard successful!")
        elif result.type == DiscardResult.GAME_OVER:
            self.logger.info("💀 Game Over during discard!")
    
    async def execute_play(self, play: Play, play_indices: List[int]):
        """Execute play action"""
        self.logger.info(f"🎯 Executing play: {play_indices}")
        
        # Validate and adjust indices
        hand = play.hand()
        valid_indices = []
        
        for idx in play_indices:
            if 0 <= idx < len(hand) and len(valid_indices) < 5:
                valid_indices.append(idx)
        
        # Ensure we have 5 cards
        while len(valid_indices) < 5 and len(valid_indices) < len(hand):
            for i in range(len(hand)):
                if i not in valid_indices:
                    valid_indices.append(i)
                    if len(valid_indices) >= 5:
                        break
        
        # Log cards being played
        self.logger.info("Playing cards:")
        for idx in valid_indices:
            if idx < len(hand) and hand[idx].card:
                card = hand[idx].card
                self.logger.info(f"  [{idx}] {card.rank.value} of {card.suit.value}")
        
        # Select cards
        play = await play.click(valid_indices)
        
        # Play the hand
        result = await play.play()
        
        if result.type == PlayResult.AGAIN:
            self.logger.info("🔄 Must play again")
        elif result.type == PlayResult.ROUND_OVER:
            await self.handle_round_over(result.overview)
        elif result.type == PlayResult.GAME_OVER:
            self.logger.info("💀 Game Over!")
            # Create a new session for next game
            self.api_client.session_id = None
    
    async def handle_round_over(self, overview: RoundOverview):
        """Handle round over screen"""
        self.logger.info("\n🎉 Round complete!")
        self.logger.info(f"Total earned: ${overview.total_earned()}")
        self.logger.info(f"Earnings breakdown: {overview.earnings()}")
        
        # Increment round counter
        self.current_round += 1
        
        await asyncio.sleep(2)
        shop = await overview.cash_out()
        self.logger.info("Cashed out to shop")
    
    async def handle_shop(self, shop: Shop):
        """Handle shop screen - same as original"""
        self.logger.info("\n▶ At Shop screen ◀")
        self.logger.info(f"Shop items: {shop.main_cards()}")
        self.logger.info(f"Vouchers: {shop.vouchers()}")
        self.logger.info(f"Boosters: {shop.boosters()}")
        
        # For now, just leave without buying
        await asyncio.sleep(1)
        select_blind = await shop.leave()
        self.logger.info("Left shop")
    
    async def cleanup(self):
        """Cleanup resources"""
        await self.api_client.close()

# ===== Entry Point =====
def main():
    """Entry point"""
    logger.info("🤖 Smart Analyzer with ONE-WORD LLM Integration")
    logger.info(f"📡 API URL: {RUNPOD_API_URL}/decide_one_word")
    logger.info("🎮 Game navigation: Local")
    logger.info("⚡ Card decisions: Ultra-fast one-word LLM responses")
    logger.info("🎯 No examples, just pure speed!")
    
    analyzer = SmartAnalyzerAPI()
    
    try:
        asyncio.run(analyzer.run())
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    finally:
        # Cleanup
        asyncio.run(analyzer.cleanup())

if __name__ == "__main__":
    main()
