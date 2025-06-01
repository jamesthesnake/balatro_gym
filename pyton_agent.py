"""
Balatro LLM Agent with RL-Ready Architecture
Supports DeepSeek Coder and other LLMs, with hooks for PPO/RL training
"""

import asyncio
import json
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

import aiohttp
import websockets

# For RL integration later
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not installed. RL features will be disabled.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ===== Game State Representations =====

class Rank(Enum):
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12

class Suit(Enum):
    SPADES = 0
    CLUBS = 1
    HEARTS = 2
    DIAMONDS = 3

@dataclass
class Card:
    rank: Rank
    suit: Suit
    
    def chip_value(self) -> int:
        chip_values = {
            Rank.TWO: 2, Rank.THREE: 3, Rank.FOUR: 4, Rank.FIVE: 5,
            Rank.SIX: 6, Rank.SEVEN: 7, Rank.EIGHT: 8, Rank.NINE: 9,
            Rank.TEN: 10, Rank.JACK: 10, Rank.QUEEN: 10, Rank.KING: 10,
            Rank.ACE: 11
        }
        return chip_values[self.rank]
    
    def to_vector(self) -> np.ndarray:
        """Convert card to one-hot vector for RL models"""
        vec = np.zeros(52)
        vec[self.rank.value + self.suit.value * 13] = 1
        return vec

@dataclass
class GameState:
    """Complete game state for decision making"""
    hand: List[Card]
    hands_left: int
    discards_left: int
    current_score: float
    blind_target: float
    money: int
    ante: int
    phase: str  # "discard" or "play"
    
    def to_observation(self) -> np.ndarray:
        """Convert to observation vector for RL models"""
        # Hand encoding (8 cards * 52 one-hot = 416)
        hand_vec = np.zeros(8 * 52)
        for i, card in enumerate(self.hand[:8]):
            hand_vec[i*52:(i+1)*52] = card.to_vector()
        
        # Game state features
        state_vec = np.array([
            self.hands_left / 4.0,  # Normalize
            self.discards_left / 4.0,
            self.current_score / 1000.0,  # Normalize by typical score
            self.blind_target / 1000.0,
            self.money / 100.0,  # Normalize
            self.ante / 10.0,
            1.0 if self.phase == "discard" else 0.0
        ])
        
        return np.concatenate([hand_vec, state_vec])


# ===== Hand Evaluation =====

class HandEvaluator:
    """Poker hand evaluation logic"""
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[int, str]:
        """Evaluate a 5-card poker hand"""
        if len(cards) != 5:
            return 0, "Invalid"
        
        # Check flush
        flush = all(c.suit == cards[0].suit for c in cards)
        
        # Check straight
        ranks = sorted([c.rank.value for c in cards])
        straight = all(ranks[i] + 1 == ranks[i + 1] for i in range(4))
        # Special case: A-2-3-4-5
        if ranks == [0, 1, 2, 3, 12]:
            straight = True
        
        # Count ranks
        rank_counts = {}
        for card in cards:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1
        
        counts = sorted(rank_counts.values(), reverse=True)
        
        # Determine hand type
        if flush and counts[0] == 5:
            return 160 * 16, "Flush Five"
        elif flush and counts == [3, 2]:
            return 140 * 14, "Flush House"
        elif counts[0] == 5:
            return 120 * 12, "Five of a Kind"
        elif straight and flush:
            return 100 * 8, "Straight Flush"
        elif counts[0] == 4:
            return 60 * 7, "Four of a Kind"
        elif counts == [3, 2]:
            return 40 * 4, "Full House"
        elif flush:
            return 35 * 4, "Flush"
        elif straight:
            return 30 * 4, "Straight"
        elif counts[0] == 3:
            return 30 * 3, "Three of a Kind"
        elif counts == [2, 2, 1]:
            return 20 * 2, "Two Pair"
        elif counts[0] == 2:
            return 10 * 2, "Pair"
        else:
            return 5 * 1, "High Card"
    
    @staticmethod
    def find_all_combinations(hand: List[Card]) -> List[Tuple[List[int], int, str]]:
        """Find all possible 5-card combinations from hand"""
        from itertools import combinations
        
        results = []
        if len(hand) >= 5:
            for combo_indices in combinations(range(len(hand)), 5):
                cards = [hand[i] for i in combo_indices]
                score, hand_type = HandEvaluator.evaluate_hand(cards)
                
                # Add chip values from scoring cards
                for card in cards:
                    score += card.chip_value()
                
                results.append((list(combo_indices), score, hand_type))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# ===== Action Space =====

@dataclass
class Action:
    """Base action class"""
    pass

@dataclass
class DiscardAction(Action):
    indices: List[int]

@dataclass
class PlayAction(Action):
    indices: List[int]

@dataclass
class SkipAction(Action):
    reason: str


# ===== LLM Agent Interface =====

class LLMAgent(ABC):
    """Abstract base class for LLM agents"""
    
    @abstractmethod
    async def decide_action(self, state: GameState) -> Action:
        pass


class DeepSeekAgent(LLMAgent):
    """DeepSeek Coder agent implementation"""
    
    def __init__(self, api_key: str, model: str = "deepseek-coder"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
    
    async def decide_action(self, state: GameState) -> Action:
        """Get action decision from DeepSeek"""
        
        # Create function definitions for DeepSeek
        functions = [
            {
                "name": "analyze_best_hand",
                "description": "Analyze all possible 5-card combinations and return the best",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "name": "discard_cards",
                "description": "Select cards to discard",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "strategy": {
                            "type": "string",
                            "enum": ["low_cards", "keep_pairs", "keep_flush", "keep_straight"],
                            "description": "Discard strategy"
                        },
                        "threshold_rank": {
                            "type": "integer",
                            "description": "For low_cards strategy, discard below this rank (0-12)"
                        }
                    },
                    "required": ["strategy"]
                }
            },
            {
                "name": "play_hand",
                "description": "Play the best 5-card combination",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        ]
        
        # Create context
        hand_info = []
        for i, card in enumerate(state.hand):
            hand_info.append({
                "index": i,
                "rank": card.rank.name,
                "suit": card.suit.name,
                "value": card.rank.value,
                "chips": card.chip_value()
            })
        
        system_prompt = """You are an expert Balatro player. Analyze the game state and choose the best action.
        
Key strategies:
- During discard phase: Remove low cards (< 10) to draw better ones
- Keep pairs, flushes, and straight possibilities
- Consider the blind target vs current score
- With limited hands, play conservatively
- High cards (10,J,Q,K,A) are valuable

Always use function calling to make decisions."""

        user_prompt = f"""Current game state:
Hand: {json.dumps(hand_info, indent=2)}
Hands left: {state.hands_left}
Discards left: {state.discards_left}
Current score: {state.current_score}
Blind target: {state.blind_target}
Money: ${state.money}
Phase: {state.phase}

What should I do?"""

        try:
            response = await self.session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "functions": functions,
                    "function_call": "auto",
                    "temperature": 0.2
                }
            )
            
            result = await response.json()
            
            # Parse function call
            if "function_call" in result["choices"][0]["message"]:
                func_call = result["choices"][0]["message"]["function_call"]
                func_name = func_call["name"]
                func_args = json.loads(func_call["arguments"])
                
                if func_name == "discard_cards":
                    indices = self._select_discard_indices(state, func_args)
                    return DiscardAction(indices=indices)
                elif func_name == "play_hand":
                    combos = HandEvaluator.find_all_combinations(state.hand)
                    if combos:
                        return PlayAction(indices=list(combos[0][0]))
                elif func_name == "analyze_best_hand":
                    combos = HandEvaluator.find_all_combinations(state.hand)
                    if combos and state.phase == "play":
                        return PlayAction(indices=list(combos[0][0]))
            
            # Fallback
            return self._fallback_action(state)
            
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return self._fallback_action(state)
    
    def _select_discard_indices(self, state: GameState, args: dict) -> List[int]:
        """Select cards to discard based on strategy"""
        strategy = args.get("strategy", "low_cards")
        
        if strategy == "low_cards":
            threshold = args.get("threshold_rank", 8)
            indices = [i for i, card in enumerate(state.hand) 
                      if card.rank.value < threshold]
            return indices[:5]  # Max 5 discards
        
        elif strategy == "keep_pairs":
            # Count ranks
            rank_indices = {}
            for i, card in enumerate(state.hand):
                rank = card.rank
                if rank not in rank_indices:
                    rank_indices[rank] = []
                rank_indices[rank].append(i)
            
            # Keep paired cards and high cards
            keep_indices = set()
            for rank, indices in rank_indices.items():
                if len(indices) >= 2:
                    keep_indices.update(indices)
            
            # Add high cards if needed
            high_cards = [(i, card) for i, card in enumerate(state.hand) 
                         if card.rank.value >= 8 and i not in keep_indices]
            high_cards.sort(key=lambda x: x[1].rank.value, reverse=True)
            
            for i, _ in high_cards:
                if len(keep_indices) >= 5:
                    break
                keep_indices.add(i)
            
            # Discard everything else
            return [i for i in range(len(state.hand)) if i not in keep_indices][:5]
        
        return []
    
    def _fallback_action(self, state: GameState) -> Action:
        """Simple fallback strategy"""
        if state.phase == "discard" and state.discards_left > 0:
            # Discard low cards
            indices = [i for i, card in enumerate(state.hand) 
                      if card.rank.value < 8]
            return DiscardAction(indices=indices[:5])
        else:
            # Play best hand
            combos = HandEvaluator.find_all_combinations(state.hand)
            if combos:
                return PlayAction(indices=list(combos[0][0]))
            return SkipAction(reason="No valid play")


# ===== RL Components (for future PPO integration) =====

if TORCH_AVAILABLE:
    class BalatroPolicy(nn.Module):
        """Neural network policy for RL agent"""
        
        def __init__(self, obs_dim: int = 423, hidden_dim: int = 256):
            super().__init__()
            self.fc1 = nn.Linear(obs_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            
            # Action heads
            self.action_type = nn.Linear(hidden_dim, 3)  # discard/play/skip
            self.card_selection = nn.Linear(hidden_dim, 8)  # which cards
            
            # Value head for PPO
            self.value = nn.Linear(hidden_dim, 1)
        
        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            x = F.relu(self.fc1(obs))
            x = F.relu(self.fc2(x))
            
            action_logits = self.action_type(x)
            card_logits = self.card_selection(x)
            value = self.value(x)
            
            return action_logits, card_logits, value
    
    class RLAgent(LLMAgent):
        """RL-based agent using trained policy"""
        
        def __init__(self, policy: BalatroPolicy):
            self.policy = policy
            self.policy.eval()
        
        async def decide_action(self, state: GameState) -> Action:
            obs = torch.FloatTensor(state.to_observation()).unsqueeze(0)
            
            with torch.no_grad():
                action_logits, card_logits, _ = self.policy(obs)
                
                # Sample action type
                action_probs = F.softmax(action_logits, dim=1)
                action_type = Categorical(action_probs).sample().item()
                
                # Sample cards
                card_probs = torch.sigmoid(card_logits)
                card_mask = (torch.rand_like(card_probs) < card_probs).squeeze()
                selected_cards = torch.where(card_mask)[0].tolist()
            
            if action_type == 0 and state.discards_left > 0:
                return DiscardAction(indices=selected_cards[:5])
            elif action_type == 1:
                # Find best valid 5-card combo including selected cards
                combos = HandEvaluator.find_all_combinations(state.hand)
                if combos:
                    return PlayAction(indices=list(combos[0][0]))
            
            return SkipAction(reason="RL policy skip")


# ===== Game Client =====

class BalatroClient:
    """WebSocket client for Balatro remotro server"""
    
    def __init__(self, host: str = "localhost", port: int = 34143):
        self.host = host
        self.port = port
        self.ws = None
    
    async def connect(self):
        self.ws = await websockets.connect(f"ws://{self.host}:{self.port}")
    
    async def close(self):
        if self.ws:
            await self.ws.close()
    
    async def send_request(self, kind: str, body: dict = None) -> dict:
        """Send request and wait for response"""
        request = {"kind": kind}
        if body:
            request["body"] = body
        
        await self.ws.send(json.dumps(request))
        
        # Wait for response
        while True:
            response = json.loads(await self.ws.recv())
            if response.get("kind", "").startswith("result/"):
                return response
    
    async def get_screen(self) -> dict:
        return await self.send_request("screen/get")
    
    async def start_run(self, deck: str = "Red", stake: str = "White"):
        return await self.send_request("main_menu/start_run", {
            "deck": deck,
            "stake": stake,
            "seed": None
        })
    
    async def select_blind(self):
        return await self.send_request("blind_select/select")
    
    async def click_cards(self, indices: List[int]):
        return await self.send_request("play/click", {"indices": indices})
    
    async def play_hand(self):
        return await self.send_request("play/play")
    
    async def discard_cards(self):
        return await self.send_request("play/discard")
    
    async def leave_shop(self):
        return await self.send_request("shop/continue")


# ===== Main Game Loop =====

async def run_agent(agent: LLMAgent, episodes: int = 100):
    """Run the agent for multiple episodes"""
    
    total_rewards = []
    
    for episode in range(episodes):
        client = BalatroClient()
        episode_reward = 0
        
        try:
            await client.connect()
            logger.info(f"Episode {episode + 1}: Connected to Balatro")
            
            while True:
                # Get current screen
                screen_resp = await client.get_screen()
                screen_data = screen_resp["body"]["Ok"]
                
                if "Menu" in screen_data:
                    # Start new run
                    await client.start_run()
                    logger.info("Started new run")
                
                elif "SelectBlind" in screen_data:
                    # Always select blind
                    await client.select_blind()
                    logger.info("Selected blind")
                
                elif "Play" in screen_data:
                    # Parse game state
                    play_data = screen_data["Play"]
                    
                    # Convert to our GameState
                    hand = []
                    for card_data in play_data["hand"]:
                        if card_data["card"]:
                            rank = Rank[card_data["card"]["rank"].upper()]
                            suit = Suit[card_data["card"]["suit"].upper()]
                            hand.append(Card(rank, suit))
                    
                    state = GameState(
                        hand=hand,
                        hands_left=play_data["hands"],
                        discards_left=play_data["discards"],
                        current_score=play_data["score"],
                        blind_target=450.0,  # Would parse from blind info
                        money=play_data["money"],
                        ante=1,
                        phase="discard" if play_data["discards"] > 0 else "play"
                    )
                    
                    # Get agent decision
                    action = await agent.decide_action(state)
                    logger.info(f"Agent action: {type(action).__name__}")
                    
                    if isinstance(action, DiscardAction) and state.discards_left > 0:
                        await client.click_cards(action.indices)
                        result = await client.discard_cards()
                        
                        if "GameOver" in result["body"]["Ok"]:
                            logger.info("Game over!")
                            break
                    
                    elif isinstance(action, PlayAction):
                        await client.click_cards(action.indices)
                        result = await client.play_hand()
                        
                        if "RoundOver" in result["body"]["Ok"]:
                            earnings = result["body"]["Ok"]["RoundOver"]["total_earned"]
                            episode_reward += earnings
                            logger.info(f"Round complete! Earned: ${earnings}")
                        elif "GameOver" in result["body"]["Ok"]:
                            logger.info("Game over!")
                            break
                
                elif "Shop" in screen_data:
                    # Skip shop for now
                    await client.leave_shop()
                    logger.info("Left shop")
                
                await asyncio.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Episode {episode + 1} error: {e}")
        
        finally:
            await client.close()
            total_rewards.append(episode_reward)
            logger.info(f"Episode {episode + 1} total reward: ${episode_reward}")
    
    return total_rewards


# ===== Entry Point =====

async def main():
    """Main entry point"""
    
    # Choose your agent
    agent_type = "deepseek"  # or "rl" if you have a trained model
    
    if agent_type == "deepseek":
        api_key = "your-deepseek-api-key"  # Set your API key
        async with DeepSeekAgent(api_key) as agent:
            rewards = await run_agent(agent, episodes=10)
            print(f"Average reward: ${np.mean(rewards):.2f}")
    
    elif agent_type == "rl" and TORCH_AVAILABLE:
        # Load trained model
        policy = BalatroPolicy()
        # policy.load_state_dict(torch.load("balatro_policy.pth"))
        agent = RLAgent(policy)
        rewards = await run_agent(agent, episodes=100)
        print(f"Average reward: ${np.mean(rewards):.2f}")


if __name__ == "__main__":
    asyncio.run(main())
