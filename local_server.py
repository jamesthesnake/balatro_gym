"""
Complete Balatro API server with example-based learning system.
Filename: local_api_server_for_live_game.py
"""
from __future__ import annotations
import json
import torch
import re
import httpx
import asyncio
import logging
import random
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datetime import datetime
from collections import Counter
from itertools import combinations

# ── Configure Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/workspace/balatro_api.log'),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# ── Configuration ------------------------------------------------------
GAME_SERVER_URL = "http://localhost:3000"
GAME_SERVER_TIMEOUT = 30.0

# ── ASCII Art Collection ───────────────────────────────────────────────
DISCARD_ASCII_ART = [
    """
    ╔═══════════════════════╗
    ║  DISCARDING CARDS...  ║
    ╚═══════════════════════╝
         ┌─┐ ┌─┐ ┌─┐
         │?│ │?│ │?│  ➜ 🗑️
         └─┘ └─┘ └─┘
    """,
    """
    ┏━━━━━━━━━━━━━━━━━━━━━━┓
    ┃   TOSSING CARDS! 🎲  ┃
    ┗━━━━━━━━━━━━━━━━━━━━━━┛
      ╱│ ╱│ ╱│
     ╱ │╱ │╱ │  ↘
    │  │  │  │   🗑️
    │  │  │  │  ╱
    └──┴──┴──┘
    """,
    """
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓ STRATEGIC DISCARD! ▓
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
       ♠ ♥ ♦ ♣
      ┌─┐┌─┐┌─┐
      │ ││ ││ │ ⟹ 🚮
      └─┘└─┘└─┘
    """,
    """
    ╭─────────────────────╮
    │  REROLL TIME! 🎰   │
    ╰─────────────────────╯
      [♠][♥][♦]
         ↓ ↓ ↓
      💨 💨 💨
      🗑️ 🗑️ 🗑️
    """
]

def get_random_discard_art():
    """Get a random discard ASCII art."""
    return random.choice(DISCARD_ASCII_ART)

# ── Example Database for Teaching Tool Usage ──────────────────────────
BALATRO_EXAMPLES = [
    {
        "scenario": "Obvious winning hand",
        "hand": ["AS", "KS", "QS", "JS", "10S", "9H", "8D", "7C"],
        "game_state": {"points_needed": 400, "hands_remaining": 2},
        
        "bad_approach": {
            "thought": "This looks like a good hand with high cards.",
            "action": "keep",
            "indices": [0, 1, 2, 3, 4],
            "result": "Won by luck, but didn't maximize score"
        },
        
        "good_approach": {
            "tools_used": [
                ("check_cards", "['AS', 'KS', 'QS', 'JS', '10S', '9H', '8D', '7C']"),
                ("get_optimal_pairs", "[{'cards': ['AS','KS','QS','JS','10S'], 'type': 'straight_flush', 'score': 800}]"),
                ("check_win_threshold", "{'required_per_hand': 200, 'current_best': 800, 'meets_threshold': true}")
            ],
            "thought": "After analysis, I have a straight flush worth 800 points, which far exceeds the 200 per hand needed.",
            "action": "keep",
            "indices": [0, 1, 2, 3, 4],
            "result": "Won with maximum score!"
        }
    },
    
    {
        "scenario": "Weak hand needing discard",
        "hand": ["2C", "5H", "7D", "9S", "JH", "KC", "AD", "3S"],
        "game_state": {"points_needed": 1000, "hands_remaining": 3, "discards_remaining": 3},
        
        "bad_approach": {
            "thought": "I have an ace and king, that's pretty good.",
            "action": "keep",
            "indices": [6, 5, 4, 3, 2],
            "result": "Lost - high card only worth 50 points, needed 333"
        },
        
        "good_approach": {
            "tools_used": [
                ("check_cards", "['2C', '5H', '7D', '9S', 'JH', 'KC', 'AD', '3S']"),
                ("get_optimal_pairs", "[{'cards': ['AD','KC','JH','9S','7D'], 'type': 'high_card', 'score': 50}]"),
                ("check_win_threshold", "{'required_per_hand': 333, 'current_best': 50, 'meets_threshold': false, 'deficit': 283}")
            ],
            "thought": "Analysis shows I only have high card worth 50 but need 333. Must discard to try for pairs/better.",
            "action": "discard",
            "indices": [0, 1, 7],
            "result": "After discard, got a pair of aces worth 150, then discarded again for three aces worth 300!"
        }
    },
    
    {
        "scenario": "Borderline decision",
        "hand": ["KH", "KD", "QS", "QC", "JH", "10D", "9S", "8C"],
        "game_state": {"points_needed": 400, "hands_remaining": 2, "discards_remaining": 1},
        
        "bad_approach": {
            "thought": "Two pair is always good, I'll keep it.",
            "action": "keep",
            "indices": [0, 1, 2, 3, 4],
            "result": "Won this hand with 200, but lost overall - needed 200 per hand average"
        },
        
        "good_approach": {
            "tools_used": [
                ("check_cards", "['KH', 'KD', 'QS', 'QC', 'JH', '10D', '9S', '8C']"),
                ("get_optimal_pairs", "[{'cards': ['KH','KD','QS','QC','JH'], 'type': 'two_pair', 'score': 200}]"),
                ("check_win_threshold", "{'required_per_hand': 200, 'current_best': 200, 'meets_threshold': true, 'surplus': 0}")
            ],
            "thought": "Two pair worth exactly 200, which just meets the threshold. With only 1 discard left, keeping is safer.",
            "action": "keep",
            "indices": [0, 1, 2, 3, 4],
            "result": "Smart decision - met exact threshold without risking a bad redraw"
        }
    }
]

# ── API Models ---------------------------------------------------------
class GameAction(BaseModel):
    action: str
    data: Optional[Dict[str, Any]] = None

class GameState(BaseModel):
    hand: List[str]
    points_needed: int
    hands_remaining: int
    current_score: int
    round: int
    discards_remaining: int

class AgentDecision(BaseModel):
    game_state: GameState
    session_id: str
    timeout: Optional[float] = 30.0

class AgentThought(BaseModel):
    thought: str
    timestamp: datetime

class AgentAction(BaseModel):
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    timestamp: datetime

class DecisionResponse(BaseModel):
    session_id: str
    action: str
    data: Dict[str, Any]
    thoughts: List[str]
    confidence: float
    display_message: Optional[str] = None  # New field for game display

class SessionState(BaseModel):
    session_id: str
    game_state: GameState
    history: List[tuple[str, str]]
    thoughts: List[AgentThought]
    actions: List[AgentAction]
    created_at: datetime
    last_updated: datetime

# ── Tool Schema --------------------------------------------------------
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]

TOOLS: List[ToolSchema] = [
    ToolSchema(
        name="analyze_hand",
        description="Analyze the current hand and get optimal combinations.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="check_win_threshold",
        description="Check if current hand meets the points threshold needed.",
        parameters={"type": "object", "properties": {}, "required": []},
    ),
    ToolSchema(
        name="decide_discard",
        description="Decide which cards to discard. Returns indices to discard.",
        parameters={
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for discarding"}
            },
            "required": ["reason"],
        },
    ),
    ToolSchema(
        name="decide_keep",
        description="Decide to keep current hand. Returns the 5 card indices to play.",
        parameters={
            "type": "object",
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 0, "maximum": 7},
                    "minItems": 5,
                    "maxItems": 5,
                },
                "reason": {"type": "string", "description": "Reason for keeping"}
            },
            "required": ["indices", "reason"],
        },
    ),
]

TOOLS_JSON = json.dumps([t.model_dump() for t in TOOLS], ensure_ascii=False)

# ── Initialize FastAPI app ---------------------------------------------
app = FastAPI(title="Balatro Agent API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session Management -------------------------------------------------
sessions: Dict[str, SessionState] = {}

# ── Model Manager ------------------------------------------------------
class ModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
    
    def load_model(self):
        if not self.model_loaded:
            logger.info("Starting model load...")
            MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
            
            import os
            possible_paths = [
                "/workspace/deepseek-model",
                "/workspace/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "/workspace/DeepSeek-R1-Distill-Llama-8B",
            ]
            
            model_path = MODEL_NAME
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"Found local model at: {model_path}")
                    break
            
            logger.info(f"Loading tokenizer from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                padding_side="left",
                local_files_only=os.path.exists(model_path)
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Loading model with 4-bit quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
                local_files_only=os.path.exists(model_path)
            )
            self.model_loaded = True
            logger.info("Model loaded successfully!")
            
            if torch.cuda.is_available():
                logger.info(f"GPU: {torch.cuda.get_device_name()}")
                logger.info(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

model_manager = ModelManager()

# ── Example Learning Functions ─────────────────────────────────────────
def calculate_similarity(game_state: GameState, example_state: Dict) -> float:
    """Calculate similarity between current game state and an example."""
    # Points per hand similarity (most important)
    current_pph = game_state.points_needed / max(1, game_state.hands_remaining)
    example_pph = example_state["points_needed"] / max(1, example_state["hands_remaining"])
    pph_similarity = 1.0 - abs(current_pph - example_pph) / 1000.0
    
    # Hands remaining similarity
    hands_diff = abs(game_state.hands_remaining - example_state["hands_remaining"])
    hands_similarity = 1.0 - (hands_diff / 10.0)
    
    # Discards remaining (if available)
    if hasattr(game_state, 'discards_remaining') and 'discards_remaining' in example_state:
        disc_diff = abs(game_state.discards_remaining - example_state.get('discards_remaining', 3))
        disc_similarity = 1.0 - (disc_diff / 5.0)
    else:
        disc_similarity = 0.5
    
    # Weighted combination
    return (pph_similarity * 0.6 + hands_similarity * 0.3 + disc_similarity * 0.1)

def find_best_examples(game_state: GameState, n: int = 3) -> List[Dict]:
    """Find examples most similar to current game state."""
    scored_examples = []
    for ex in BALATRO_EXAMPLES:
        similarity = calculate_similarity(game_state, ex["game_state"])
        scored_examples.append((similarity, ex))
    
    # Return top n most similar - use key to avoid comparing dicts
    return [ex for _, ex in sorted(scored_examples, key=lambda x: x[0], reverse=True)[:n]]

def build_example_based_prompt(game_state: GameState) -> str:
    """Build a prompt that includes relevant examples."""
    # Find most relevant examples
    relevant_examples = find_best_examples(game_state, n=2)
    
    # Build the prompt
    prompt = """You are a Balatro poker bot that MUST use tools to analyze hands properly.

CRITICAL: Learn from these examples of good vs bad play:

"""
    
    for ex in relevant_examples:
        prompt += f"""
Example: {ex['scenario']}
Hand: {ex['hand']}
Need: {ex['game_state']['points_needed']} points in {ex['game_state']['hands_remaining']} hands

❌ BAD APPROACH (without tools):
Thought: {ex['bad_approach']['thought']}
Decision: {ex['bad_approach']['action']} {ex['bad_approach']['indices']}
Result: {ex['bad_approach']['result']}

✅ GOOD APPROACH (with tools):
"""
        for tool, result in ex['good_approach']['tools_used']:
            prompt += f"Used {tool}: {result}\n"
        
        prompt += f"""Thought: {ex['good_approach']['thought']}
Decision: {ex['good_approach']['action']} {ex['good_approach']['indices']}
Result: {ex['good_approach']['result']}

"""
    
    prompt += f"""
Now analyze YOUR situation:
Hand: {game_state.hand}
Points needed: {game_state.points_needed}
Hands remaining: {game_state.hands_remaining}
Discards remaining: {game_state.discards_remaining}

You MUST use tools like in the good examples. Start with check_cards."""
    
    return prompt

# ── Poker Hand Analysis Function ---------------------------------------
def analyze_poker_hand(hand: List[str]) -> Dict[str, Any]:
    """Analyze poker hand and return best combination using Balatro scoring."""
    if len(hand) < 5:
        return {"hand_type": "incomplete", "best_score": 0, "best_indices": list(range(len(hand)))}
    
    # Parse card format helper
    def parse_card(card: str) -> tuple[str, str]:
        """Parse card string to extract rank and suit, handling both formats."""
        # Handle format like "King of Hearts" or "10 of Spades"
        if " of " in card:
            rank_part, suit_part = card.split(" of ")
            # Convert suit names to single letters
            suit_map = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
            suit = suit_map.get(suit_part, suit_part[0])
            # Convert rank names
            rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
            rank = rank_map.get(rank_part, rank_part)
            return rank, suit
        # Handle format like "KH" or "10S"
        else:
            if len(card) == 2:
                return card[0], card[1]
            elif len(card) == 3:  # Like "10S"
                return card[:2], card[2]
            else:
                # Default fallback
                return card[:-1], card[-1]
    
    # Convert string cards to analyzable format
    rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10, 
                   'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11}
    for i in range(2, 10):
        rank_values[str(i)] = i
    
    # Card chip values in Balatro
    card_chip_values = {'A': 11, 'K': 10, 'Q': 10, 'J': 10,
                        'Ace': 11, 'King': 10, 'Queen': 10, 'Jack': 10}
    for i in range(2, 11):
        card_chip_values[str(i)] = i
    
    best_hand = None
    best_score = 0
    best_indices = []
    
    # Try all possible 5-card combinations from the 8 cards
    from itertools import combinations
    for combo in combinations(range(min(8, len(hand))), 5):
        indices = list(combo)
        selected_cards = [hand[i] for i in indices]
        
        # Parse cards for this combination
        parsed_cards = []
        for card in selected_cards:
            rank, suit = parse_card(card)
            parsed_cards.append((rank, suit))
        
        # Evaluate this 5-card hand
        chips, mult, hand_type = evaluate_balatro_hand(parsed_cards, rank_values, card_chip_values)
        score = chips * mult
        
        if score > best_score:
            best_score = score
            best_hand = hand_type
            best_indices = indices
    
    # Also analyze the full hand for strategic information
    all_parsed = []
    for i, card in enumerate(hand[:8]):
        rank, suit = parse_card(card)
        all_parsed.append((i, rank, suit))
    
    strategic_info = analyze_hand_strategy(all_parsed, rank_values)
    
    return {
        "hand_type": best_hand,
        "best_score": best_score,
        "best_indices": best_indices,
        "strategic_info": strategic_info
    }

def analyze_hand_strategy(parsed_hand: List[tuple[int, str, str]], rank_values: Dict) -> Dict[str, Any]:
    """Analyze hand for strategic discard decisions."""
    # Count ranks and suits
    rank_counts = {}
    suit_counts = {}
    rank_indices = {}
    suit_indices = {}
    
    for idx, rank, suit in parsed_hand:
        # Track rank occurrences
        if rank not in rank_counts:
            rank_counts[rank] = 0
            rank_indices[rank] = []
        rank_counts[rank] += 1
        rank_indices[rank].append(idx)
        
        # Track suit occurrences
        if suit not in suit_counts:
            suit_counts[suit] = 0
            suit_indices[suit] = []
        suit_counts[suit] += 1
        suit_indices[suit].append(idx)
    
    # Find pairs, trips, etc.
    pairs = [(rank, indices) for rank, indices in rank_indices.items() if len(indices) == 2]
    trips = [(rank, indices) for rank, indices in rank_indices.items() if len(indices) == 3]
    quads = [(rank, indices) for rank, indices in rank_indices.items() if len(indices) >= 4]
    
    # Find flush draws (4 of same suit)
    flush_draws = [(suit, indices) for suit, indices in suit_indices.items() if len(indices) == 4]
    flushes = [(suit, indices) for suit, indices in suit_indices.items() if len(indices) >= 5]
    
    # Find straight draws
    straight_draws = find_straight_draws(parsed_hand, rank_values)
    
    # Identify cards to definitely keep
    keep_cards = set()
    
    # Keep quads
    for rank, indices in quads:
        keep_cards.update(indices)
    
    # Keep trips
    for rank, indices in trips:
        keep_cards.update(indices)
    
    # Keep pairs (prioritize higher pairs)
    sorted_pairs = sorted(pairs, key=lambda x: rank_values.get(x[0], rank_values.get(x[0].capitalize(), 0)), reverse=True)
    for rank, indices in sorted_pairs[:2]:  # Keep up to 2 pairs
        keep_cards.update(indices)
    
    # Keep flush draws if we have 4 of a suit
    for suit, indices in flush_draws:
        keep_cards.update(indices)
    
    # Keep cards that contribute to straight draws
    for draw_indices in straight_draws:
        if len(draw_indices) >= 4:
            keep_cards.update(draw_indices)
    
    return {
        "pairs": pairs,
        "trips": trips,
        "quads": quads,
        "flush_draws": flush_draws,
        "flushes": flushes,
        "straight_draws": straight_draws,
        "keep_cards": list(keep_cards),
        "rank_counts": rank_counts,
        "suit_counts": suit_counts
    }

def find_straight_draws(parsed_hand: List[tuple[int, str, str]], rank_values: Dict) -> List[List[int]]:
    """Find potential straight draws in the hand."""
    # Get unique ranks with their indices
    rank_to_indices = {}
    for idx, rank, _ in parsed_hand:
        rank_val = rank_values.get(rank, rank_values.get(rank.capitalize(), 0))
        if rank_val not in rank_to_indices:
            rank_to_indices[rank_val] = []
        rank_to_indices[rank_val].append(idx)
    
    straight_draws = []
    unique_ranks = sorted(rank_to_indices.keys())
    
    # Check for consecutive ranks
    for i in range(len(unique_ranks)):
        consecutive = [unique_ranks[i]]
        indices = rank_to_indices[unique_ranks[i]].copy()
        
        for j in range(i + 1, len(unique_ranks)):
            if unique_ranks[j] == consecutive[-1] + 1:
                consecutive.append(unique_ranks[j])
                indices.extend(rank_to_indices[unique_ranks[j]])
                if len(consecutive) >= 4:  # 4 cards to a straight
                    straight_draws.append(indices[:5])  # Keep max 5 cards
            else:
                break
    
    # Check for ace-low straight possibility
    if 14 in rank_to_indices and all(r in rank_to_indices for r in [2, 3, 4, 5]):
        indices = []
        for r in [14, 2, 3, 4, 5]:
            indices.extend(rank_to_indices[r])
        straight_draws.append(indices[:5])
    
    return straight_draws

def evaluate_balatro_hand(cards: List[tuple[str, str]], rank_values: Dict, card_chip_values: Dict) -> tuple[int, int, str]:
    """Evaluate a 5-card hand using Balatro's scoring system."""
    chips = 0
    mult = 0
    hand_type = "high_card"
    
    ranks = [card[0] for card in cards]
    suits = [card[1] for card in cards]
    
    # Check for flush
    flush = len(set(suits)) == 1
    
    # Check for straight
    sorted_rank_values = sorted([rank_values.get(r, rank_values.get(r.capitalize(), 0)) for r in ranks])
    straight = False
    
    # Regular straight check
    if sorted_rank_values == list(range(sorted_rank_values[0], sorted_rank_values[0] + 5)):
        straight = True
    # Check for ace-low straight (A,2,3,4,5)
    elif sorted_rank_values == [2, 3, 4, 5, 14]:
        straight = True
    
    # Count ranks
    rank_counts = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    
    # Sort by count and rank value
    sorted_counts = sorted(rank_counts.items(), 
                          key=lambda x: (x[1], rank_values.get(x[0], rank_values.get(x[0].capitalize(), 0))), 
                          reverse=True)
    primary_rank, primary_count = sorted_counts[0] if sorted_counts else (None, 0)
    secondary_rank, secondary_count = sorted_counts[1] if len(sorted_counts) > 1 else (None, 0)
    
    # Determine hand type and base chips/mult according to Balatro rules
    if flush and primary_count == 5:  # Flush Five
        chips = 160
        mult = 16
        hand_type = "flush_five"
        scoring_cards = cards
    elif flush and primary_count == 3 and secondary_count == 2:  # Flush House
        chips = 140
        mult = 14
        hand_type = "flush_house"
        scoring_cards = cards
    elif primary_count == 5:  # Five of a Kind
        chips = 120
        mult = 12
        hand_type = "five_of_a_kind"
        scoring_cards = cards
    elif straight and flush:  # Straight Flush
        chips = 100
        mult = 8
        hand_type = "straight_flush"
        scoring_cards = cards
    elif primary_count == 4:  # Four of a Kind
        chips = 60
        mult = 7
        hand_type = "four_of_a_kind"
        scoring_cards = [c for c in cards if c[0] == primary_rank]
    elif primary_count == 3 and secondary_count == 2:  # Full House
        chips = 40
        mult = 4
        hand_type = "full_house"
        scoring_cards = cards
    elif flush:  # Flush
        chips = 35
        mult = 4
        hand_type = "flush"
        scoring_cards = cards
    elif straight:  # Straight
        chips = 30
        mult = 4
        hand_type = "straight"
        scoring_cards = cards
    elif primary_count == 3:  # Three of a Kind
        chips = 30
        mult = 3
        hand_type = "three_of_a_kind"
        scoring_cards = [c for c in cards if c[0] == primary_rank]
    elif primary_count == 2 and secondary_count == 2:  # Two Pair
        chips = 20
        mult = 2
        hand_type = "two_pair"
        scoring_cards = [c for c in cards if c[0] in [primary_rank, secondary_rank]]
    elif primary_count == 2:  # Pair
        chips = 10
        mult = 2
        hand_type = "pair"
        scoring_cards = [c for c in cards if c[0] == primary_rank]
    else:  # High Card
        chips = 5
        mult = 1
        hand_type = "high_card"
        # Find highest value card
        max_card = max(cards, key=lambda c: rank_values.get(c[0], rank_values.get(c[0].capitalize(), 0)))
        scoring_cards = [max_card]
    
    # Add chip values for scoring cards
    for card in scoring_cards:
        rank = card[0]
        chip_value = card_chip_values.get(rank, card_chip_values.get(rank.capitalize(), 0))
        chips += chip_value
    
    return chips, mult, hand_type

# ── Decision Processing Functions --------------------------------------
async def process_agent_decision(session: SessionState) -> DecisionResponse:
    """Process decision by forcing tool use through strict prompting."""
    logger.info(f"Processing decision for session: {session.session_id}")
    
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    game_state = session.game_state
    
    # Analyze the hand
    analysis = analyze_poker_hand(game_state.hand)
    logger.info(f"Analysis: {analysis}")
    
    # Check threshold
    threshold_check = {
        "points_needed": game_state.points_needed,
        "hands_remaining": game_state.hands_remaining,
        "required_per_hand": game_state.points_needed / max(1, game_state.hands_remaining),
        "current_best_value": analysis["best_score"],
        "meets_threshold": analysis["best_score"] >= (game_state.points_needed / max(1, game_state.hands_remaining))
    }
    logger.info(f"Threshold: {threshold_check}")
    
    # Make decision based on analysis
    if threshold_check["meets_threshold"] or game_state.discards_remaining == 0:
        display_msg = f"✅ {analysis['hand_type'].replace('_', ' ').title()}: {analysis['best_score']} pts"
        if threshold_check["meets_threshold"]:
            display_msg += f" (Need {threshold_check['required_per_hand']:.0f})"
        else:
            display_msg = f"⚠️ Last chance! {analysis['hand_type'].replace('_', ' ').title()}: {analysis['best_score']} pts"
        
        return DecisionResponse(
            session_id=session.session_id,
            action="keep",
            data={
                "indices": analysis["best_indices"],
                "cards": [game_state.hand[i] for i in analysis["best_indices"]]
            },
            thoughts=[
                f"Hand analysis: {analysis['hand_type']} worth {analysis['best_score']} points",
                f"Threshold: need {threshold_check['required_per_hand']:.0f} per hand",
                f"Decision: {'Meets threshold' if threshold_check['meets_threshold'] else 'Last chance, keeping best'}"
            ],
            confidence=0.9 if threshold_check["meets_threshold"] else 0.6,
            display_message=display_msg
        )
    else:
        # SMART DISCARD DECISION
        logger.info("\n" + get_random_discard_art())
        
        strategic_info = analysis.get("strategic_info", {})
        keep_cards = set(strategic_info.get("keep_cards", []))
        
        # Build display message for game screen
        display_parts = []
        display_parts.append(f"Current: {analysis['hand_type'].replace('_', ' ').title()} ({analysis['best_score']} pts)")
        display_parts.append(f"Need: {threshold_check['required_per_hand']:.0f} pts/hand")
        
        # If we have a good base (pair, trips, flush draw), keep those cards
        if keep_cards:
            discard_indices = [i for i in range(len(game_state.hand)) if i not in keep_cards]
            
            # Add strategy info to display
            if strategic_info.get("trips"):
                display_parts.append(f"Keeping: Three {strategic_info['trips'][0][0]}s")
            elif strategic_info.get("pairs"):
                pair_names = [f"Pair of {r}s" for r, _ in strategic_info['pairs'][:2]]
                display_parts.append(f"Keeping: {', '.join(pair_names)}")
            elif strategic_info.get("flush_draws"):
                suit_name = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
                suit = strategic_info['flush_draws'][0][0]
                display_parts.append(f"Going for: {suit_name.get(suit, suit)} Flush")
            elif strategic_info.get("straight_draws") and strategic_info['straight_draws']:
                display_parts.append("Going for: Straight")
        else:
            # No good combinations - keep high value cards
            parsed_hand = []
            for i, card in enumerate(game_state.hand[:8]):
                if " of " in card:
                    rank_part, _ = card.split(" of ")
                    rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
                    rank = rank_map.get(rank_part, rank_part)
                else:
                    rank = card[:-1] if len(card) <= 3 else card[:-1]
                parsed_hand.append((i, rank))
            
            # Sort by rank value and keep top cards
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
                          'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11}
            for i in range(2, 10):
                rank_values[str(i)] = i
            
            sorted_cards = sorted(parsed_hand, 
                                key=lambda x: rank_values.get(x[1], rank_values.get(x[1].capitalize(), int(x[1]) if x[1].isdigit() else 0)), 
                                reverse=True)
            
            # Keep 2-4 highest cards depending on their values
            keep_count = 2
            if len([c for c in sorted_cards[:4] if rank_values.get(c[1], 0) >= 10]) >= 3:
                keep_count = 3
            elif len([c for c in sorted_cards[:4] if rank_values.get(c[1], 0) >= 11]) >= 2:
                keep_count = 2
                
            keep_indices = [c[0] for c in sorted_cards[:keep_count]]
            discard_indices = [i for i in range(len(game_state.hand)) if i not in keep_indices]
            
            display_parts.append("Keeping: High cards")
        
        display_parts.append(f"Discarding: {len(discard_indices)} cards")
        display_message = " | ".join(display_parts)
        
        # Build detailed decision explanation
        thoughts = [f"Current best: {analysis['hand_type']} worth {analysis['best_score']} points"]
        thoughts.append(f"Need {threshold_check['required_per_hand']:.0f} per hand (deficit: {threshold_check['required_per_hand'] - analysis['best_score']:.0f})")
        
        if strategic_info.get("pairs"):
            thoughts.append(f"Keeping pairs: {[r for r, _ in strategic_info['pairs']]}")
        if strategic_info.get("trips"):
            thoughts.append(f"Keeping three of a kind: {[r for r, _ in strategic_info['trips']]}")
        if strategic_info.get("flush_draws"):
            thoughts.append(f"Keeping flush draw: {len(strategic_info['flush_draws'][0][1])} cards of {strategic_info['flush_draws'][0][0]}")
        
        thoughts.append(f"Discarding {len(discard_indices)} cards for better combinations")
        
        # Log the discard details with style
        logger.info(f"""
╔═══════════════════════════════════════╗
║      SMART DISCARD DECISION! 🎯       ║
╠═══════════════════════════════════════╣
║ Current: {analysis['hand_type']:<20} ║
║ Score: {analysis['best_score']:<23} ║
║ Need: {threshold_check['required_per_hand']:<24.0f} ║
║ Deficit: {threshold_check['required_per_hand'] - analysis['best_score']:<21.0f} ║
╠═══════════════════════════════════════╣
║ Discarding {len(discard_indices)} cards:                   ║
║ {str([game_state.hand[i] for i in discard_indices[:5]]):<37} ║
║ Keeping {len(game_state.hand) - len(discard_indices)} cards:                      ║
║ {str([game_state.hand[i] for i in range(len(game_state.hand)) if i not in discard_indices][:5]):<37} ║
╚═══════════════════════════════════════╝
        """)
        
        return DecisionResponse(
            session_id=session.session_id,
            action="discard",
            data={"indices": discard_indices[:5]},  # Max 5 discards
            thoughts=thoughts,
            confidence=0.8,
            display_message=display_message
        )

async def process_decision_with_examples(session: SessionState) -> DecisionResponse:
    """Process decision using example-based learning."""
    logger.info(f"Processing with examples for session: {session.session_id}")
    
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    game_state = session.game_state
    
    # Build example-based prompt
    prompt = build_example_based_prompt(game_state)
    
    # Add explicit tool request to prompt
    prompt += """

Based on the examples above, please respond with:
1. Which tools you want to use (e.g., analyze_hand, check_win_threshold, decide_discard, decide_keep)
2. Your strategic thinking about this hand
3. Your recommended action (keep or discard) and why

Format your response like:
TOOLS: [list the tools you'd use]
STRATEGY: [your strategic analysis]
ACTION: [keep/discard]
REASON: [why this action]
"""
    
    logger.info(f"Example-based prompt length: {len(prompt)}")
    
    # Get LLM response
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    
    with torch.no_grad():
        out = model_manager.model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.3,
            do_sample=True,
            pad_token_id=model_manager.tokenizer.eos_token_id,
        )
    
    response = model_manager.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    logger.info(f"LLM Response:\n{response}")
    
    # Parse LLM response
    llm_tools = []
    llm_strategy = ""
    llm_action = ""
    llm_reason = ""
    
    # Extract components from response
    lines = response.split('\n')
    for line in lines:
        if line.startswith("TOOLS:"):
            tools_text = line.replace("TOOLS:", "").strip()
            # Extract tool names from the text
            tool_names = ["analyze_hand", "check_win_threshold", "decide_discard", "decide_keep"]
            for tool in tool_names:
                if tool in tools_text.lower():
                    llm_tools.append(tool)
        elif line.startswith("STRATEGY:"):
            llm_strategy = line.replace("STRATEGY:", "").strip()
        elif line.startswith("ACTION:"):
            llm_action = line.replace("ACTION:", "").strip().lower()
        elif line.startswith("REASON:"):
            llm_reason = line.replace("REASON:", "").strip()
    
    # Log what the LLM wants
    logger.info(f"LLM requested tools: {llm_tools}")
    logger.info(f"LLM strategy: {llm_strategy}")
    logger.info(f"LLM action: {llm_action}")
    
    # Do our own analysis as backup/verification
    analysis = analyze_poker_hand(game_state.hand)
    threshold_check = {
        "required_per_hand": game_state.points_needed / max(1, game_state.hands_remaining),
        "current_best_value": analysis["best_score"],
        "meets_threshold": analysis["best_score"] >= (game_state.points_needed / max(1, game_state.hands_remaining))
    }
    
    # Build response with both LLM and analytical insights
    thoughts = []
    
    # Add LLM's strategic thinking if available
    if llm_strategy:
        thoughts.append(f"LLM Strategy: {llm_strategy}")
    
    # Add our analysis
    thoughts.append(f"Analysis: {analysis['hand_type']} worth {analysis['best_score']} points")
    thoughts.append(f"Threshold check: Need {threshold_check['required_per_hand']:.0f} per hand")
    
    # Add LLM's reasoning if available
    if llm_reason:
        thoughts.append(f"LLM Reasoning: {llm_reason}")
    
    # Add comparison to examples
    best_examples = find_best_examples(game_state, n=1)
    if best_examples:
        thoughts.append(f"Similar to example: {best_examples[0]['scenario']}")
    
    # Determine action - blend LLM suggestion with analytical decision
    should_keep = threshold_check["meets_threshold"] or game_state.discards_remaining == 0
    
    # If LLM has a strong opinion and it makes sense, consider it
    if llm_action in ["keep", "discard"]:
        if llm_action == "keep" and analysis["best_score"] >= 100:  # Decent hand
            should_keep = True
            thoughts.append("Following LLM's keep recommendation")
        elif llm_action == "discard" and analysis["best_score"] < threshold_check["required_per_hand"]:
            should_keep = False
            thoughts.append("Following LLM's discard recommendation")
        else:
            thoughts.append(f"LLM suggested {llm_action}, but analysis suggests {'keep' if should_keep else 'discard'}")
    
    if should_keep:
        display_msg = f"✅ {analysis['hand_type'].replace('_', ' ').title()}: {analysis['best_score']} pts"
        if llm_strategy:
            display_msg += f" | {llm_strategy[:30]}..."
            
        return DecisionResponse(
            session_id=session.session_id,
            action="keep",
            data={
                "indices": analysis["best_indices"],
                "cards": [game_state.hand[i] for i in analysis["best_indices"]]
            },
            thoughts=thoughts,
            confidence=0.9,
            display_message=display_msg
        )
    else:
        # SMART DISCARD WITH ASCII ART!
        logger.info("\n" + get_random_discard_art())
        
        strategic_info = analysis.get("strategic_info", {})
        keep_cards = set(strategic_info.get("keep_cards", []))
        
        # Build display message
        display_parts = []
        display_parts.append(f"Current: {analysis['hand_type'].replace('_', ' ').title()} ({analysis['best_score']} pts)")
        display_parts.append(f"Need: {threshold_check['required_per_hand']:.0f} pts")
        
        # Add LLM strategy to display if available
        if llm_strategy:
            display_parts.append(f"AI: {llm_strategy[:40]}...")
        
        # Determine what to discard based on strategy
        if keep_cards:
            discard_indices = [i for i in range(min(8, len(game_state.hand))) if i not in keep_cards]
            
            # Add strategy info to display
            if strategic_info.get("trips"):
                display_parts.append(f"Keeping: Three {strategic_info['trips'][0][0]}s")
            elif strategic_info.get("pairs"):
                display_parts.append(f"Keeping: Pair of {strategic_info['pairs'][0][0]}s")
            elif strategic_info.get("flush_draws"):
                suit_map = {"H": "♥️", "D": "♦️", "C": "♣️", "S": "♠️"}
                suit = strategic_info['flush_draws'][0][0]
                display_parts.append(f"Going for: {suit_map.get(suit, suit)} Flush")
        else:
            # No good combinations - keep high value cards
            parsed_hand = []
            for i, card in enumerate(game_state.hand[:8]):
                if " of " in card:
                    rank_part, _ = card.split(" of ")
                    rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
                    rank = rank_map.get(rank_part, rank_part)
                else:
                    rank = card[:-1] if len(card) <= 3 else card[:-1]
                parsed_hand.append((i, rank))
            
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
                          'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11}
            for i in range(2, 10):
                rank_values[str(i)] = i
            
            sorted_cards = sorted(parsed_hand, 
                                key=lambda x: rank_values.get(x[1], int(x[1]) if x[1].isdigit() else 0), 
                                reverse=True)
            
            keep_count = 3 if len([c for c in sorted_cards[:4] if rank_values.get(c[1], 0) >= 10]) >= 3 else 2
            keep_indices = [c[0] for c in sorted_cards[:keep_count]]
            discard_indices = [i for i in range(min(8, len(game_state.hand))) if i not in keep_indices]
            
            display_parts.append("Keeping: High cards")
        
        display_parts.append(f"Discarding: {len(discard_indices[:5])} cards")
        display_message = " | ".join(display_parts)
        
        # Add strategic reasoning to thoughts
        keeping_description = "High value cards"
        if strategic_info.get("trips"):
            rank = strategic_info['trips'][0][0]
            keeping_description = f"Three {rank}s! 🎰"
            thoughts.append(f"Keeping three of a kind: {rank}")
        elif strategic_info.get("pairs"):
            if len(strategic_info['pairs']) >= 2:
                ranks = [r for r, _ in strategic_info['pairs'][:2]]
                keeping_description = f"Two Pair: {ranks[0]}s & {ranks[1]}s"
                thoughts.append(f"Keeping two pair: {ranks}")
            else:
                rank = strategic_info['pairs'][0][0]
                keeping_description = f"Pair of {rank}s"
                thoughts.append(f"Keeping pair: {rank}")
        elif strategic_info.get("flush_draws"):
            suit_symbols = {"H": "♥️", "D": "♦️", "C": "♣️", "S": "♠️"}
            suit = strategic_info['flush_draws'][0][0]
            suit_count = len(strategic_info['flush_draws'][0][1])
            keeping_description = f"{suit_count} {suit_symbols.get(suit, suit)} Flush Draw"
            thoughts.append("Keeping flush draw opportunity")
        
        # Get actual cards for display
        discard_cards = [game_state.hand[i] for i in discard_indices[:5]]
        keep_cards_list = [game_state.hand[i] for i in range(min(8, len(game_state.hand))) if i not in discard_indices]
        
        # Format cards for ASCII display
        def format_card_short(card):
            if " of " in card:
                rank, suit = card.split(" of ")
                suit_symbols = {"Hearts": "♥", "Diamonds": "♦", "Clubs": "♣", "Spades": "♠"}
                rank_short = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}.get(rank, rank)
                return f"{rank_short}{suit_symbols.get(suit, suit[0])}"
            return card
        
        discard_display = ', '.join([format_card_short(c) for c in discard_cards])
        keep_display = ', '.join([format_card_short(c) for c in keep_cards_list])
            
        logger.info(f"""
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃      EXAMPLE-BASED SMART DISCARD! 📚🎯       ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Learning from: {best_examples[0]['scenario'] if best_examples else 'General strategy':<29} ┃
┃ Current hand: {analysis['hand_type'].replace('_', ' ').title():<30} ┃
┃ Points: {analysis['best_score']:<37} ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ LLM SAYS: {llm_strategy[:34] if llm_strategy else 'No specific strategy':<34} ┃
┃ STRATEGY: {keeping_description:<34} ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃ Keeping: {keep_display[:35]:<35} ┃
┃ Tossing: {discard_display[:35]:<35} ┃
┃ Tools requested: {', '.join(llm_tools[:2]) if llm_tools else 'None':<27} ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
        """)
        
        return DecisionResponse(
            session_id=session.session_id,
            action="discard",
            data={"indices": discard_indices[:5]},  # Max 5 discards
            thoughts=thoughts,
            confidence=0.85,
            display_message=display_message
        )

# ── Learning from Results ----------------------------------------------
class ExampleCollector:
    """Collect real game results to build more examples."""
    
    def __init__(self):
        self.game_history = []
    
    def record_decision(self, game_state: GameState, decision: DecisionResponse, outcome: str):
        """Record a decision and its outcome for future learning."""
        self.game_history.append({
            "timestamp": datetime.now(),
            "game_state": game_state.model_dump(),
            "decision": decision.model_dump(),
            "outcome": outcome
        })
    
    def generate_new_examples(self) -> List[Dict]:
        """Generate new training examples from game history."""
        new_examples = []
        
        for i, game1 in enumerate(self.game_history):
            for game2 in self.game_history[i+1:]:
                if self._similar_situation(game1, game2):
                    if game1["outcome"] == "won" and game2["outcome"] == "lost":
                        new_examples.append(self._create_example(game2, game1))
        
        return new_examples
    
    def _similar_situation(self, game1, game2) -> bool:
        """Check if two game states are similar enough to compare."""
        gs1, gs2 = game1["game_state"], game2["game_state"]
        return abs(gs1["points_needed"] - gs2["points_needed"]) < 200
    
    def _create_example(self, bad_game, good_game) -> Dict:
        """Create a training example from two games."""
        return {
            "scenario": f"Learned from game {bad_game['timestamp']}",
            "hand": bad_game["game_state"]["hand"],
            "game_state": bad_game["game_state"],
            "bad_approach": {
                "thought": bad_game["decision"]["thoughts"][0] if bad_game["decision"]["thoughts"] else "No analysis",
                "action": bad_game["decision"]["action"],
                "indices": bad_game["decision"]["data"].get("indices", []),
                "result": bad_game["outcome"]
            },
            "good_approach": {
                "tools_used": [],  # Would need to track this
                "thought": good_game["decision"]["thoughts"][0] if good_game["decision"]["thoughts"] else "Analyzed properly",
                "action": good_game["decision"]["action"],
                "indices": good_game["decision"]["data"].get("indices", []),
                "result": good_game["outcome"]
            }
        }

# Global instance for collecting examples
example_collector = ExampleCollector()

# ── API Endpoints ------------------------------------------------------
@app.get("/")
async def root():
    """Health check endpoint."""
    logger.info("Health check requested")
    return {
        "status": "ok",
        "service": "Balatro Agent API v2",
        "model_loaded": model_manager.model_loaded
    }

@app.post("/decide", response_model=DecisionResponse)
async def make_decision(request: AgentDecision):
    """Main endpoint - game server sends state, agent returns decision."""
    logger.info(f"\n{'='*60}")
    logger.info(f"NEW DECISION REQUEST")
    logger.info(f"{'='*60}")
    logger.info(f"Session ID: {request.session_id}")
    logger.info(f"Hand: {request.game_state.hand}")
    logger.info(f"Points needed: {request.game_state.points_needed}")
    logger.info(f"Hands remaining: {request.game_state.hands_remaining}")
    
    # Create or update session
    if request.session_id in sessions:
        session = sessions[request.session_id]
        session.game_state = request.game_state
        logger.info("Updating existing session")
    else:
        session = SessionState(
            session_id=request.session_id,
            game_state=request.game_state,
            history=[],
            thoughts=[],
            actions=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        sessions[request.session_id] = session
        logger.info("Created new session")
    
    # Process decision
    try:
        decision = await asyncio.wait_for(
            process_agent_decision(session),
            timeout=request.timeout
        )
        logger.info(f"Decision successful: {decision.action}")
        logger.info(f"{'='*60}\n")
        return decision
    except asyncio.TimeoutError:
        logger.error("Agent timeout!")
        raise HTTPException(status_code=504, detail="Agent timeout")
    except Exception as e:
        logger.error(f"Error in decision endpoint: {e}", exc_info=True)
        raise

@app.post("/decide_with_examples", response_model=DecisionResponse)
async def make_decision_with_examples(request: AgentDecision):
    """Decision endpoint that uses example-based learning."""
    logger.info("Using example-based decision making")
    
    # Create or update session
    if request.session_id in sessions:
        session = sessions[request.session_id]
        session.game_state = request.game_state
    else:
        session = SessionState(
            session_id=request.session_id,
            game_state=request.game_state,
            history=[],
            thoughts=[],
            actions=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        sessions[request.session_id] = session
    
    decision = await process_decision_with_examples(session)
    return decision

@app.post("/decide_hybrid", response_model=DecisionResponse)
async def make_hybrid_decision(request: AgentDecision):
    """Use examples 50% of the time to compare effectiveness."""
    
    # Create or update session
    if request.session_id in sessions:
        session = sessions[request.session_id]
        session.game_state = request.game_state
    else:
        session = SessionState(
            session_id=request.session_id,
            game_state=request.game_state,
            history=[],
            thoughts=[],
            actions=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        sessions[request.session_id] = session
    
    if random.random() < 0.5:
        logger.info("Hybrid: Using example-based approach")
        decision = await process_decision_with_examples(session)
    else:
        logger.info("Hybrid: Using direct approach")
        decision = await process_agent_decision(session)
    
    return decision

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full session history for debugging."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "game_state": session.game_state.model_dump(),
        "thoughts": [t.model_dump() for t in session.thoughts],
        "actions": [a.model_dump() for a in session.actions],
        "created_at": session.created_at,
        "last_updated": session.last_updated
    }

@app.post("/record_outcome/{session_id}")
async def record_outcome(session_id: str, outcome: str):
    """Record the outcome of a decision for learning."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # Get the last decision made
    if session.actions:
        last_action = session.actions[-1]
        decision = DecisionResponse(
            session_id=session_id,
            action=last_action.name,
            data=last_action.arguments,
            thoughts=[t.thought for t in session.thoughts[-3:]],  # Last few thoughts
            confidence=0.5
        )
        
        example_collector.record_decision(session.game_state, decision, outcome)
        
        return {"status": "recorded", "total_history": len(example_collector.game_history)}
    
    return {"status": "no_actions_to_record"}

@app.get("/examples")
async def get_examples():
    """Get all training examples."""
    return {
        "static_examples": len(BALATRO_EXAMPLES),
        "learned_examples": len(example_collector.game_history),
        "examples": BALATRO_EXAMPLES[:3]  # First 3 as preview
    }

@app.post("/decide_one_word", response_model=DecisionResponse)
async def make_one_word_decision(request: AgentDecision):
    """Ultra-fast endpoint - LLM just says keep or discard, no examples."""
    logger.info(f"\n{'='*60}")
    logger.info(f"ONE WORD DECISION REQUEST")
    logger.info(f"{'='*60}")
    logger.info(f"Session ID: {request.session_id}")
    logger.info(f"Hand: {request.game_state.hand}")
    logger.info(f"Points needed: {request.game_state.points_needed}")
    logger.info(f"Hands remaining: {request.game_state.hands_remaining}")
    
    # Create or update session
    if request.session_id in sessions:
        session = sessions[request.session_id]
        session.game_state = request.game_state
        logger.info("Updating existing session")
    else:
        session = SessionState(
            session_id=request.session_id,
            game_state=request.game_state,
            history=[],
            thoughts=[],
            actions=[],
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        sessions[request.session_id] = session
        logger.info("Created new session")
    
    # Process decision
    try:
        decision = await asyncio.wait_for(
            process_one_word_decision(session),
            timeout=request.timeout
        )
        logger.info(f"Decision successful: {decision.action}")
        logger.info(f"{'='*60}\n")
        return decision
    except asyncio.TimeoutError:
        logger.error("Agent timeout!")
        raise HTTPException(status_code=504, detail="Agent timeout")
    except Exception as e:
        logger.error(f"Error in decision endpoint: {e}", exc_info=True)
        raise

async def process_one_word_decision(session: SessionState) -> DecisionResponse:
    """Process decision with one-word LLM response - no examples."""
    logger.info(f"Processing one-word decision for session: {session.session_id}")
    
    if not model_manager.model_loaded:
        model_manager.load_model()
    
    game_state = session.game_state
    
    # First, analyze the hand
    logger.info(f"⚙️  Analyzing hand...")
    analysis = analyze_poker_hand(game_state.hand)
    
    # Calculate threshold
    threshold_check = {
        "required_per_hand": game_state.points_needed / max(1, game_state.hands_remaining),
        "current_best_score": analysis["best_score"],
        "meets_threshold": analysis["best_score"] >= (game_state.points_needed / max(1, game_state.hands_remaining)),
        "deficit": (game_state.points_needed / max(1, game_state.hands_remaining)) - analysis["best_score"]
    }
    
    logger.info(f"   • Best: {analysis['hand_type']} ({analysis['best_score']} pts)")
    logger.info(f"   • Need: {threshold_check['required_per_hand']:.0f} pts/hand")
    if threshold_check['meets_threshold']:
        logger.info(f"   • ✅ MEETS threshold")
    else:
        logger.info(f"   • ❌ BELOW by {threshold_check['deficit']:.0f}")
    
    # Ultra-simple prompt - NO EXAMPLES
    prompt = f"""You are a Balatro poker bot. Respond with ONLY ONE WORD: "keep" or "discard"

Hand: {game_state.hand[:8]}
Best: {analysis['hand_type']} = {analysis['best_score']} points
Need: {threshold_check['required_per_hand']:.0f} points
Discards left: {game_state.discards_remaining}

ONE WORD:"""
    
    # Get LLM's one-word response
    logger.info(f"📤 Asking LLM...")
    
    inputs = model_manager.tokenizer(prompt, return_tensors="pt").to(model_manager.model.device)
    
    with torch.no_grad():
        out = model_manager.model.generate(
            **inputs,
            max_new_tokens=5,  # Just one word
            temperature=0.3,   # Slightly higher for more variation
            do_sample=True,
            pad_token_id=model_manager.tokenizer.eos_token_id,
        )
    
    response = model_manager.tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    ).strip().lower()
    
    # Extract first word
    llm_word = response.split()[0] if response else "keep"
    logger.info(f"🎯 LLM says: {llm_word.upper()}")
    
    # Execute decision
    if llm_word == "discard" and game_state.discards_remaining > 0 and not threshold_check["meets_threshold"]:
        # Find optimal discard
        strategic_info = analysis.get("strategic_info", {})
        keep_cards = set(strategic_info.get("keep_cards", []))
        
        if keep_cards:
            discard_indices = [i for i in range(min(8, len(game_state.hand))) if i not in keep_cards][:5]
        else:
            # Keep high cards
            parsed_hand = []
            for i, card in enumerate(game_state.hand[:8]):
                if " of " in card:
                    rank_part, _ = card.split(" of ")
                    rank_map = {"Ace": "A", "King": "K", "Queen": "Q", "Jack": "J"}
                    rank = rank_map.get(rank_part, rank_part)
                else:
                    rank = card[:-1] if len(card) <= 3 else card[:-1]
                parsed_hand.append((i, rank))
            
            rank_values = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, '10': 10,
                          'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11}
            for i in range(2, 10):
                rank_values[str(i)] = i
            
            sorted_cards = sorted(parsed_hand, 
                                key=lambda x: rank_values.get(x[1], int(x[1]) if x[1].isdigit() else 0), 
                                reverse=True)
            
            keep_count = 2 if len([c for c in sorted_cards[:3] if rank_values.get(c[1], 0) >= 10]) < 2 else 3
            keep_indices = [c[0] for c in sorted_cards[:keep_count]]
            discard_indices = [i for i in range(min(8, len(game_state.hand))) if i not in keep_indices][:5]
        
        logger.info(f"   Discarding: {discard_indices}")
        
        return DecisionResponse(
            session_id=session.session_id,
            action="discard",
            data={"indices": discard_indices},
            thoughts=[f"Current: {analysis['hand_type']} ({analysis['best_score']})", 
                     f"Need: {threshold_check['required_per_hand']:.0f}",
                     "LLM: discard"],
            confidence=0.8,
            display_message=f"Discard {len(discard_indices)} | {analysis['hand_type']} → ?"
        )
    else:
        # Keep/play optimal hand
        logger.info(f"   Playing: {analysis['best_indices']}")
        
        return DecisionResponse(
            session_id=session.session_id,
            action="keep",
            data={
                "indices": analysis["best_indices"],
                "cards": [game_state.hand[i] for i in analysis["best_indices"]]
            },
            thoughts=[f"Best: {analysis['hand_type']} ({analysis['best_score']})",
                     f"Need: {threshold_check['required_per_hand']:.0f}",
                     "LLM: keep"],
            confidence=0.9 if threshold_check["meets_threshold"] else 0.7,
            display_message=f"✅ {analysis['hand_type'].replace('_', ' ').title()}: {analysis['best_score']} pts"
        )

# ── Run the server -----------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Balatro Agent API v2 with Example Learning...")
    model_manager.load_model()
    uvicorn.run(app, host="0.0.0.0", port=8000)
