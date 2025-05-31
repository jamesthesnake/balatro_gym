import numpy as np
from itertools import combinations
from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import encode_select
from balatro_gym.balatro_game import Card, BalatroGame
from typing import List, Tuple

# ------------------------------------------------------------------
# Heuristic parameters
# ------------------------------------------------------------------
# We'll discard cards whose "rank index" is below this threshold.
# Card indices run 0..51, with rank = idx % 13 (0 → 2, 1 → 3, …, 8 → T, 9 → J, 10 → Q, 11 → K, 12 → A).
# If threshold_rank = 8, then anything with rank < 8 (i.e. 2..9) gets discarded;  T, J, Q, K, A are kept.
THRESHOLD_RANK = 8

# How many episodes (hands) to simulate:
NUM_EPISODES = 1000
# ------------------------------------------------------------------


def rank_of_card(card_idx: int) -> int:
    """
    Given a card index 0..51, return its rank index 0..12
      0  → "2"
      1  → "3"
      ...
      8  → "T"
      9  → "J"
      10 → "Q"
      11 → "K"
      12 → "A"
    """
    return card_idx % 13


def card_idx_to_card_object(card_idx: int) -> Card:
    """
    Convert a card index (0-51) to a Card object.
    """
    rank_idx = card_idx % 13
    suit_idx = card_idx // 13
    
    rank = list(Card.Ranks)[rank_idx]
    suit = list(Card.Suits)[suit_idx]
    
    return Card(rank, suit)


def make_discard_action(hand: np.ndarray, threshold_rank: int) -> int:
    """
    Given `hand`, a length‐8 np.ndarray of card indices (0..51),
    build a bitmask action (0..255) that discards every card
    whose rank < threshold_rank.
    """
    ranks = hand % 13  # array of shape (8,), values 0..12
    discard_positions: List[int] = [i for i in range(8) if ranks[i] < threshold_rank]
    # Build bitmask: set bit i if we discard card at index i
    mask = 0
    for i in discard_positions:
        mask |= (1 << i)
    return mask  # this is an integer in [0..255]


def evaluate_all_combinations(hand: np.ndarray) -> Tuple[int, int]:
    """
    Given `hand`, a length‐8 np.ndarray of card indices,
    evaluate all possible 5-card combinations (C(8,5) = 56) 
    and return the best score along with the positions of the best hand.
    
    Returns:
        Tuple of (best_score, best_positions_tuple)
    """
    best_score = -1
    best_positions = None
    
    # Generate all possible 5-card combinations from 8 cards
    for positions in combinations(range(8), 5):
        # Convert card indices to Card objects
        card_indices = [hand[pos] for pos in positions]
        card_objects = [card_idx_to_card_object(idx) for idx in card_indices]
        
        # Evaluate this 5-card hand using BalatroGame's evaluation function
        score = BalatroGame._evaluate_hand(card_objects)
        
        if score > best_score:
            best_score = score
            best_positions = positions
    
    return best_score, best_positions


def make_select_action(hand: np.ndarray) -> int:
    """
    Given `hand`, a length‐8 np.ndarray of card indices after drawing new cards,
    evaluate all 56 possible 5-card combinations and select the best one.
    Returns the appropriate action ID [256..311].
    """
    best_score, best_positions = evaluate_all_combinations(hand)
    
    # Sort the positions for consistent encoding
    keep_positions: Tuple[int, ...] = tuple(sorted(best_positions))
    return encode_select(keep_positions)  # map that 5‐tuple to [256..311]


def run_one_episode(threshold_rank: int) -> float:
    """
    Run a single hand of EightCardDrawEnv using the two‐phase heuristic:
     1) Discard all cards whose rank < threshold_rank.
     2) From the resulting 8 cards, evaluate all 56 possible 5-card hands 
        and keep the one with the highest score.
    Returns the episode reward (hand score ∈ [0,1]).
    """
    env = EightCardDrawEnv()
    obs, _ = env.reset()
    # Access the raw hand array: env.hand is shape (8,) with card indices 0..51
    hand: np.ndarray = env.hand.copy()

    # ---------------------- Phase 0: Discard ----------------------
    discard_action = make_discard_action(hand, threshold_rank)
    # Step through discard. This draws replacement cards automatically.
    obs2, reward0, terminated0, truncated0, info0 = env.step(discard_action)
    # terminated0 should be False, because we haven't scored yet (phase → 1)
    assert not terminated0

    # ----------------------- Phase 1: Select‐Five -----------------------
    # Now env.hand has 8 cards after the draw.
    hand2: np.ndarray = env.hand.copy()
    select_action = make_select_action(hand2)
    obs3, reward1, terminated1, truncated1, info1 = env.step(select_action)
    # terminated1 should be True, because we just scored
    assert terminated1

    return reward1  # final poker score ∈ [0,1]


def main():
    # Run many episodes and record the rewards
    rewards = []
    for i in range(NUM_EPISODES):
        r = run_one_episode(THRESHOLD_RANK)
        rewards.append(r)
        
        # Print progress every 100 episodes
        if (i + 1) % 100 == 0:
            print(f"Completed {i + 1}/{NUM_EPISODES} episodes...")

    rewards = np.array(rewards, dtype=np.float32)
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    max_reward = float(np.max(rewards))
    min_reward = float(np.min(rewards))

    print(f"\nRan {NUM_EPISODES} episodes with THRESHOLD_RANK = {THRESHOLD_RANK}")
    print(f"Using exhaustive evaluation of all 56 possible 5-card combinations")
    print(f"Average hand score: {avg_reward:.4f}  ± {std_reward:.4f}")
    print(f"Min hand score: {min_reward:.4f}   Max hand score: {max_reward:.4f}")


if __name__ == "__main__":
    main()
