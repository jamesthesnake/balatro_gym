import numpy as np
from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import encode_select
from typing import List, Tuple

# ------------------------------------------------------------------
# Heuristic parameters
# ------------------------------------------------------------------
# We’ll discard cards whose “rank index” is below this threshold.
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


def make_select_action(hand: np.ndarray) -> int:
    """
    Given `hand`, a length‐8 np.ndarray of card indices after drawing new cards,
    select the 5 highest-ranked cards and return the appropriate action ID [256..311].
    We break ties by taking lower index first if ranks are equal.
    """
    ranks = hand % 13  # array of 8 rank‐indices
    # Sort positions 0..7 by rank descending, tie‐break on card index ascending
    sorted_positions = sorted(range(8), key=lambda i: (ranks[i], hand[i]), reverse=True)
    # Take the top 5 positions
    keep_positions: Tuple[int, ...] = tuple(sorted(sorted_positions[:5]))
    return encode_select(keep_positions)  # map that 5‐tuple to [256..311]


def run_one_episode(threshold_rank: int) -> float:
    """
    Run a single hand of EightCardDrawEnv using the two‐phase heuristic:
     1) Discard all cards whose rank < threshold_rank.
     2) From the resulting 8 cards, keep the 5 highest‐ranked cards to score.
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
    # terminated0 should be False, because we haven’t scored yet (phase → 1)
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
    for _ in range(NUM_EPISODES):
        r = run_one_episode(THRESHOLD_RANK)
        rewards.append(r)

    rewards = np.array(rewards, dtype=np.float32)
    avg_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    max_reward = float(np.max(rewards))
    min_reward = float(np.min(rewards))

    print(f"Ran {NUM_EPISODES} episodes with THRESHOLD_RANK = {THRESHOLD_RANK}")
    print(f"Average hand score: {avg_reward:.4f}  ± {std_reward:.4f}")
    print(f"Min hand score: {min_reward:.4f}   Max hand score: {max_reward:.4f}")


if __name__ == "__main__":
    main()

