import numpy as np
import pickle
from datetime import datetime

# Ensure our patch is applied before we import EpochCardDrawEnv
import balatro_gym.patch_balatro_env  # applies the monkey‐patch to use BalatroGame scoring

from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import decode_discard, decode_select
from balatro_gym.score_with_balatro import int_to_card  # to convert idx→Card object
from balatro_gym.balatro_game import BalatroGame

# ──────────────────────────────────────────────────────────────────────── #
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────── #
NUM_ROUNDS = 500000           # how many rounds (each round = up to 4 hands) to collect
HANDS_PER_ROUND = 4          # define a round as exactly 4 hands
DISCARDS_PER_ROUND = 4       # total discard budget across HANDS_PER_ROUND hands
PASS_THRESHOLD = 300         # raw‐chip score threshold for pass/fail at the end of a round

# ──────────────────────────────────────────────────────────────────────── #
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────── #
def compute_raw_chip_value(card_ids: np.ndarray) -> int:
    """
    Given a length-5 array of card IDs (0..51), convert each to a Card object
    and call BalatroGame._evaluate_hand(...) to get the raw chip integer.
    """
    cards = [int_to_card(int(idx)) for idx in card_ids]
    return BalatroGame._evaluate_hand(cards)


def collect_trajectories(num_rounds: int, hands_per_round: int, discard_budget: int):
    """
    Run `num_rounds` rounds, where each round is exactly `hands_per_round` independent hands
    using a random-but-budgeted discard policy on EightCardDrawEnv. Each hand still consists
    of two phases:
      - Phase 0: discard-step
      - Phase 1: select-step (and raw-chip evaluation)

    We re-instantiate the env before each hand so that the deck is reset. We also enforce that
    across the entire round, you may discard at most `discard_budget` cards total.

    Returns:
        List[List[List[ dict ]]], where:
          - Outer list: rounds (length = num_rounds).
          - For each round: list of exactly `hands_per_round` hands.
          - For each hand: a list of exactly 2 transition-dicts:
            {
              "hand_before":         np.ndarray(8,) of card IDs 0..51,
              "phase":               int 0 or 1,
              "action":              int in [0..311],
              "reward":              float normalized [0,1],
              "hand_after":          np.ndarray(8,) of card IDs 0..51 OR None,
              "keep_indices":        tuple of 5 ints OR None,
              "balatro_raw_score":   int chip value OR None,
              "num_discards":        int (# cards discarded in this action) OR 0 if Phase 1,
              "done":                bool,

              # ADDED FIELDS (in Phase 1 transition only):
              "round_score_so_far":  int (sum of balatro_raw_score for all completed hands so far),
              "round_pass":         bool (only present on the very last hand of the round)
            }
    """
    all_rounds = []

    for rnd in range(num_rounds):
        remaining_discards = discard_budget
        round_trajectories = []
        round_raw_score = 0  # Running total of raw chip values within this round

        for hand_idx in range(hands_per_round):
            # Initialize a fresh env so the deck is reset for each hand
            env = EightCardDrawEnv()
            obs, _ = env.reset()

            hand_transitions = []
            done = False

            while not done:
                phase = int(obs["phase"])
                mask = obs["action_mask"]  # shape (312,)

                if phase == 0:
                    # ── Phase 0: Choose a discard action, subject to remaining_budget ──
                    valid_actions = np.flatnonzero(mask == 1)  # indices of valid Phase 0 moves

                    # Filter by how many cards each action would discard
                    filtered = []
                    for a in valid_actions:
                        discard_indices = decode_discard(int(a))
                        num_to_discard = len(discard_indices)
                        if num_to_discard <= remaining_discards:
                            filtered.append(a)

                    # If no action fits the budget, force “discard zero” if available
                    if not filtered:
                        for a in valid_actions:
                            if len(decode_discard(int(a))) == 0:
                                filtered = [a]
                                break

                    action = int(np.random.choice(filtered))
                    num_discarded = len(decode_discard(action))
                    remaining_discards -= num_discarded

                else:
                    # ── Phase 1: Select which 5 cards to keep (no discard cost) ──
                    valid_actions = np.flatnonzero(mask == 1)
                    action = int(np.random.choice(valid_actions))
                    num_discarded = 0

                # ── Step the environment ─────────────────────────────────────────
                hand_before = env.hand.copy()  # np.ndarray of shape (8,)
                next_obs, reward, done, truncated, info = env.step(action)
                hand_after = env.hand.copy() if phase == 0 else None

                keep_indices = None
                balatro_raw_score = None
                if phase == 1:
                    keep_indices = decode_select(action)
                    kept_ids = env.hand[list(keep_indices)]  # shape (5,)
                    balatro_raw_score = compute_raw_chip_value(kept_ids)
                    round_raw_score += balatro_raw_score  # add to round total

                transition = {
                    "hand_before":       hand_before,        # np.ndarray(8,)
                    "phase":             phase,              # 0 or 1
                    "action":            action,             # int 0..311
                    "reward":            reward,             # float normalized [0,1]
                    "hand_after":        hand_after,         # np.ndarray(8,) or None
                    "keep_indices":      keep_indices,       # tuple of length 5 or None
                    "balatro_raw_score": balatro_raw_score,  # int or None
                    "num_discards":      num_discarded,      # how many cards we discarded
                    "done":              done                # bool
                }

                hand_transitions.append(transition)
                obs = next_obs

            # ── End of one hand (two transitions) ──
            # Add “round_score_so_far” to the Phase 1 transition (index 1)
            # Phase 1 is always the second transition in hand_transitions.
            phase1_transition = hand_transitions[1]
            phase1_transition["round_score_so_far"] = round_raw_score

            # If this is the last hand of the round, also record pass/fail
            if hand_idx == hands_per_round - 1:
                phase1_transition["round_pass"] = (round_raw_score > PASS_THRESHOLD)

            round_trajectories.append(hand_transitions)

        # ── End of a round (HANDS_PER_ROUND hands) ──
        all_rounds.append(round_trajectories)

    return all_rounds


# ──────────────────────────────────────────────────────────────────────── #
# MAIN & SAVE
# ──────────────────────────────────────────────────────────────────────── #
def main():
    # 1) Collect trajectories for NUM_ROUNDS rounds
    trajectories = collect_trajectories(NUM_ROUNDS, HANDS_PER_ROUND, DISCARDS_PER_ROUND)
    print(f"Collected {len(trajectories)} rounds; each round has {HANDS_PER_ROUND} hands.")

    # 2) Build a timestamped filename under pickles/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pickles/rounds_{timestamp}.pkl"

    # 3) Save entire list of rounds as pickle
    with open(filename, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved rounds to {filename}")


if __name__ == "__main__":
    main()

