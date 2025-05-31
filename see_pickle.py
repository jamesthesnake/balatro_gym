import sys
import pickle
import numpy as np
import pathlib
from itertools import combinations

# ────────────────────────────────────────────────────────────────────── #
# Utility: Convert a card ID (0–51) to a string like "A♠"
# ────────────────────────────────────────────────────────────────────── #
RANKS = "23456789TJQKA"
SUITS = ["♠", "♥", "♦", "♣"]

def id_to_card(idx: int) -> str:
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    return f"{rank}{suit}"

def decode_discard(mask: int):
    """Return list of indices (0–7) discarded given a bitmask 0–255."""
    return [i for i in range(8) if (mask >> i) & 1]

def decode_select(action_id: int):
    """Return the 5‐card indices (0–7) chosen by an action in [256..311]."""
    FIVE_CARD_COMBOS = list(combinations(range(8), 5))
    return FIVE_CARD_COMBOS[action_id - 256]


# ────────────────────────────────────────────────────────────────────── #
# Print a single transition
# ────────────────────────────────────────────────────────────────────── #
def print_transition(step_i, transition):
    phase = transition["phase"]
    action = transition["action"]
    reward = transition["reward"]
    done   = transition["done"]
    hand_before = transition["hand_before"]           # np.ndarray of 8 ints
    hand_after  = transition["hand_after"]            # np.ndarray of 8 ints or None
    keep_indices = transition["keep_indices"]         # tuple of 5 ints or None
    balatro_raw_score = transition.get("balatro_raw_score", None)

    # Convert the 8‐card hand_before to human‐readable strings
    hand_before_strs = [id_to_card(int(cid)) for cid in hand_before]

    print(f"      Step {step_i}: phase={phase}, action={action}, reward={reward:.4f}, done={done}")
    print(f"        hand_before: {hand_before_strs}")

    if phase == 0:
        # Phase 0 = discard step
        discard_positions = decode_discard(action)
        if discard_positions:
            discarded_cards = [hand_before_strs[i] for i in discard_positions]
        else:
            discarded_cards = ["None"]
        print(f"        cards_discarded: {discarded_cards}")
        if hand_after is not None:
            hand_after_strs = [id_to_card(int(cid)) for cid in hand_after]
            print(f"        hand_after_discard: {hand_after_strs}")

    else:
        # Phase 1 = selection step
        if keep_indices is not None:
            kept_cards = [hand_before_strs[i] for i in keep_indices]
            print(f"        cards_kept: {kept_cards}")
            print(f"        balatro_raw_score: {balatro_raw_score}")

            # New field: round_score_so_far (always present in Phase 1)
            if "round_score_so_far" in transition:
                rsf = transition["round_score_so_far"]
                print(f"        round_score_so_far: {rsf}")

            # New field: round_pass (only on last hand’s Phase 1)
            if "round_pass" in transition:
                rp = transition["round_pass"]
                print(f"        round_pass: {rp}")


# ────────────────────────────────────────────────────────────────────── #
# Main: accept one or more pickle filenames as command‐line arguments
# ────────────────────────────────────────────────────────────────────── #
def main():
    if len(sys.argv) < 2:
        print("Usage: python see_pickle.py <pickle_file_path> [<pickle_file_path> ...]")
        sys.exit(1)

    for filename in sys.argv[1:]:
        path = pathlib.Path(filename)
        if not path.exists() or not path.is_file():
            print(f"File not found: {filename}\n")
            continue

        print(f"\n=== Reading '{filename}' ===")
        with open(path, "rb") as f:
            try:
                rounds = pickle.load(f)
            except Exception as e:
                print(f"  Error loading pickle '{filename}': {e}\n")
                continue

        if not isinstance(rounds, list) or not rounds:
            print("  Unexpected format (not a non‐empty list). Skipping.\n")
            continue

        # Print up to 5 rounds for brevity
        num_rounds_to_print = min(5, len(rounds))
        for rnd_idx in range(num_rounds_to_print):
            round_data = rounds[rnd_idx]
            print(f"\nRound {rnd_idx} has {len(round_data)} hands:")
            for hand_idx, hand in enumerate(round_data):
                print(f"  Hand {hand_idx} (2 steps):")
                for step_i, transition in enumerate(hand):
                    print_transition(step_i, transition)
                print("    -------------------------")
            print("\n=============================")
        print("")  # blank line after each file

if __name__ == "__main__":
    main()

