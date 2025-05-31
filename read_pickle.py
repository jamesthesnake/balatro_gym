import pickle
import pathlib
from itertools import combinations

import numpy as np

# Utility to convert card ID (0-51) to human-readable string
RANKS = "23456789TJQKA"
SUITS = ["♠", "♥", "♦", "♣"]

def id_to_card(idx: int) -> str:
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    return f"{rank}{suit}"

def decode_discard(mask: int):
    """Return list of indices (0-7) that were discarded given bitmask."""
    return [i for i in range(8) if (mask >> i) & 1]

def decode_select(action_id: int):
    """Return tuple of 5 indices kept for select-five action (256-311)."""
    five_combos = list(combinations(range(8), 5))
    return five_combos[action_id - 256]

def main():
    pickles_dir = pathlib.Path("pickles")
    if not pickles_dir.exists() or not pickles_dir.is_dir():
        print("Directory 'pickles' not found.")
        return

    pickle_files = sorted(pickles_dir.glob("*.pkl"))
    if not pickle_files:
        print("No .pkl files found in 'pickles' directory.")
        return

    for pfile in pickle_files:
        print(f"\n--- Reading '{pfile.name}' ---")
        with open(pfile, "rb") as f:
            try:
                trajectories = pickle.load(f)
            except Exception as e:
                print(f"  Failed to load: {e}")
                continue

        # Process first two episodes if present
        for ep_idx in range(2):
            if ep_idx >= len(trajectories):
                print(f"  Episode {ep_idx} not found in this file.")
                continue

            episode = trajectories[ep_idx]
            print(f"  Episode {ep_idx} has {len(episode)} steps:")

            for step_i, transition in enumerate(episode):
                hand_before = transition["hand_before"]  # shape (8,), ints
                phase = transition["phase"]
                action = transition["action"]
                reward = transition["reward"]
                hand_after = transition["hand_after"]    # shape (8,) or None
                keep_indices = transition["keep_indices"]  # tuple or None
                done = transition["done"]

                # Convert hand_before IDs to strings
                hand_before_strs = [id_to_card(int(cid)) for cid in hand_before]

                print(f"    Step {step_i}: phase={phase}, action={action}, reward={reward:.4f}, done={done}")
                print(f"      Hand before: {hand_before_strs}")

                if phase == 0:
                    # Discard step
                    discarded_positions = decode_discard(action)
                    discarded_cards = [hand_before_strs[i] for i in discarded_positions] if discarded_positions else ["None"]
                    print(f"      Cards discarded (indices): {discarded_positions} → {discarded_cards}")
                    if hand_after is not None:
                        hand_after_strs = [id_to_card(int(cid)) for cid in hand_after]
                        print(f"      Hand after discard: {hand_after_strs}")
                else:
                    # Select-five step
                    if keep_indices is not None:
                        kept_cards = [id_to_card(int(hand_before[i])) for i in keep_indices]
                        print(f"      Cards kept (indices): {keep_indices} → {kept_cards}")

            print("    --------------------------")

if __name__ == "__main__":
    main()

