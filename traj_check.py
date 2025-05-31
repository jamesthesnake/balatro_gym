import numpy as np
import balatro_gym.patch_balatro_env   # this applies the patch immediately

from balatro_gym.env import EightCardDrawEnv
from gymnasium import ObservationWrapper, spaces

# ───── 1) Helper: Convert card index [0..51] → human string "A♠", "T♦", etc. ─────
RANKS = "23456789TJQKA"
SUITS  = ["♠", "♥", "♦", "♣"]

def card_to_str(idx: int) -> str:
    rank = RANKS[idx % 13]
    suit = SUITS[idx // 13]
    return f"{rank}{suit}"

# ───── 2) Wrapper so that the env returns ONLY the 8‐card “cards” array ─────
class CardsOnlyWrapper(ObservationWrapper):
    """
    Wrap an EightCardDrawEnv so that:
      - reset()/step() return only obs["cards"] (shape = (8, 52)).
      - observation_space is set accordingly.
    """
    def __init__(self, env):
        super().__init__(env)
        # Grab one sample to inspect “cards” shape & dtype
        example_obs, _ = env.reset()
        cards_arr = example_obs["cards"]  # shape (8, 52)
        # Build a new observation_space: Box matching (8, 52)
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=cards_arr.shape,  # (8, 52)
            dtype=cards_arr.dtype,
        )

    def observation(self, obs_dict):
        # Only return the “cards” field, discarding “phase” and “action_mask”
        return obs_dict["cards"]


# ───── 3) Build a single wrapped environment ─────
env = CardsOnlyWrapper(EightCardDrawEnv())

# ───── 4) Simple smoke test: reset, random action, step ─────
obs, _ = env.reset()

# Before anything else, convert the initial 8 cards (one‐hot) into card IDs then strings
initial_ids = env.env.hand.copy()  # shape (8,)
initial_strs = [card_to_str(c) for c in initial_ids]
print("\n=== Smoke Test: One Random Step Each Phase ===")
print("Initial 8 cards:", initial_strs)

# Phase 0: discard
phase = env.env.phase  # should be 0 after reset
action_mask = env.env._action_mask()  # shape (312,)
valid_indices = np.flatnonzero(action_mask)
discard_action = int(np.random.choice(valid_indices))
print(f"Phase 0: Discard action = {discard_action}")

# Decode which positions (0–7) were thrown away
discard_positions = [i for i in range(8) if (discard_action >> i) & 1]
discarded_cards = [initial_strs[i] for i in discard_positions] if discard_positions else ["None"]
print("  Cards discarded:", discarded_cards)

# Step through Phase 0
next_obs, reward0, done0, truncated0, info0 = env.step(discard_action)
assert reward0 == 0.0 and not done0
print("  Reward after discard (always 0):", reward0)

# Now env.env.hand has the new 8 cards after drawing replacements
hand_after_ids = env.env.hand.copy()
hand_after_strs = [card_to_str(c) for c in hand_after_ids]
print("  New 8 cards:", hand_after_strs)

# Phase 1: select‐five
phase = env.env.phase  # should now be 1
action_mask = env.env._action_mask()
valid_indices = np.flatnonzero(action_mask)
select_action = int(np.random.choice(valid_indices))
print(f"Phase 1: Select action = {select_action}")

# Decode which 5 positions (0–7) are kept
from balatro_gym.actions import decode_select
keep_positions = decode_select(select_action)
kept_cards = [hand_after_strs[i] for i in keep_positions]
print("  Cards kept (final 5):", kept_cards)

# Step through Phase 1
final_obs, reward1, done1, truncated1, info1 = env.step(select_action)
print("  Final normalized poker score (reward):", f"{reward1:.4f}")
print("  Done:", done1)
print("=== End of Smoke Test ===\n")


# ───── 5) (Optional) Run a short random rollout of up to 10 steps ─────
print("=== Short Random Rollout ===")
env.reset()
for t in range(10):
    phase = env.env.phase
    mask = env.env._action_mask()
    action = int(np.random.choice(np.flatnonzero(mask)))

    if phase == 0:
        # Phase 0: show 8 cards before discard
        before_ids = env.env.hand.copy()
        before_strs = [card_to_str(c) for c in before_ids]
        discard_positions = [i for i in range(8) if (action >> i) & 1]
        discard_strs = [before_strs[i] for i in discard_positions] if discard_positions else ["None"]
        print(f"Step {t:02d} | Phase 0 (Discard) | Before: {before_strs}")
        print(f"             → Discard action = {action}, discarding {discard_strs}")

    else:
        # Phase 1: show 8 cards before selecting five
        before_ids = env.env.hand.copy()
        before_strs = [card_to_str(c) for c in before_ids]
        keep_positions = decode_select(action)
        keep_strs = [before_strs[i] for i in keep_positions]
        print(f"Step {t:02d} | Phase 1 (Select)  | Before: {before_strs}")
        print(f"             → Select action = {action}, keeping {keep_strs}")

    obs_t, rew_t, term_t, trunc_t, _ = env.step(action)
    print(f"         Reward: {rew_t:.4f}, Done: {term_t}\n")

    if term_t:
        break

print("=== End of Rollout ===")

