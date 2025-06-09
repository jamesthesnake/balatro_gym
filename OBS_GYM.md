# Observation-space Contract  
_All shapes follow NumPy / Gymnasium conventions (row-major).  
Unless noted, `dtype=int32` (Gym’s default for `Box`) or `int8` (for `MultiBinary`)._

---

## 1. `BalatroEnv`  (`balatro_env.py`)

| Key | Space | Shape | dtype | Semantics |
|-----|-------|-------|-------|-----------|
| **deck.cards** | `Box(low=0, high=52)` | `(52,)` | `int32` | Encodes the entire 52-card draw pile; each element is the *index* (0–51) or 0 if slot empty. |
| **deck.cards_played** | `MultiBinary(52)` | `(52,)` | `int8` | Bitmask of cards already seen this round. |
| **hand** | `Box(low=0, high=51)` | `(8,)` | `int32` | Current 8-card hand; unused slots are `0`. |
| **highlighted** | `Box(low=0, high=51)` | `(5,)` | `int32` | Cards currently selected for a “play” action; zero-padded. |
| **blinds** | `Box(low=0, high=10000)` | `(3,)` | `int32` | Chip value of the three blinds (small → big). |
| **blind_index** | `Discrete(3)` | `()` | `int64` | Which blind is active (0, 1, 2). |
| **hands** | `Discrete(10)` | `()` | `int64` | Hands remaining in this round. |
| **discards** | `Discrete(8)` | `()` | `int64` | Discards remaining in this round. |
| **round_score** | `Discrete(100 000)` | `()` | `int64` | Chips earned so far this round. |
| **round_hands** | `Discrete(10)` | `()` | `int64` | Max hands allowed this round. |
| **round_discards** | `Discrete(8)` | `()` | `int64` | Max discards allowed this round. |

---

## 2. `BalatroSmallEnv`  (`balatro_small_env.py`)

| Key | Space | Shape | dtype | Notes |
|-----|-------|-------|-------|-------|
| **deck.cards** | `Box(0, 52)` | `(52,)` | `int32` | Same as **BalatroEnv**. |
| **deck.cards_played** | `MultiBinary(52)` | `(52,)` | `int8` | Same as **BalatroEnv**. |
| **hand** | `Box(0, 51)` | `(8,)` | `int32` | Current hand. |
| **highlighted** | `Box(0, 51)` | `(5,)` | `int32` | Selected cards for play/discard. |
| **round_score** | `Discrete(100 000)` | `()` | `int64` | Chips earned this *mini* round. |
| **round_hands** | `Discrete(10)` | `()` | `int64` | Hands left this round. |
| **round_discards** | `Discrete(8)` | `()` | `int64` | Discards left this round. |

*Differences from* `BalatroEnv`:  
– No blind info (`blinds`, `blind_index`).  
– No global counters (`hands`, `discards`) because the *small* env self-terminates early.  

---

## 3. `EightCardDrawEnv`  (`env.py`)

> A minimal toy env for 8-card draw poker with two phases (discard → select-five).

| Key | Space | Shape | dtype | Meaning |
|-----|-------|-------|-------|---------|
| **cards** | `MultiBinary(8 × 52)` | `(8, 52)` | `int8` | One-hot matrix; row *i* is a 52-dim card vector for card *i* of hand. |
| **phase** | `Discrete(2)` | `()` | `int64` | `0 = discard`, `1 = select-five`. |
| **action_mask** | `MultiBinary(312)` | `(312,)` | `int8` | Mask of legal actions (size = full `Discrete(312)` space). |

---

## Conventions & Tips

* **Card Encoding**  
  *Index* format `(rank × 4 + suit)` where rank ∈ `0-12` (2 … Ace), suit ∈ `0-3` (♣,♦,♥,♠).

* **Empty Slots**  
  `Box` spaces are zero-padded; `0` means “no card”.

* **dtype**  
  Gymnasium uses `np.int8` for `MultiBinary`; `np.int64` for `Discrete`. All `Box` spaces here are `np.int32` to save memory but remain wide enough for 0-52.

* **Extensibility**  
  When you add features (jokers, consumables, shop, etc.), extend this contract and bump a doc version header so downstream agents can detect schema changes.

---

_Last updated : 2025-06-09_
