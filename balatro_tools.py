#!/usr/bin/env python3
"""
Utility layer exposing BalatroGame functions as callable tools.
* Adds keep_cards() that returns raw-chip reward.
* Automatically resets global GAME when round ends.
"""
from __future__ import annotations
from typing import List, Tuple
import itertools
from balatro_game import BalatroGame, Card

# ── global self-renewing game ------------------------------------------
def _new_game() -> BalatroGame:
    print("[BalatroTools]  >>> NEW GAME <<<")
    return BalatroGame()

GAME: BalatroGame = _new_game()

# ── helpers -------------------------------------------------------------
RANK_CODE = {
    Card.Ranks.TWO: "2", Card.Ranks.THREE: "3", Card.Ranks.FOUR: "4",
    Card.Ranks.FIVE: "5", Card.Ranks.SIX: "6",  Card.Ranks.SEVEN: "7",
    Card.Ranks.EIGHT: "8", Card.Ranks.NINE: "9", Card.Ranks.TEN: "T",
    Card.Ranks.JACK: "J", Card.Ranks.QUEEN: "Q", Card.Ranks.KING: "K",
    Card.Ranks.ACE: "A",
}
SUIT_CODE = {
    Card.Suits.SPADES: "S",
    Card.Suits.CLUBS: "C",
    Card.Suits.HEARTS: "H",
    Card.Suits.DIAMONDS: "D",
}
def _code(idx: int) -> str:
    c = GAME.deck[idx]
    return f"{RANK_CODE[c.rank]}{SUIT_CODE[c.suit]}"

def _reset_if_end():
    global GAME
    if GAME.state in {BalatroGame.State.WIN, BalatroGame.State.LOSS}:
        GAME = _new_game()

# ── tool: read current 8-card hand -------------------------------------
def check_cards() -> str:
    return " ".join(_code(i) for i in GAME.hand_indexes)

# ── tool: brute-force best 5-card subsets ------------------------------
def get_optimal_pairs(top_k: int = 3) -> List[Tuple[List[int], int]]:
    best = []
    for keep in itertools.combinations(range(8), 5):
        cards = [GAME.deck[GAME.hand_indexes[i]] for i in keep]
        raw = BalatroGame._evaluate_hand(cards)
        best.append((list(keep), raw))
    return sorted(best, key=lambda t: t[1], reverse=True)[:top_k]

# ── tool: discard and redraw -------------------------------------------
def discard_cards(indices: List[int]) -> str:
    for pos in sorted(indices, reverse=True):
        deck_idx = GAME.hand_indexes.pop(pos)
        GAME.deck[deck_idx].played = False
    GAME._draw_cards()
    return check_cards()

# ── tool: keep five cards, compute reward & maybe reset -----------------
def keep_cards(indices: List[int]) -> dict:
    cards = [GAME.deck[GAME.hand_indexes[i]] for i in indices]
    raw = BalatroGame._evaluate_hand(cards)
    GAME.round_score += raw
    GAME.round_hands -= 1
    if GAME.round_score >= GAME.blinds[GAME.blind_index]:
        GAME._end_round()
    elif GAME.round_hands == 0:
        GAME.state = BalatroGame.State.LOSS
        _reset_if_end()
    summary = {
        "kept": " ".join(_code(i) for i in indices),
        "raw_score": raw,
        "round_score": GAME.round_score,
        "state": GAME.state.name,
    }
    return summary

