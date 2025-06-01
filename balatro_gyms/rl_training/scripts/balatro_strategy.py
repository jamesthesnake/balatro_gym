# balatro_strategy.py
from enum import Enum, auto

class Target(Enum):
    ANY      = auto()
    FLUSH    = auto()
    STRAIGHT = auto()
    HIGHPAIR = auto()

_current_target = Target.ANY
_budget         = 4         # discard budget this round

def set_strategy(target: str, budget: int = 4):
    """
    Called once at the *beginning* of a new round.
    target: 'FLUSH' | 'STRAIGHT' | 'HIGHPAIR' | 'ANY'
    budget: 0–4 allowed discards across the whole round
    """
    global _current_target, _budget
    _current_target = Target[target]
    _budget         = budget
    return {"ok": True}

def get_context() -> str:
    """String token appended to every prompt."""
    return f"<STRATEGY={_current_target.name}|BUDGET={_budget}>"

def consume_discards(n: int):
    """Called by discard_cards(); keeps budget in sync."""
    global _budget
    _budget = max(0, _budget - n)

