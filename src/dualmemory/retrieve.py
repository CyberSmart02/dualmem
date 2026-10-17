import math
from dataclasses import dataclass
from datetime import datetime

from dualmemory.models import Episode, Fact


@dataclass
class ScoredItem:
    """A fact or episode with its retrieval score, unified for packing.
    kind distinguishes them for the render step; the scorer treats them alike."""
    kind: str          # 'fact' | 'episode'
    id: int
    text: str
    time: datetime     # event_time (episode) or valid_from (fact)
    score: float
    tokens: int


# Default scoring weights (EVALUATION.md §11 initial values; tuned on dev later)
ALPHA = 0.7    # similarity weight
BETA = 0.2     # recency weight
GAMMA = 0.1    # importance weight
TAU_R_DAYS = 30.0


def _recency(t_q: datetime, event_time: datetime, tau_days: float) -> float:
    """exp(-Δdays / tau). 1.0 at t_q, decays with age. Clamped at 0 for future."""
    delta_days = (t_q - event_time).total_seconds() / 86400.0
    if delta_days < 0:
        return 1.0                      # shouldn't happen post-filter; be safe
    return math.exp(-delta_days / tau_days)


def score_item(sim: float, t_q: datetime, event_time: datetime,
               importance: float, *, alpha=ALPHA, beta=BETA, gamma=GAMMA,
               tau_days=TAU_R_DAYS) -> float:
    """R4: weighted blend of similarity, recency, importance."""
    return (alpha * sim
            + beta * _recency(t_q, event_time, tau_days)
            + gamma * importance)


def fact_is_valid_at(fact: Fact, t_q: datetime) -> bool:
    """R3 validity: valid_from <= t_q and (valid_to is None or t_q < valid_to)."""
    if fact.valid_from > t_q:
        return False
    if fact.valid_to is not None and t_q >= fact.valid_to:
        return False
    return True


def episode_is_visible_at(ep: Episode, t_q: datetime) -> bool:
    """R3: an episode can't be retrieved before it happened."""
    return ep.event_time <= t_q


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Real tokenizer (tiktoken) at eval."""
    return max(1, len(text) // 4)


def pack(items: list[ScoredItem], token_budget: int) -> list[ScoredItem]:
    """R6: greedy by score/tokens, take best value-per-token until budget full."""
    ranked = sorted(items, key=lambda it: it.score / it.tokens, reverse=True)
    selected: list[ScoredItem] = []
    used = 0
    for it in ranked:
        if used + it.tokens <= token_budget:
            selected.append(it)
            used += it.tokens
    return selected


def render(items: list[ScoredItem]) -> str:
    """R7 frozen template: Facts first, then Events (chronological). Ties by id."""
    facts = [it for it in items if it.kind == "fact"]
    events = [it for it in items if it.kind == "episode"]

    lines: list[str] = []
    if facts:
        lines.append("Facts:")
        for it in sorted(facts, key=lambda i: (i.time, i.id)):
            lines.append(f"- {it.text} (since {it.time.date().isoformat()})")
    if events:
        lines.append("Events:")
        for it in sorted(events, key=lambda i: (i.time, i.id)):
            lines.append(f"- [{it.time.date().isoformat()}] {it.text}")
    return "\n".join(lines) if lines else "Facts:\n(none)\nEvents:\n(none)"