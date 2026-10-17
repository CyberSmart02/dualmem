import json
import logging

import numpy as np
from pydantic import ValidationError

from dualmemory.models import CandidateFact, Episode
from dualmemory.providers import LLM

logger = logging.getLogger(__name__)

# EVALUATION.md §11 initial values; tuned on dev later.
TAU_CLUSTER = 0.8
N_MIN = 3


def cluster_vectors(vectors: np.ndarray, tau: float = TAU_CLUSTER) -> list[list[int]]:
    """Group vectors whose cosine similarity chains together above tau.

    Single-linkage agglomerative clustering with a distance threshold, done via
    union-find: any pair with cos >= tau gets merged. Returns clusters as lists
    of row indices. Dependency-free (no sklearn) and fine at one-conversation scale."""
    n = len(vectors)
    if n == 0:
        return []

    v = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0                 # avoid divide-by-zero on zero vectors
    v = v / norms
    sim = v @ v.T                           # pairwise cosine matrix

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def is_eligible(cluster: list[Episode], linked_episode_ids: set[int],
                n_min: int = N_MIN) -> bool:
    """A cluster may be promoted to a candidate fact only if it is big enough,
    spans at least two source messages, and no member already backs a fact."""
    if len(cluster) < n_min:
        return False
    sources = {m for ep in cluster for m in ep.source_message_ids}
    if len(sources) < 2:
        return False
    return not any(ep.id in linked_episode_ids for ep in cluster)

CONSOLIDATE_PROMPT = """These events from one conversation are closely related.
If together they imply ONE durable fact about a subject (a standing preference,
habit, attribute, or relationship), state it. If they only describe one-off
events, return null.

Events:
{events}

Return ONLY JSON: {{"subject": str, "text": str, "confidence": float}} or null.
"""


class Consolidator:
    """Turns an eligible cluster of episodes into at most one CandidateFact.
    LLM proposes, pydantic validates; anything malformed is skipped and logged."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def propose(self, cluster: list[Episode]) -> CandidateFact | None:
        ordered = sorted(cluster, key=lambda e: (e.event_time, e.id))
        events = "\n".join(
            f"- [{e.event_time.date().isoformat()}] {e.text}" for e in ordered
        )
        raw = self.llm.complete(CONSOLIDATE_PROMPT.format(events=events))
        try:
            parsed = json.loads(raw)
            if parsed is None:
                return None                      # LLM found nothing durable
            return CandidateFact(
                subject=parsed["subject"],
                text=parsed["text"],
                confidence=parsed.get("confidence", 0.7),
                # the fact holds from the LAST supporting event: that is when the
                # pattern was established
                event_time=ordered[-1].event_time,
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError,
                ValidationError) as exc:
            logger.warning("consolidation parse failed, skipping cluster: %s", exc)
            return None