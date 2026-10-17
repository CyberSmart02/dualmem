from typing import Protocol

import numpy as np


class Embedder(Protocol):
    """Contract: anything that turns text into a fixed-length vector."""
    dim: int

    def embed(self, text: str) -> np.ndarray: ...


class FakeEmbedder:
    """Deterministic, offline embedder for tests. No API, no model.

    Hashes text into a reproducible vector so the same text always
    maps to the same point, and similar calls are stable across runs.
    Satisfies the Embedder Protocol structurally (no inheritance).
    """

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def embed(self, text: str) -> np.ndarray:
        # Seed an RNG from the text hash → deterministic vector per text.
        seed = abs(hash(text)) % (2**32)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dim).astype("float32")
        return vec

class LLM(Protocol):
    """Contract for any text model: prompt in, text out.
    Core code depends on this, not a concrete model — so fake now, real later,
    with zero changes to extraction/adjudication logic."""

    def complete(self, prompt: str) -> str: ...


class FakeLLM:
    """Offline LLM for tests. Returns a canned reply chosen by matching a
    substring of the prompt; defaults to an empty JSON list.

    Satisfies the LLM Protocol structurally (no inheritance). Lets the whole
    extract -> parse -> validate -> store pipe run deterministically, no API."""

    def __init__(self, responses: dict[str, str] | None = None) -> None:
        # map: substring to look for in the prompt -> canned reply to return
        self._responses = responses or {}

    def complete(self, prompt: str) -> str:
        for key, reply in self._responses.items():
            if key in prompt:
                return reply
        return "[]"   # default: empty JSON list (nothing extracted)