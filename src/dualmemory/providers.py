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