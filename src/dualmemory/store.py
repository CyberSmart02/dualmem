import faiss
import numpy as np

from dualmemory.models import Episode


class EpisodeStore:
    """Holds episodes and their vectors; returns nearest episodes for a query.

    Skeleton version: episodes live in a dict (later: SQLite), vectors in a
    FAISS index. SQLite-as-source-of-truth comes later; for now the dict is it.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._episodes: dict[int, Episode] = {}
        # IndexFlatIP = exact inner product; IndexIDMap = we own the ids.
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def _prep(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype="float32")
        if v.ndim == 1:
            v = v.reshape(1, -1)          # single vector -> (1, dim)
        faiss.normalize_L2(v)             # inner product == cosine
        return v

    def add(self, episode: Episode, vector: np.ndarray) -> None:
        self._episodes[episode.id] = episode
        v = self._prep(vector)
        ids = np.array([episode.id], dtype="int64")
        self._index.add_with_ids(v, ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[Episode, float]]:
        if self._index.ntotal == 0:
            return []
        q = self._prep(query_vec)
        k = min(k, self._index.ntotal)             # can't ask for more than exist
        scores, ids = self._index.search(q, k)
        results: list[tuple[Episode, float]] = []
        for eid, score in zip(ids[0], scores[0]):
            if eid == -1:                          # empty slot, skip
                continue
            results.append((self._episodes[int(eid)], float(score)))
        return results