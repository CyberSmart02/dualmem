import faiss
import numpy as np


class VectorStore:
    """FAISS index over one memory type. Stores (id -> vector), returns nearest ids.

    Knows nothing about episodes/facts — pure vectors in, ids + scores out.
    This is a rebuildable cache; SQLite is the source of truth.
    """

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))

    def _prep(self, vec: np.ndarray) -> np.ndarray:
        v = np.asarray(vec, dtype="float32")
        if v.ndim == 1:
            v = v.reshape(1, -1)
        faiss.normalize_L2(v)
        return v

    def add(self, id_: int, vector: np.ndarray) -> None:
        v = self._prep(vector)
        ids = np.array([id_], dtype="int64")
        self._index.add_with_ids(v, ids)

    def search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if self._index.ntotal == 0:
            return []
        q = self._prep(query_vec)
        k = min(k, self._index.ntotal)
        scores, ids = self._index.search(q, k)
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def remove(self, id_: int) -> None:
        self._index.remove_ids(faiss.IDSelectorArray(np.array([id_], dtype="int64")))

    @property
    def ids(self) -> set[int]:
        """All ids currently in the index — used to check SQLite/FAISS agreement (I7)."""
        # IndexIDMap stores ids in id_map; reconstruct as a set.
        return set(faiss.vector_to_array(self._index.id_map).tolist())

    def reset(self) -> None:
        """Empty the index (used by rebuild_index)."""
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))