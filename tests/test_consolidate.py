from datetime import datetime

import numpy as np

from dualmemory.consolidate import cluster_vectors, is_eligible
from dualmemory.models import Episode


def _ep(id, sources):
    return Episode(id=id, conversation_id="c", text=f"e{id}",
                   event_time=datetime(2026, 1, id), source_message_ids=sources)


def test_cluster_groups_similar_and_separates_dissimilar():
    vecs = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.05, 0.0],   # near the first
        [0.98, 0.0, 0.05],   # near the first
        [0.0, 1.0, 0.0],     # orthogonal
    ])
    clusters = cluster_vectors(vecs, tau=0.9)
    assert sorted(sorted(c) for c in clusters) == [[0, 1, 2], [3]]


def test_cluster_empty():
    assert cluster_vectors(np.zeros((0, 3))) == []


def test_eligible_big_enough_multi_source_unlinked():
    cluster = [_ep(1, [10]), _ep(2, [11]), _ep(3, [12])]
    assert is_eligible(cluster, linked_episode_ids=set(), n_min=3)


def test_not_eligible_too_small():
    cluster = [_ep(1, [10]), _ep(2, [11])]
    assert not is_eligible(cluster, set(), n_min=3)


def test_not_eligible_single_source_message():
    cluster = [_ep(1, [10]), _ep(2, [10]), _ep(3, [10])]
    assert not is_eligible(cluster, set(), n_min=3)    # one observation x3


def test_not_eligible_if_already_linked():
    cluster = [_ep(1, [10]), _ep(2, [11]), _ep(3, [12])]
    assert not is_eligible(cluster, linked_episode_ids={2}, n_min=3)