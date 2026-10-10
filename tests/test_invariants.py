from datetime import datetime

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder


def _msg(id: int, content: str, day: int) -> Message:
    return Message(id=id, conversation_id="c1", role="user",
                   content=content, timestamp=datetime(2026, 1, day))


def _mem() -> Memory:
    return Memory(embedder=FakeEmbedder(dim=16))


# I1 — every episode has >=1 source message.
# (Skeleton: add() always sets one. We assert the produced episodes satisfy it.)
def test_I1_every_episode_has_a_source_message():
    mem = _mem()
    eps = mem.add([_msg(1, "hello", 1), _msg(2, "world", 2)])
    assert all(len(e.source_message_ids) >= 1 for e in eps)


# I7 — the SQLite id set and the FAISS id set match.
def test_I7_sqlite_and_faiss_ids_match():
    mem = _mem()
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2), _msg(3, "c", 3)])
    assert mem.sql.all_episode_ids() == mem.vectors.ids


# I7 recovery — after FAISS drifts (manually emptied), rebuild_index restores it.
def test_rebuild_index_recovers_faiss_from_sqlite():
    mem = _mem()
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2)])
    mem.vectors.reset()                      # simulate FAISS drift / loss
    assert mem.vectors.ids == set()          # FAISS now empty
    assert mem.sql.all_episode_ids() == {1, 2}   # SQLite still has truth

    mem.rebuild_index()                      # recover
    assert mem.sql.all_episode_ids() == mem.vectors.ids == {1, 2}


# Delete keeps the two stores consistent (I7 holds after removal).
def test_delete_keeps_stores_consistent():
    mem = _mem()
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2)])
    mem.sql.delete_episode(1)
    mem.vectors.remove(1)
    assert mem.sql.all_episode_ids() == mem.vectors.ids == {2}