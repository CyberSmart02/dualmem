from datetime import datetime

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder, FakeLLM


def _llm_for(texts: list[str]) -> FakeLLM:
    """A FakeLLM whose extraction returns one episode per given text."""
    import json
    eps = [{"text": t, "event_time": "2026-01-01T00:00:00"} for t in texts]
    payload = json.dumps({"episodes": eps, "facts": []})
    return FakeLLM({"Extract memory": payload})


def _msg(id: int, content: str, day: int) -> Message:
    return Message(id=id, conversation_id="c1", role="user",
                   content=content, timestamp=datetime(2026, 1, day))


def _mem(texts: list[str]) -> Memory:
    return Memory(embedder=FakeEmbedder(dim=16), llm=_llm_for(texts))


def test_I1_every_episode_has_a_source_message():
    mem = _mem(["hello", "world"])
    eps, _ = mem.add([_msg(1, "hello", 1), _msg(2, "world", 2)])
    assert all(len(e.source_message_ids) >= 1 for e in eps)


def test_I7_sqlite_and_faiss_ids_match():
    mem = _mem(["a", "b", "c"])
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2), _msg(3, "c", 3)])
    assert mem.sql.all_episode_ids() == mem.vectors.ids


def test_rebuild_index_recovers_faiss_from_sqlite():
    mem = _mem(["a", "b"])
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2)])
    ids_before = mem.sql.all_episode_ids()
    mem.vectors.reset()
    assert mem.vectors.ids == set()
    assert mem.sql.all_episode_ids() == ids_before
    mem.rebuild_index()
    assert mem.sql.all_episode_ids() == mem.vectors.ids == ids_before


def test_delete_keeps_stores_consistent():
    mem = _mem(["a", "b"])
    mem.add([_msg(1, "a", 1), _msg(2, "b", 2)])
    some_id = next(iter(mem.sql.all_episode_ids()))
    mem.sql.delete_episode(some_id)
    mem.vectors.remove(some_id)
    assert mem.sql.all_episode_ids() == mem.vectors.ids