import json
from datetime import datetime

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder, FakeLLM


def _mem(episode_texts: list[str]) -> Memory:
    eps = [{"text": t, "event_time": "2026-01-01T00:00:00"} for t in episode_texts]
    llm = FakeLLM({"Extract memory": json.dumps({"episodes": eps, "facts": []})})
    return Memory(embedder=FakeEmbedder(dim=16), llm=llm)


def _msg(id: int, content: str, day: int) -> Message:
    return Message(id=id, conversation_id="c1", role="user",
                   content=content, timestamp=datetime(2026, 1, day))


def test_add_extracts_and_search_renders():
    mem = _mem(["flat white coffee ordered", "rainy weekend"])
    episodes, _ = mem.add([_msg(1, "I ordered coffee", 1),
                           _msg(2, "weather was rainy", 2)])
    assert len(episodes) == 2

    result = mem.search("coffee", k=2)
    assert "Events:" in result
    assert "flat white coffee ordered" in result


def test_search_on_empty_memory():
    mem = _mem([])
    result = mem.search("anything", k=5)
    assert result == "Events:\n(none)"