from datetime import datetime

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder


def _msg(id: int, content: str, day: int) -> Message:
    return Message(
        id=id,
        conversation_id="conv-1",
        role="user",
        content=content,
        timestamp=datetime(2026, 1, day),
    )


def test_add_then_search_returns_relevant_episode():
    mem = Memory(embedder=FakeEmbedder(dim=16))

    messages = [
        _msg(1, "I ordered a flat white coffee this morning", day=1),
        _msg(2, "The weather was rainy all weekend", day=2),
    ]
    episodes = mem.add(messages)

    # add() created one episode per message
    assert len(episodes) == 2

    # search returns a rendered Events block containing stored content
    result = mem.search("coffee", k=2)
    assert "Events:" in result
    assert "flat white coffee" in result


def test_search_on_empty_memory():
    mem = Memory(embedder=FakeEmbedder(dim=16))
    result = mem.search("anything", k=5)
    assert result == "Events:\n(none)"