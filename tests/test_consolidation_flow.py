import json
from datetime import datetime

import numpy as np

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeLLM


class TopicEmbedder:
    """Deterministic test embedder: texts containing 'coffee' share one direction,
    everything else goes to another. Gives clusters we control."""
    dim = 4

    def embed(self, text: str) -> np.ndarray:
        if "coffee" in text:
            return np.array([1.0, 0.1, 0.0, 0.0], dtype="float32")
        return np.array([0.0, 0.0, 1.0, 0.1], dtype="float32")


def _extraction(text: str, day: int) -> str:
    ev = f"2026-01-{day:02d}T00:00:00"
    return json.dumps({"episodes": [{"text": text, "event_time": ev}], "facts": []})


def _msg(id: int, content: str, day: int) -> Message:
    return Message(id=id, conversation_id="c1", role="user",
                   content=content, timestamp=datetime(2026, 1, day))


def _mem_with_three_coffee_episodes() -> Memory:
    mem = Memory(embedder=TopicEmbedder(), llm=FakeLLM())
    for i, text in enumerate(["ordered coffee", "coffee with oat milk",
                              "coffee before gym"], start=1):
        mem.extractor.llm = FakeLLM({"Extract memory": _extraction(text, i)})
        mem.add([_msg(i, text, i)])
    return mem


def test_consolidate_promotes_cluster_to_fact():
    mem = _mem_with_three_coffee_episodes()
    mem.consolidator.llm = mem.adjudicator.llm = FakeLLM({
        "closely related": json.dumps(
            {"subject": "user", "text": "drinks coffee regularly", "confidence": 0.8}),
    })
    facts = mem.consolidate("c1")

    assert len(facts) == 1
    assert facts[0].text == "drinks coffee regularly"
    assert facts[0].valid_from == datetime(2026, 1, 3)    # last supporting event
    assert len(mem.sql.edges_for_fact(facts[0].id)) >= 1   # provenance (I2)


def test_consolidate_null_from_llm_creates_nothing():
    mem = _mem_with_three_coffee_episodes()
    mem.consolidator.llm = FakeLLM({"closely related": "null"})
    assert mem.consolidate("c1") == []


def test_consolidate_skips_small_clusters():
    mem = Memory(embedder=TopicEmbedder(), llm=FakeLLM())
    mem.extractor.llm = FakeLLM({"Extract memory": _extraction("ordered coffee", 1)})
    mem.add([_msg(1, "ordered coffee", 1)])
    assert mem.consolidate("c1") == []                     # 1 episode < N_MIN


def test_consolidate_is_idempotent():
    mem = _mem_with_three_coffee_episodes()
    mem.consolidator.llm = mem.adjudicator.llm = FakeLLM({
        "closely related": json.dumps(
            {"subject": "user", "text": "drinks coffee regularly", "confidence": 0.8}),
    })
    first = mem.consolidate("c1")
    second = mem.consolidate("c1")
    assert len(first) == 1
    assert second == []        # cluster now has a linked episode, so it's skipped