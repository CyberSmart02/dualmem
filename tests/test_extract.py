from datetime import datetime

from dualmemory.extract import Extractor
from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder, FakeLLM

CANNED = """{
  "episodes": [{"text": "ordered coffee", "event_time": "2026-01-03T10:00:00"}],
  "facts": [{"subject": "user", "text": "likes coffee",
             "confidence": 0.8, "event_time": "2026-01-03T10:00:00"}]
}"""


def _msg():
    return [Message(id=1, conversation_id="c", role="user",
                    content="I ordered coffee", timestamp=datetime(2026, 1, 3, 10))]


def test_extract_parses_episodes_and_candidates():
    llm = FakeLLM({"Extract memory": CANNED})
    ex = Extractor(llm)
    episodes, candidates = ex.extract(_msg(), next_episode_id=1)
    assert len(episodes) == 1
    assert episodes[0].text == "ordered coffee"
    assert episodes[0].source_message_ids == [1]
    assert len(candidates) == 1
    assert candidates[0].subject == "user"
    assert candidates[0].confidence == 0.8


def test_extract_handles_garbage_output():
    llm = FakeLLM({"Extract memory": "sorry, I can't do that"})
    ex = Extractor(llm)
    episodes, candidates = ex.extract(_msg(), next_episode_id=1)
    assert episodes == []
    assert candidates == []


def test_extract_skips_one_malformed_item_keeps_rest():
    canned = """{"episodes": [
        {"text": "good one", "event_time": "2026-01-03T10:00:00"},
        {"event_time": "2026-01-03T10:00:00"}
    ], "facts": []}"""
    llm = FakeLLM({"Extract memory": canned})
    ex = Extractor(llm)
    episodes, _ = ex.extract(_msg(), next_episode_id=1)
    assert len(episodes) == 1
    assert episodes[0].text == "good one"


def test_add_stores_episodes_and_facts():
    llm = FakeLLM({"Extract memory": CANNED})
    mem = Memory(embedder=FakeEmbedder(dim=16), llm=llm)
    episodes, facts = mem.add(_msg())

    assert len(episodes) == 1
    assert mem.sql.all_episode_ids() == mem.vectors.ids == {episodes[0].id}

    assert len(facts) == 1
    stored = mem.sql.get_fact(facts[0].id)
    assert stored.text == "likes coffee"
    assert stored.valid_from == datetime(2026, 1, 3, 10)
    assert stored.valid_to is None