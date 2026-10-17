import json
from datetime import datetime

from dualmemory.memory import Memory
from dualmemory.models import Message
from dualmemory.providers import FakeEmbedder, FakeLLM


def _msg(id: int, content: str, day: int) -> Message:
    return Message(id=id, conversation_id="c1", role="user",
                   content=content, timestamp=datetime(2026, 1, day))


def _extraction(fact_text: str, subject: str = "user", day: int = 1) -> str:
    """Canned extraction: one episode + one candidate fact."""
    ev = f"2026-01-{day:02d}T00:00:00"
    return json.dumps({
        "episodes": [{"text": f"said: {fact_text}", "event_time": ev}],
        "facts": [{"subject": subject, "text": fact_text,
                   "confidence": 0.9, "event_time": ev}],
    })


def _mem(responses: dict[str, str]) -> Memory:
    return Memory(embedder=FakeEmbedder(dim=16), llm=FakeLLM(responses))


# I2 — every committed fact has a derived_from edge to an episode.
def test_I2_fact_has_derived_from_edge():
    mem = _mem({"Extract memory": _extraction("is vegetarian")})
    _, facts = mem.add([_msg(1, "I'm vegetarian", 1)])
    assert len(facts) == 1
    edges = mem.sql.edges_for_fact(facts[0].id)
    assert len(edges) >= 1
    assert all(e.kind == "derived_from" for e in edges)


# ADD path — first fact about a subject, nothing to conflict with.
def test_add_creates_new_fact():
    mem = _mem({"Extract memory": _extraction("is vegetarian")})
    _, facts = mem.add([_msg(1, "I'm vegetarian", 1)])
    assert facts[0].text == "is vegetarian"
    assert facts[0].valid_to is None          # currently valid


# SUPERSEDE — a contradicting fact closes the old one (I3: old row kept).
def test_supersede_closes_old_fact_keeps_row():
    # First add: "is vegetarian" (ADD, no existing facts → no adjudication call)
    mem = _mem({"Extract memory": _extraction("is vegetarian", day=1)})
    _, first = mem.add([_msg(1, "I'm vegetarian", 1)])
    old_id = first[0].id

    # Second add: "eats meat" + adjudicator says SUPERSEDE the old fact.
    decision = json.dumps({"action": "SUPERSEDE", "target_id": old_id})
    mem.extractor.llm = mem.adjudicator.llm = FakeLLM({
        "Extract memory": _extraction("eats meat", day=6),
        "ONE action": decision,   # matches ADJUDICATE_PROMPT text
    })
    _, second = mem.add([_msg(2, "actually I eat meat now", 6)])

    # old fact: still present (I3), window closed, points to new
    old = mem.sql.get_fact(old_id)
    assert old is not None                       # NOT deleted (I3)
    assert old.valid_to == datetime(2026, 1, 6)  # closed at change time
    assert old.superseded_by == second[0].id

    # new fact: valid, open window
    assert second[0].text == "eats meat"
    assert second[0].valid_to is None


# EVIDENCE — a restatement adds an edge, creates no new fact.
def test_evidence_adds_edge_no_new_fact():
    mem = _mem({"Extract memory": _extraction("is vegetarian", day=1)})
    _, first = mem.add([_msg(1, "I'm vegetarian", 1)])
    fact_id = first[0].id
    edges_before = len(mem.sql.edges_for_fact(fact_id))

    decision = json.dumps({"action": "EVIDENCE", "target_id": fact_id})
    mem.extractor.llm = mem.adjudicator.llm = FakeLLM({
        "Extract memory": _extraction("is vegetarian", day=5),
        "ONE action": decision,
    })
    _, second = mem.add([_msg(2, "yeah I don't eat meat", 5)])

    # no new fact created; existing fact returned
    assert second[0].id == fact_id
    # but a new derived_from edge was added (more provenance)
    assert len(mem.sql.edges_for_fact(fact_id)) == edges_before + 1


# I6 — only the two edge kinds ever exist.
def test_I6_only_valid_edge_kinds():
    mem = _mem({"Extract memory": _extraction("is vegetarian")})
    mem.add([_msg(1, "I'm vegetarian", 1)])
    assert mem.sql.all_edge_kinds() <= {"derived_from", "relates_to"}

def test_superseded_fact_not_retrieved_at_query_time():
    # vegetarian (Jan 1), then eats meat (Jan 6) → SUPERSEDE
    mem = _mem({"Extract memory": _extraction("is vegetarian", day=1)})
    _, first = mem.add([_msg(1, "I'm vegetarian", 1)])
    old_id = first[0].id

    decision = json.dumps({"action": "SUPERSEDE", "target_id": old_id})
    mem.extractor.llm = mem.adjudicator.llm = FakeLLM({
        "Extract memory": _extraction("eats meat", day=6),
        "ONE action": decision,
    })
    mem.add([_msg(2, "actually I eat meat now", 6)])

    result = mem.search("what does the user eat", token_budget=2000,
                        t_q=datetime(2026, 1, 10))

    # Split the Facts section from the Events section.
    facts_section = result.split("Events:")[0]

    # The valid fact is present; the superseded fact is filtered OUT of Facts.
    assert "eats meat" in facts_section
    assert "is vegetarian" not in facts_section   # stale fact excluded by validity
    # (the historical episode "said: is vegetarian" may still appear under Events —
    #  that's correct: events are immutable, only facts get superseded)