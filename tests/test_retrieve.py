from datetime import datetime

from dualmemory.models import Episode, Fact
from dualmemory.retrieve import (
    ScoredItem,
    episode_is_visible_at,
    fact_is_valid_at,
    pack,
    render,
)


def _fact(id, text, vf, vt=None):
    return Fact(id=id, conversation_id="c", subject="user", text=text,
                confidence=1.0, valid_from=vf, valid_to=vt,
                created_at=datetime(2026, 1, 1))


# --- R3 validity: the core payoff ------------------------------------
def test_valid_fact_at_query_time():
    f = _fact(1, "is vegetarian", datetime(2026, 1, 1))     # open window
    assert fact_is_valid_at(f, datetime(2026, 3, 1))        # valid later


def test_superseded_fact_excluded_after_window_closes():
    # vegetarian, but window closed on Feb 1 (superseded)
    f = _fact(1, "is vegetarian", datetime(2026, 1, 1), vt=datetime(2026, 2, 1))
    assert fact_is_valid_at(f, datetime(2026, 1, 15))       # valid before change
    assert not fact_is_valid_at(f, datetime(2026, 3, 1))    # EXCLUDED after change


def test_fact_not_valid_before_it_started():
    f = _fact(1, "is vegetarian", datetime(2026, 2, 1))
    assert not fact_is_valid_at(f, datetime(2026, 1, 1))    # before valid_from


def test_episode_not_visible_before_it_happened():
    ep = Episode(id=1, conversation_id="c", text="x",
                 event_time=datetime(2026, 2, 1), source_message_ids=[1])
    assert episode_is_visible_at(ep, datetime(2026, 3, 1))
    assert not episode_is_visible_at(ep, datetime(2026, 1, 1))


# --- R6 packing -------------------------------------------------------
def test_pack_respects_budget():
    items = [
        ScoredItem("fact", 1, "a", datetime(2026, 1, 1), score=0.9, tokens=100),
        ScoredItem("fact", 2, "b", datetime(2026, 1, 1), score=0.8, tokens=100),
        ScoredItem("fact", 3, "c", datetime(2026, 1, 1), score=0.7, tokens=100),
    ]
    packed = pack(items, token_budget=250)   # only 2 fit
    assert len(packed) == 2
    assert {it.id for it in packed} == {1, 2}   # highest score/token first


def test_pack_empty():
    assert pack([], token_budget=1000) == []


# --- R7 render --------------------------------------------------------
def test_render_facts_before_events():
    items = [
        ScoredItem("episode", 1, "ordered coffee", datetime(2026, 1, 2),
                   score=0.5, tokens=5),
        ScoredItem("fact", 2, "likes coffee", datetime(2026, 1, 1),
                   score=0.9, tokens=5),
    ]
    out = render(items)
    assert out.index("Facts:") < out.index("Events:")    # facts first
    assert "likes coffee (since 2026-01-01)" in out
    assert "[2026-01-02] ordered coffee" in out