import json
import logging
from datetime import datetime

from dualmemory.models import CandidateFact, Fact
from dualmemory.providers import LLM
from dualmemory.store.sql import SqlStore

logger = logging.getLogger(__name__)


ADJUDICATE_PROMPT = """A new candidate fact may relate to existing facts about
the same subject. Choose exactly ONE action.

Candidate: "{candidate}"

Existing facts (id: text):
{existing}

Actions:
- EVIDENCE: candidate means the SAME as an existing fact. (restatement)
- UPDATE: candidate refines the same truth with more detail, nothing contradicted.
- SUPERSEDE: candidate CONTRADICTS / replaces an existing fact.
- ADD: nothing relevant — it's a new fact.

Return ONLY JSON: {{"action": "EVIDENCE|UPDATE|SUPERSEDE|ADD", "target_id": <id or null>}}
"""


class Adjudicator:
    """Decides and executes EVIDENCE/UPDATE/SUPERSEDE/ADD for candidate facts.

    Decision = LLM (judgment). Execution = SqlStore ops (mechanical).
    Returns the committed Fact (or the existing one, for EVIDENCE)."""

    def __init__(self, llm: LLM, sql: SqlStore) -> None:
        self.llm = llm
        self.sql = sql

    def adjudicate(
        self, candidate: CandidateFact, conversation_id: str,
        source_episode_id: int, new_fact_id: int,
    ) -> Fact | None:
        existing = self.sql.valid_facts_for_subject(conversation_id, candidate.subject)

        # No conflict possible → straight ADD, no LLM call.
        if not existing:
            return self._do_add(candidate, conversation_id, source_episode_id,
                                 new_fact_id)

        action, target = self._decide(candidate, existing)

        if action == "EVIDENCE" and target is not None:
            # same truth → just link the existing fact to this new episode
            self.sql.add_edge(target.id, source_episode_id, "derived_from")
            return target
        if action == "UPDATE" and target is not None:
            self.sql.add_revision(target.id, target.text, datetime.now())
            self.sql.update_fact_text(target.id, candidate.text)
            self.sql.add_edge(target.id, source_episode_id, "derived_from")
            return self.sql.get_fact(target.id)
        if action == "SUPERSEDE" and target is not None:
            new = self._do_add(candidate, conversation_id, source_episode_id,
                               new_fact_id)
            self.sql.supersede_fact(target.id, candidate.event_time, new.id)
            return new
        # default / ADD
        return self._do_add(candidate, conversation_id, source_episode_id,
                            new_fact_id)

    # --- decision (LLM) --------------------------------------------------
    def _decide(self, candidate: CandidateFact,
                existing: list[Fact]) -> tuple[str, Fact | None]:
        listing = "\n".join(f"{f.id}: {f.text}" for f in existing)
        prompt = ADJUDICATE_PROMPT.format(candidate=candidate.text, existing=listing)
        raw = self.llm.complete(prompt)
        try:
            parsed = json.loads(raw)
            action = parsed["action"]
            tid = parsed.get("target_id")
            target = next((f for f in existing if f.id == tid), None)
            return action, target
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as exc:
            logger.warning("adjudication parse failed, defaulting to ADD: %s", exc)
            return "ADD", None
    # --- execution: ADD --------------------------------------------------
    def _do_add(self, candidate: CandidateFact, conversation_id: str,
                source_episode_id: int, new_fact_id: int) -> Fact:
        fact = Fact(
            id=new_fact_id,
            conversation_id=conversation_id,
            subject=candidate.subject,
            text=candidate.text,
            confidence=candidate.confidence,
            valid_from=candidate.event_time,
            created_at=datetime.now(),
        )
        self.sql.add_fact(fact)
        self.sql.add_edge(fact.id, source_episode_id, "derived_from")  # provenance (I2)
        return fact