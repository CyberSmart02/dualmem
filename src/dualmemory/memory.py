from datetime import datetime

from dualmemory.extract import Extractor
from dualmemory.models import CandidateFact, Episode, Fact, Message
from dualmemory.providers import LLM, Embedder
from dualmemory.store.sql import SqlStore
from dualmemory.store.vectors import VectorStore


class Memory:
    """Public API. Orchestrates extraction + storage.

    add(): messages -> Extractor -> episodes + candidate facts -> stored.
           Episodes go to SQLite + FAISS. Candidates become facts (naive ADD
           for now; real adjudication is the next milestone)."""

    def __init__(self, embedder: Embedder, llm: LLM,
                 db_url: str = "sqlite:///:memory:") -> None:
        self.embedder = embedder
        self.sql = SqlStore(db_url)
        self.vectors = VectorStore(dim=embedder.dim)      # episode vectors
        self.extractor = Extractor(llm)
        self._next_episode_id = 1
        self._next_fact_id = 1

    def add(self, messages: list[Message]) -> tuple[list[Episode], list[Fact]]:
        episodes, candidates = self.extractor.extract(
            messages, next_episode_id=self._next_episode_id
        )

        # store episodes: SQLite first (truth), FAISS second (invariant I7)
        for ep in episodes:
            self.sql.add_episode(ep)
            self.vectors.add(ep.id, self.embedder.embed(ep.text))
            self._next_episode_id = max(self._next_episode_id, ep.id + 1)

        # naive ADD: each candidate becomes a committed fact (no adjudication yet)
        facts = [self._commit_candidate(c, messages) for c in candidates]
        return episodes, facts

    def _commit_candidate(self, c: CandidateFact, messages: list[Message]) -> Fact:
        fact = Fact(
            id=self._next_fact_id,
            conversation_id=messages[0].conversation_id,
            subject=c.subject,
            text=c.text,
            confidence=c.confidence,
            valid_from=c.event_time,          # fact holds from when it was stated
            created_at=datetime.now(),
        )
        self._next_fact_id += 1
        self.sql.add_fact(fact)
        return fact

    def search(self, query: str, k: int = 5) -> str:
        q_vec = self.embedder.embed(query)
        hits = self.vectors.search(q_vec, k)
        episodes = [self.sql.get_episode(i) for i, _ in hits]
        episodes = [(e, s) for (e, (_, s)) in zip(episodes, hits) if e is not None]
        return self._render(episodes)

    def rebuild_index(self) -> None:
        self.vectors.reset()
        for ep in self.sql.all_episodes():
            self.vectors.add(ep.id, self.embedder.embed(ep.text))

    @staticmethod
    def _render(hits: list[tuple[Episode, float]]) -> str:
        if not hits:
            return "Events:\n(none)"
        lines = ["Events:"]
        for ep, _score in sorted(hits, key=lambda h: h[0].event_time):
            lines.append(f"- [{ep.event_time.date().isoformat()}] {ep.text}")
        return "\n".join(lines)