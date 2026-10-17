from dualmemory.extract import Extractor
from dualmemory.link import Adjudicator
from dualmemory.models import Episode, Fact, Message
from dualmemory.providers import LLM, Embedder
from dualmemory.store.sql import SqlStore
from dualmemory.store.vectors import VectorStore


class Memory:
    """Public API. Orchestrates extraction + adjudication + storage.

    add(): messages -> extractor -> episodes + candidate facts.
           Episodes -> SQLite + FAISS. Candidates -> adjudicator, which
           decides EVIDENCE/UPDATE/SUPERSEDE/ADD against existing facts."""

    def __init__(self, embedder: Embedder, llm: LLM,
                 db_url: str = "sqlite:///:memory:") -> None:
        self.embedder = embedder
        self.sql = SqlStore(db_url)
        self.vectors = VectorStore(dim=embedder.dim)
        self.extractor = Extractor(llm)
        self.adjudicator = Adjudicator(llm, self.sql)
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

        # attribute candidates to the first episode of this batch (v0 simplification)
        source_ep_id = episodes[0].id if episodes else None

        facts: list[Fact] = []
        for c in candidates:
            if source_ep_id is None:
                continue   # no episode to attach provenance to; skip
            result = self.adjudicator.adjudicate(
                candidate=c,
                conversation_id=messages[0].conversation_id,
                source_episode_id=source_ep_id,
                new_fact_id=self._next_fact_id,
            )
            if result is not None:
                facts.append(result)
                if result.id == self._next_fact_id:   # only bump on a NEW fact
                    self._next_fact_id += 1
        return episodes, facts

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