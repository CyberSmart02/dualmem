from dualmemory.models import Episode, Message
from dualmemory.providers import Embedder
from dualmemory.store.sql import SqlStore
from dualmemory.store.vectors import VectorStore


class Memory:
    """Public API. SQLite (source of truth) + FAISS (vector index) + embedder."""

    def __init__(self, embedder: Embedder, db_url: str = "sqlite:///:memory:") -> None:
        self.embedder = embedder
        self.sql = SqlStore(db_url)
        self.vectors = VectorStore(dim=embedder.dim)
        self._next_id = 1

    def add(self, messages: list[Message]) -> list[Episode]:
        created: list[Episode] = []
        for msg in messages:
            ep = Episode(
                id=self._next_id,
                conversation_id=msg.conversation_id,
                text=msg.content,
                event_time=msg.timestamp,
                source_message_ids=[msg.id],
            )
            self._next_id += 1
            # Write order: SQLite FIRST (source of truth), FAISS SECOND (invariant I7).
            self.sql.add_episode(ep)
            self.vectors.add(ep.id, self.embedder.embed(ep.text))
            created.append(ep)
        return created

    def search(self, query: str, k: int = 5) -> str:
        q_vec = self.embedder.embed(query)
        hits = self.vectors.search(q_vec, k)          # [(id, score), ...]
        episodes = [(self.sql.get_episode(i), s) for i, s in hits]
        episodes = [(e, s) for e, s in episodes if e is not None]
        return self._render(episodes)

    def rebuild_index(self) -> None:
        """Recreate FAISS from SQLite (recovers drift — invariant I7)."""
        self.vectors.reset()
        for ep in self.sql.all_episodes():
            self.vectors.add(ep.id, self.embedder.embed(ep.text))

    @staticmethod
    def _render(hits: list[tuple[Episode, float]]) -> str:
        if not hits:
            return "Events:\n(none)"
        lines = ["Events:"]
        for ep, _score in sorted(hits, key=lambda h: h[0].event_time):
            date = ep.event_time.date().isoformat()
            lines.append(f"- [{date}] {ep.text}")
        return "\n".join(lines)