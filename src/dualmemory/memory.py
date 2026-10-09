
from dualmemory.models import Episode, Message
from dualmemory.providers import Embedder
from dualmemory.store import EpisodeStore


class Memory:
    """The public API. Coordinates embedding (providers) and storage (store).

    Skeleton behaviour:
      - add(): turn each user message into one episode (no real extraction yet),
               embed it, store it.
      - search(): embed the query, retrieve top-k episodes, render to text.
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.store = EpisodeStore(dim=embedder.dim)
        self._next_id = 1

    def add(self, messages: list[Message]) -> list[Episode]:
        created: list[Episode] = []
        for msg in messages:
            # Skeleton stub: one episode per message, text = content verbatim.
            # Real extraction (LLM, atomic statements) replaces this later.
            ep = Episode(
                id=self._next_id,
                conversation_id=msg.conversation_id,
                text=msg.content,
                event_time=msg.timestamp,
                source_message_ids=[msg.id],
            )
            self._next_id += 1
            vec = self.embedder.embed(ep.text)
            self.store.add(ep, vec)
            created.append(ep)
        return created

    def search(self, query: str, k: int = 5) -> str:
        q_vec = self.embedder.embed(query)
        hits = self.store.search(q_vec, k)
        return self._render(hits)

    @staticmethod
    def _render(hits: list[tuple[Episode, float]]) -> str:
        """Frozen-template render (skeleton subset of EVALUATION.md §7)."""
        if not hits:
            return "Events:\n(none)"
        lines = ["Events:"]
        for ep, _score in sorted(hits, key=lambda h: h[0].event_time):
            date = ep.event_time.date().isoformat()
            lines.append(f"- [{date}] {ep.text}")
        return "\n".join(lines)