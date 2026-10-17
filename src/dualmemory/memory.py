from datetime import datetime

import numpy as np

from dualmemory.consolidate import Consolidator, cluster_vectors, is_eligible
from dualmemory.extract import Extractor
from dualmemory.link import Adjudicator
from dualmemory.models import Episode, Fact, Message
from dualmemory.providers import LLM, Embedder
from dualmemory.retrieve import (
    ScoredItem,
    episode_is_visible_at,
    estimate_tokens,
    fact_is_valid_at,
    pack,
    render,
    score_item,
)
from dualmemory.store.sql import SqlStore
from dualmemory.store.vectors import VectorStore


class Memory:
    """Public API. Orchestrates extraction + adjudication + retrieval + storage.

    add():    messages -> extractor -> episodes + candidate facts. Episodes ->
              SQLite + episode FAISS. Candidates -> adjudicator; committed facts
              indexed in the fact FAISS. One index per memory type (DESIGN.md §1).
    search(): R1-R7 — embed, retrieve episodes+facts, filter by validity/visibility
              at t_q, score, pack under token budget, render."""

    def __init__(self, embedder: Embedder, llm: LLM,
                 db_url: str = "sqlite:///:memory:") -> None:
        self.embedder = embedder
        self.sql = SqlStore(db_url)
        self.vectors = VectorStore(dim=embedder.dim)        # episode vectors
        self.fact_vectors = VectorStore(dim=embedder.dim)   # fact vectors
        self.extractor = Extractor(llm)
        self.adjudicator = Adjudicator(llm, self.sql)
        self.consolidator = Consolidator(llm)
        self._next_episode_id = 1
        self._next_fact_id = 1

    # ------------------------------------------------------------------ write
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
                # index the fact for retrieval (superseded facts keep their vector;
                # the validity filter excludes them at read time)
                self.fact_vectors.add(result.id, self.embedder.embed(result.text))
        return episodes, facts

    def consolidate(self, conversation_id: str) -> list[Fact]:
        """P2: cluster episodes, promote eligible clusters to candidate facts,
        run each through the same adjudicator as extraction."""
        episodes = [e for e in self.sql.all_episodes()
                    if e.conversation_id == conversation_id]
        if not episodes:
            return []

        vectors = np.stack([self.embedder.embed(e.text) for e in episodes])
        linked = self.sql.episode_ids_with_facts()

        facts: list[Fact] = []
        for idx_group in cluster_vectors(vectors):
            cluster = [episodes[i] for i in idx_group]
            if not is_eligible(cluster, linked):
                continue
            candidate = self.consolidator.propose(cluster)
            if candidate is None:
                continue
            result = self.adjudicator.adjudicate(
                candidate=candidate,
                conversation_id=conversation_id,
                source_episode_id=cluster[-1].id,   # provenance: newest supporting episode
                new_fact_id=self._next_fact_id,
            )
            if result is not None:
                facts.append(result)
                if result.id == self._next_fact_id:
                    self._next_fact_id += 1
                self.fact_vectors.add(result.id, self.embedder.embed(result.text))
        return facts
    # ------------------------------------------------------------------ read
    def search(self, query: str, token_budget: int = 2000,
               t_q: datetime | None = None, n_candidates: int = 50) -> str:
        if t_q is None:
            t_q = datetime.now()
        q_vec = self.embedder.embed(query)

        items: list[ScoredItem] = []

        # episodes: retrieve, filter by visibility at t_q, score
        for eid, sim in self.vectors.search(q_vec, n_candidates):
            ep = self.sql.get_episode(eid)
            if ep is None or not episode_is_visible_at(ep, t_q):
                continue
            items.append(ScoredItem(
                kind="episode", id=ep.id, text=ep.text, time=ep.event_time,
                score=score_item(sim, t_q, ep.event_time, importance=0.5),
                tokens=estimate_tokens(ep.text),
            ))

        # facts: retrieve, filter by validity at t_q, score
        for fid, sim in self.fact_vectors.search(q_vec, n_candidates):
            f = self.sql.get_fact(fid)
            if f is None or not fact_is_valid_at(f, t_q):   # supersession pays off
                continue
            items.append(ScoredItem(
                kind="fact", id=f.id, text=f.text, time=f.valid_from,
                score=score_item(sim, t_q, f.valid_from, importance=f.confidence),
                tokens=estimate_tokens(f.text),
            ))

        return render(pack(items, token_budget))

    # ------------------------------------------------------------------ maint
    def rebuild_index(self) -> None:
        self.vectors.reset()
        for ep in self.sql.all_episodes():
            self.vectors.add(ep.id, self.embedder.embed(ep.text))
        self.fact_vectors.reset()
        for fid in self.sql.all_fact_ids():
            f = self.sql.get_fact(fid)
            self.fact_vectors.add(f.id, self.embedder.embed(f.text))