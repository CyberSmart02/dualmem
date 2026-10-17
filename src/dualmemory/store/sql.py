from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from dualmemory.models import (
    Base,
    Edge,
    EdgeRow,
    Episode,
    EpisodeRow,
    Fact,
    FactRevisionRow,
    FactRow,
)


class SqlStore:
    """SQLite source-of-truth for episodes and facts. Converts between the
    pydantic models (logic/transport) and the table rows (persistence)."""

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        # ":memory:" = ephemeral DB (great for tests). A path = a real file.
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)   # create tables if absent

    # --- episode conversions ---------------------------------------------
    @staticmethod
    def _to_row(ep: Episode) -> EpisodeRow:
        return EpisodeRow(
            id=ep.id,
            conversation_id=ep.conversation_id,
            text=ep.text,
            event_time=ep.event_time,
            # list[int] -> "10,11" for SQLite
            source_message_ids=",".join(str(i) for i in ep.source_message_ids),
        )

    @staticmethod
    def _to_episode(row: EpisodeRow) -> Episode:
        ids = [int(x) for x in row.source_message_ids.split(",") if x]
        return Episode(
            id=row.id,
            conversation_id=row.conversation_id,
            text=row.text,
            event_time=row.event_time,
            source_message_ids=ids,
        )

    # --- episode CRUD ----------------------------------------------------
    def add_episode(self, ep: Episode) -> None:
        with Session(self.engine) as session:
            session.add(self._to_row(ep))
            session.commit()

    def get_episode(self, id_: int) -> Episode | None:
        with Session(self.engine) as session:
            row = session.get(EpisodeRow, id_)
            return self._to_episode(row) if row else None

    def delete_episode(self, id_: int) -> None:
        with Session(self.engine) as session:
            row = session.get(EpisodeRow, id_)
            if row:
                session.delete(row)
                session.commit()

    def all_episode_ids(self) -> set[int]:
        """Every episode id in SQLite — used to check FAISS agreement (I7)."""
        with Session(self.engine) as session:
            rows = session.execute(select(EpisodeRow.id)).scalars().all()
            return set(rows)

    def all_episodes(self) -> list[Episode]:
        """All episodes — used by rebuild_index() to re-embed into FAISS."""
        with Session(self.engine) as session:
            rows = session.execute(select(EpisodeRow)).scalars().all()
            return [self._to_episode(r) for r in rows]

    # --- fact conversions ------------------------------------------------
    @staticmethod
    def _fact_to_row(f: Fact) -> FactRow:
        return FactRow(
            id=f.id,
            conversation_id=f.conversation_id,
            subject=f.subject,
            text=f.text,
            confidence=f.confidence,
            valid_from=f.valid_from,
            valid_to=f.valid_to,
            superseded_by=f.superseded_by,
            created_at=f.created_at,
        )

    @staticmethod
    def _row_to_fact(row: FactRow) -> Fact:
        return Fact(
            id=row.id,
            conversation_id=row.conversation_id,
            subject=row.subject,
            text=row.text,
            confidence=row.confidence,
            valid_from=row.valid_from,
            valid_to=row.valid_to,
            superseded_by=row.superseded_by,
            created_at=row.created_at,
        )

    # --- fact CRUD -------------------------------------------------------
    def add_fact(self, f: Fact) -> None:
        with Session(self.engine) as session:
            session.add(self._fact_to_row(f))
            session.commit()

    def get_fact(self, id_: int) -> Fact | None:
        with Session(self.engine) as session:
            row = session.get(FactRow, id_)
            return self._row_to_fact(row) if row else None

    def all_fact_ids(self) -> set[int]:
        """Every fact id in SQLite."""
        with Session(self.engine) as session:
            rows = session.execute(select(FactRow.id)).scalars().all()
            return set(rows)

      # --- edges -----------------------------------------------------------
    def add_edge(self, src: int, dst: int, kind: str, weight: float = 1.0) -> None:
        with Session(self.engine) as session:
            session.add(EdgeRow(src=src, dst=dst, kind=kind, weight=weight))
            session.commit()

    def edges_for_fact(self, fact_id: int) -> list[Edge]:
        """All derived_from edges from this fact (its supporting episodes)."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(EdgeRow).where(
                    EdgeRow.src == fact_id, EdgeRow.kind == "derived_from"
                )
            ).scalars().all()
            return [Edge(src=r.src, dst=r.dst, kind=r.kind, weight=r.weight)
                    for r in rows]

    def all_edge_kinds(self) -> set[str]:
        """Distinct edge kinds in the store — used to enforce invariant I6."""
        with Session(self.engine) as session:
            rows = session.execute(select(EdgeRow.kind)).scalars().all()
            return set(rows)

    def episode_ids_with_facts(self) -> set[int]:
        """Episode ids that some fact already derives from (derived_from dst)."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(EdgeRow.dst).where(EdgeRow.kind == "derived_from")
            ).scalars().all()
            return set(rows)
    # --- fact updates & revisions ---------------------------------------
    def add_revision(self, fact_id: int, previous_text: str,
                     revised_at: datetime) -> None:
        with Session(self.engine) as session:
            session.add(FactRevisionRow(
                fact_id=fact_id, previous_text=previous_text, revised_at=revised_at
            ))
            session.commit()

    def update_fact_text(self, fact_id: int, new_text: str) -> None:
        """Rewrite a fact's text in place (UPDATE action). Keeps id + valid_from."""
        with Session(self.engine) as session:
            row = session.get(FactRow, fact_id)
            if row:
                row.text = new_text
                session.commit()

    def supersede_fact(self, old_id: int, valid_to: datetime,
                       new_fact_id: int) -> None:
        """Close an old fact's validity window and point it at its replacement
        (SUPERSEDE action). The old row is NEVER deleted (invariant I3)."""
        with Session(self.engine) as session:
            row = session.get(FactRow, old_id)
            if row:
                row.valid_to = valid_to
                row.superseded_by = new_fact_id
                session.commit()

    def valid_facts_for_subject(self, conversation_id: str,
                                subject: str) -> list[Fact]:
        """Currently-valid facts (valid_to is NULL) for a subject — the
        candidates the adjudicator compares a new candidate against."""
        with Session(self.engine) as session:
            rows = session.execute(
                select(FactRow).where(
                    FactRow.conversation_id == conversation_id,
                    FactRow.subject == subject,
                    FactRow.valid_to.is_(None),
                )
            ).scalars().all()
            return [self._row_to_fact(r) for r in rows]