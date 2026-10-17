from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from dualmemory.models import Base, Episode, EpisodeRow, Fact, FactRow


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