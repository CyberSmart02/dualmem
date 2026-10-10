from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from dualmemory.models import Base, Episode, EpisodeRow


class SqlStore:
    """SQLite source-of-truth for episodes. Converts between the pydantic
    Episode (logic/transport) and the EpisodeRow table (persistence)."""

    def __init__(self, url: str = "sqlite:///:memory:") -> None:
        # ":memory:" = ephemeral DB (great for tests). A path = a real file.
        self.engine = create_engine(url)
        Base.metadata.create_all(self.engine)   # create tables if absent

    # --- conversions between pydantic Episode <-> EpisodeRow -------------
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

    # --- CRUD ------------------------------------------------------------
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