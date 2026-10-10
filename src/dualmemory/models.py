from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# ---------------------------------------------------------------------------
# Pydantic models — validation & transport (boundaries, in-memory logic)
# ---------------------------------------------------------------------------

class Message(BaseModel):
    id: int
    conversation_id: str
    role: str
    content: str
    timestamp: datetime


class Episode(BaseModel):
    id: int
    conversation_id: str
    text: str
    event_time: datetime
    source_message_ids: list[int] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# SQLAlchemy tables — persistence (the SQLite layer only)
# One Base that all tables inherit from.
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class EpisodeRow(Base):
    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String)
    event_time: Mapped[datetime] = mapped_column(DateTime)
    # source_message_ids is a list → store as comma-separated string for v0.
    # (Simple + SQLite-friendly; a JSON column or child table comes later.)
    source_message_ids: Mapped[str] = mapped_column(String, default="")