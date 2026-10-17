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

class Fact(BaseModel):
    id: int
    conversation_id: str
    subject: str
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_from: datetime
    valid_to: datetime | None = None          # None = still valid now
    superseded_by: int | None = None          # id of the fact that replaced this
    created_at: datetime


class CandidateFact(BaseModel):
    """A proposed fact from the extractor, before adjudication assigns
    id / valid_from / supersession. subject+text is the claim; event_time
    is when (per the source message) the fact started holding."""
    subject: str
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    event_time: datetime
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
class FactRow(Base):
    __tablename__ = "facts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(String)
    subject: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(String)
    confidence: Mapped[float] = mapped_column(default=1.0)
    valid_from: Mapped[datetime] = mapped_column(DateTime)
    valid_to: Mapped[datetime | None] = mapped_column(DateTime, default=None)
    superseded_by: Mapped[int | None] = mapped_column(Integer, default=None)
    created_at: Mapped[datetime] = mapped_column(DateTime)