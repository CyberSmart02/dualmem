from datetime import datetime

from pydantic import BaseModel, Field


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