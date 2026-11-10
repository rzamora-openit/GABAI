from dataclasses import dataclass, field
from datetime import datetime

from Model.base import BaseModel


@dataclass(kw_only=True)
class Message(BaseModel):
    role: str = ""  # e.g. "user", "assistant", "system"
    content: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    conversation_id: int | None = None
