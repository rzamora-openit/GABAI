from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from Model.base import BaseModel

if TYPE_CHECKING:
    from Model.folder import Folder
    from Model.message import Message


@dataclass(kw_only=True)
class Conversation(BaseModel):
    guid: uuid.UUID = field(default_factory=uuid.uuid4)
    title: str = ""

    messages: list[Message] = field(default_factory=list)
    last_message: datetime = field(default_factory=datetime.now)

    folder: Folder | None = None
    folder_id: int | None = None

    user_id: str = ""
    token: int = 0
