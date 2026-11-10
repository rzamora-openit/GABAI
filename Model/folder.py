from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from Model.base import BaseModel

if TYPE_CHECKING:
    from Model.conversation import Conversation


@dataclass(kw_only=True)
class Folder(BaseModel):
    name: str = ""
    user_id: str = ""
    conversations: list[Conversation] = field(default_factory=list)
