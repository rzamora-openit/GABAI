"""Common interface for chat LLM providers.

Every provider takes the same message history and tool list, and streams the
same three event types back: ThinkingDelta, ContentDelta and ToolCall.

Message history format (provider-neutral; each provider converts it):
    {"role": "system" | "user", "content": str}
    {"role": "assistant", "content": str, "thinking": str (optional),
     "tool_calls": [{"id": str, "name": str, "arguments": dict}] (optional)}
    {"role": "tool", "content": str, "name": str, "tool_call_id": str}

Tools use the OpenAI function-tool schema:
    {"type": "function", "function": {"name", "description", "parameters"}}
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Union


@dataclass
class ThinkingDelta:
    """A piece of the model's reasoning text."""
    text: str


@dataclass
class ContentDelta:
    """A piece of the model's reply text."""
    text: str


@dataclass
class ToolCall:
    """A complete tool call requested by the model."""
    name: str
    arguments: dict = field(default_factory=dict)
    id: str = ""


StreamEvent = Union[ThinkingDelta, ContentDelta, ToolCall]


class ChatProvider(ABC):
    """Base class for all chat providers."""

    #: True if this provider streams ThinkingDelta events.
    supports_thinking: bool = False

    @abstractmethod
    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamEvent]:
        """Send the conversation and yield events as they arrive."""
