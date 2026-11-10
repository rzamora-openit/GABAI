"""Ollama implementation of ChatProvider (local models via /api/chat)."""
import json
import uuid
from typing import Iterator

import requests

from LLM.base import ChatProvider, ContentDelta, StreamEvent, ThinkingDelta, ToolCall
from Utility.env import env


class OllamaProvider(ChatProvider):
    def __init__(self, model: str, think: bool | None = None, base_url: str | None = None, timeout: int = 600):
        """
        think: True  -> ask for reasoning (model must support thinking)
               False -> turn reasoning off (for models that think by default)
               None  -> don't send the option (models without thinking)
        """
        self.model = model
        self.think = think
        self.supports_thinking = bool(think)
        self.base_url = (base_url or env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout

    def _convert(self, messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m["role"] == "assistant":
                if m.get("thinking") and self.supports_thinking:
                    msg["thinking"] = m["thinking"]
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"function": {"name": tc["name"], "arguments": tc["arguments"]}} for tc in m["tool_calls"]
                    ]
            elif m["role"] == "tool":
                msg["tool_name"] = m.get("name", "")
            out.append(msg)
        return out

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamEvent]:
        payload = {"model": self.model, "messages": self._convert(messages), "stream": True}
        if tools:
            payload["tools"] = tools
        if self.think is not None:
            payload["think"] = self.think

        with requests.post(f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"Ollama {response.status_code}: {response.text}")
            for line in response.iter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(f"Ollama: {chunk['error']}")
                message = chunk.get("message", {})
                if message.get("thinking"):
                    yield ThinkingDelta(message["thinking"])
                if message.get("content"):
                    yield ContentDelta(message["content"])
                for tc in message.get("tool_calls") or []:
                    fn = tc["function"]
                    args = fn.get("arguments") or {}
                    if isinstance(args, str):
                        args = json.loads(args or "{}")
                    # Ollama may omit ids; generate one so the history also works with OpenAI-style providers
                    yield ToolCall(name=fn["name"], arguments=args, id=tc.get("id") or f"call_{uuid.uuid4().hex[:12]}")
