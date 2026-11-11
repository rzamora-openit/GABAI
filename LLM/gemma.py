"""Gemma 4 (local, Ollama /api/chat). Supports tools; thinking is always on."""
import json
import uuid
from typing import Iterator

import requests

from LLM.base import ChatProvider, ContentDelta, StreamEvent, ThinkingDelta, ToolCall
from Utility.config import NUM_CTX, OLLAMA_BASE_URL


class Gemma(ChatProvider):
    model = "gemma4:latest"
    supports_thinking = True

    def __init__(self, options: dict | None = None, timeout: int = 600):
        """options: Ollama model options, e.g. {"temperature": 0, "num_ctx": 16384}. num_ctx defaults to NUM_CTX."""
        self.options = {"num_ctx": NUM_CTX, **(options or {})}
        self.timeout = timeout

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            msg = {"role": m["role"], "content": m.get("content") or ""}
            if m["role"] == "assistant":
                if m.get("thinking"):
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
        payload = {
            "model": self.model,
            "messages": self._convert(messages),
            "stream": True,
            "think": True,
            "options": self.options,
        }
        if tools:
            payload["tools"] = tools

        url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as response:
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
