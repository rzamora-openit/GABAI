"""Azure OpenAI (Chat Completions REST API, streamed over SSE).

Settings come from .env:
    AZURE_OPENAI_ENDPOINT     e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT   the deployment name, not the model name
    AZURE_OPENAI_API_VERSION  optional, defaults to 2024-10-21
    AZURE_OPENAI_REASONING_EFFORT  set only for reasoning models (e.g. gpt-5, o-series): low | medium | high.
                              Leave empty for non-reasoning models (e.g. gpt-4o), which reject the option.
                              When tools are sent it is forced to "none", because Chat Completions does
                              not allow function tools together with reasoning.
"""
import json
from typing import Iterator

import requests

from LLM.base import ChatProvider, ContentDelta, StreamEvent, ToolCall
from Utility.env import env


class AzureOpenAI(ChatProvider):
    supports_thinking = False  # Chat Completions does not stream reasoning text

    def __init__(self, options: dict | None = None, timeout: int = 600):
        """options: extra Chat Completions request fields, e.g. {"temperature": 0, "max_tokens": 4000}."""
        self.options = options or {}
        self.deployment = env("AZURE_OPENAI_DEPLOYMENT", required=True)
        self.endpoint = env("AZURE_OPENAI_ENDPOINT", required=True).rstrip("/")
        self.api_key = env("AZURE_OPENAI_API_KEY", required=True)
        self.api_version = env("AZURE_OPENAI_API_VERSION", "2024-10-21")
        self.reasoning_effort = env("AZURE_OPENAI_REASONING_EFFORT")
        self.timeout = timeout

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out = []
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m.get("content") or None}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}
                        for tc in m["tool_calls"]
                    ]
            elif m["role"] == "tool":
                msg = {"role": "tool", "tool_call_id": m["tool_call_id"], "content": m.get("content") or ""}
            else:
                msg = {"role": m["role"], "content": m.get("content") or ""}
            out.append(msg)
        return out

    def stream(self, messages: list[dict], tools: list[dict] | None = None) -> Iterator[StreamEvent]:
        url = (f"{self.endpoint}/openai/deployments/{self.deployment}/chat/completions"
               f"?api-version={self.api_version}")
        payload = {**self.options, "messages": self._convert(messages), "stream": True}
        if tools:
            payload["tools"] = tools
        if self.reasoning_effort:
            # reasoning models reject function tools unless reasoning is off for that call
            payload["reasoning_effort"] = "none" if tools else self.reasoning_effort

        pending: dict[int, dict] = {}  # tool calls arrive in pieces, keyed by index
        with requests.post(url, json=payload, headers={"api-key": self.api_key}, stream=True,
                           timeout=self.timeout) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"Azure OpenAI {response.status_code}: {response.text}")
            for raw in response.iter_lines():
                line = raw.decode("utf-8").strip() if raw else ""
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                for choice in json.loads(data).get("choices", []):
                    delta = choice.get("delta") or {}
                    if delta.get("content"):
                        yield ContentDelta(delta["content"])
                    for tc in delta.get("tool_calls") or []:
                        call = pending.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                        fn = tc.get("function") or {}
                        call["id"] = tc.get("id") or call["id"]
                        call["name"] = fn.get("name") or call["name"]
                        call["arguments"] += fn.get("arguments") or ""

        for call in pending.values():
            yield ToolCall(name=call["name"], arguments=json.loads(call["arguments"] or "{}"), id=call["id"])
