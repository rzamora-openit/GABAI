"""Streamlit drawing for LangGraph runs and LangChain messages, shared by the chat, LaaJ and test-runner pages."""
from typing import Callable, Iterable

import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage

STREAM_MODES = ["messages", "custom", "values"]


def thinking_of(message: BaseMessage) -> str:
    """The model's reasoning text (Ollama thinking models put it in additional_kwargs)."""
    return message.additional_kwargs.get("reasoning_content") or ""


def show_message(message: BaseMessage) -> None:
    """Draw one saved message as a chat bubble; tool results and system messages are not shown."""
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.text)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            if thinking_of(message):
                with st.expander("Thinking"):
                    st.markdown(thinking_of(message))
            for call in message.tool_calls:
                st.caption(f"Tool call: `{call['name']}({call['args']})`")
            if message.text:
                st.markdown(message.text)


class _Bubble:
    """One streaming assistant reply: a thinking expander and the reply text."""

    def __init__(self, container):
        with container:
            with st.chat_message("assistant"):
                self.thinking_slot = st.empty()
                self.text_slot = st.empty()
                self.calls_slot = st.container()
        self.thinking, self.text, self.thinking_body, self.calls = "", "", None, {}

    def add(self, chunk: AIMessageChunk) -> None:
        if thinking_of(chunk):
            if self.thinking_body is None:
                # first thinking chunk: open the "Thinking" expander and stream into it
                self.thinking_body = self.thinking_slot.expander("Thinking", expanded=True).empty()
            self.thinking += thinking_of(chunk)
            self.thinking_body.markdown(self.thinking + "▌")
        if chunk.text:
            self.text += chunk.text
            self.text_slot.markdown(self.text + "▌")
        for call in chunk.tool_call_chunks:
            if call.get("name"):
                self.calls[call.get("index", len(self.calls))] = call["name"]

    def finish(self) -> None:
        if self.thinking:
            # thinking is done: collapse the expander
            self.thinking_slot.expander("Thinking", expanded=False).markdown(self.thinking)
        self.text_slot.markdown(self.text)
        for name in self.calls.values():
            self.calls_slot.caption(f"Tool call: `{name}`")


def render_stream(events: Iterable, container, nodes: tuple[str, ...], on_values: Callable[[dict], None] | None = None) -> dict:
    """Draw a LangGraph stream made with stream_mode=STREAM_MODES into a Streamlit container and return the last state.

    nodes: the graph nodes whose model tokens are shown (models called inside tools, e.g. the re-ranker, are not).
    Progress lines a tool writes with its stream writer ({"progress": text}) are shown as captions.
    on_values: called with the full graph state after each step.
    """
    bubble, bubble_id, state = None, None, {}
    try:
        for mode, data in events:
            if mode == "messages":
                chunk, metadata = data
                if metadata.get("langgraph_node") not in nodes or not isinstance(chunk, AIMessageChunk):
                    continue
                if bubble is None or chunk.id != bubble_id:
                    if bubble:
                        bubble.finish()
                    bubble, bubble_id = _Bubble(container), chunk.id
                bubble.add(chunk)
            elif mode == "custom":
                if isinstance(data, dict) and "progress" in data:
                    container.caption(data["progress"])
            elif mode == "values":
                state = data
                if on_values:
                    on_values(data)
    finally:
        if bubble:
            bubble.finish()
    return state
