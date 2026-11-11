"""LLM query layer: one class per model. Usage:

    from LLM.azure_openai import AzureOpenAI

    for event in AzureOpenAI(options={"temperature": 0}).stream(messages, tools):
        ...  # ThinkingDelta, ContentDelta or ToolCall

Pages that let the user pick a model use MODELS: `MODELS[name](options=...)`.
"""
from functools import partial

from LLM.azure_openai import AzureOpenAI
from LLM.base import ChatProvider, ContentDelta, StreamEvent, ThinkingDelta, ToolCall
from LLM.embeddings import OllamaEmbedder
from LLM.gemma import Gemma
from LLM.llama import Llama
from LLM.phi import Phi
from LLM.qwen import Qwen

# Display name -> model class, for model pickers and name-based settings. All listed models support tools.
MODELS = {
    "Qwen 3.5 4B (thinking)": partial(Qwen, think=True),
    "Qwen 3.5 4B (no thinking)": Qwen,
    "Gemma 4 (thinking)": Gemma,
    "Llama 3.2": Llama,
    "Phi-4 mini": Phi,
    "Azure OpenAI": AzureOpenAI,
}

__all__ = [
    "MODELS", "AzureOpenAI", "Gemma", "Llama", "Phi", "Qwen", "OllamaEmbedder",
    "ChatProvider", "StreamEvent", "ThinkingDelta", "ContentDelta", "ToolCall",
]
