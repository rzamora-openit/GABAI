"""LLM query layer. Usage:

    from LLM import PROVIDERS, get_provider, ThinkingDelta, ContentDelta, ToolCall

    for event in get_provider("Llama 3.2").stream(messages, tools):
        ...
"""
from LLM.azure_openai_provider import AzureOpenAIProvider
from LLM.base import ChatProvider, ContentDelta, StreamEvent, ThinkingDelta, ToolCall
from LLM.ollama_provider import OllamaProvider

# Display name -> factory. Factories are lazy so missing settings (e.g. Azure) only fail when selected.
# All listed Ollama models support tools.
PROVIDERS = {
    "Qwen 3.5 4B (thinking)": lambda: OllamaProvider("qwen3.5:4b", think=True),
    "Qwen 3.5 4B (no thinking)": lambda: OllamaProvider("qwen3.5:4b", think=False),
    "Gemma 4 (thinking)": lambda: OllamaProvider("gemma4:latest", think=True),
    "Llama 3.2": lambda: OllamaProvider("llama3.2:latest"),
    "Phi-4 mini": lambda: OllamaProvider("phi4-mini:latest"),
    "Azure OpenAI": lambda: AzureOpenAIProvider(),
}


def get_provider(name: str) -> ChatProvider:
    return PROVIDERS[name]()


__all__ = [
    "PROVIDERS", "get_provider", "ChatProvider", "OllamaProvider", "AzureOpenAIProvider",
    "StreamEvent", "ThinkingDelta", "ContentDelta", "ToolCall",
]
