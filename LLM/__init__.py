"""LLM layer: LangChain chat models and embeddings. Usage:

    from LLM import azure_openai

    llm = azure_openai(temperature=0)          # a LangChain BaseChatModel
    llm.bind_tools(tools).invoke(messages)     # or use it in a LangGraph graph

Pages that let the user pick a model use MODELS: `MODELS[name](**kwargs)` returns a chat model.
"""
from functools import partial

from LLM.chat_models import AzureChat, azure_openai, gemma, llama, ollama_chat, phi, qwen
from LLM.embeddings import OllamaEmbedder

# Display name -> chat model factory, for model pickers and name-based settings. All listed models support tools.
MODELS = {
    "Qwen 3.5 4B (thinking)": partial(qwen, think=True),
    "Qwen 3.5 4B (no thinking)": qwen,
    "Gemma 4 (thinking)": gemma,
    "Llama 3.2": llama,
    "Phi-4 mini": phi,
    "Azure OpenAI": azure_openai,
}

__all__ = ["MODELS", "AzureChat", "azure_openai", "gemma", "llama", "ollama_chat", "phi", "qwen", "OllamaEmbedder"]
