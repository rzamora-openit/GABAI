"""Chat models, as LangChain chat models (BaseChatModel). Every factory returns a model that supports tools,
`bind_tools` and `with_structured_output`, so agents and LangGraph graphs can use any of them interchangeably.

    from LLM.chat_models import azure_openai
    reply = azure_openai(temperature=0).invoke("Hello")

Keyword arguments are passed to the LangChain model class, e.g. temperature=0 or num_ctx=16384 for Ollama models.
"""
from typing import Any, ClassVar

import openai
from langchain_ollama import ChatOllama
from langchain_openai import AzureChatOpenAI

from Utility.config import NUM_CTX, OLLAMA_BASE_URL
from Utility.env import env


def ollama_chat(model: str, reasoning: bool | None = None, timeout: int = 600, **kwargs: Any) -> ChatOllama:
    """A local Ollama model. reasoning: True streams thinking (in additional_kwargs["reasoning_content"]), False turns
    it off, None keeps the model's default. num_ctx defaults to NUM_CTX: Ollama's own default is often 4096, which
    silently drops the oldest messages."""
    settings = {"num_ctx": NUM_CTX, "base_url": OLLAMA_BASE_URL, "client_kwargs": {"timeout": timeout}, **kwargs}
    return ChatOllama(model=model, reasoning=reasoning, **settings)


def qwen(think: bool = False, **kwargs: Any) -> ChatOllama:
    """Qwen 3.5 4B. Thinking is off unless think=True."""
    return ollama_chat("qwen3.5:4b", reasoning=think, **kwargs)


def gemma(**kwargs: Any) -> ChatOllama:
    """Gemma 4. Thinking is always on."""
    return ollama_chat("gemma4:latest", reasoning=True, **kwargs)


def llama(**kwargs: Any) -> ChatOllama:
    """Llama 3.2. No thinking."""
    return ollama_chat("llama3.2:latest", **kwargs)


def phi(**kwargs: Any) -> ChatOllama:
    """Phi-4 mini. No thinking."""
    return ollama_chat("phi4-mini:latest", **kwargs)


class AzureChat(AzureChatOpenAI):
    """AzureChatOpenAI (Chat Completions) with two fixes for reasoning deployments (e.g. gpt-5, o-series):

    - Chat Completions does not allow function tools together with reasoning, so whenever tools are bound
      (bind_tools, and with_structured_output, which binds a tool) reasoning_effort is set to "none" for those calls.
    - Reasoning deployments only accept the default temperature: when a call is rejected because of the temperature
      it is retried once without it, and the deployment is remembered so temperature is left out from then on.
    """

    rejects_temperature: ClassVar[set[str]] = set()

    def bind_tools(self, tools, **kwargs):
        if self.reasoning_effort:
            kwargs.setdefault("reasoning_effort", "none")
        return super().bind_tools(tools, **kwargs)

    def _get_request_payload(self, input_, *, stop=None, **kwargs) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.deployment_name in self.rejects_temperature:
            payload.pop("temperature", None)
        return payload

    def _drop_temperature(self, error: openai.BadRequestError) -> bool:
        """True if the error is about the temperature and the call should be retried without it."""
        if "temperature" not in str(error) or self.deployment_name in self.rejects_temperature:
            return False
        self.rejects_temperature.add(self.deployment_name)
        return True

    def _generate(self, *args, **kwargs):
        try:
            return super()._generate(*args, **kwargs)
        except openai.BadRequestError as error:
            if not self._drop_temperature(error):
                raise
            return super()._generate(*args, **kwargs)

    def _stream(self, *args, **kwargs):
        # A rejected request fails before the first chunk, so nothing has been yielded when it is retried.
        try:
            yield from super()._stream(*args, **kwargs)
        except openai.BadRequestError as error:
            if not self._drop_temperature(error):
                raise
            yield from super()._stream(*args, **kwargs)


def azure_openai(timeout: int = 600, **kwargs: Any) -> AzureChat:
    """Azure OpenAI. Settings come from .env:

    AZURE_OPENAI_ENDPOINT     e.g. https://my-resource.openai.azure.com
    AZURE_OPENAI_API_KEY
    AZURE_OPENAI_DEPLOYMENT   the deployment name, not the model name
    AZURE_OPENAI_API_VERSION  optional, defaults to 2024-10-21
    AZURE_OPENAI_REASONING_EFFORT  set only for reasoning models (e.g. gpt-5, o-series): low | medium | high.
                              Leave empty for non-reasoning models (e.g. gpt-4o), which reject the option.
    """
    settings = {
        "azure_endpoint": env("AZURE_OPENAI_ENDPOINT", required=True).rstrip("/"),
        "api_key": env("AZURE_OPENAI_API_KEY", required=True),
        "azure_deployment": env("AZURE_OPENAI_DEPLOYMENT", required=True),
        "api_version": env("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        "reasoning_effort": env("AZURE_OPENAI_REASONING_EFFORT") or None,
        "use_responses_api": False,
        "timeout": timeout,
        **kwargs,
    }
    return AzureChat(**settings)
