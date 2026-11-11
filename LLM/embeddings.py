"""Ollama embeddings (local models via /api/embed)."""
import requests

from Utility.config import EMBED_MODEL
from Utility.env import env

BATCH_SIZE = 16


class OllamaEmbedder:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
        document_prefix: str = "search_document: ",
        query_prefix: str = "search_query: ",
    ):
        """The default prefixes are the ones nomic-embed-text expects; pass "" for models that need none."""
        self.model = model or EMBED_MODEL
        self.base_url = (base_url or env("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = timeout
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        # truncate=False makes Ollama return an error for over-long input instead of silently cutting it.
        payload = {"model": self.model, "input": inputs, "truncate": False}
        response = requests.post(f"{self.base_url}/api/embed", json=payload, timeout=self.timeout)
        if response.status_code >= 400:
            raise RuntimeError(f"Ollama {response.status_code}: {response.text}")
        return response.json()["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = [self.document_prefix + text for text in texts[start:start + BATCH_SIZE]]
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]
