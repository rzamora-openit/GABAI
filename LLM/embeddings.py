"""Ollama embeddings (local models), as a LangChain Embeddings."""
import ollama
from langchain_ollama import OllamaEmbeddings

from Utility.config import EMBED_MODEL, OLLAMA_BASE_URL


class OllamaEmbedder(OllamaEmbeddings):
    """OllamaEmbeddings with task prefixes, batching, and an error for over-long input instead of silent truncation.

    The default prefixes are the ones nomic-embed-text expects; pass "" for models that need none.
    """

    model: str = EMBED_MODEL
    base_url: str | None = OLLAMA_BASE_URL
    client_kwargs: dict | None = {"timeout": 120}
    document_prefix: str = "search_document: "
    query_prefix: str = "search_query: "
    batch_size: int = 16

    def _embed(self, inputs: list[str]) -> list[list[float]]:
        # truncate=False makes Ollama return an error for over-long input instead of silently cutting it.
        try:
            response = self._client.embed(
                self.model, inputs, truncate=False, dimensions=self.dimensions,
                options=self._default_params, keep_alive=self.keep_alive,
            )
        except ollama.ResponseError as e:
            raise RuntimeError(f"Ollama {e.status_code}: {e.error}") from e
        return response["embeddings"]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed([self.document_prefix + text for text in texts[start:start + self.batch_size]]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([self.query_prefix + text])[0]
