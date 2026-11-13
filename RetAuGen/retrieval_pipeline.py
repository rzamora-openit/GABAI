"""Retrieval pipeline, as LangChain components:

    ChunkRetriever  intent -> query vector -> top N chunk names from Postgres -> chunk Documents read from the folder
    LLMReranker     a re-ranker agent that keeps only the chunks that help answer the intent, most relevant first

    docs = ChunkRetriever(embeddings=OllamaEmbedder(), chunk_folder="KnowledgeBase/chunks", k=5).invoke(intent)
    relevant = LLMReranker(llm=azure_openai(temperature=0)).compress_documents(docs, intent)

Every Document has the chunk's file name in metadata["filename"].
"""
from pathlib import Path
from typing import Callable, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import BaseDocumentCompressor, Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, ConfigDict, Field

from Utility.config import RAG_TOP_N
from Utility.vector_db_context import VectorDbContext


class ChunkRetriever(BaseRetriever):
    """The k chunks closest to the query (the intention behind the user's question), closest first."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Embeddings
    chunk_folder: Path
    k: int = RAG_TOP_N
    db_factory: Callable[[], VectorDbContext] = VectorDbContext  # opened per search and closed afterwards

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        vector = self.embeddings.embed_query(query)
        db = self.db_factory()
        try:
            filenames = db.top_n(vector, self.k)
        finally:
            db.close()
        return [
            Document(page_content=(Path(self.chunk_folder) / name).read_text(encoding="utf-8"), metadata={"filename": name})
            for name in filenames
        ]


RERANK_PROMPT = """You re-rank retrieved document chunks for a RAG system.
You get the intent behind a user's question and numbered chunks.
Pick only the chunks that contain information that helps answer the intent, most relevant first.
If no chunk helps, pick none."""


class RankedChunks(BaseModel):
    """The relevant chunk numbers, most relevant first. Empty if no chunk is relevant."""

    chunk_numbers: list[int] = Field(description="Relevant chunk numbers, most relevant first.")


class LLMReranker(BaseDocumentCompressor):
    """Re-ranker agent: keeps the documents that help answer the query, most relevant first. An empty result means no
    information was found."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm: BaseChatModel

    def compress_documents(self, documents: Sequence[Document], query: str, callbacks=None) -> list[Document]:
        documents = list(documents)
        if not documents:
            return []
        chunks = "\n\n".join(f"[Chunk {i}]\n{doc.page_content}" for i, doc in enumerate(documents, start=1))
        messages = [SystemMessage(RERANK_PROMPT), HumanMessage(f"Intent: {query}\n\n{chunks}")]
        ranked = self.llm.with_structured_output(RankedChunks, method="function_calling").invoke(
            messages, config={"callbacks": callbacks}
        )
        if ranked is None:  # the model did not answer with the structured output
            return []
        numbers = dict.fromkeys(ranked.chunk_numbers)  # drops repeats, keeps order
        return [documents[i - 1] for i in numbers if 1 <= i <= len(documents)]
