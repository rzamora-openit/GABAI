"""Retrieval pipeline: intent -> query vector -> top N chunk names from Postgres -> chunk texts from the folder
-> re-ranker agent keeps the relevant ones."""
from pathlib import Path

from LLM import ChatProvider, OllamaEmbedder, ToolCall
from RetAuGen.vector_db_context import VectorDbContext


def retrieve(intent: str, n: int, chunk_folder: str | Path, embedder: OllamaEmbedder, db: VectorDbContext) -> list[str]:
    """Return the texts of the n chunks closest to the intent (the intention behind the user's query), closest first."""
    vector = embedder.embed_query(intent)
    filenames = db.top_n(vector, n)
    return [(Path(chunk_folder) / filename).read_text(encoding="utf-8") for filename in filenames]


def rerank_agent(intent: str, texts: list[str], llm: ChatProvider) -> list[str]:
    """Return the texts that help answer the intent, most relevant first. An empty list means no information was found.

    llm: the model to use, e.g. AzureOpenAI(options={"temperature": 0}).
    """

    system_prompt = """You re-rank retrieved document chunks for a RAG system.
You get the intent behind a user's question and numbered chunks.
Pick only the chunks that contain information that helps answer the intent, most relevant first.
If no chunk helps, pick none.
Return your answer by calling the rank_chunks tool once."""

    rank_chunks_tool = {
        "type": "function",
        "function": {
            "name": "rank_chunks",
            "description": "Return the relevant chunk numbers, most relevant first. Empty if no chunk is relevant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunk_numbers": {"type": "array", "items": {"type": "integer"}, "description": "Relevant chunk numbers, most relevant first."},
                },
                "required": ["chunk_numbers"],
            },
        },
    }

    chunks = "\n\n".join(f"[Chunk {i}]\n{text}" for i, text in enumerate(texts, start=1))
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Intent: {intent}\n\n{chunks}"}]
    for event in llm.stream(messages, [rank_chunks_tool]):
        if isinstance(event, ToolCall) and event.name == "rank_chunks":
            numbers = dict.fromkeys(event.arguments["chunk_numbers"])  # drops repeats, keeps order
            return [texts[i - 1] for i in numbers if 1 <= i <= len(texts)]
    return []
