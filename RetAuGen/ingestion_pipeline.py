"""Ingestion pipeline: PDFs -> extracted text -> chunk markdown files (LLM agent) -> vectors in Postgres.

Run from the project root:   python.exe -m RetAuGen.ingestion_pipeline
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from LLM import AzureOpenAI, ChatProvider, OllamaEmbedder, ToolCall
from RetAuGen.vector_db_context import VectorDbContext


def ingest(pdf_folder: str | Path, extracted_folder: str | Path, chunk_folder: str | Path, llm: ChatProvider,
           embedder: OllamaEmbedder, db: VectorDbContext) -> None:
    """Extract every PDF in pdf_folder into extracted_folder, chunk it into chunk_folder, embed the chunks
    and save them to the db.

    A PDF that was ingested before is replaced: its old chunk files and db rows are removed first.
    """
    texts = extract(pdf_folder, extracted_folder)
    for source in texts:
        for old_chunk in Path(chunk_folder).glob(f"{source}.chunk_*.md"):
            old_chunk.unlink()
        db.delete(source)
    chunk_documents(texts, llm, chunk_folder)
    embed_chunks(chunk_folder, embedder, db)


def extract(folder: str | Path, out_dir: str | Path) -> dict[str, str]:
    """Extract the text of every PDF in the folder and save it as out_dir/{filename}.txt.

    Returns {pdf file name without extension: extracted text}.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    texts = {}
    for pdf_file in sorted(Path(folder).glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_file)).load()
        texts[pdf_file.stem] = "\n\n".join(page.page_content for page in pages)
        (out_dir / f"{pdf_file.stem}.txt").write_text(texts[pdf_file.stem], encoding="utf-8")
    return texts


def chunk_documents(texts: dict[str, str], llm: ChatProvider, out_dir: str | Path) -> None:
    """Run the agent on each extracted document and save each chunk as out_dir/{filename}.chunk_{N}.md.

    texts: {filename: extracted text}, as returned by extract().
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in texts.items():
        chunks = chunking_agent(text, llm)
        for n, chunk in enumerate(chunks, start=1):
            (out_dir / f"{filename}.chunk_{n}.md").write_text(chunk, encoding="utf-8")


def chunking_agent(text: str, llm: ChatProvider) -> list[str]:
    """Send the system prompt and the text; return the chunks from the agent's save_chunks call.

    llm: the model to use, e.g. AzureOpenAI(options={"temperature": 0}).
    """

    system_prompt = """You split documents into chunks for RAG ingestion.
Split the document by section. Each chunk is one self-contained section in markdown, starting with its heading.
Keep the original wording; do not summarize, skip or invent text.
Return all chunks by calling the save_chunks tool once."""

    save_chunks_tool = {
        "type": "function",
        "function": {
            "name": "save_chunks",
            "description": "Save the document's chunks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "chunks": {"type": "array", "items": {"type": "string"}, "description": "The chunks, in document order."},
                },
                "required": ["chunks"],
            },
        },
    }

    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}]
    for event in llm.stream(messages, [save_chunks_tool]):
        if isinstance(event, ToolCall) and event.name == "save_chunks":
            return event.arguments["chunks"]
    return []


def embed_chunks(folder: str | Path, embedder: OllamaEmbedder, db: VectorDbContext) -> None:
    """Read every *.chunk_*.md file in the folder, embed its text, and save (filename, vector) to the db."""
    files = sorted(Path(folder).glob("*.chunk_*.md"))
    texts = [file.read_text(encoding="utf-8") for file in files]
    vectors = embedder.embed_documents(texts)
    for file, vector in zip(files, vectors):
        db.save(file.name, vector)


if __name__ == "__main__":
    db = VectorDbContext()
    try:
        ingest("KnowledgeBase", "KnowledgeBase/extracted", "KnowledgeBase/chunks", AzureOpenAI(options={"temperature": 0}), OllamaEmbedder(), db)
    finally:
        db.close()
