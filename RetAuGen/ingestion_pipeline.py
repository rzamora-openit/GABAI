"""Ingestion pipeline: PDFs -> extracted text -> chunk markdown files (LLM agent) -> vectors in Postgres.

Run from the project root:   python.exe -m RetAuGen.ingestion_pipeline
"""
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader

from LLM import AzureOpenAI, ChatProvider, OllamaEmbedder, ToolCall
from RetAuGen.vector_db_context import VectorDbContext


def ingest(pdf_folder: str | Path, extracted_folder: str | Path, chunk_folder: str | Path, llm: ChatProvider,
           embedder: OllamaEmbedder, db: VectorDbContext) -> None:
    """Ingest every PDF in pdf_folder, running only the steps whose output is missing:

    - no extracted_folder/{source}.txt     -> extract the PDF
    - no chunk_folder/{source}.chunk_*.md  -> chunk the extracted text (old db rows of that PDF are removed first)
    - chunk files not in the db            -> embed and save those files

    First, the files and db rows of PDFs that are no longer in pdf_folder are removed.
    """
    remove_deleted_pdfs(pdf_folder, extracted_folder, chunk_folder, db)

    for pdf_file in sorted(Path(pdf_folder).glob("*.pdf")):
        source = pdf_file.stem

        extracted_file = Path(extracted_folder) / f"{source}.txt"
        if not extracted_file.exists():
            extract(pdf_file, extracted_file)

        chunk_files = sorted(Path(chunk_folder).glob(f"{source}.chunk_*.md"))
        if not chunk_files:
            db.delete(source)
            chunk_files = chunk_document(source, extracted_file.read_text(encoding="utf-8"), llm, chunk_folder)

        stored = db.filenames(source)
        embed_chunks([file for file in chunk_files if file.name not in stored], embedder, db)


def remove_deleted_pdfs(pdf_folder: str | Path, extracted_folder: str | Path, chunk_folder: str | Path,
                        db: VectorDbContext) -> None:
    """Delete the extracted text, chunk files and db rows of every source whose PDF is gone from pdf_folder."""
    sources = {pdf_file.stem for pdf_file in Path(pdf_folder).glob("*.pdf")}

    for file in Path(extracted_folder).glob("*.txt"):
        if file.stem not in sources:
            file.unlink()

    for file in Path(chunk_folder).glob("*.chunk_*.md"):
        if source_of(file.name) not in sources:
            file.unlink()

    for source in {source_of(filename) for filename in db.filenames()} - sources:
        db.delete(source)


def source_of(chunk_filename: str) -> str:
    """'Doc.chunk_3.md' -> 'Doc'."""
    return chunk_filename.rsplit(".chunk_", 1)[0]


def extract(pdf_file: Path, out_file: Path) -> None:
    """Extract the PDF's text and save it to out_file."""
    pages = PyPDFLoader(str(pdf_file)).load()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n\n".join(page.page_content for page in pages), encoding="utf-8")


def chunk_document(source: str, text: str, llm: ChatProvider, out_dir: str | Path) -> list[Path]:
    """Run the agent on the text and save each chunk as out_dir/{source}.chunk_{N}.md. Returns the saved files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for n, chunk in enumerate(chunking_agent(text, llm), start=1):
        file = out_dir / f"{source}.chunk_{n}.md"
        file.write_text(chunk, encoding="utf-8")
        files.append(file)
    return files


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


def embed_chunks(files: list[Path], embedder: OllamaEmbedder, db: VectorDbContext) -> None:
    """Embed the text of each chunk file and save (filename, vector) to the db."""
    if not files:
        return
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
