"""Ingestion pipeline, as a LangGraph graph: PDFs -> extracted text -> chunk markdown files (LLM agent) -> vectors in Postgres.

    START -> remove_deleted -> next_pdf --(no PDF left)--> END
                                  ^   \\
                                  |    extract -> chunk -> embed
                                  +---------------------------/

Every step runs only when its output is missing:
- no extracted_folder/{source}.txt     -> extract the PDF
- no chunk_folder/{source}.chunk_*.md  -> chunk the extracted text (old db rows of that PDF are removed first)
- chunk files not in the db            -> embed and save those files

Run from the project root:   python.exe -m RetAuGen.ingestion_pipeline
"""
from pathlib import Path
from typing import TypedDict

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from Utility.vector_db_context import VectorDbContext


class IngestState(TypedDict, total=False):
    queue: list[str]  # sources (PDF names without .pdf) still to ingest
    source: str  # the source being ingested; "" when there is none left
    ingested: list[str]  # sources done, in order


def build_ingestion_graph(pdf_folder: str | Path, extracted_folder: str | Path, chunk_folder: str | Path,
                          llm: BaseChatModel, embedder: Embeddings, db: VectorDbContext):
    """The compiled ingestion graph. Invoke it with {} (see ingest)."""
    pdf_folder, extracted_folder, chunk_folder = Path(pdf_folder), Path(extracted_folder), Path(chunk_folder)

    def remove_deleted(state: IngestState) -> IngestState:
        remove_deleted_pdfs(pdf_folder, extracted_folder, chunk_folder, db)
        return {"queue": sorted(pdf.stem for pdf in pdf_folder.glob("*.pdf")), "ingested": []}

    def next_pdf(state: IngestState) -> IngestState:
        queue = state["queue"]
        return {"source": queue[0] if queue else "", "queue": queue[1:]}

    def extract_step(state: IngestState) -> IngestState:
        extracted_file = extracted_folder / f"{state['source']}.txt"
        if not extracted_file.exists():
            extract(pdf_folder / f"{state['source']}.pdf", extracted_file)
        return {}

    def chunk_step(state: IngestState) -> IngestState:
        source = state["source"]
        if not list(chunk_folder.glob(f"{source}.chunk_*.md")):
            db.delete(source)
            chunk_document(source, (extracted_folder / f"{source}.txt").read_text(encoding="utf-8"), llm, chunk_folder)
        return {}

    def embed_step(state: IngestState) -> IngestState:
        source = state["source"]
        stored = db.filenames(source)
        chunk_files = sorted(chunk_folder.glob(f"{source}.chunk_*.md"))
        embed_chunks([file for file in chunk_files if file.name not in stored], embedder, db)
        return {"ingested": [*state["ingested"], source]}

    graph = StateGraph(IngestState)
    graph.add_node("remove_deleted", remove_deleted)
    graph.add_node("next_pdf", next_pdf)
    graph.add_node("extract", extract_step)
    graph.add_node("chunk", chunk_step)
    graph.add_node("embed", embed_step)
    graph.add_edge(START, "remove_deleted")
    graph.add_edge("remove_deleted", "next_pdf")
    graph.add_conditional_edges("next_pdf", lambda state: "extract" if state["source"] else END, ["extract", END])
    graph.add_edge("extract", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "next_pdf")
    return graph.compile()


def ingest(pdf_folder: str | Path, extracted_folder: str | Path, chunk_folder: str | Path, llm: BaseChatModel,
           embedder: Embeddings, db: VectorDbContext) -> list[str]:
    """Ingest every PDF in pdf_folder (see the module docstring). Returns the sources ingested, in order.

    First, the files and db rows of PDFs that are no longer in pdf_folder are removed.
    """
    graph = build_ingestion_graph(pdf_folder, extracted_folder, chunk_folder, llm, embedder, db)
    pdf_count = len(list(Path(pdf_folder).glob("*.pdf")))
    # each PDF takes 4 steps (next_pdf, extract, chunk, embed); the default limit of 25 would stop at about 6 PDFs
    final = graph.invoke({}, {"recursion_limit": 4 * pdf_count + 10})
    return final.get("ingested", [])


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


def chunk_document(source: str, text: str, llm: BaseChatModel, out_dir: str | Path) -> list[Path]:
    """Run the agent on the text and save each chunk as out_dir/{source}.chunk_{N}.md. Returns the saved files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for n, chunk in enumerate(chunking_agent(text, llm), start=1):
        file = out_dir / f"{source}.chunk_{n}.md"
        file.write_text(chunk, encoding="utf-8")
        files.append(file)
    return files


CHUNKING_PROMPT = """You split documents into chunks for RAG ingestion.
Split the document by section. Each chunk is one self-contained section in markdown, starting with its heading.
Keep the original wording; do not summarize, skip or invent text.
Return all chunks at once."""


class SaveChunks(BaseModel):
    """Save the document's chunks."""

    chunks: list[str] = Field(description="The chunks, in document order.")


def chunking_agent(text: str, llm: BaseChatModel) -> list[str]:
    """Send the system prompt and the text; return the chunks the agent returned (empty if it returned none).

    llm: the model to use, e.g. azure_openai(temperature=0).
    """
    result = llm.with_structured_output(SaveChunks, method="function_calling").invoke(
        [SystemMessage(CHUNKING_PROMPT), HumanMessage(text)]
    )
    return result.chunks if result else []


def embed_chunks(files: list[Path], embedder: Embeddings, db: VectorDbContext) -> None:
    """Embed the text of each chunk file and save (filename, vector) to the db."""
    if not files:
        return
    texts = [file.read_text(encoding="utf-8") for file in files]
    vectors = embedder.embed_documents(texts)
    for file, vector in zip(files, vectors):
        db.save(file.name, vector)


if __name__ == "__main__":
    from LLM import OllamaEmbedder, azure_openai

    db = VectorDbContext()
    try:
        ingest("KnowledgeBase", "KnowledgeBase/extracted", "KnowledgeBase/chunks", azure_openai(temperature=0), OllamaEmbedder(), db)
    finally:
        db.close()
