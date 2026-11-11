"""Shared settings read from the environment (.env)."""
from pathlib import Path

from Utility.env import env

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")
LANGSMITH_API_KEY = env("LANGSMITH_API_KEY")
HF_TOKEN = env("HF_TOKEN")

# ---- RAG settings ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_PDF_DIR = PROJECT_ROOT / "KnowledgeBase"
RAG_MARKDOWN_DIR = RAG_PDF_DIR / "markdown"

EMBED_MODEL = env("RAG_EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = 768

RAG_TOP_N = int(env("RAG_TOP_N", "10"))  # pieces returned by the vector search
RAG_TOP_K = int(env("RAG_TOP_K", "3"))  # pieces kept by the re-ranker

# Names are keys of LLM.MODELS.
INGEST_PROVIDER = env("RAG_INGEST_PROVIDER", "Qwen 3.5 4B (no thinking)")
RERANK_PROVIDER = env("RAG_RERANK_PROVIDER", "Qwen 3.5 4B (no thinking)")

MAX_WINDOW_CHARS = 4000  # sized so window + carried-over section + output fit a 4096-token context
MAX_PIECE_CHARS = 3000  # longest text embedded as one piece
MAX_CARRY_CHARS = 2000  # a section longer than this is not carried into the next window (keeps prompts bounded)
MAX_SAVE_ROUNDS = 40  # tool-call rounds allowed for one window (small models save one section per round)
# Default context window (tokens) for every Ollama model (qwen, gemma, llama, phi in LLM/); override with options={"num_ctx": ...}. Ollama's own
# default is often 4096, which silently drops the oldest messages; a larger value costs more GPU/RAM.
NUM_CTX = int(env("RAG_NUM_CTX", "8192"))
FIDELITY_MIN_COVERAGE = 0.95
FIDELITY_MAX_EXTRA = 0.05

_POSTGRES_KEYS = {
    "host": "host",
    "port": "port",
    "username": "user",
    "user id": "user",
    "password": "password",
    "database": "dbname",
}


def parse_postgres_line(line: str) -> dict[str, str]:
    """Parse 'Host=h;Port=5432;Username=u;Password=p;Database=d' into psycopg keyword arguments.

    Keys are case-insensitive and unknown keys are ignored. Host, user and database are required.
    """
    result: dict[str, str] = {}
    for piece in line.split(";"):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        mapped = _POSTGRES_KEYS.get(key.strip().lower())
        if mapped:
            result[mapped] = value.strip().strip("\"'")
    missing = [name for name in ("host", "user", "dbname") if not result.get(name)]
    if missing:
        raise ValueError(f"Postgres connection line is missing: {', '.join(missing)}")
    return result


def postgres_conninfo() -> dict[str, str]:
    """Connection settings from the owner's .env line (`Host=...;Port=...;Username=...;Password=...;Database=...`).

    That line has no variable name, so python-dotenv loads it as the variable `Host` whose value is the rest of the line.
    """
    value = env("Host")
    if not value:
        raise RuntimeError("Postgres connection line (Host=...;Port=...;Username=...;Password=...;Database=...) not found in .env")
    return parse_postgres_line("Host=" + value)


if __name__ == "__main__":
    # Never print secret values; only show whether they are set.
    for key in ("OLLAMA_BASE_URL", "LANGSMITH_API_KEY", "HF_TOKEN"):
        print(f"{key}: {'set' if env(key) else 'NOT set'}")
