"""Shared settings read from the environment (.env)."""
from Utility.env import env

OLLAMA_BASE_URL = env("OLLAMA_BASE_URL", "http://localhost:11434")
LANGSMITH_API_KEY = env("LANGSMITH_API_KEY")
HF_TOKEN = env("HF_TOKEN")


if __name__ == "__main__":
    # Never print secret values; only show whether they are set.
    for key in ("OLLAMA_BASE_URL", "LANGSMITH_API_KEY", "HF_TOKEN"):
        print(f"{key}: {'set' if env(key) else 'NOT set'}")
