"""Environment variable access. Usage: from Utility.env import env; env("HF_TOKEN", "default")"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (parent of Utility), regardless of the current working directory.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    """Return the value of an environment variable, or `default` if unset or empty.

    With required=True, raise RuntimeError when no value (or default) is available.
    """
    value = os.getenv(name) or default
    if required and not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
