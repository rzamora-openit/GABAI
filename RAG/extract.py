"""PDF text extraction and page windows for the ingestion agent."""
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_pages(pdf_path: Path) -> list[str]:
    """Return one string per page; a page without extractable text gives ''."""
    reader = PdfReader(str(pdf_path))
    return [page.extract_text() or "" for page in reader.pages]


def has_text(pages: list[str]) -> bool:
    """True if any page has non-whitespace text (False for a scanned, image-only PDF)."""
    return any(page.strip() for page in pages)


def make_windows(pages: list[str], max_chars: int) -> list[str]:
    """Pack whole pages, in order, into windows of at most max_chars (pages joined by a blank line).

    Blank pages are skipped. A page longer than max_chars becomes a window of its own.
    """
    windows: list[str] = []
    current = ""
    for page in pages:
        text = page.strip()
        if not text:
            continue
        candidate = f"{current}\n\n{text}" if current else text
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                windows.append(current)
            current = text
    if current:
        windows.append(current)
    return windows
