"""Markdown sections: parse a document into sections, pop the last one, split long ones for embedding."""
import re
from dataclasses import dataclass

_HEADING = re.compile(r"^## (.*)$", re.MULTILINE)  # "## Title"; deeper levels ("### x") do not start a section


@dataclass
class Section:
    title: str
    body: str
    order: int  # 0-based position in the document


def parse_markdown(markdown: str, default_title: str) -> list[Section]:
    """Split Markdown at `## ` headings.

    A leading `# ` title line is ignored. Text before the first heading (or all the text when there is no
    heading) becomes a section called `default_title`. Sections with an empty body are dropped.
    """
    matches = list(_HEADING.finditer(markdown))
    preface = markdown[: matches[0].start()] if matches else markdown
    preface = "\n".join(line for line in preface.splitlines() if not line.startswith("# ")).strip()

    raw: list[tuple[str, str]] = []
    if preface:
        raw.append((default_title, preface))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        raw.append((match.group(1).strip(), markdown[match.end():end].strip()))

    kept = [(title, body) for title, body in raw if body]
    return [Section(title, body, order) for order, (title, body) in enumerate(kept)]


def pop_last_section(markdown: str) -> tuple[str, str]:
    """Return (markdown without its last `## ` block, that block including its heading line).

    With no `## ` heading there is nothing to pop: returns (markdown, "").
    """
    matches = list(_HEADING.finditer(markdown))
    if not matches:
        return markdown, ""
    start = matches[-1].start()
    return markdown[:start], markdown[start:].strip()


def _cut(paragraph: str, max_chars: int) -> list[str]:
    """Cut one paragraph into chunks of at most max_chars, at the last whitespace before the limit."""
    chunks = []
    while len(paragraph) > max_chars:
        cut = max(paragraph.rfind(ch, 0, max_chars + 1) for ch in " \n\t")
        if cut <= 0:
            cut = max_chars  # no whitespace to cut at
        chunks.append(paragraph[:cut].rstrip())
        paragraph = paragraph[cut:].lstrip()
    if paragraph:
        chunks.append(paragraph)
    return chunks


def split_section(section: Section, max_chars: int) -> list[str]:
    """Split a section body into pieces of at most max_chars, at paragraph boundaries where possible."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", section.body) if p.strip()]
    pieces: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for chunk in _cut(paragraph, max_chars):
            candidate = f"{current}\n\n{chunk}" if current else chunk
            if len(candidate) <= max_chars:
                current = candidate
            else:
                pieces.append(current)
                current = chunk
    if current:
        pieces.append(current)
    return pieces
