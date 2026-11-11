"""Fidelity check: did the ingestion agent keep the PDF's text, without dropping or inventing any?"""
import re
from collections import Counter
from dataclasses import dataclass

from Utility.config import FIDELITY_MAX_EXTRA, FIDELITY_MIN_COVERAGE


@dataclass
class FidelityReport:
    coverage: float  # share of the source's words that appear in the Markdown
    extra: float  # share of the Markdown's words that are not in the source
    passed: bool


def _words(text: str) -> Counter:
    return Counter(re.findall(r"\w+", text.lower()))


def check_fidelity(
    source_text: str,
    markdown: str,
    min_coverage: float = FIDELITY_MIN_COVERAGE,
    max_extra: float = FIDELITY_MAX_EXTRA,
) -> FidelityReport:
    """Compare the words of the source text with the words of the Markdown (as multisets, order ignored)."""
    source, produced = _words(source_text), _words(markdown)
    n_source, n_produced = sum(source.values()), sum(produced.values())
    if n_source == 0:
        return FidelityReport(0.0, 1.0 if n_produced else 0.0, False)
    overlap = sum((source & produced).values())
    coverage = overlap / n_source
    extra = 1 - overlap / n_produced if n_produced else 0.0
    return FidelityReport(coverage, extra, coverage >= min_coverage and extra <= max_extra)
