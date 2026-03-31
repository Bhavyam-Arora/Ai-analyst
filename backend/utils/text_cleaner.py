"""
utils/text_cleaner.py — Strip noise from raw PDF-extracted text

WHY THIS EXISTS:
PyMuPDF extracts text per page, but PDFs often embed repeating headers
(company name, doc title) and footers (page numbers, confidentiality notices)
on every page. If we chunk without cleaning, those strings pollute every chunk,
waste embedding space, and confuse the LLM with irrelevant content.

APPROACH:
- Remove lines that look like page numbers (standalone digits or "Page X of Y")
- Remove very short repeated lines (likely headers/footers)
- Normalize whitespace so the chunker sees clean paragraph breaks
"""

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)


def clean_page_text(text: str) -> str:
    """
    Clean a single page's raw extracted text.

    Steps:
    1. Remove standalone page-number lines  (e.g. "3", "Page 3 of 10")
    2. Remove lines that are purely non-alphanumeric (horizontal rules, etc.)
    3. Collapse 3+ consecutive blank lines into two (preserves paragraph breaks
       that the RecursiveCharacterTextSplitter relies on)
    4. Strip leading/trailing whitespace

    Args:
        text: Raw text string from PyMuPDF for one page.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()

        # Skip standalone page numbers: "3", "- 3 -", "Page 3", "Page 3 of 10"
        if re.fullmatch(r"[-–—]?\s*\d+\s*[-–—]?", stripped):
            continue
        if re.fullmatch(r"[Pp]age\s+\d+(\s+of\s+\d+)?", stripped):
            continue

        # Skip lines that contain only punctuation/symbols (horizontal rules)
        if stripped and not re.search(r"[A-Za-z0-9]", stripped):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Collapse excessive blank lines (3+) down to 2
    # Two blank lines = paragraph separator, which LangChain splitter uses
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def remove_repeated_headers_footers(pages: list[dict]) -> list[dict]:
    """
    Detect and remove lines that repeat across many pages — these are almost
    certainly headers or footers (e.g., company name, document title, date).

    WHY: A line like "CONFIDENTIAL — ACME CORP" on every page will appear in
    every chunk and add meaningless tokens to every embedding. Removing it
    reduces noise and improves retrieval precision.

    ALGORITHM:
    - Count how many pages each non-empty line appears on
    - If a line appears on > 40% of pages AND the doc has > 3 pages, flag it
    - Remove flagged lines from all pages

    Args:
        pages: List of {"page_num": int, "text": str} dicts from pdf_parser.

    Returns:
        Same structure with repeated lines stripped out.
    """
    if len(pages) <= 3:
        # Too short to reliably detect headers/footers — skip
        return pages

    # Count occurrences of each line across all pages
    line_counter: Counter = Counter()
    for page in pages:
        unique_lines = set(
            line.strip() for line in page["text"].split("\n") if line.strip()
        )
        line_counter.update(unique_lines)

    # A line is a "repeated header/footer" if it appears on > 40% of pages
    threshold = len(pages) * 0.4
    repeated_lines = {
        line for line, count in line_counter.items() if count >= threshold
    }

    if repeated_lines:
        logger.info(
            "Detected %d repeated header/footer lines across %d pages — removing them.",
            len(repeated_lines),
            len(pages),
        )

    # Strip the flagged lines from each page
    cleaned_pages = []
    for page in pages:
        lines = page["text"].split("\n")
        filtered = [
            line for line in lines if line.strip() not in repeated_lines
        ]
        cleaned_pages.append(
            {"page_num": page["page_num"], "text": "\n".join(filtered)}
        )

    return cleaned_pages
