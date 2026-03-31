"""
rag/pdf_parser.py — Extract text from PDF and DOCX files, page by page

WHY PAGE-BY-PAGE EXTRACTION:
Every chunk we store in Pinecone needs a `page_num` metadata field so that
when the LLM cites sources, it can say "see Page 4" instead of nothing.
If we concatenated all text first and split later, we'd lose that page boundary
information. So we extract per page and carry the page number through the
entire pipeline.

LIBRARIES USED:
- PyMuPDF (imported as `fitz`): fast, accurate PDF parsing with page access
- python-docx: DOCX paragraph-level extraction (DOCX has no native "pages",
  so we simulate pages by grouping every N paragraphs)
"""

import logging
import fitz  # PyMuPDF
from docx import Document as DocxDocument
from pathlib import Path

from utils.text_cleaner import clean_page_text, remove_repeated_headers_footers

logger = logging.getLogger(__name__)

# For DOCX files: how many paragraphs we treat as one "virtual page"
# WHY: DOCX has no real page concept at the text level. We group paragraphs
# so we can still attach meaningful page_num metadata to each chunk.
DOCX_PARAGRAPHS_PER_PAGE = 20


def parse_pdf(file_path: str) -> list[dict]:
    """
    Extract text from a PDF file, one dict per page.

    Process:
    1. Open the PDF with PyMuPDF
    2. Iterate each page, extract plain text
    3. Clean the text (remove headers/footers, normalize whitespace)
    4. Return list of {page_num, text} dicts

    WHY WE USE get_text("text") AND NOT "html" OR "blocks":
    Plain text mode is the most portable for downstream NLP. HTML adds markup
    we'd have to strip anyway. "blocks" mode is useful for layout analysis
    but overkill for our RAG use case.

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        List of {"page_num": int, "text": str} dicts (1-indexed page numbers).

    Raises:
        ValueError: If the file cannot be opened or has no extractable text.
    """
    pages = []

    try:
        # fitz.open() loads the PDF into memory; it handles encrypted/corrupt
        # files gracefully by raising exceptions we can catch
        doc = fitz.open(file_path)
        logger.info("Opened PDF: %s — %d pages", file_path, len(doc))

        for page_index in range(len(doc)):
            page = doc[page_index]

            # extract_text returns the raw string for this page
            # flags=0 means no extra formatting hints — clean plain text
            raw_text = page.get_text("text")

            # Clean the raw text: remove page numbers, normalize whitespace
            cleaned = clean_page_text(raw_text)

            # Only include pages with meaningful content (skip blank pages)
            if cleaned.strip():
                pages.append(
                    {
                        # 1-indexed so citations say "Page 1" not "Page 0"
                        "page_num": page_index + 1,
                        "text": cleaned,
                    }
                )

        doc.close()

    except Exception as e:
        logger.error("Failed to parse PDF '%s': %s", file_path, e)
        raise ValueError(f"Could not parse PDF: {e}") from e

    if not pages:
        raise ValueError("PDF produced no extractable text. It may be image-only (scanned).")

    # Remove repeated header/footer lines across pages
    pages = remove_repeated_headers_footers(pages)

    logger.info(
        "PDF parse complete: %d pages with content extracted from '%s'",
        len(pages),
        Path(file_path).name,
    )
    return pages


def parse_docx(file_path: str) -> list[dict]:
    """
    Extract text from a DOCX file, grouped into virtual pages.

    DOCX files don't have a native page structure accessible via python-docx
    (page breaks are a rendering concern, not a text structure). So we group
    every DOCX_PARAGRAPHS_PER_PAGE paragraphs into a "virtual page". This keeps
    the page_num metadata useful without being perfectly accurate.

    Args:
        file_path: Absolute path to the DOCX file.

    Returns:
        List of {"page_num": int, "text": str} dicts.

    Raises:
        ValueError: If the file cannot be opened.
    """
    pages = []

    try:
        doc = DocxDocument(file_path)
        logger.info("Opened DOCX: %s", file_path)

        # Collect all non-empty paragraphs
        paragraphs = [
            para.text.strip()
            for para in doc.paragraphs
            if para.text.strip()
        ]

        # Group into virtual pages
        virtual_page = 1
        for i in range(0, len(paragraphs), DOCX_PARAGRAPHS_PER_PAGE):
            chunk_paragraphs = paragraphs[i : i + DOCX_PARAGRAPHS_PER_PAGE]
            page_text = "\n\n".join(chunk_paragraphs)
            cleaned = clean_page_text(page_text)

            if cleaned.strip():
                pages.append({"page_num": virtual_page, "text": cleaned})

            virtual_page += 1

    except Exception as e:
        logger.error("Failed to parse DOCX '%s': %s", file_path, e)
        raise ValueError(f"Could not parse DOCX: {e}") from e

    if not pages:
        raise ValueError("DOCX produced no extractable text.")

    logger.info(
        "DOCX parse complete: %d virtual pages from '%s'",
        len(pages),
        Path(file_path).name,
    )
    return pages


def parse_document(file_path: str) -> list[dict]:
    """
    Route to the correct parser based on file extension.

    This is the single entry point for Phase 2 ingestion. The caller doesn't
    need to know whether they have a PDF or DOCX — they just call this function.

    Args:
        file_path: Absolute path to a .pdf or .docx file.

    Returns:
        List of {"page_num": int, "text": str} dicts.

    Raises:
        ValueError: If the file type is unsupported.
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        return parse_pdf(file_path)
    elif ext == ".docx":
        return parse_docx(file_path)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Only .pdf and .docx are accepted."
        )
