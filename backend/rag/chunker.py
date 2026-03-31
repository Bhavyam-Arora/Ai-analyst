"""
rag/chunker.py — Split parsed pages into overlapping chunks for embedding

WHY CHUNKING MATTERS (The Core RAG Trade-off):
When we embed a document, we can't send all 50 pages to OpenAI at once —
that would produce one massive, blurry embedding that represents "the whole
document" rather than any specific clause or section.

Instead, we split the text into small, focused chunks (~900 tokens each),
embed each chunk independently, and store them in Pinecone. At query time,
we only retrieve the 5 most relevant chunks. This gives the LLM:
- Precise, relevant context (not irrelevant pages)
- Grounded answers (we know exactly which page each chunk came from)

CHUNK SIZE DECISION (900 tokens / 150 overlap):
- Too small (< 300 tokens): Chunks lose context (a clause split mid-sentence)
- Too large (> 1500 tokens): Chunks are too general, retrieval is imprecise
- 900 tokens ≈ 3–4 paragraphs — ideal for contract clauses
- 150-token overlap: Prevents clause boundaries being split across chunks
  without the LLM having both halves

SPLITTER HIERARCHY (why RecursiveCharacterTextSplitter):
LangChain's RecursiveCharacterTextSplitter tries each separator in order:
  1. "\n\n" — paragraph breaks (preferred, keeps semantics intact)
  2. "\n"   — line breaks (fallback if paragraph is too long)
  3. ". "   — sentence boundaries (fallback for run-on paragraphs)
  4. " "    — word boundaries (last resort)
  5. ""     — character-level (absolute last resort, almost never used)

This hierarchy means we respect document structure as much as possible.
"""

import logging
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils.token_counter import count_tokens

logger = logging.getLogger(__name__)

# Read chunk config from environment (set in .env — see CLAUDE.md)
# Using env vars makes it easy to tune without code changes
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))        # tokens per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))  # overlap between chunks


def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    Build and return the LangChain text splitter.

    WHY length_function=count_tokens:
    By default, LangChain measures chunk size in characters. But OpenAI's
    embedding model has a 8191 TOKEN limit, not a character limit. A character
    count of 900 could be 200 tokens (simple words) or 1200 tokens (dense
    legal terms). Using tiktoken's count_tokens() gives accurate measurements.

    Returns:
        Configured RecursiveCharacterTextSplitter instance.
    """
    return RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=count_tokens,  # measure in tokens, not characters
        is_separator_regex=False,       # treat separators as literal strings
    )


def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split a list of parsed pages into overlapping chunks with metadata.

    For each page, we run the splitter and record:
    - chunk_text: the actual text for this chunk (what gets embedded)
    - page_num:   which page this chunk came from (used in citations)
    - chunk_index: sequential index across the whole document (used as part of
                   the Pinecone vector ID to ensure uniqueness)
    - section_title: best-effort heading detection (first line if it looks like
                     a heading) — helps the LLM reference the correct section

    WHY WE CHUNK PER-PAGE (not the whole document at once):
    If we concatenated all pages first and then split, a chunk might straddle
    two pages and we'd lose exact page attribution. By chunking per-page, every
    chunk has a definitive page_num.

    The trade-off: very long single paragraphs (spanning a whole page) get
    split by sentence boundaries, which is fine — the overlap ensures continuity.

    Args:
        pages: List of {"page_num": int, "text": str} from pdf_parser.

    Returns:
        List of chunk dicts:
        {
            "chunk_text":    str,   # the text content of this chunk
            "page_num":      int,   # source page number (1-indexed)
            "chunk_index":   int,   # global sequential index (0-indexed)
            "section_title": str,   # detected heading or empty string
        }
    """
    splitter = _build_splitter()
    all_chunks = []
    global_index = 0  # tracks chunk_index across all pages

    for page in pages:
        page_num = page["page_num"]
        text = page["text"]

        # Detect a section title: if the first non-empty line is short and
        # doesn't end with a period, it's likely a heading
        # WHY: Storing section_title in Pinecone metadata helps agents answer
        # "what does section 4 say?" without needing to read every chunk
        section_title = _detect_section_title(text)

        # Split this page's text into chunks
        # split_text() returns a list of strings
        raw_chunks = splitter.split_text(text)

        for raw_chunk in raw_chunks:
            # Skip chunks that are pure whitespace or too short to be useful
            # (these can appear if a page starts/ends with lots of blank lines)
            if len(raw_chunk.strip()) < 20:
                continue

            all_chunks.append(
                {
                    "chunk_text": raw_chunk.strip(),
                    "page_num": page_num,
                    "chunk_index": global_index,
                    "section_title": section_title,
                }
            )
            global_index += 1

    logger.info(
        "Chunking complete: %d pages → %d chunks (size=%d tokens, overlap=%d tokens)",
        len(pages),
        len(all_chunks),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )

    return all_chunks


def _detect_section_title(text: str) -> str:
    """
    Heuristically detect a section/heading title from the start of page text.

    HEURISTIC: A line is likely a heading if:
    - It's the first non-empty line of the page
    - It's <= 80 characters (headings are short)
    - It doesn't end with a period (body text usually ends with ".")
    - It's not ALL lowercase (headings are usually title-case or ALL CAPS)

    This is intentionally simple — a perfect heading detector would need an
    NLP classifier, which is overkill for metadata enrichment.

    Args:
        text: Full page text string.

    Returns:
        Detected section title string, or empty string if none found.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    first_line = lines[0]

    is_short = len(first_line) <= 80
    no_trailing_period = not first_line.endswith(".")
    not_all_lowercase = first_line != first_line.lower()

    if is_short and no_trailing_period and not_all_lowercase:
        return first_line

    return ""
