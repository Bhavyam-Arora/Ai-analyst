"""
rag/retriever.py — Convert a natural language query into formatted LLM context

WHY A SEPARATE RETRIEVER MODULE:
Every agent (extraction, risk, Q&A) needs to retrieve relevant chunks before
calling GPT-4o. Rather than duplicating the embed → query → format logic in
each agent, we centralize it here. Agents just call:

    context = await retrieve_context(doc_id, query)

And they get back a formatted string ready to inject into a prompt.

THE RETRIEVAL FLOW:
1. Embed the query string using the same model as document ingestion
   (critical: query and document embeddings MUST use the same model)
2. Query Pinecone for the top_k most similar chunks in the doc's namespace
3. Format the chunks into a readable context block for the LLM prompt

FORMAT DESIGN:
Each chunk is labeled with its page number so the LLM can include citations
like [Page 4] in its response. The format is:

    [Page 3 — Section: Termination Clauses]
    Either party may terminate this agreement with 30 days written notice...

    [Page 5 — Section: Governing Law]
    This agreement shall be governed by the laws of the State of California...

This format is deliberately readable — the LLM can parse "Page 3" from it
easily, and a human reviewing the prompt can also understand it.
"""

import logging
import os
from rag.embedder import embed_query
from rag.pinecone_client import retrieve_chunks

logger = logging.getLogger(__name__)


async def retrieve_context(
    doc_id: str,
    query: str,
    top_k: int | None = None,
) -> str:
    """
    Retrieve the most relevant document chunks for a query and format them
    as a context string for LLM prompt injection.

    This is the main function agents call. It combines embedder + pinecone_client
    into a single, simple interface.

    Args:
        doc_id: The document namespace to search within.
        query: Natural language query string (e.g., "What are the termination clauses?")
        top_k: Number of chunks to retrieve. Defaults to PINECONE_TOP_K env var.

    Returns:
        Multi-line string with all retrieved chunks, formatted for LLM injection.
        Returns empty string if no chunks are found (agents should handle this).

    Raises:
        RuntimeError: If embedding or Pinecone query fails.
    """
    if top_k is None:
        top_k = int(os.getenv("PINECONE_TOP_K", "5"))

    logger.info(
        "Retrieving context for doc_id=%s, query='%s...', top_k=%d",
        doc_id,
        query[:60],
        top_k,
    )

    # Step 1: Embed the query
    # WHY: We can't compare text to vectors directly — we must convert the
    # query to the same vector space as the stored chunks
    query_embedding = await embed_query(query)

    # Step 2: Retrieve top_k closest chunks from Pinecone
    chunks = retrieve_chunks(doc_id, query_embedding, top_k=top_k)

    if not chunks:
        logger.warning(
            "No chunks retrieved for doc_id=%s, query='%s...'",
            doc_id,
            query[:60],
        )
        return ""

    # Step 3: Format chunks into a readable context block
    context = _format_chunks_as_context(chunks)

    logger.info(
        "Context assembled: %d chunks, %d characters",
        len(chunks),
        len(context),
    )
    return context


def _format_chunks_as_context(chunks: list[dict]) -> str:
    """
    Format a list of retrieved chunk dicts into a single context string.

    FORMATTING DECISIONS:
    - Sort by page_num so the LLM reads content in document order
      (retrieval returns by relevance score, not page order)
    - Label each chunk with page number and section title
    - Separate chunks with a blank line so they're visually distinct
    - Include the similarity score as a comment (optional, helps debugging)

    Args:
        chunks: List of chunk dicts from retrieve_chunks():
                [{"chunk_text": str, "page_num": int, "section_title": str, "score": float}]

    Returns:
        Formatted context string.
    """
    # Sort by page number so the context reads chronologically
    sorted_chunks = sorted(chunks, key=lambda c: c["page_num"])

    formatted_parts = []
    for chunk in sorted_chunks:
        page_num = chunk.get("page_num", "?")
        section_title = chunk.get("section_title", "")
        chunk_text = chunk.get("chunk_text", "")

        # Build the header line: "[Page 3]" or "[Page 3 — Section Title]"
        if section_title:
            header = f"[Page {page_num} — {section_title}]"
        else:
            header = f"[Page {page_num}]"

        formatted_parts.append(f"{header}\n{chunk_text}")

    # Join chunks with double newline for visual separation in the prompt
    return "\n\n".join(formatted_parts)


async def retrieve_context_multi_query(
    doc_id: str,
    queries: list[str],
    top_k_per_query: int = 3,
) -> str:
    """
    Retrieve context using multiple query strings and deduplicate results.

    WHY MULTI-QUERY:
    The extraction agent needs to find many different things: parties, dates,
    termination clauses, payment terms, etc. A single query like "extract all
    key information" is too vague and may miss specific sections.

    By running targeted sub-queries (e.g., "parties to the agreement",
    "payment due dates", "termination conditions"), we retrieve more precise
    chunks for each information type.

    DEDUPLICATION:
    The same chunk might be returned for multiple queries (e.g., a section
    mentioning both payment terms and termination). We deduplicate by
    chunk_index to avoid repeating the same text in the context.

    Args:
        doc_id: Document namespace.
        queries: List of targeted query strings.
        top_k_per_query: Chunks to retrieve per query (lower than single-query
                         top_k since we're running multiple queries).

    Returns:
        Deduplicated, formatted context string.
    """
    import asyncio

    # Run all query embeddings concurrently (not sequentially)
    # asyncio.gather() fires all embed_query calls at the same time
    query_embeddings = await asyncio.gather(
        *[embed_query(q) for q in queries]
    )

    # Retrieve chunks for each query embedding
    all_chunks = []
    seen_chunk_indices = set()

    from rag.pinecone_client import retrieve_chunks as _retrieve

    for query, embedding in zip(queries, query_embeddings):
        chunks = _retrieve(doc_id, embedding, top_k=top_k_per_query)

        for chunk in chunks:
            chunk_index = chunk.get("chunk_index")
            # Deduplicate: skip if we've already included this chunk
            if chunk_index not in seen_chunk_indices:
                seen_chunk_indices.add(chunk_index)
                all_chunks.append(chunk)

    logger.info(
        "Multi-query retrieval: %d queries → %d unique chunks for doc_id=%s",
        len(queries),
        len(all_chunks),
        doc_id,
    )

    if not all_chunks:
        return ""

    return _format_chunks_as_context(all_chunks)
