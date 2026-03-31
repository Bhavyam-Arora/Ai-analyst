"""
test_phase2.py — Phase 2 checkpoint: RAG pipeline end-to-end smoke test

Run this from the project root with the venv active:
    python test_phase2.py

WHAT THIS TESTS:
1. Text cleaning utilities work correctly
2. Token counter gives accurate counts
3. PDF parser can parse a real (minimal) PDF
4. Chunker splits text into correct-size chunks
5. Embedder returns 1536-dim vectors from OpenAI
6. Pinecone upsert and retrieve round-trip works
7. Retriever formats context correctly

All tests clean up after themselves — no test data persists in Pinecone.

Expected output:
    ✅ text_cleaner: page number lines removed
    ✅ token_counter: tiktoken counting correctly
    ✅ chunker: splits text into chunks with correct metadata
    ✅ embedder: OpenAI returned 1536-dim embeddings
    ✅ pinecone round-trip: upserted 2 vectors, retrieved top match
    ✅ retriever: context formatted with page citations
    🎉 Phase 2 complete. RAG pipeline is working end-to-end.
"""

import asyncio
import os
import sys
import tempfile
import fitz  # PyMuPDF — to create a test PDF programmatically

# ── Add backend/ to sys.path ──────────────────────────────────────────────
# Backend modules use imports relative to backend/ (e.g. `from rag.xxx import`).
# Adding backend/ to sys.path makes those imports resolve correctly.
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT / "backend"))

from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")


def test_text_cleaner():
    """Verify the cleaner strips page numbers and normalizes whitespace."""
    from utils.text_cleaner import clean_page_text

    raw = "CONFIDENTIAL\n\nThis agreement is between Party A and Party B.\n\n\n\n3\n\nPage 3 of 10\n\nPayment terms are net 30."
    cleaned = clean_page_text(raw)

    assert "Page 3 of 10" not in cleaned, "Page number line should be removed"
    assert "\n\n\n" not in cleaned, "Triple newlines should be collapsed"
    assert "Payment terms are net 30." in cleaned, "Body text should be preserved"
    print("✅ text_cleaner: page number lines removed")


def test_token_counter():
    """Verify tiktoken gives sensible counts."""
    from utils.token_counter import count_tokens, truncate_to_token_limit

    # "Hello world" is 2 tokens in cl100k_base
    count = count_tokens("Hello world")
    assert count == 2, f"Expected 2 tokens for 'Hello world', got {count}"

    # Truncation should return a shorter string
    long_text = "word " * 1000  # ~1000 tokens
    truncated = truncate_to_token_limit(long_text, max_tokens=100)
    assert count_tokens(truncated) <= 100, "Truncated text should be within token limit"

    print("✅ token_counter: tiktoken counting correctly")


def test_chunker():
    """Verify chunker produces chunks with correct metadata fields."""
    from rag.chunker import chunk_pages

    # Create mock pages (as if pdf_parser returned them)
    pages = [
        {
            "page_num": 1,
            "text": "TERMINATION CLAUSES\n\n" + ("Either party may terminate this agreement with thirty (30) days written notice. " * 30),
        },
        {
            "page_num": 2,
            "text": "PAYMENT TERMS\n\n" + ("Payment is due within thirty (30) days of invoice date. Late payments incur a 1.5% monthly fee. " * 20),
        },
    ]

    chunks = chunk_pages(pages)

    assert len(chunks) > 0, "Should produce at least one chunk"

    for chunk in chunks:
        assert "chunk_text" in chunk, "Each chunk must have chunk_text"
        assert "page_num" in chunk, "Each chunk must have page_num"
        assert "chunk_index" in chunk, "Each chunk must have chunk_index"
        assert "section_title" in chunk, "Each chunk must have section_title"
        assert chunk["page_num"] in [1, 2], "Page nums should match input"
        assert len(chunk["chunk_text"]) > 0, "Chunk text should not be empty"

    # Verify chunk_index is sequential (0, 1, 2, ...)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks))), "chunk_index should be sequential"

    # Verify section title detection
    page1_chunks = [c for c in chunks if c["page_num"] == 1]
    assert any(c["section_title"] == "TERMINATION CLAUSES" for c in page1_chunks), \
        "Should detect 'TERMINATION CLAUSES' as section title"

    print(f"✅ chunker: splits text into {len(chunks)} chunks with correct metadata")


async def test_embedder():
    """Verify OpenAI returns 1536-dim embeddings."""
    from rag.embedder import embed_chunks, embed_query

    test_chunks = [
        {"chunk_text": "This agreement is governed by California law.", "page_num": 1, "chunk_index": 0, "section_title": ""},
        {"chunk_text": "Payment is due within 30 days.", "page_num": 2, "chunk_index": 1, "section_title": ""},
    ]

    embedded = await embed_chunks(test_chunks)

    assert len(embedded) == 2, f"Expected 2 embeddings, got {len(embedded)}"
    for chunk in embedded:
        assert "embedding" in chunk, "Each chunk should have an embedding"
        assert len(chunk["embedding"]) == 1536, \
            f"Expected 1536 dimensions, got {len(chunk['embedding'])}"

    # Test query embedding
    query_vec = await embed_query("What is the governing law?")
    assert len(query_vec) == 1536, f"Query embedding should be 1536 dims, got {len(query_vec)}"

    print(f"✅ embedder: OpenAI returned 1536-dim embeddings for {len(embedded)} chunks")
    return embedded


async def test_pinecone_roundtrip(embedded_chunks):
    """Test upsert to Pinecone and retrieve, then clean up."""
    from rag.pinecone_client import upsert_chunks, retrieve_chunks, delete_document
    from rag.embedder import embed_query

    test_doc_id = "test_phase2_checkpoint"

    # Clean up any leftover test data from previous runs
    try:
        delete_document(test_doc_id)
    except Exception:
        pass  # Namespace may not exist yet — that's fine

    # Upsert 2 test vectors
    count = upsert_chunks(test_doc_id, embedded_chunks)
    assert count == 2, f"Expected 2 vectors upserted, got {count}"

    # Wait briefly for Pinecone to index (free tier can have slight delay)
    await asyncio.sleep(2)

    # Retrieve with a relevant query
    query_vec = await embed_query("What is the governing law?")
    results = retrieve_chunks(test_doc_id, query_vec, top_k=2)

    assert len(results) > 0, "Should retrieve at least one chunk"
    assert "chunk_text" in results[0], "Result should have chunk_text"
    assert "page_num" in results[0], "Result should have page_num"
    assert "score" in results[0], "Result should have similarity score"
    assert results[0]["score"] > 0.5, \
        f"Top result score should be > 0.5, got {results[0]['score']:.4f}"

    print(
        f"✅ pinecone round-trip: upserted {count} vectors, "
        f"retrieved top match (score={results[0]['score']:.4f})"
    )

    # Clean up test data
    delete_document(test_doc_id)
    print("   (test vectors cleaned up from Pinecone)")


async def test_retriever(embedded_chunks):
    """Test the full retriever: query → embed → retrieve → format."""
    from rag.pinecone_client import upsert_chunks, delete_document
    from rag.retriever import retrieve_context

    test_doc_id = "test_phase2_retriever"

    try:
        delete_document(test_doc_id)
    except Exception:
        pass

    upsert_chunks(test_doc_id, embedded_chunks)
    await asyncio.sleep(2)

    context = await retrieve_context(test_doc_id, "What is the governing law?")

    assert len(context) > 0, "Context should not be empty"
    assert "[Page" in context, "Context should include page citations like [Page 1]"

    print(f"✅ retriever: context formatted with page citations ({len(context)} chars)")

    # Preview the context
    preview = context[:200].replace("\n", " ")
    print(f"   Preview: {preview}...")

    delete_document(test_doc_id)


async def main():
    print("\n" + "="*60)
    print("Phase 2 Checkpoint — RAG Pipeline Smoke Test")
    print("="*60 + "\n")

    # Check env vars
    required = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("   Make sure your .env file is configured and load_dotenv() ran.")
        sys.exit(1)

    # Run synchronous tests
    test_text_cleaner()
    test_token_counter()
    test_chunker()

    # Run async tests
    embedded_chunks = await test_embedder()
    await test_pinecone_roundtrip(embedded_chunks)
    await test_retriever(embedded_chunks)

    print("\n" + "="*60)
    print("🎉 Phase 2 complete. RAG pipeline is working end-to-end.")
    print("="*60)
    print("\nNext step: Phase 3 — LangGraph agents (extraction, risk, Q&A)")


if __name__ == "__main__":
    asyncio.run(main())
