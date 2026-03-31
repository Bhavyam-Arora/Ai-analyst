"""
rag/ingest.py — Orchestrate the full document ingestion pipeline

WHY AN ORCHESTRATOR:
The RAG pipeline has 4 distinct steps: parse → chunk → embed → upsert.
Each step produces output consumed by the next. By putting the orchestration
logic here (instead of inside the upload API route), we keep each component
single-responsibility and make it easy to test each step independently.

THE PIPELINE:
1. pdf_parser.parse_document() → list of {page_num, text} per page
2. chunker.chunk_pages()       → list of {chunk_text, page_num, chunk_index, ...}
3. embedder.embed_chunks()     → same list + "embedding" key on each chunk
4. pinecone_client.upsert_chunks() → vectors stored in Pinecone

The result is an `IngestionResult` with metadata about what was processed,
useful for logging and for the API response to the frontend.
"""

import asyncio
import logging
import os
import uuid
from pathlib import Path

from rag.pdf_parser import parse_document
from rag.chunker import chunk_pages
from rag.embedder import embed_chunks
from rag.pinecone_client import upsert_chunks, delete_document

logger = logging.getLogger(__name__)


class IngestionResult:
    """
    Simple data class holding the outcome of a document ingestion run.

    Having a structured result (rather than a raw dict) makes it easy to:
    - Return a typed Pydantic response from the API
    - Log exactly what happened at each stage
    - Detect partial failures (parsed OK but embedding failed, etc.)
    """

    def __init__(
        self,
        doc_id: str,
        filename: str,
        page_count: int,
        chunk_count: int,
        vector_count: int,
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.page_count = page_count
        self.chunk_count = chunk_count
        self.vector_count = vector_count

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "vector_count": self.vector_count,
        }


async def ingest_document(file_path: str, filename: str) -> IngestionResult:
    """
    Run the complete parse → chunk → embed → upsert pipeline for one document.

    A unique doc_id is generated here (UUID4). This ID:
    - Becomes the Pinecone namespace for all this document's vectors
    - Is returned to the frontend so subsequent /analyze and /chat calls
      can specify which document they're working on
    - Acts as the "session key" throughout the multi-agent pipeline

    WHY UUID4 (not a hash of the file):
    UUID4 is random and guaranteed unique. A hash would be the same for
    identical files, which could cause namespace collisions in Pinecone if
    two users upload the same document. UUID4 gives each upload its own
    isolated namespace.

    Args:
        file_path: Absolute path to the saved upload file.
        filename: Original filename (for logging and the API response).

    Returns:
        IngestionResult with doc_id, page_count, chunk_count, vector_count.

    Raises:
        ValueError: If parsing produces no content.
        RuntimeError: If embedding or Pinecone upsert fails.
    """
    # Generate a unique ID for this document ingestion
    doc_id = str(uuid.uuid4())

    logger.info(
        "=== Ingestion START: doc_id=%s, file=%s ===", doc_id, filename
    )

    # ── STEP 1: Parse ──────────────────────────────────────────────────────────
    # parse_document() handles both PDF and DOCX based on file extension
    logger.info("[1/4] Parsing document: %s", filename)
    pages = parse_document(file_path)
    page_count = len(pages)
    logger.info("[1/4] Parse complete: %d pages with content", page_count)

    # ── STEP 2: Chunk ──────────────────────────────────────────────────────────
    # chunk_pages() splits pages into overlapping token-bounded chunks
    logger.info("[2/4] Chunking %d pages...", page_count)
    chunks = chunk_pages(pages)
    chunk_count = len(chunks)
    logger.info("[2/4] Chunking complete: %d chunks generated", chunk_count)

    if chunk_count == 0:
        raise ValueError(f"Document '{filename}' produced no chunks after parsing.")

    # ── STEP 3: Embed ──────────────────────────────────────────────────────────
    # embed_chunks() calls OpenAI in batches and attaches embeddings to chunk dicts
    logger.info("[3/4] Embedding %d chunks via OpenAI...", chunk_count)
    embedded_chunks = await embed_chunks(chunks)
    logger.info("[3/4] Embedding complete: %d vectors generated", len(embedded_chunks))

    # ── STEP 4: Upsert ─────────────────────────────────────────────────────────
    # upsert_chunks() stores vectors in Pinecone under the doc_id namespace
    logger.info("[4/4] Upserting to Pinecone (namespace=%s)...", doc_id)
    vector_count = upsert_chunks(doc_id, embedded_chunks)
    logger.info("[4/4] Upsert complete: %d vectors stored", vector_count)

    result = IngestionResult(
        doc_id=doc_id,
        filename=filename,
        page_count=page_count,
        chunk_count=chunk_count,
        vector_count=vector_count,
    )

    logger.info(
        "=== Ingestion COMPLETE: doc_id=%s | pages=%d | chunks=%d | vectors=%d ===",
        doc_id,
        page_count,
        chunk_count,
        vector_count,
    )

    return result


async def reingest_document(
    doc_id: str, file_path: str, filename: str
) -> IngestionResult:
    """
    Re-ingest a document that was already ingested (e.g., re-upload by user).

    Process:
    1. Delete all existing Pinecone vectors for this doc_id
    2. Re-run the full ingest pipeline

    WHY THIS IS SEPARATE FROM ingest_document:
    We want to be explicit about deleting old data before re-ingesting.
    A simple re-ingest would upsert on top of existing vectors (same vector IDs
    overwrite), which works for Pinecone upserts, but we might also have
    chunks from the old version that no longer exist in the new version.
    Deleting first ensures a clean slate.

    Args:
        doc_id: Existing document ID to replace.
        file_path: Path to the new file.
        filename: New filename.

    Returns:
        IngestionResult with same doc_id and updated counts.
    """
    logger.info("Re-ingesting doc_id=%s — deleting old vectors first", doc_id)
    delete_document(doc_id)

    # Run the normal pipeline but with the existing doc_id instead of a new UUID
    logger.info("=== Re-ingestion START: doc_id=%s, file=%s ===", doc_id, filename)

    pages = parse_document(file_path)
    chunks = chunk_pages(pages)
    embedded_chunks = await embed_chunks(chunks)
    vector_count = upsert_chunks(doc_id, embedded_chunks)

    result = IngestionResult(
        doc_id=doc_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
        vector_count=vector_count,
    )

    logger.info(
        "=== Re-ingestion COMPLETE: doc_id=%s | vectors=%d ===",
        doc_id,
        vector_count,
    )
    return result
