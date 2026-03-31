"""
rag/pinecone_client.py — Upsert and retrieve vectors from Pinecone

WHY PINECONE (not a local vector DB like FAISS):
For a portfolio project, Pinecone's free tier gives us:
- Managed infrastructure (no server to run)
- Real-time upsert/query (no index rebuild delay)
- Metadata filtering (filter by doc_id without scanning all vectors)
- Persistence (data survives server restarts)

NAMESPACING STRATEGY:
Pinecone supports "namespaces" — logical partitions within an index.
We use doc_id as the namespace so that:
1. A query for Document A NEVER returns chunks from Document B
2. We can delete all chunks for a document by deleting its namespace
3. Multiple users' documents don't contaminate each other's retrieval

Without namespaces, we'd need to add a metadata filter `doc_id == X` to
every query, which is slower and more error-prone.

VECTOR ID FORMAT:
Each vector needs a unique ID within its namespace.
We use: `{doc_id}_{chunk_index}` — e.g., "abc123_0", "abc123_1"
This makes it easy to reconstruct which document a vector belongs to.
"""

import logging
import os
from pinecone import Pinecone

logger = logging.getLogger(__name__)

# How many vectors to upsert per Pinecone API call
# Pinecone recommends batches of 100 vectors for the free tier
UPSERT_BATCH_SIZE = 100


def get_pinecone_index():
    """
    Initialize and return the Pinecone index client.

    WHY NOT A GLOBAL:
    We don't initialize Pinecone at import time because:
    1. Tests can run without real API keys if this isn't called
    2. The connection is established lazily when first needed
    3. If the API key is missing, we get a clear error at call time, not startup

    Returns:
        Pinecone Index object ready for upsert/query calls.
    """
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "legal-docs")

    if not api_key:
        raise RuntimeError("PINECONE_API_KEY environment variable is not set.")

    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)
    return index


def upsert_chunks(doc_id: str, embedded_chunks: list[dict]) -> int:
    """
    Upsert embedded chunk vectors into Pinecone under a doc_id namespace.

    Each vector is stored with rich metadata so we can reconstruct the
    original chunk text during retrieval (without querying a separate DB).

    METADATA STORED PER VECTOR:
    - doc_id:        which document this came from
    - page_num:      for citations ("see Page 4")
    - chunk_index:   sequential index, useful for ordering results
    - chunk_text:    the actual text — stored here so retrieval returns text directly
    - section_title: detected heading, useful for Q&A answers

    WHY STORE chunk_text IN PINECONE:
    Pinecone returns metadata with query results. If we store chunk_text here,
    a single Pinecone query gives us both the relevant vectors AND the text.
    The alternative (storing text separately in PostgreSQL and joining) adds
    latency and complexity that isn't worth it for this project's scale.

    Metadata size limit: Pinecone allows up to 40KB per vector's metadata.
    A 900-token chunk is ~3600 characters — well within the limit.

    Args:
        doc_id: Unique document identifier (used as Pinecone namespace).
        embedded_chunks: List of chunk dicts with "embedding" key attached.

    Returns:
        Total number of vectors upserted.

    Raises:
        RuntimeError: If Pinecone upsert fails.
    """
    if not embedded_chunks:
        logger.warning("upsert_chunks called with empty list for doc_id=%s", doc_id)
        return 0

    index = get_pinecone_index()
    total_upserted = 0

    # Build Pinecone vector format: list of (id, values, metadata) tuples
    vectors = []
    for chunk in embedded_chunks:
        vector_id = f"{doc_id}_{chunk['chunk_index']}"
        vectors.append(
            {
                "id": vector_id,
                "values": chunk["embedding"],  # the 1536-dim float list
                "metadata": {
                    "doc_id": doc_id,
                    "page_num": chunk["page_num"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["chunk_text"],
                    "section_title": chunk.get("section_title", ""),
                },
            }
        )

    # Upsert in batches to stay within Pinecone's request size limits
    total_batches = (len(vectors) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE
    logger.info(
        "Upserting %d vectors for doc_id=%s in %d batch(es)",
        len(vectors),
        doc_id,
        total_batches,
    )

    for batch_start in range(0, len(vectors), UPSERT_BATCH_SIZE):
        batch = vectors[batch_start : batch_start + UPSERT_BATCH_SIZE]

        try:
            # namespace=doc_id ensures vectors are isolated per document
            result = index.upsert(vectors=batch, namespace=doc_id)
            total_upserted += result.upserted_count
            logger.info(
                "Upserted batch: %d vectors (running total: %d)",
                result.upserted_count,
                total_upserted,
            )

        except Exception as e:
            logger.error(
                "Pinecone upsert failed for doc_id=%s, batch starting at %d: %s",
                doc_id,
                batch_start,
                e,
            )
            raise RuntimeError(f"Pinecone upsert failed: {e}") from e

    logger.info(
        "Upsert complete for doc_id=%s: %d total vectors stored",
        doc_id,
        total_upserted,
    )
    return total_upserted


def retrieve_chunks(
    doc_id: str,
    query_embedding: list[float],
    top_k: int | None = None,
) -> list[dict]:
    """
    Query Pinecone for the most semantically similar chunks to a query embedding.

    HOW SIMILARITY SEARCH WORKS:
    Pinecone computes cosine similarity between the query vector and all stored
    vectors in the namespace. It returns the top_k most similar ones.

    Cosine similarity = 1 means identical direction (perfect match)
    Cosine similarity = 0 means perpendicular (no relation)
    Cosine similarity < 0 means opposite (rare in practice)

    We filter by namespace (doc_id) so we only search within this document.

    Args:
        doc_id: Document namespace to search within.
        query_embedding: 1536-dim float list from embed_query().
        top_k: Number of results to return. Defaults to PINECONE_TOP_K env var (5).

    Returns:
        List of match dicts sorted by similarity score (descending):
        [{"chunk_text": str, "page_num": int, "score": float, ...}]

    Raises:
        RuntimeError: If Pinecone query fails.
    """
    if top_k is None:
        top_k = int(os.getenv("PINECONE_TOP_K", "5"))

    index = get_pinecone_index()

    try:
        # include_metadata=True returns the metadata dict we stored at upsert time
        # This is how we get chunk_text back from the vector store
        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=doc_id,
            include_metadata=True,
            include_values=False,  # we don't need the raw vectors back
        )

    except Exception as e:
        logger.error(
            "Pinecone query failed for doc_id=%s: %s", doc_id, e
        )
        raise RuntimeError(f"Pinecone query failed: {e}") from e

    matches = result.matches
    logger.info(
        "Retrieved %d chunks for doc_id=%s (top score: %.4f)",
        len(matches),
        doc_id,
        matches[0].score if matches else 0.0,
    )

    # Flatten Pinecone's match format into simple dicts
    # Each match has .id, .score, .metadata
    chunks = []
    for match in matches:
        metadata = match.metadata or {}
        chunks.append(
            {
                "chunk_text": metadata.get("chunk_text", ""),
                "page_num": metadata.get("page_num", 0),
                "chunk_index": metadata.get("chunk_index", 0),
                "section_title": metadata.get("section_title", ""),
                "score": match.score,  # cosine similarity [0, 1]
            }
        )

    return chunks


def delete_document(doc_id: str) -> None:
    """
    Delete all vectors for a document by deleting its namespace.

    WHY THIS MATTERS:
    Pinecone free tier has a 100k vector limit. When a user re-uploads a
    document or we want to clean up, we need to delete the old vectors.
    Deleting a namespace removes ALL vectors in it atomically.

    Args:
        doc_id: The document namespace to delete.
    """
    index = get_pinecone_index()

    try:
        index.delete(delete_all=True, namespace=doc_id)
        logger.info("Deleted all vectors for doc_id=%s", doc_id)
    except Exception as e:
        logger.error("Failed to delete namespace for doc_id=%s: %s", doc_id, e)
        raise RuntimeError(f"Pinecone delete failed: {e}") from e
