"""
rag/embedder.py — Convert text chunks into vector embeddings via OpenAI

WHY EMBEDDINGS:
A vector embedding is a list of ~1536 numbers that encodes the *semantic meaning*
of a text. Two texts about the same concept will have embeddings that are "close"
in vector space (high cosine similarity), even if they use different words.

This is how RAG retrieval works:
1. We embed every chunk and store those vectors in Pinecone
2. At query time, we embed the user's question
3. Pinecone finds the stored chunk-vectors that are closest to the question-vector
4. We return those chunks to the LLM as context

MODEL CHOICE (text-embedding-3-small):
- Dimension: 1536 — matches our Pinecone index configuration
- Cost: ~$0.02 per 1M tokens — very cheap for a document pipeline
- Quality: Better than ada-002 for retrieval tasks at lower cost
- NOT text-embedding-3-large (3072 dims, 5x cost) — overkill for legal docs

BATCHING (why we batch API calls):
OpenAI's Embeddings API accepts up to 2048 input strings per request.
For a 50-page legal contract, we might have 150+ chunks. Sending them all
in one request is fine. But for very large documents, batching prevents
timeouts and lets us log progress.
"""

import asyncio
import logging
import os
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Model name as a constant so it's easy to update in one place
EMBEDDING_MODEL = "text-embedding-3-small"

# How many chunks to send per API request
# 100 is safe for the free tier; increase to 2048 for production
EMBEDDING_BATCH_SIZE = 100


async def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed all chunk texts and attach the embedding vectors to each chunk dict.

    Process:
    1. Extract chunk texts
    2. Send to OpenAI in batches of EMBEDDING_BATCH_SIZE
    3. Attach the returned embedding vector back to each chunk dict

    WHY ASYNC:
    Each OpenAI API call takes ~500ms–2s. For 150 chunks in 2 batches, async
    lets FastAPI continue handling other requests while we wait. Using
    `await client.embeddings.create(...)` releases the event loop during the
    network call.

    Args:
        chunks: List of chunk dicts from chunker.py:
                [{"chunk_text": str, "page_num": int, "chunk_index": int, ...}]

    Returns:
        Same list with "embedding" key added to each dict:
        [{"chunk_text": str, ..., "embedding": list[float]}]

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    if not chunks:
        return []

    # AsyncOpenAI reads OPENAI_API_KEY from environment automatically
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    texts = [chunk["chunk_text"] for chunk in chunks]
    all_embeddings: list[list[float]] = []

    total_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE
    logger.info(
        "Embedding %d chunks in %d batch(es) using %s",
        len(chunks),
        total_batches,
        EMBEDDING_MODEL,
    )

    for batch_num, batch_start in enumerate(range(0, len(texts), EMBEDDING_BATCH_SIZE)):
        batch_texts = texts[batch_start : batch_start + EMBEDDING_BATCH_SIZE]

        try:
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch_texts,
                # encoding_format="float" is the default — returns list[float]
                # Use "base64" only if you need to minimize network payload
                encoding_format="float",
            )

            # Response.data is sorted by index, matching our input order
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            logger.info(
                "Batch %d/%d complete — %d embeddings (dim=%d)",
                batch_num + 1,
                total_batches,
                len(batch_embeddings),
                len(batch_embeddings[0]) if batch_embeddings else 0,
            )

        except Exception as e:
            logger.error(
                "OpenAI embedding failed on batch %d: %s", batch_num + 1, e
            )
            raise RuntimeError(f"Embedding batch {batch_num + 1} failed: {e}") from e

    # Attach embeddings back to the original chunk dicts
    # zip() pairs each chunk with its embedding by position
    enriched_chunks = []
    for chunk, embedding in zip(chunks, all_embeddings):
        enriched_chunks.append({**chunk, "embedding": embedding})

    logger.info("Embedding complete: %d chunks embedded", len(enriched_chunks))
    return enriched_chunks


async def embed_query(query_text: str) -> list[float]:
    """
    Embed a single query string for similarity search.

    WHY A SEPARATE FUNCTION:
    Query embedding uses the same model as document embedding (critical —
    query and document vectors MUST be in the same embedding space for
    cosine similarity to work). But queries are single strings, not batches.

    Args:
        query_text: The user's natural language question.

    Returns:
        List of floats (1536-dimensional embedding vector).

    Raises:
        RuntimeError: If the OpenAI API call fails.
    """
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    try:
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[query_text],
            encoding_format="float",
        )
        return response.data[0].embedding

    except Exception as e:
        logger.error("Failed to embed query '%s...': %s", query_text[:50], e)
        raise RuntimeError(f"Query embedding failed: {e}") from e
