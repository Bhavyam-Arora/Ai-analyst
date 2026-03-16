"""
test_connections.py — Phase 1 Checkpoint Script

Run this BEFORE writing any business logic.
It verifies that all three external services are properly connected:
  1. OpenAI — embed a test sentence
  2. Pinecone — upsert the embedding, then query it back
  3. FastAPI — confirmed separately by running main.py

Usage:
    cd backend
    python test_connections.py

Expected output:
    ✅ OpenAI embedding OK — vector length: 1536
    ✅ Pinecone upsert OK — upserted 1 vector
    ✅ Pinecone query OK — top match score: 0.99...
    🎉 All connections verified. Ready for Phase 2.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def test_openai_embedding() -> list[float]:
    """
    Step 1: Call OpenAI embeddings API with a test string.
    This verifies your OPENAI_API_KEY is valid and the model is accessible.
    
    text-embedding-3-small produces a 1536-dimensional vector.
    Each number represents a position in semantic space.
    """
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input="This agreement is entered into between Party A and Party B.",
    )

    vector = response.data[0].embedding
    assert len(vector) == 1536, f"Expected 1536 dimensions, got {len(vector)}"
    logger.info(f"✅ OpenAI embedding OK — vector length: {len(vector)}")
    return vector


def test_pinecone(vector: list[float]) -> None:
    """
    Step 2: Upsert the test vector into Pinecone, then query it back.
    
    Key concepts used here:
    - Namespace: we use 'test-namespace' to isolate this test data
    - Upsert: insert-or-update a vector by ID
    - Query: find top-k most similar vectors to a given vector
    - Metadata: structured data stored alongside the vector (used for citations later)
    """
    from pinecone import Pinecone

    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

    # Upsert test vector with metadata
    # In production: doc_id becomes the namespace, chunk_text goes in metadata
    upsert_response = index.upsert(
        vectors=[
            {
                "id": "test-vector-001",
                "values": vector,
                "metadata": {
                    "doc_id": "test-doc",
                    "page_num": 1,
                    "chunk_index": 0,
                    "chunk_text": "This agreement is entered into between Party A and Party B.",
                },
            }
        ],
        namespace="test-namespace",
    )
    assert upsert_response.upserted_count == 1
    logger.info(f"✅ Pinecone upsert OK — upserted {upsert_response.upserted_count} vector")

    # Query back — should return our test vector as top result
    # top_k=1 since we only upserted one vector
    query_response = index.query(
        vector=vector,
        top_k=1,
        namespace="test-namespace",
        include_metadata=True,
    )

    assert len(query_response.matches) > 0, "No matches returned from Pinecone query"
    top_match = query_response.matches[0]
    logger.info(f"✅ Pinecone query OK — top match score: {top_match.score:.4f}")
    logger.info(f"   Metadata: {top_match.metadata}")

    # Clean up test data
    index.delete(ids=["test-vector-001"], namespace="test-namespace")
    logger.info("   Test vector cleaned up from Pinecone")


def main():
    # Validate env vars are loaded
    required_vars = ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        raise EnvironmentError(
            f"Missing environment variables: {missing}\n"
            "Make sure you have a .env file in /backend with all required keys."
        )

    logger.info("Running Phase 1 connection tests...\n")

    try:
        vector = test_openai_embedding()
        test_pinecone(vector)
        print("\n🎉 All connections verified. Ready for Phase 2.")
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        raise


if __name__ == "__main__":
    main()
