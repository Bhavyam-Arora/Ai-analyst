"""
main.py — FastAPI application entry point

This is the root of the backend. It:
1. Creates the FastAPI app instance
2. Registers route handlers from /api/
3. Configures CORS so the React frontend can call it
4. Adds rate limiting via slowapi
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else so all os.getenv() calls across the app see the keys
load_dotenv(Path(__file__).parent / ".env")

# ── Add backend/ to sys.path ───────────────────────────────────────────────
# The backend modules (rag/, api/, models/, utils/) use imports relative to
# the backend/ directory (e.g. `from rag.ingest import ...`). Adding backend/
# to sys.path means Python resolves those imports correctly whether we run
# `uvicorn main:app` from the project root or from inside backend/.
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Phase 2: Upload route (ingest pipeline)
from api.upload import router as upload_router
# Phase 3: Analysis pipeline + Q&A chat
from api.analyze import router as analyze_router
from api.chat import router as chat_router

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter (slowapi) ─────────────────────────────────────
# This identifies users by IP address for rate limiting
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan handler: runs startup logic before the server accepts requests,
    and cleanup logic when it shuts down.

    Phase 2 addition: verify Pinecone index and OpenAI API are reachable
    at startup so we fail fast rather than on the first upload request.
    """
    logger.info("🚀 AI Legal Analyst backend starting up...")

    # ── Phase 2 startup checks ─────────────────────────────────────────────
    # Verify Pinecone connection by describing the index stats
    # If this fails, the server starts but logs a clear warning
    try:
        import os
        from pinecone import Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY", ""))
        index = pc.Index(os.getenv("PINECONE_INDEX_NAME", "legal-docs"))
        stats = index.describe_index_stats()
        logger.info(
            "✅ Pinecone connected — index '%s', total vectors: %d",
            os.getenv("PINECONE_INDEX_NAME", "legal-docs"),
            stats.total_vector_count,
        )
    except Exception as e:
        logger.warning("⚠️  Pinecone startup check failed: %s", e)

    yield
    logger.info("🛑 Backend shutting down...")


# ── App instance ───────────────────────────────────────────────
app = FastAPI(
    title="AI Legal Document Analyst",
    description="RAG pipeline + multi-agent legal document analysis",
    version="0.1.0",
    lifespan=lifespan,
)

# Attach rate limiter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────
# Allows the React dev server (localhost:5173) to call this API.
# In production, replace "*" with your actual frontend domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ─────────────────────────────────────────
# Phase 2: Upload + ingestion pipeline
app.include_router(upload_router, prefix="/api", tags=["upload"])
# Phase 3: Analysis pipeline + Q&A chat
app.include_router(analyze_router, prefix="/api", tags=["analyze"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


# ── Health check ───────────────────────────────────────────────
@app.get("/", tags=["health"])
async def health_check():
    """
    Simple health check. Confirm this returns 200 before Phase 2.
    Run: curl http://localhost:8000/
    """
    return {"status": "ok", "service": "ai-legal-analyst-backend"}
