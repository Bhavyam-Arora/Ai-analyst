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
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else so all os.getenv() calls across the app see the keys
# main.py lives inside backend/, so .env is right next to it at backend/.env
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from api.upload import router as upload_router
from api.analyze import router as analyze_router
from api.chat import router as chat_router

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Rate limiter (slowapi) ─────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 AI Legal Analyst backend starting up...")

    try:
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

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Route registration ─────────────────────────────────────────
app.include_router(upload_router, prefix="/api", tags=["upload"])
app.include_router(analyze_router, prefix="/api", tags=["analyze"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


# ── Health check ───────────────────────────────────────────────
@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "service": "ai-legal-analyst-backend"}
