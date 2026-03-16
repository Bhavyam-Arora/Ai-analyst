"""
main.py — FastAPI application entry point

This is the root of the backend. It:
1. Creates the FastAPI app instance
2. Registers route handlers from /api/
3. Configures CORS so the React frontend can call it
4. Adds rate limiting via slowapi
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# We'll add route imports here as we build each phase
# from api.upload import router as upload_router
# from api.analyze import router as analyze_router
# from api.chat import router as chat_router

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
    Add Pinecone connection check and OpenAI ping here in Phase 2.
    """
    logger.info("🚀 AI Legal Analyst backend starting up...")
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
# Uncomment these as you build each phase:
# app.include_router(upload_router, prefix="/api", tags=["upload"])
# app.include_router(analyze_router, prefix="/api", tags=["analyze"])
# app.include_router(chat_router, prefix="/api", tags=["chat"])


# ── Health check ───────────────────────────────────────────────
@app.get("/", tags=["health"])
async def health_check():
    """
    Simple health check. Confirm this returns 200 before Phase 2.
    Run: curl http://localhost:8000/
    """
    return {"status": "ok", "service": "ai-legal-analyst-backend"}
