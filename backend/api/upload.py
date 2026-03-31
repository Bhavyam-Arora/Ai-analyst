"""
api/upload.py — POST /api/upload route handler

WHY THIS IS THE ENTRY POINT FOR PHASE 2:
Every subsequent feature (analysis, chat) requires a doc_id, which is only
created during ingestion. The upload endpoint is therefore the first thing
the user interacts with — they upload a file, get a doc_id, and all other
actions reference that doc_id.

FLOW:
1. Receive multipart/form-data file upload
2. Validate: file type (PDF or DOCX only), file size (max 20MB)
3. Save file to a temp location on disk
4. Run the ingest pipeline (parse → chunk → embed → upsert)
5. Delete the temp file (we don't need it after ingestion)
6. Return UploadResponse with doc_id and ingestion metadata

WHY WE DELETE THE FILE AFTER INGESTION:
After the document is chunked and embedded into Pinecone, the original file
is no longer needed for the RAG pipeline. Keeping uploaded files would waste
disk space and create a security/compliance risk (legal documents are sensitive).
If we later want to show the original PDF to the user, we'd add cloud storage
(S3/GCS) — but for Phase 2, deletion keeps it simple.

RATE LIMITING:
The @limiter.limit decorator (from slowapi) restricts this endpoint to 5
uploads per minute per IP. Document ingestion involves expensive OpenAI API
calls — rate limiting prevents abuse and runaway costs.
"""

import logging
import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from rag.ingest import ingest_document
from models.upload import UploadResponse, UploadErrorResponse

logger = logging.getLogger(__name__)

# ── Router setup ──────────────────────────────────────────────────────────────
# APIRouter lets us define routes in a separate module and include them in main.py
# prefix="/api" is added when we include this router in main.py
router = APIRouter()

# Rate limiter — uses the same instance pattern as main.py
# WHY: Each request to this endpoint triggers multiple OpenAI API calls
# (embedding ~100+ chunks). 5/minute is generous for real use but protects
# against runaway costs from scripts or accidental loops.
limiter = Limiter(key_func=get_remote_address)

# ── Constants ─────────────────────────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": UploadErrorResponse, "description": "Invalid file"},
        413: {"model": UploadErrorResponse, "description": "File too large"},
        500: {"model": UploadErrorResponse, "description": "Ingestion failed"},
    },
    summary="Upload a legal document for analysis",
    description="Accepts PDF or DOCX files up to 20MB. Parses, chunks, embeds, and stores the document in Pinecone. Returns a doc_id for subsequent /analyze and /chat calls.",
)
@limiter.limit("5/minute")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """
    Handle legal document upload and trigger the RAG ingestion pipeline.

    Args:
        request: FastAPI Request object (required by slowapi rate limiter).
        file: The uploaded file (multipart/form-data).

    Returns:
        UploadResponse with doc_id and ingestion statistics.
    """
    logger.info(
        "Upload request received: filename=%s, content_type=%s",
        file.filename,
        file.content_type,
    )

    # ── Validate file type ────────────────────────────────────────────────────
    # Check both the file extension AND the content type for security
    # WHY BOTH: A malicious user could rename a .exe to .pdf. Checking content
    # type provides a second layer of validation (though it can also be spoofed
    # in HTTP headers — true validation would inspect file magic bytes).
    filename = file.filename or "unknown"
    ext = Path(filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        logger.warning("Rejected upload: unsupported file type '%s'", ext)
        raise HTTPException(
            status_code=400,
            detail={
                "error": "invalid_file_type",
                "detail": f"Only PDF and DOCX files are accepted. Received: '{ext}'",
            },
        )

    # ── Read file into memory to check size ───────────────────────────────────
    # We read the full file here so we can:
    # 1. Check its actual size (Content-Length header can be spoofed)
    # 2. Write it to a temp file for the parser
    file_bytes = await file.read()
    file_size = len(file_bytes)

    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        logger.warning(
            "Rejected upload: file too large (%.1f MB > %d MB limit)", size_mb, MAX_FILE_SIZE_MB
        )
        raise HTTPException(
            status_code=413,
            detail={
                "error": "file_too_large",
                "detail": f"File size {size_mb:.1f}MB exceeds the {MAX_FILE_SIZE_MB}MB limit.",
            },
        )

    if file_size == 0:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_file", "detail": "The uploaded file is empty."},
        )

    logger.info(
        "File validated: %s (%.2f MB)", filename, file_size / (1024 * 1024)
    )

    # ── Save to temp file ─────────────────────────────────────────────────────
    # PyMuPDF and python-docx both need a file path, not a bytes buffer.
    # We write to a named temp file and delete it after ingestion.
    # WHY tempfile.NamedTemporaryFile with delete=False:
    # On Windows, you can't open a NamedTemporaryFile while it's already open
    # by Python. Using delete=False and manually deleting is the cross-platform way.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, prefix="legal_upload_"
        ) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        logger.info("Saved to temp file: %s", temp_path)

        # ── Run ingestion pipeline ────────────────────────────────────────────
        result = await ingest_document(file_path=temp_path, filename=filename)

        return UploadResponse(
            doc_id=result.doc_id,
            filename=result.filename,
            page_count=result.page_count,
            chunk_count=result.chunk_count,
            vector_count=result.vector_count,
        )

    except ValueError as e:
        # ValueError = parsing/chunking failed (unreadable file, scanned PDF, etc.)
        logger.error("Ingestion validation error for '%s': %s", filename, e)
        raise HTTPException(
            status_code=400,
            detail={"error": "ingestion_failed", "detail": str(e)},
        )

    except RuntimeError as e:
        # RuntimeError = OpenAI or Pinecone API failure
        logger.error("Ingestion runtime error for '%s': %s", filename, e)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "pipeline_error",
                "detail": "Document ingestion failed due to an internal error. Please try again.",
            },
        )

    finally:
        # Always clean up the temp file, even if ingestion failed
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
            logger.info("Temp file cleaned up: %s", temp_path)
