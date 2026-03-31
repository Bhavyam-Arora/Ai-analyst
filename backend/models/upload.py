"""
models/upload.py — Pydantic request/response models for the upload endpoint

WHY PYDANTIC MODELS:
FastAPI uses Pydantic to automatically:
1. Validate incoming request data (type checking, required fields)
2. Serialize outgoing response data (dict → JSON)
3. Generate OpenAPI schema (visible at /docs)

By defining explicit models, we get free input validation, clear API
documentation, and type safety throughout the codebase. Pydantic v2
(which we use) is ~5-10x faster than v1 due to Rust-backed validation.
"""

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """
    Response returned by POST /api/upload after successful ingestion.

    The `doc_id` field is the most important — the frontend stores this
    in Zustand state and includes it in every subsequent /analyze and /chat
    request to identify which document is being processed.
    """

    doc_id: str = Field(
        description="Unique identifier for this document. Store this for all subsequent API calls."
    )
    filename: str = Field(
        description="Original name of the uploaded file."
    )
    page_count: int = Field(
        description="Number of pages with extractable content."
    )
    chunk_count: int = Field(
        description="Number of text chunks created from the document."
    )
    vector_count: int = Field(
        description="Number of embedding vectors stored in Pinecone."
    )
    message: str = Field(
        default="Document uploaded and indexed successfully.",
        description="Human-readable status message."
    )


class UploadErrorResponse(BaseModel):
    """
    Structured error response for upload failures.

    WHY STRUCTURED ERRORS:
    Never return raw Python exceptions to the client — they leak implementation
    details and make frontend error handling inconsistent. Every error path
    through the API should return this shape so the frontend can handle it uniformly.
    """

    error: str = Field(
        description="Machine-readable error type (e.g., 'invalid_file_type', 'file_too_large')."
    )
    detail: str = Field(
        description="Human-readable error description."
    )
