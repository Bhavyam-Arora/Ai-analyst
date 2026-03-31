"""
models/chat.py — Pydantic models for the Q&A chat endpoint

The chat endpoint powers the "Ask a question about this document" feature.
It accepts a question, retrieves relevant chunks, and streams back a
grounded answer with page citations.
"""

from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body for POST /api/chat"""

    doc_id: str = Field(
        description="The document ID to query against."
    )
    question: str = Field(
        description="The user's natural language question about the document.",
        min_length=3,
        max_length=1000,
    )


class Citation(BaseModel):
    """
    A source citation pointing to a specific page in the document.

    WHY CITATIONS MATTER:
    Without citations, the user has no way to verify the LLM's answer.
    By including [Page X] references, users can open the PDF and confirm
    the answer is grounded in real document content — not hallucinated.
    """

    page_num: int = Field(description="Page number the cited content appears on.")
    chunk_text: str = Field(
        description="The relevant excerpt from that page that supports the answer."
    )
    section_title: Optional[str] = Field(
        default=None,
        description="Section or heading this excerpt belongs to, if detectable."
    )


class ChatResponse(BaseModel):
    """Response from POST /api/chat"""

    answer: str = Field(
        description="The LLM's answer, grounded in retrieved document excerpts."
    )
    citations: list[Citation] = Field(
        default_factory=list,
        description="Source pages and excerpts that support the answer."
    )
    doc_id: str = Field(
        description="The document that was queried."
    )
