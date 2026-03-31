"""
models/analysis.py — Pydantic models for document analysis request/response

These models validate the data flowing through the /api/analyze endpoint
and serve as the contract between the frontend and the LangGraph agent pipeline.
"""

from typing import Optional
from pydantic import BaseModel, Field


class AnalysisRequest(BaseModel):
    """Request body for POST /api/analyze"""

    doc_id: str = Field(
        description="The document ID returned by the upload endpoint."
    )


class ExtractedData(BaseModel):
    """
    Structured extraction output from the extraction agent.

    All fields are Optional because legal documents vary widely —
    not all contracts have all these fields. Returning null (None in Python)
    is preferable to hallucinating a value that isn't in the document.

    WHY OPTIONAL FOR ALL FIELDS:
    The extraction agent's system prompt instructs GPT-4o to return null for
    missing fields. If we made fields required, a document without an expiry_date
    would fail Pydantic validation and the whole analysis would error.
    """

    parties: Optional[list[str]] = Field(
        default=None,
        description="Names of all parties to the agreement."
    )
    effective_date: Optional[str] = Field(
        default=None,
        description="Date the agreement becomes effective (ISO format preferred)."
    )
    expiry_date: Optional[str] = Field(
        default=None,
        description="Date the agreement expires or terminates."
    )
    payment_terms: Optional[str] = Field(
        default=None,
        description="Summary of payment obligations, amounts, and schedules."
    )
    obligations: Optional[list[str]] = Field(
        default=None,
        description="List of key obligations for each party."
    )
    termination_clauses: Optional[list[str]] = Field(
        default=None,
        description="Conditions under which either party may terminate the agreement."
    )
    jurisdiction: Optional[str] = Field(
        default=None,
        description="Jurisdiction where disputes are resolved."
    )
    governing_law: Optional[str] = Field(
        default=None,
        description="State/country law that governs the agreement."
    )


class RiskItem(BaseModel):
    """
    A single identified legal risk from the risk agent.

    SEVERITY LEVELS (defined in CLAUDE.md):
    - HIGH:   Unlimited liability, IP loss, irreversible obligations
    - MEDIUM: One-sided terms, vague conditions, unusual jurisdiction
    - LOW:    Missing standard clauses, informal language
    """

    severity: str = Field(
        description="Risk severity: HIGH, MEDIUM, or LOW."
    )
    clause_text: str = Field(
        description="The exact clause or text that poses the risk."
    )
    page_reference: Optional[int] = Field(
        default=None,
        description="Page number where this clause appears."
    )
    explanation: str = Field(
        description="Why this clause is a risk."
    )
    recommendation: str = Field(
        description="Suggested action or alternative wording."
    )


class AnalysisResponse(BaseModel):
    """
    Full analysis response returned by POST /api/analyze.

    Contains both extraction results and risk assessment, plus metadata
    about the analysis run.
    """

    doc_id: str
    extracted_data: Optional[ExtractedData] = Field(
        default=None,
        description="Structured key information extracted from the document."
    )
    risks: Optional[list[RiskItem]] = Field(
        default=None,
        description="List of identified legal risks with severity ratings."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Plain-English summary of the document's key points."
    )
    status: str = Field(
        default="completed",
        description="Analysis status: 'completed', 'partial', or 'failed'."
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if analysis partially or fully failed."
    )
