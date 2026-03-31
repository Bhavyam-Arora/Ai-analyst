"""
api/analyze.py — POST /api/analyze endpoint

This endpoint receives a doc_id, runs the full LangGraph analysis pipeline
(extraction → risk → summary), and returns structured results.

WHY THE PIPELINE CAN TAKE 30-60 SECONDS:
Each agent makes one or more GPT-4o API calls. GPT-4o's latency is typically
2-8 seconds per call, and we run 3 agents sequentially. On a large document
with many retrieved chunks, the LLM processes more tokens → higher latency.

In a production system you'd run this asynchronously (Celery + Redis task queue)
and poll for results. For this portfolio demo, we run synchronously and let the
HTTP connection stay open. FastAPI handles this fine with async/await.

HOW THE RESPONSE IS BUILT:
The LangGraph graph returns a final GraphState dict. We map it to an
AnalysisResponse Pydantic model:
- extracted_data (dict) → ExtractedData Pydantic model
- risks (list of dicts) → list of RiskItem Pydantic models
- summary (str) → plain string
- error (str) → surfaced in AnalysisResponse.status = "partial" or "failed"
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from agents.graph import analysis_graph
from models.analysis import AnalysisRequest, AnalysisResponse, ExtractedData, RiskItem

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Analysis is expensive (multiple LLM calls) — rate limit more conservatively than upload
RATE_LIMIT = "10/minute"


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    summary="Analyze a document with the LangGraph agent pipeline",
    description=(
        "Runs extraction, risk analysis, and summarization agents on a previously uploaded document. "
        "Returns structured key information, legal risks with severity ratings, and a plain-English summary."
    ),
)
@limiter.limit(RATE_LIMIT)
async def analyze_document(request: Request, body: AnalysisRequest) -> AnalysisResponse:
    """
    POST /api/analyze

    Accepts a doc_id (returned by /api/upload) and runs the full LangGraph
    analysis pipeline. Returns structured extraction, risk assessment, and summary.

    The pipeline runs 3 LangGraph nodes sequentially:
    1. extraction_node — identifies parties, dates, obligations, etc.
    2. risk_node — identifies legal risks with severity ratings
    3. summary_node — generates a plain-English summary

    If extraction fails, risk and summary are skipped (conditional edge in graph).
    """
    doc_id = body.doc_id
    logger.info("[analyze] Starting analysis pipeline for doc_id=%s", doc_id)

    # Build initial GraphState
    # WHY ONLY doc_id: The graph starts with just the document identifier.
    # Each agent reads from Pinecone and writes its own output to state.
    # None values for optional fields are intentional — agents populate them.
    initial_state = {
        "doc_id": doc_id,
        "extracted_data": None,
        "risks": None,
        "summary": None,
        "error": None,
    }

    try:
        # Run the compiled LangGraph — await the async invoke
        # ainvoke() runs the graph asynchronously, respecting await points in each node
        final_state = await analysis_graph.ainvoke(initial_state)

    except Exception as e:
        logger.error(
            "[analyze] Graph execution failed for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Analysis pipeline failed: {str(e)}",
        )

    # Map GraphState → AnalysisResponse
    error = final_state.get("error")
    extracted_raw = final_state.get("extracted_data")
    risks_raw = final_state.get("risks") or []
    summary = final_state.get("summary")

    # Determine status
    # "completed": all three agents ran successfully
    # "partial": some agents ran but one failed (error set, but partial results present)
    # "failed": extraction failed, no useful results
    if error and not extracted_raw:
        status = "failed"
    elif error:
        status = "partial"
    else:
        status = "completed"

    # Convert raw dicts to Pydantic models
    # WHY: AnalysisResponse expects typed models, not raw dicts.
    # We validate here to catch any schema drift between agent output and our models.
    extracted_data = None
    if extracted_raw:
        try:
            extracted_data = ExtractedData(**extracted_raw)
        except Exception as e:
            logger.warning("[analyze] ExtractedData validation failed: %s", e)

    risk_items = []
    for raw_risk in risks_raw:
        try:
            risk_items.append(RiskItem(**raw_risk))
        except Exception as e:
            logger.warning("[analyze] RiskItem validation failed: %s — %s", raw_risk, e)

    logger.info(
        "[analyze] Analysis complete for doc_id=%s — status=%s, risks=%d",
        doc_id,
        status,
        len(risk_items),
    )

    return AnalysisResponse(
        doc_id=doc_id,
        extracted_data=extracted_data,
        risks=risk_items if risk_items else None,
        summary=summary,
        status=status,
        error=error,
    )
