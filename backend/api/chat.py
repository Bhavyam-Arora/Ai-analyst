"""
api/chat.py — POST /api/chat endpoint

Powers the "Ask a question about this document" feature. Each call:
1. Retrieves the most relevant chunks from Pinecone for the question
2. Passes them to GPT-4o with a strict grounding prompt
3. Returns the answer with structured page citations

WHY NOT STREAMING (SSE) YET:
Streaming via Server-Sent Events would let the frontend display the answer
word-by-word as GPT-4o generates it. This is a Phase 5 enhancement.
For Phase 3, we return the complete answer as a standard JSON response.
The endpoint is designed so switching to SSE later only requires wrapping
the chat completion in a StreamingResponse — the qa_agent logic stays the same.

RATE LIMITING:
Chat is cheaper than analysis (1 LLM call vs. 3) but still costs money.
We allow 20 questions/minute — enough for an active user session without
burning through the API budget.

CITATION FLOW:
The qa_agent returns a ChatResponse with:
- answer: the LLM's response with [Page X] inline references
- citations: list of Citation objects with page_num, chunk_text, section_title
The frontend uses citations to highlight source passages in the PDF viewer.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from agents.qa_agent import run_qa
from models.chat import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

RATE_LIMIT = "20/minute"


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a natural language question about a document",
    description=(
        "Retrieves relevant document excerpts from Pinecone and uses GPT-4o to answer "
        "the user's question. Returns a grounded answer with page-level citations."
    ),
)
@limiter.limit(RATE_LIMIT)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    """
    POST /api/chat

    Accepts a doc_id and a natural language question. Returns a grounded answer
    with page citations pointing to the source passages in the document.

    The answer is strictly grounded — the LLM is instructed to answer ONLY from
    retrieved document excerpts and to say "I could not find this information"
    if the answer isn't present, rather than hallucinating.
    """
    doc_id = body.doc_id
    question = body.question

    logger.info(
        "[chat] Question for doc_id=%s: '%s...'",
        doc_id,
        question[:80],
    )

    try:
        response = await run_qa(doc_id=doc_id, question=question)
        return response

    except Exception as e:
        logger.error(
            "[chat] Q&A failed for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Q&A agent failed: {str(e)}",
        )
