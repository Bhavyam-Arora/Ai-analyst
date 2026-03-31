"""
agents/qa_agent.py — Answer user questions with exact page citations

WHY A SEPARATE Q&A AGENT (NOT PART OF THE MAIN GRAPH):
The main pipeline (extraction → risk → summary) runs once per document upload.
Q&A is conversational — it runs once per user question, potentially many times
per document session.

The Q&A agent is a single-function agent invoked directly by the /api/chat
endpoint, rather than part of the StateGraph. This keeps the main graph simple
and the Q&A path fast (no graph compilation overhead per request).

CITATION EXTRACTION:
The LLM is instructed to include [Page X] references in its answer.
After the LLM responds, we parse these references and match them against the
retrieved chunks to build structured Citation objects for the frontend.

WHY STRICT GROUNDING:
Legal Q&A is high-stakes. If the model answers from its training data rather
than the actual document, it could give wrong legal advice. The system prompt
explicitly restricts answers to provided excerpts and instructs the model to
say "I could not find this information" rather than guessing.

HALLUCINATION PREVENTION PATTERN:
- Only retrieved chunks are passed to the LLM (no free-form context)
- System prompt forbids inference not grounded in excerpts
- Citations allow the user to verify every statement in the source document
- If no chunks are retrieved, we return a fallback response (not an LLM guess)
"""

import logging
import re

from openai import AsyncOpenAI

from models.chat import ChatResponse, Citation
from rag.retriever import retrieve_context
from rag.embedder import embed_query
from rag.pinecone_client import retrieve_chunks

logger = logging.getLogger(__name__)
client = AsyncOpenAI()

SYSTEM_PROMPT = """You are a legal document assistant. Answer the user's question using ONLY the provided
document excerpts below. If the answer cannot be found in the excerpts, respond with exactly:
"I could not find this information in the provided document."

Rules:
1. Answer ONLY from the provided excerpts — do not use your training knowledge
2. For every factual statement, cite the source using [Page X] notation
3. If multiple pages support the answer, cite all of them: [Page 2, Page 5]
4. Keep answers concise and direct — 2-5 sentences unless the question requires more detail
5. Use plain English — avoid legal jargon unless quoting directly from the document
6. Never infer, assume, or extrapolate beyond what the document explicitly states"""


async def run_qa(doc_id: str, question: str) -> ChatResponse:
    """
    Run the Q&A agent for a single user question.

    This is the main entry point called by the /api/chat endpoint.

    Flow:
    1. Embed the question and retrieve relevant chunks from Pinecone
    2. Format chunks as context (with page labels)
    3. Call GPT-4o with strict grounding prompt
    4. Parse [Page X] citations from the answer and match to retrieved chunks
    5. Return ChatResponse with answer + structured Citation list

    Args:
        doc_id: The document namespace to query.
        question: The user's natural language question.

    Returns:
        ChatResponse with grounded answer and source citations.
    """
    logger.info(
        "[qa_agent] Question received for doc_id=%s: '%s...'",
        doc_id,
        question[:80],
    )

    try:
        # Step 1: Retrieve relevant chunks
        # WHY: We retrieve here (not just in retriever.py) so we keep the raw chunk
        # objects for citation matching later.
        query_embedding = await embed_query(question)
        chunks = retrieve_chunks(doc_id, query_embedding, top_k=5)

        if not chunks:
            logger.warning(
                "[qa_agent] No chunks retrieved for doc_id=%s — returning fallback response",
                doc_id,
            )
            return ChatResponse(
                answer="I could not find this information in the provided document.",
                citations=[],
                doc_id=doc_id,
            )

        # Step 2: Format chunks as context for the LLM prompt
        context = _format_context(chunks)

        # Step 3: Call GPT-4o
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Document Excerpts:\n{context}\n\nQuestion: {question}",
                },
            ],
            temperature=0,      # Q&A is deterministic — we want the same answer each time
            max_tokens=800,     # Keep answers focused
        )

        answer = response.choices[0].message.content.strip()

        logger.info(
            "[qa_agent] Answer generated for doc_id=%s — tokens used: %d",
            doc_id,
            response.usage.total_tokens,
        )

        # Step 4: Extract page citations from the answer and build Citation objects
        citations = _extract_citations(answer, chunks)

        return ChatResponse(
            answer=answer,
            citations=citations,
            doc_id=doc_id,
        )

    except Exception as e:
        logger.error(
            "[qa_agent] Error for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        raise


def _format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks as a labeled context string for the LLM.

    Each chunk gets a [Page X] label so the LLM can reference specific pages
    in its answer. Sorted by page number for readable, document-order context.
    """
    sorted_chunks = sorted(chunks, key=lambda c: c.get("page_num", 0))
    parts = []
    for chunk in sorted_chunks:
        page_num = chunk.get("page_num", "?")
        section = chunk.get("section_title", "")
        text = chunk.get("chunk_text", "")

        header = f"[Page {page_num}]" if not section else f"[Page {page_num} — {section}]"
        parts.append(f"{header}\n{text}")

    return "\n\n".join(parts)


def _extract_citations(answer: str, chunks: list[dict]) -> list[Citation]:
    """
    Parse [Page X] references from the LLM answer and match them to retrieved chunks.

    WHY MATCH TO CHUNKS:
    The Citation object includes chunk_text and section_title, not just a page number.
    By matching the cited page number back to the retrieved chunks, we can return
    the exact excerpt that supports the answer — enabling the frontend to highlight
    it in the PDF viewer.

    Args:
        answer: The LLM's answer string, expected to contain [Page X] references.
        chunks: The retrieved chunks that were passed to the LLM.

    Returns:
        List of Citation objects, one per unique cited page (matched to chunks).
    """
    # Find all page numbers mentioned in the answer: [Page 3], [Page 2, Page 5], etc.
    cited_pages = set(int(p) for p in re.findall(r"\[Page\s+(\d+)\]", answer))

    if not cited_pages:
        # No explicit citations in the answer — return all retrieved chunks as context
        # This is a fallback for answers that don't follow the citation format
        return [
            Citation(
                page_num=c.get("page_num", 0),
                chunk_text=c.get("chunk_text", ""),
                section_title=c.get("section_title"),
            )
            for c in chunks
        ]

    # Build a map of page_num → chunk for fast lookup
    chunk_by_page: dict[int, dict] = {}
    for chunk in chunks:
        page = chunk.get("page_num")
        if page is not None and page not in chunk_by_page:
            chunk_by_page[page] = chunk

    # Build Citation for each cited page (only pages we actually retrieved)
    citations = []
    for page_num in sorted(cited_pages):
        chunk = chunk_by_page.get(page_num)
        if chunk:
            citations.append(
                Citation(
                    page_num=page_num,
                    chunk_text=chunk.get("chunk_text", ""),
                    section_title=chunk.get("section_title"),
                )
            )
        else:
            # LLM cited a page we didn't retrieve — include minimal citation
            logger.debug(
                "[qa_agent] LLM cited page %d which wasn't in retrieved chunks", page_num
            )
            citations.append(
                Citation(
                    page_num=page_num,
                    chunk_text="",
                    section_title=None,
                )
            )

    return citations
