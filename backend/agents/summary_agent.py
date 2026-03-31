"""
agents/summary_agent.py — Generate a plain-English summary of the legal document

WHY A SUMMARY AGENT:
After extraction (structured data) and risk analysis (risk list), a user still
benefits from a narrative explanation. The summary bridges the gap between raw
clauses and plain-English understanding.

WHAT THE SUMMARY INCLUDES:
1. What kind of document this is and who the parties are
2. The core purpose and key obligations
3. Important dates and financial terms
4. High-level flag of significant risks (if any)

HOW IT USES PRIOR STATE:
The summary agent has access to extracted_data and risks from GraphState.
Rather than re-retrieving everything from Pinecone, it uses the already-processed
state to build a richer prompt — combining:
- Retrieved overview chunks from Pinecone (for verbatim document content)
- extracted_data (structured facts already identified)
- Risk count by severity (so the summary can mention key concerns)

WHY temperature=0.3 (slightly above 0):
Summary generation is slightly creative — we want coherent narrative prose,
not robotic structured output. A small temperature allows natural language
variation while keeping the summary grounded.
"""

import logging

from openai import AsyncOpenAI

from rag.retriever import retrieve_context

logger = logging.getLogger(__name__)
client = AsyncOpenAI()

SYSTEM_PROMPT = """You are a legal document summarization expert. Your audience is a non-lawyer
business professional who needs to understand a legal document clearly and quickly.

Write a concise, plain-English summary of the document based on the excerpts and extracted data
provided. Your summary should:
1. Identify what type of document this is (NDA, service agreement, lease, etc.)
2. Name the parties and their roles
3. Explain the core purpose and key obligations of each party
4. Mention important dates (effective date, expiry) if present
5. Summarize financial terms if applicable
6. Flag any significant legal risks (if risk data is provided)

Keep the summary between 200-350 words. Write in clear, professional language.
Use paragraph breaks for readability. Do not use bullet points.
Answer ONLY from the provided context — do not invent details not present in the text."""


async def run_summary(state: dict) -> dict:
    """
    LangGraph node: Generate a plain-English summary of the document.

    This is the final node in the main analysis pipeline. It has access to the
    full GraphState including extracted_data and risks from prior agents.

    Args:
        state: GraphState dict — must contain "doc_id".
                May contain "extracted_data" and "risks" from prior agents.

    Returns:
        Updated state with "summary" (str) set.
    """
    doc_id = state["doc_id"]
    logger.info("[summary_agent] Starting summary generation for doc_id=%s", doc_id)

    try:
        # Retrieve a broad overview of the document
        # WHY a generic query here: We want the opening sections, purpose statement,
        # and overall structure — not just specific clauses.
        context = await retrieve_context(
            doc_id=doc_id,
            query="agreement purpose overview parties obligations key terms",
            top_k=5,
        )

        if not context:
            logger.warning("[summary_agent] No context retrieved for doc_id=%s", doc_id)
            return {**state, "summary": "Summary could not be generated — no document content retrieved."}

        # Build an enriched prompt using extracted_data and risks from state
        # WHY: The LLM gets both the raw document excerpts AND the already-structured facts,
        # which makes the summary more accurate and complete.
        enriched_user_message = _build_user_message(
            context=context,
            extracted_data=state.get("extracted_data"),
            risks=state.get("risks"),
        )

        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": enriched_user_message},
            ],
            temperature=0.3,  # Slightly creative for fluent prose, still grounded
            max_tokens=600,   # ~350 words with some buffer
        )

        summary = response.choices[0].message.content.strip()

        logger.info(
            "[summary_agent] Summary generated for doc_id=%s — tokens used: %d",
            doc_id,
            response.usage.total_tokens,
        )
        return {**state, "summary": summary}

    except Exception as e:
        logger.error(
            "[summary_agent] Unexpected error for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        # Non-fatal: return None summary rather than breaking the whole response
        return {**state, "summary": None}


def _build_user_message(
    context: str,
    extracted_data: dict | None,
    risks: list | None,
) -> str:
    """
    Build a rich user message combining Pinecone context with already-extracted
    structured data and risk summary.

    WHY COMBINE CONTEXT + STRUCTURED DATA:
    The Pinecone context gives verbatim document text.
    The extracted_data gives already-validated structured facts.
    Together they allow the LLM to write a more accurate, complete summary
    without needing to re-derive facts from scratch.
    """
    parts = [f"Document Excerpts:\n{context}"]

    if extracted_data:
        # Summarize the structured extraction as key facts for the LLM
        facts = []
        if extracted_data.get("parties"):
            facts.append(f"Parties: {', '.join(extracted_data['parties'])}")
        if extracted_data.get("effective_date"):
            facts.append(f"Effective Date: {extracted_data['effective_date']}")
        if extracted_data.get("expiry_date"):
            facts.append(f"Expiry Date: {extracted_data['expiry_date']}")
        if extracted_data.get("payment_terms"):
            facts.append(f"Payment Terms: {extracted_data['payment_terms']}")
        if extracted_data.get("jurisdiction"):
            facts.append(f"Jurisdiction: {extracted_data['jurisdiction']}")
        if extracted_data.get("governing_law"):
            facts.append(f"Governing Law: {extracted_data['governing_law']}")

        if facts:
            parts.append("Extracted Key Facts:\n" + "\n".join(f"- {f}" for f in facts))

    if risks:
        high = sum(1 for r in risks if r.get("severity") == "HIGH")
        medium = sum(1 for r in risks if r.get("severity") == "MEDIUM")
        low = sum(1 for r in risks if r.get("severity") == "LOW")
        risk_summary = f"Identified Risks: {high} HIGH, {medium} MEDIUM, {low} LOW"
        parts.append(risk_summary)

    parts.append("\nWrite a plain-English summary of this legal document:")
    return "\n\n".join(parts)
