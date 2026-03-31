"""
agents/extraction_agent.py — Extract structured key information from a legal document

WHY RETRIEVAL BEFORE EXTRACTION:
The naive approach would be to dump the entire document text into a GPT-4o prompt.
This fails for large documents (context window limits) and is expensive.

Instead, we use targeted retrieval:
1. Run 7 specific queries (parties, dates, payment terms, etc.) against Pinecone
2. Each query pulls the 3 most relevant chunks for that specific field
3. Deduplicate overlapping chunks
4. Feed only the relevant excerpts to GPT-4o

This means GPT-4o sees dense, relevant content for each field rather than
wading through pages of boilerplate.

WHY temperature=0:
Extraction is deterministic — we want consistent, structured output every time.
Temperature 0 makes GPT-4o less creative and more faithful to the prompt.

WHY response_format={"type": "json_object"}:
GPT-4o has a JSON mode that guarantees the output is parseable JSON.
This eliminates the most common failure mode: LLM adding markdown code fences
or explanatory text around the JSON.
"""

import json
import logging

from openai import AsyncOpenAI

from models.analysis import ExtractedData
from rag.retriever import retrieve_context_multi_query

logger = logging.getLogger(__name__)

# Shared async client — reused across all requests in this process
client = AsyncOpenAI()

# WHY MULTIPLE QUERIES:
# Each legal field lives in a different part of the document.
# "parties" are often in the preamble, "termination" in a dedicated clause section,
# "payment terms" in their own section. A single generic query returns a mixed bag.
# Targeted queries retrieve the right sections for each field.
EXTRACTION_QUERIES = [
    "parties to the agreement names of companies individuals signatories",
    "effective date commencement date agreement start date",
    "expiry date termination date end date agreement duration",
    "payment terms fees compensation amount due schedule",
    "obligations duties responsibilities of each party",
    "termination clauses conditions for termination notice period",
    "jurisdiction governing law applicable law dispute resolution",
]

SYSTEM_PROMPT = """You are a legal document analysis expert. You will be given excerpts from a legal document.
Extract the following information ONLY from the provided text. If a field is not present
in the text, return null for that field. Do NOT infer or assume information not explicitly stated.
Return your response as a valid JSON object with exactly these keys:
parties, effective_date, expiry_date, payment_terms, obligations,
termination_clauses, jurisdiction, governing_law.

Field definitions:
- parties: array of strings — full names of all parties to the agreement
- effective_date: string — the date the agreement becomes effective, as written in the document
- expiry_date: string — the expiry or end date, as written in the document
- payment_terms: string — summary of payment obligations, amounts, and schedules
- obligations: array of strings — key obligations, one per party or obligation type
- termination_clauses: array of strings — each condition that allows termination
- jurisdiction: string — jurisdiction where disputes are resolved
- governing_law: string — state or country law that governs the agreement

Return ONLY the JSON object. No explanation, no markdown, no code fences."""

# WHY A SEPARATE RETRY PROMPT:
# If the first call returns malformed JSON (rare with json_object mode, but possible),
# we retry with an even more explicit template showing the exact expected structure.
# CLAUDE.md mandates: "JSON parsing from LLM responses must have a retry with a
# more explicit prompt on failure."
RETRY_SYSTEM_PROMPT = """You are a JSON extraction assistant. Return ONLY a valid JSON object
with exactly these keys (use null for any field not found in the document):

{{
  "parties": null,
  "effective_date": null,
  "expiry_date": null,
  "payment_terms": null,
  "obligations": null,
  "termination_clauses": null,
  "jurisdiction": null,
  "governing_law": null
}}

Document Excerpts:
{context}

Return ONLY the JSON object. No markdown, no explanation."""


async def run_extraction(state: dict) -> dict:
    """
    LangGraph node: Extract structured key information from the document.

    This is the first agent in the main pipeline. It reads the doc_id from state,
    retrieves relevant document chunks, and writes extracted_data back to state.

    Args:
        state: GraphState dict — must contain "doc_id"

    Returns:
        Updated state with "extracted_data" (dict) or "error" (str) set.
    """
    doc_id = state["doc_id"]
    logger.info("[extraction_agent] Starting extraction for doc_id=%s", doc_id)

    try:
        # Step 1: Retrieve context with multiple targeted queries
        # retrieve_context_multi_query runs all queries concurrently via asyncio.gather,
        # then deduplicates chunks by chunk_index.
        context = await retrieve_context_multi_query(
            doc_id=doc_id,
            queries=EXTRACTION_QUERIES,
            top_k_per_query=3,  # 7 queries × 3 chunks = up to 21 unique chunks
        )

        if not context:
            logger.warning(
                "[extraction_agent] No context retrieved for doc_id=%s — document may be empty",
                doc_id,
            )
            return {**state, "error": "No document content could be retrieved for extraction."}

        # Step 2: Call GPT-4o with retry logic
        extracted_dict = await _call_llm_with_retry(context)

        if extracted_dict is None:
            return {
                **state,
                "error": "Extraction failed: LLM did not return valid JSON after retry.",
            }

        # Step 3: Validate against Pydantic model
        # WHY: We validate before writing to GraphState so downstream agents
        # receive guaranteed-clean data, not raw LLM output.
        # Pydantic coerces types (e.g., "null" string → None) and catches schema mismatches.
        extracted_data = ExtractedData(**extracted_dict)

        logger.info(
            "[extraction_agent] Extraction complete for doc_id=%s — parties=%s",
            doc_id,
            extracted_data.parties,
        )

        # model_dump() converts the Pydantic model back to a plain dict for GraphState
        return {**state, "extracted_data": extracted_data.model_dump()}

    except Exception as e:
        logger.error(
            "[extraction_agent] Unexpected error for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        return {**state, "error": f"Extraction agent error: {str(e)}"}


async def _call_llm_with_retry(context: str) -> dict | None:
    """
    Call GPT-4o for structured extraction and parse the JSON response.
    Retries once with an explicit template prompt if the first parse fails.

    Returns:
        Parsed dict, or None if both attempts fail.
    """
    user_message = f"Context:\n{context}\n\nExtract the requested fields:"

    # --- First attempt ---
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        logger.info(
            "[extraction_agent] LLM call complete — tokens used: %d",
            response.usage.total_tokens,
        )
        return json.loads(raw)

    except json.JSONDecodeError:
        logger.warning("[extraction_agent] First JSON parse failed — retrying with explicit template")
    except Exception as e:
        logger.error("[extraction_agent] LLM call failed: %s", e)
        raise

    # --- Retry with stricter prompt (truncate context to avoid prompt bloat) ---
    try:
        retry_response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": RETRY_SYSTEM_PROMPT.format(context=context[:4000]),
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw_retry = retry_response.choices[0].message.content
        return json.loads(raw_retry)

    except json.JSONDecodeError:
        logger.error("[extraction_agent] Retry also produced invalid JSON — giving up")
        return None
    except Exception as e:
        logger.error("[extraction_agent] Retry LLM call failed: %s", e)
        raise
