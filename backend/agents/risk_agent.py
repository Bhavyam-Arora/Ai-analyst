"""
agents/risk_agent.py — Identify legal risks with severity ratings

WHY A DEDICATED RISK AGENT:
The extraction agent asks "what IS in this document?" — it pulls facts.
The risk agent asks "what SHOULD worry me?" — it applies legal judgment.

These are different reasoning tasks and benefit from different prompts and
different retrieval strategies:
- Extraction: multi-query for specific fields across the whole document
- Risk: broad retrieval of clauses + focus on liability, IP, obligations, jurisdiction

WHY RETRIEVE EVEN WITH EXTRACTED DATA IN STATE:
The extraction agent's output (extracted_data) gives us structured summaries, but
it loses the exact clause wording. Risk analysis needs the verbatim clause text to:
1. Quote the exact problematic wording in clause_text
2. Reference the page number where it appears
So we retrieve fresh chunks specifically focused on risk-prone sections.

SEVERITY RUBRIC (from CLAUDE.md):
- HIGH:   Unlimited liability, IP loss, irreversible obligations
- MEDIUM: One-sided terms, vague conditions, unusual jurisdiction
- LOW:    Missing standard clauses, informal language
"""

import json
import logging

from openai import AsyncOpenAI

from models.analysis import RiskItem
from rag.retriever import retrieve_context_multi_query

logger = logging.getLogger(__name__)
client = AsyncOpenAI()

# Risk-focused retrieval queries target the most legally sensitive sections
RISK_QUERIES = [
    "liability indemnification indemnity unlimited liability",
    "intellectual property ownership IP assignment rights",
    "termination for convenience penalty damages breach",
    "non-compete non-solicitation restrictive covenant",
    "governing law jurisdiction dispute resolution arbitration",
    "warranty disclaimer limitation of liability",
    "confidentiality obligations data protection privacy",
    "payment obligations penalty interest late fees",
]

SYSTEM_PROMPT = """You are a legal risk analyst. Review the following contract clauses and identify risks.
For each risk found, classify it as HIGH, MEDIUM, or LOW severity.

Severity definitions:
HIGH:   Unlimited liability, complete IP assignment, irreversible obligations, waiver of all remedies
MEDIUM: One-sided terms, vague or ambiguous conditions, unusual jurisdiction, broad indemnification
LOW:    Missing standard protective clauses, informal language, minor imbalances

For each risk, return a JSON object in a "risks" array with exactly these keys:
- severity: "HIGH", "MEDIUM", or "LOW"
- clause_text: the exact quote from the document that is problematic (verbatim, keep it concise)
- page_reference: the page number where this clause appears (integer), or null if not determinable
- explanation: why this clause is a risk (2-3 sentences)
- recommendation: what to do about it — suggested action or alternative wording

Return a JSON object with a "risks" key containing the array.
If no risks are found, return {"risks": []}.
Return ONLY the JSON object. No explanation, no markdown."""

RETRY_SYSTEM_PROMPT = """You are a JSON formatting assistant. Return ONLY a valid JSON object.

Format:
{{
  "risks": [
    {{
      "severity": "HIGH" | "MEDIUM" | "LOW",
      "clause_text": "exact quote from document",
      "page_reference": 5,
      "explanation": "why this is a risk",
      "recommendation": "what to do"
    }}
  ]
}}

Document Clauses:
{context}

Return ONLY the JSON. No markdown."""


async def run_risk(state: dict) -> dict:
    """
    LangGraph node: Identify legal risks from the document.

    Runs after extraction_node. Uses retrieved chunks (not just extracted_data)
    to preserve exact clause wording needed for citations.

    Args:
        state: GraphState dict — must contain "doc_id". May contain "extracted_data".

    Returns:
        Updated state with "risks" (list of dicts) set, or "error" if failed.
    """
    doc_id = state["doc_id"]
    logger.info("[risk_agent] Starting risk analysis for doc_id=%s", doc_id)

    try:
        # Retrieve risk-focused context
        context = await retrieve_context_multi_query(
            doc_id=doc_id,
            queries=RISK_QUERIES,
            top_k_per_query=3,
        )

        if not context:
            logger.warning(
                "[risk_agent] No context retrieved for doc_id=%s — skipping risk analysis",
                doc_id,
            )
            # Not a fatal error — return empty risks list rather than halting the pipeline
            return {**state, "risks": []}

        # Call GPT-4o with retry
        risks_data = await _call_llm_with_retry(context)

        if risks_data is None:
            logger.error("[risk_agent] Failed to get valid JSON for doc_id=%s", doc_id)
            # Partial failure: set risks to empty but don't set error (let summary continue)
            return {**state, "risks": []}

        # Validate each risk item against the Pydantic model
        risk_items = []
        for raw_risk in risks_data.get("risks", []):
            try:
                risk_item = RiskItem(**raw_risk)
                risk_items.append(risk_item.model_dump())
            except Exception as e:
                logger.warning("[risk_agent] Skipping invalid risk item: %s — %s", raw_risk, e)

        logger.info(
            "[risk_agent] Risk analysis complete for doc_id=%s — %d risks identified",
            doc_id,
            len(risk_items),
        )
        return {**state, "risks": risk_items}

    except Exception as e:
        logger.error(
            "[risk_agent] Unexpected error for doc_id=%s: %s",
            doc_id,
            e,
            exc_info=True,
        )
        # Non-fatal: return empty risks so summary agent can still run
        return {**state, "risks": [], "error": f"Risk agent error: {str(e)}"}


async def _call_llm_with_retry(context: str) -> dict | None:
    """
    Call GPT-4o for risk identification and parse the JSON response.

    Returns:
        Dict with a "risks" key, or None if both attempts fail.
    """
    # First attempt
    try:
        response = await client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Clauses to review:\n{context}",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        logger.info(
            "[risk_agent] LLM call complete — tokens used: %d",
            response.usage.total_tokens,
        )
        return json.loads(raw)

    except json.JSONDecodeError:
        logger.warning("[risk_agent] First JSON parse failed — retrying")
    except Exception as e:
        logger.error("[risk_agent] LLM call failed: %s", e)
        raise

    # Retry
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
        return json.loads(retry_response.choices[0].message.content)

    except json.JSONDecodeError:
        logger.error("[risk_agent] Retry also produced invalid JSON")
        return None
    except Exception as e:
        logger.error("[risk_agent] Retry LLM call failed: %s", e)
        raise
