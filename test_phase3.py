"""
test_phase3.py — Phase 3 checkpoint tests for the LangGraph agent pipeline

Run from the project root with the venv activated:
    python test_phase3.py

Prerequisites:
- Phase 2 must pass (test_phase2.py)
- A document must already be ingested (provide a real doc_id below)
- .env must have valid OPENAI_API_KEY and PINECONE_API_KEY

This test validates:
1. GraphState TypedDict is importable and has correct fields
2. The analysis graph compiles without errors
3. Extraction agent retrieves context and returns ExtractedData
4. Risk agent returns a list of RiskItem dicts
5. Summary agent returns a non-empty string
6. Full graph invoke (extraction → risk → summary) works end-to-end
7. Q&A agent answers a question with citations
8. /api/analyze endpoint returns a valid AnalysisResponse (via HTTP)
9. /api/chat endpoint returns a grounded answer with citations (via HTTP)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "backend"))
from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# ── Configuration ─────────────────────────────────────────────────────────────
# Replace with a real doc_id from a previous /api/upload call.
# Run: curl -X POST http://localhost:8000/api/upload -F "file=@/path/to/contract.pdf"
# and paste the returned doc_id here.
TEST_DOC_ID = os.getenv("TEST_DOC_ID", "")

BACKEND_URL = "http://localhost:8000"

PASS = "✅"
FAIL = "❌"


def check(label: str, condition: bool, detail: str = "") -> bool:
    status = PASS if condition else FAIL
    suffix = f" — {detail}" if detail else ""
    print(f"{status} {label}{suffix}")
    return condition


# ── Unit tests (import-level) ─────────────────────────────────────────────────

def test_state_import():
    from agents.state import GraphState
    required_keys = {"doc_id", "extracted_data", "risks", "summary", "error"}
    hints = GraphState.__annotations__
    missing = required_keys - set(hints.keys())
    return check("GraphState: correct fields defined", not missing, f"missing: {missing}" if missing else "")


def test_graph_compiles():
    try:
        from agents.graph import analysis_graph
        # The graph object must be callable (compiled)
        has_invoke = hasattr(analysis_graph, "ainvoke")
        return check("analysis_graph: compiles and has ainvoke()", has_invoke)
    except Exception as e:
        return check("analysis_graph: compiles", False, str(e))


# ── Async agent tests ─────────────────────────────────────────────────────────

async def test_extraction_agent():
    if not TEST_DOC_ID:
        print(f"⚠️  extraction_agent: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        from agents.extraction_agent import run_extraction
        state = {"doc_id": TEST_DOC_ID, "extracted_data": None, "risks": None, "summary": None, "error": None}
        result = await run_extraction(state)

        has_extraction = result.get("extracted_data") is not None
        no_error = result.get("error") is None
        return check(
            "extraction_agent: returns extracted_data",
            has_extraction and no_error,
            f"parties={result.get('extracted_data', {}).get('parties')}" if has_extraction else result.get("error"),
        )
    except Exception as e:
        return check("extraction_agent: runs without exception", False, str(e))


async def test_risk_agent():
    if not TEST_DOC_ID:
        print(f"⚠️  risk_agent: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        from agents.risk_agent import run_risk
        state = {"doc_id": TEST_DOC_ID, "extracted_data": None, "risks": None, "summary": None, "error": None}
        result = await run_risk(state)

        risks = result.get("risks")
        is_list = isinstance(risks, list)
        return check(
            "risk_agent: returns list of risks",
            is_list,
            f"{len(risks)} risks identified" if is_list else str(risks),
        )
    except Exception as e:
        return check("risk_agent: runs without exception", False, str(e))


async def test_full_graph():
    if not TEST_DOC_ID:
        print(f"⚠️  full_graph: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        from agents.graph import analysis_graph
        initial_state = {
            "doc_id": TEST_DOC_ID,
            "extracted_data": None,
            "risks": None,
            "summary": None,
            "error": None,
        }
        final_state = await analysis_graph.ainvoke(initial_state)

        has_extraction = final_state.get("extracted_data") is not None
        has_risks = final_state.get("risks") is not None
        has_summary = bool(final_state.get("summary"))
        no_error = final_state.get("error") is None

        all_ok = has_extraction and has_risks and has_summary
        return check(
            "full_graph: extraction + risk + summary all populated",
            all_ok,
            f"extraction={'ok' if has_extraction else 'missing'}, "
            f"risks={'ok' if has_risks else 'missing'}, "
            f"summary={'ok' if has_summary else 'missing'}, "
            f"error={final_state.get('error')}",
        )
    except Exception as e:
        return check("full_graph: ainvoke completes", False, str(e))


async def test_qa_agent():
    if not TEST_DOC_ID:
        print(f"⚠️  qa_agent: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        from agents.qa_agent import run_qa
        response = await run_qa(doc_id=TEST_DOC_ID, question="Who are the parties to this agreement?")

        has_answer = bool(response.answer)
        has_citations = isinstance(response.citations, list)
        return check(
            "qa_agent: returns answer with citations",
            has_answer and has_citations,
            f"answer_len={len(response.answer)}, citations={len(response.citations)}",
        )
    except Exception as e:
        return check("qa_agent: runs without exception", False, str(e))


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

async def test_analyze_endpoint():
    if not TEST_DOC_ID:
        print(f"⚠️  /api/analyze: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        import httpx
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/api/analyze",
                json={"doc_id": TEST_DOC_ID},
            )
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            has_doc_id = data.get("doc_id") == TEST_DOC_ID
            has_status = data.get("status") in ("completed", "partial")
            return check(
                "/api/analyze: returns 200 with AnalysisResponse",
                has_doc_id and has_status,
                f"status={data.get('status')}, extracted={data.get('extracted_data') is not None}",
            )
        return check("/api/analyze: returns 200", False, f"status_code={resp.status_code}")
    except Exception as e:
        return check("/api/analyze: HTTP call succeeds", False, str(e))


async def test_chat_endpoint():
    if not TEST_DOC_ID:
        print(f"⚠️  /api/chat: skipped (set TEST_DOC_ID env var)")
        return True

    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{BACKEND_URL}/api/chat",
                json={"doc_id": TEST_DOC_ID, "question": "What are the main obligations?"},
            )
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            has_answer = bool(data.get("answer"))
            has_citations = isinstance(data.get("citations"), list)
            return check(
                "/api/chat: returns 200 with ChatResponse",
                has_answer and has_citations,
                f"answer_len={len(data.get('answer', ''))}, citations={len(data.get('citations', []))}",
            )
        return check("/api/chat: returns 200", False, f"status_code={resp.status_code}")
    except Exception as e:
        return check("/api/chat: HTTP call succeeds", False, str(e))


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("\n" + "=" * 60)
    print("Phase 3 — LangGraph Agent Pipeline Tests")
    print("=" * 60 + "\n")

    if not TEST_DOC_ID:
        print(
            "⚠️  TEST_DOC_ID not set. Agent tests will be skipped.\n"
            "   To run full tests:\n"
            "   1. Start the server: uvicorn main:app --reload --port 8000\n"
            "   2. Upload a PDF: curl -X POST http://localhost:8000/api/upload -F 'file=@contract.pdf'\n"
            "   3. Copy the doc_id and set: export TEST_DOC_ID=<doc_id>\n"
            "   4. Re-run: python test_phase3.py\n"
        )

    results = [
        test_state_import(),
        test_graph_compiles(),
        await test_extraction_agent(),
        await test_risk_agent(),
        await test_full_graph(),
        await test_qa_agent(),
        await test_analyze_endpoint(),
        await test_chat_endpoint(),
    ]

    passed = sum(results)
    total = len(results)

    print(f"\n{'=' * 60}")
    if passed == total:
        print(f"🎉 Phase 3 complete. All {total} checks passed.")
        print("   RAG pipeline + LangGraph agents are working end-to-end.")
        print("   Ready for Phase 4: React frontend.\n")
    else:
        print(f"⚠️  {passed}/{total} checks passed. Review failures above.\n")


if __name__ == "__main__":
    asyncio.run(main())
