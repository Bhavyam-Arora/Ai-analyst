"""
agents/graph.py — Assemble the LangGraph StateGraph for the main analysis pipeline

WHY LANGGRAPH OVER A SIMPLE FUNCTION CHAIN:
You could just call extraction → risk → summary sequentially in a function.
LangGraph adds:
1. Explicit state transitions — the execution path is visible and debuggable
2. Conditional edges — skip downstream agents if an upstream one fails
3. Graph visualization — LangGraph can render the graph for documentation
4. Future extensibility — adding a new agent is one add_node + add_edge call

THE GRAPH STRUCTURE:
    START
      ↓
  extraction_node  — retrieves context, calls GPT-4o, writes extracted_data
      ↓ (conditional)
      ├── [if error] → END  (skip risk + summary if extraction failed)
      └── [if ok]   → risk_node
                          ↓
                     summary_node
                          ↓
                         END

WHY A CONDITIONAL EDGE AFTER EXTRACTION:
If extraction fails (e.g., Pinecone is down, GPT-4o times out), running risk and
summary is pointless — they depend on the retrieved context and extracted data.
The conditional edge routes to END immediately, returning a partial AnalysisResponse
with the error message, rather than running 3 more expensive LLM calls.

COMPILED GRAPH (MODULE-LEVEL):
We compile the graph once at import time and reuse the compiled object for every
request. Compilation builds the execution plan (validates edges, resolves node
functions). This avoids re-compiling on every POST /api/analyze call.
"""

import logging

from langgraph.graph import END, StateGraph

from agents.extraction_agent import run_extraction
from agents.risk_agent import run_risk
from agents.state import GraphState
from agents.summary_agent import run_summary

logger = logging.getLogger(__name__)


def _route_after_extraction(state: GraphState) -> str:
    """
    Conditional routing function after extraction_node.

    LangGraph calls this function with the current state after extraction_node
    completes. It returns a string key that maps to the next node in the
    conditional_edges mapping.

    Logic:
    - If extraction set an error: route to "end" (skip risk + summary)
    - Otherwise: route to "risk"

    WHY CHECK state.get("error"):
    Extraction might fail if Pinecone returns no chunks, or if the LLM
    response can't be parsed as JSON (even after retry). In either case,
    extraction_node sets state["error"] rather than raising an exception,
    so the graph continues to this routing function.
    """
    if state.get("error"):
        logger.info(
            "[graph] Extraction failed with error — routing to END: %s",
            state.get("error"),
        )
        return "end"
    return "risk"


def build_analysis_graph():
    """
    Build and compile the LangGraph StateGraph for document analysis.

    Returns:
        Compiled LangGraph CompiledStateGraph ready for invocation via .ainvoke()
    """
    workflow = StateGraph(GraphState)

    # Register nodes
    # Each node is an async function: async def node_fn(state: dict) -> dict
    # LangGraph merges the returned dict into the shared state
    workflow.add_node("extraction", run_extraction)
    workflow.add_node("risk", run_risk)
    # WHY "summarize" NOT "summary":
    # LangGraph 0.1.x forbids node names that match GraphState keys.
    # "summary" is a state key, so the node must have a different name.
    workflow.add_node("summarize", run_summary)

    # Set the entry point — first node to execute
    workflow.set_entry_point("extraction")

    # Conditional edge after extraction:
    # _route_after_extraction returns "risk" or "end"
    # The dict maps those return values to actual node names or END
    workflow.add_conditional_edges(
        "extraction",
        _route_after_extraction,
        {
            "risk": "risk",
            "end": END,
        },
    )

    # Linear edges: risk → summarize → END
    workflow.add_edge("risk", "summarize")
    workflow.set_finish_point("summarize")

    compiled = workflow.compile()
    logger.info("[graph] Analysis graph compiled successfully")
    return compiled


# Compile once at import time — reused for all requests
# WHY MODULE-LEVEL: Avoids re-compiling the graph on every POST /api/analyze.
# The graph structure never changes at runtime.
analysis_graph = build_analysis_graph()
