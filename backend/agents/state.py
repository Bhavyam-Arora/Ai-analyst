"""
agents/state.py — Shared state definition for the LangGraph pipeline

WHY A SHARED STATE (The Core LangGraph Concept):
LangGraph is a state machine framework. Every agent (node) in the graph
reads from and writes to a single shared state object — the GraphState.
Think of it like a baton in a relay race: each agent picks it up, does
its work, adds its output to the state, and passes it to the next agent.

WHY TypedDict AND NOT A PYDANTIC MODEL:
LangGraph expects state to be a TypedDict (or dataclass). Pydantic models
add validation overhead and LangGraph needs to merge state updates using
dict operations. TypedDict gives us type hints for IDE autocomplete without
the overhead.

STATE FLOW:
    START
      ↓
  ingest_node   → sets: doc_id (already set before graph starts)
      ↓
  extraction_node → reads: doc_id | writes: extracted_data
      ↓ (only if extraction succeeded — conditional edge)
  risk_node       → reads: doc_id, extracted_data | writes: risks
      ↓
  summary_node    → reads: doc_id, extracted_data, risks | writes: summary
      ↓
    END

The qa_node is a SEPARATE single-node subgraph invoked per chat question,
not part of the main analysis pipeline.
"""

from typing import Optional
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Shared state passed between all agents in the LangGraph pipeline.

    Every key is Optional except doc_id — agents only populate their
    own output key and leave others untouched.

    FIELDS:
    - doc_id:         The Pinecone namespace for this document. Set before
                      the graph starts; never modified by agents.
    - extracted_data: Output of extraction_agent — structured key-value info.
    - risks:          Output of risk_agent — list of risk dicts.
    - summary:        Output of summary_agent — plain-English summary string.
    - error:          Set by any agent that fails — halts downstream agents
                      via conditional routing in graph.py.
    """

    doc_id: str
    extracted_data: Optional[dict]
    risks: Optional[list]
    summary: Optional[str]
    error: Optional[str]
