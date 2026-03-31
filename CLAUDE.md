# AI Legal Document Analyst — Claude Code Project Context

## Project Summary

A production-quality, full-stack GenAI application that lets users upload legal documents
(contracts, NDAs, lease deeds, agreements), then automatically:
- Extracts structured key information (parties, dates, obligations, termination clauses)
- Identifies legal risks with severity ratings (HIGH / MEDIUM / LOW)
- Answers natural language questions with exact source citations

The system is built on a RAG (Retrieval-Augmented Generation) pipeline backed by a
multi-agent architecture using LangGraph.

---

## Developer Context

The developer has 1.5 years of Full Stack experience with Angular + TypeScript and is
actively transitioning into a GenAI / LLM Application Developer role. This means:

- **Frontend code** can be written at a senior level — clean, modular React/TypeScript
- **Backend / AI pipeline code** should be written with clear explanations of *why* each
  design decision is made, not just *what* the code does
- Always explain GenAI-specific concepts (RAG, embeddings, chunking, LangGraph state
  machines) briefly before diving into implementation
- Prefer production patterns over shortcuts — this is a portfolio piece for GenAI roles
  such as: GenAI Developer, LLM Application Engineer, AI Full Stack Engineer

---

## Tech Stack

### Backend
| Tool | Version / Notes |
|---|---|
| Python | 3.11+ |
| FastAPI | REST API, async-native |
| LangGraph | Multi-agent orchestration (state machine) |
| LangChain | Chunking utilities, prompt templates, retrievers |
| OpenAI SDK | GPT-4o for reasoning, text-embedding-3-small for embeddings |
| Pinecone | Vector database — use doc_id as namespace |
| PyMuPDF (fitz) | PDF parsing with page number extraction |
| python-docx | DOCX parsing |
| Pydantic v2 | Request/response validation models |
| Uvicorn | ASGI server |
| python-dotenv | API key management via .env |
| slowapi | Rate limiting on FastAPI endpoints |

### Frontend
| Tool | Version / Notes |
|---|---|
| React 18 | With TypeScript |
| Vite | Dev server + bundler |
| Tailwind CSS | Utility-first styling |
| React Router v6 | Client-side routing |
| TanStack Query | Server state (fetching, caching, loading states) |
| Zustand | Global client state (current doc_id, analysis state) |
| Axios | HTTP client for FastAPI calls |
| React Dropzone | Drag-and-drop file upload |
| React PDF | In-browser PDF viewer with highlight support |
| EventSource / SSE | Streaming responses from FastAPI (native browser API) |

### Infrastructure
| Tool | Purpose |
|---|---|
| Pinecone Free Tier | Vector DB — 1 index, up to 100k vectors |
| OpenAI API | GPT-4o + embeddings — set a $10 budget cap |
| Docker + Docker Compose | Local dev containerization |
| Git + GitHub | Version control — public repo for portfolio |
| Railway.app or Render.com | Backend deployment (free tier) |
| Vercel or Netlify | Frontend deployment (free tier) |

---

## Folder Structure

```
ai-legal-analyst/              ← monorepo root
├── CLAUDE.md                  ← this file
├── README.md
├── docker-compose.yml
├── .gitignore
│
├── backend/
│   ├── main.py                ← FastAPI app entry point
│   ├── .env                   ← NEVER commit this
│   ├── requirements.txt
│   │
│   ├── api/                   ← Route handlers
│   │   ├── upload.py          ← POST /upload
│   │   ├── analyze.py         ← POST /analyze
│   │   └── chat.py            ← POST /chat
│   │
│   ├── agents/                ← LangGraph agent definitions
│   │   ├── state.py           ← Shared GraphState TypedDict
│   │   ├── extraction_agent.py
│   │   ├── risk_agent.py
│   │   ├── summary_agent.py
│   │   ├── qa_agent.py
│   │   └── graph.py           ← StateGraph assembly + compile
│   │
│   ├── rag/                   ← RAG pipeline components
│   │   ├── pdf_parser.py      ← PyMuPDF → [{page_num, text}]
│   │   ├── chunker.py         ← LangChain splitter → [{chunk_text, page_num, chunk_index}]
│   │   ├── embedder.py        ← OpenAI embeddings in batches
│   │   ├── pinecone_client.py ← upsert_chunks() + retrieve_chunks()
│   │   ├── ingest.py          ← Orchestrates parse → chunk → embed → upsert
│   │   └── retriever.py       ← query string → formatted context string
│   │
│   ├── models/                ← Pydantic request/response models
│   │   ├── upload.py
│   │   ├── analysis.py
│   │   └── chat.py
│   │
│   └── utils/
│       ├── text_cleaner.py    ← Strip headers/footers from parsed PDF text
│       └── token_counter.py   ← Count tokens before LLM calls
│
└── frontend/
    ├── index.html
    ├── vite.config.ts
    ├── tailwind.config.js
    ├── tsconfig.json
    └── src/
        ├── main.tsx
        ├── App.tsx
        ├── store/
        │   └── useDocumentStore.ts   ← Zustand store
        ├── services/
        │   └── api.ts                ← Axios calls to FastAPI
        ├── pages/
        │   ├── UploadPage.tsx
        │   └── AnalysisPage.tsx
        └── components/
            ├── UploadZone.tsx
            ├── AnalysisDashboard.tsx
            ├── ExtractedDataCard.tsx
            ├── RiskPanel.tsx
            ├── ChatInterface.tsx
            └── CitationCard.tsx
```

---

## RAG Pipeline — Core Rules

These rules must be followed in every agent and every retrieval call:

1. **Parse first** — PyMuPDF extracts text per page, preserving page numbers
2. **Chunk with overlap** — 800–1000 tokens per chunk, 100–150 token overlap,
   split by `\n\n` → `\n` → `. ` using LangChain RecursiveCharacterTextSplitter
3. **Embed every chunk** — OpenAI text-embedding-3-small (1536 dimensions)
4. **Store with metadata** — every Pinecone vector must include:
   `{ doc_id, page_num, chunk_index, chunk_text, section_title }`
5. **Namespace by doc_id** — never query across documents; always filter by namespace
6. **Retrieve before reasoning** — agents ALWAYS run Pinecone similarity search
   BEFORE calling GPT-4o. No LLM calls without retrieved context.
7. **top_k = 5** — retrieve 5 chunks per query unless there is a specific reason to change
8. **Ground the prompt** — every system prompt must instruct the model to answer ONLY
   from provided context; return null for missing fields; never infer or hallucinate

---

## Agent Architecture — LangGraph

### Shared GraphState (state.py)
```python
class GraphState(TypedDict):
    doc_id: str
    extracted_data: Optional[dict]
    risks: Optional[list]
    summary: Optional[str]
    error: Optional[str]
```

### Agent Execution Order (main pipeline)
```
START → ingest_node → extraction_node → risk_node → summary_node → END
```
- `risk_node` only runs if `extraction_node` succeeded (conditional edge)
- `qa_node` is a separate subgraph, invoked per user question via POST /chat

### Agent Prompts

**Extraction Agent System Prompt:**
```
You are a legal document analysis expert. You will be given excerpts from a legal document.
Extract the following information ONLY from the provided text. If a field is not present
in the text, return null for that field. Do NOT infer or assume information not explicitly stated.
Return your response as a valid JSON object with exactly these keys:
parties, effective_date, expiry_date, payment_terms, obligations,
termination_clauses, jurisdiction, governing_law.
Context: {retrieved_chunks}
Extract the requested fields:
```

**Risk Agent System Prompt:**
```
You are a legal risk analyst. Review the following contract clauses and identify risks.
For each risk found, classify it as HIGH, MEDIUM, or LOW severity.
HIGH: Unlimited liability, IP loss, irreversible obligations
MEDIUM: One-sided terms, vague conditions, unusual jurisdiction
LOW: Missing standard clauses, informal language
Return a JSON array of risk objects: [{severity, clause_text, page_reference, explanation, recommendation}]
Clauses to review: {retrieved_chunks}
```

**Q&A Agent System Prompt:**
```
You are a legal document assistant. Answer the user's question using ONLY the provided
document excerpts below. If the answer cannot be found in the excerpts, respond with:
'I could not find this information in the provided document.'
For every statement you make, cite the source using [Page X] notation.
Document Excerpts: {retrieved_chunks}
Question: {user_question}
```

---

## Code Standards

### Error Handling
- All LLM calls must be wrapped in `try/except`
- JSON parsing from LLM responses must have a retry with a more explicit prompt on failure
- FastAPI endpoints must return structured error responses, never raw exceptions
- Log all errors with context (doc_id, agent name, error message)

### Async
- All FastAPI route handlers must be `async def`
- All LLM and Pinecone calls must use `await` — never block the event loop

### Validation
- All request bodies validated via Pydantic models in `/backend/models/`
- File uploads: max 20MB, PDF or DOCX only
- All agent JSON outputs validated against Pydantic models before storing in state

### Logging
- Use Python's `logging` module (not print statements) in backend
- Log: ingestion start/end, chunk counts, embedding batch sizes, retrieval results count,
  agent start/end, LLM token usage

---

## Environment Variables

```env
# backend/.env — NEVER commit this file
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
PINECONE_INDEX_NAME=legal-docs
PINECONE_ENVIRONMENT=...
```

The `.env` file must be in `.gitignore` before the first commit.

---

## Build Phases

| Phase | Focus | Est. Time |
|---|---|---|
| Phase 1 | Project setup, API connections, env validation | 3–4 days |
| Phase 2 | RAG pipeline: ingest, chunk, embed, retrieve | 5–7 days |
| Phase 3 | LangGraph agents: extraction, risk, summary, Q&A | 7–10 days |
| Phase 4 | React frontend: upload, dashboard, chat, citations | 7–8 days |
| Phase 5 | Polish, testing with real docs, deployment | 4–5 days |

**Rule:** Never start the next
 phase until the current phase passes its checkpoint test.

---

## Key Interview Talking Points (keep in mind while building)

- RAG > fine-tuning for document Q&A: grounded in real content, cheaper, no retraining
- Hallucination prevention: strict system prompts + null for missing fields + citations
- LangGraph > CrewAI: explicit state transitions, debuggable, production-ready
- Chunking trade-off: 800 tokens with 150 overlap — tested on real contractss
- Scaling path: Celery + Redis for async, Pinecone serverless, PostgreSQL for persistence

---

## Current Phase

**PHASE 1 — Foundation & Setup**

Next action: Create the folder structure, requirements.txt, .env template, and
verify all API connections (OpenAI embed test + Pinecone upsert/query test).
