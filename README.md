# AI Legal Document Analyst

> RAG Pipeline + Multi-Agent Legal Document Analysis  
> Stack: React · TypeScript · FastAPI · LangGraph · OpenAI · Pinecone

---

## Phase 1 Setup — Run These Commands in Order

### 1. Clone and structure

```bash
git clone <your-repo-url>
cd ai-legal-analyst
```

### 2. Backend setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Install all dependencies
pip install -r requirements.txt

# Create your .env file from template
cp .env.example .env
# Now open .env and fill in your actual API keys
```

### 3. Get your API keys

| Service | Where to sign up | What to get |
|---|---|---|
| OpenAI | platform.openai.com | API key — set a $10 usage limit |
| Pinecone | pinecone.io | API key + create index named `legal-docs`, dimension=1536, metric=cosine |

### 4. Verify all connections work

```bash
# From /backend with venv activated
python test_connections.py
```

Expected output:
```
✅ OpenAI embedding OK — vector length: 1536
✅ Pinecone upsert OK — upserted 1 vector
✅ Pinecone query OK — top match score: 0.9999
🎉 All connections verified. Ready for Phase 2.
```

### 5. Start the FastAPI server

```bash
uvicorn main:app --reload --port 8000
```

Verify: open http://localhost:8000 — should return `{"status": "ok"}`  
API docs: http://localhost:8000/docs

### 6. Frontend setup

```bash
cd ../frontend

# Create Vite + React + TypeScript app
npm create vite@latest . -- --template react-ts

# Install all dependencies
npm install react-router-dom @tanstack/react-query zustand axios react-dropzone react-pdf

# Install and configure Tailwind
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Verify: `npm run dev` → open http://localhost:5173

---

## Phase 1 Checkpoint ✅

Before moving to Phase 2, confirm ALL of these:
- [ ] `python test_connections.py` passes with no errors
- [ ] `http://localhost:8000` returns `{"status": "ok"}`
- [ ] `http://localhost:5173` shows the default Vite + React page
- [ ] `.env` is in `.gitignore` and NOT committed to git

---

## Phase 2 — RAG Pipeline

### Files added in Phase 2

```
backend/
├── rag/
│   ├── pdf_parser.py       ← PyMuPDF + python-docx → [{page_num, text}]
│   ├── chunker.py          ← LangChain splitter → [{chunk_text, page_num, chunk_index, section_title}]
│   ├── embedder.py         ← OpenAI text-embedding-3-small in batches
│   ├── pinecone_client.py  ← upsert_chunks() + retrieve_chunks() namespaced by doc_id
│   ├── ingest.py           ← Orchestrates parse → chunk → embed → upsert
│   └── retriever.py        ← query string → formatted context string for LLM prompts
├── api/
│   └── upload.py           ← POST /api/upload (validates file, runs ingestion, returns doc_id)
├── models/
│   ├── upload.py           ← UploadResponse, UploadErrorResponse
│   ├── analysis.py         ← ExtractedData, RiskItem, AnalysisResponse
│   └── chat.py             ← ChatRequest, ChatResponse, Citation
└── utils/
    ├── text_cleaner.py     ← Strip page numbers, repeated headers/footers
    └── token_counter.py    ← tiktoken-based token counting and truncation
```

### Test the RAG pipeline

```bash
# From project root with venv active
python test_phase2.py
```

Expected output:
```
✅ text_cleaner: page number lines removed
✅ token_counter: tiktoken counting correctly
✅ chunker: splits text into N chunks with correct metadata
✅ embedder: OpenAI returned 1536-dim embeddings for 2 chunks
✅ pinecone round-trip: upserted 2 vectors, retrieved top match (score=0.9xxx)
✅ retriever: context formatted with page citations
🎉 Phase 2 complete. RAG pipeline is working end-to-end.
```

### Test the upload endpoint

```bash
# Start the server
uvicorn main:app --reload --port 8000

# Upload a PDF (in a separate terminal)
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/your/contract.pdf"
```

Expected response:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "filename": "contract.pdf",
  "page_count": 12,
  "chunk_count": 47,
  "vector_count": 47,
  "message": "Document uploaded and indexed successfully."
}
```

## Phase 2 Checkpoint ✅

- [ ] `python test_phase2.py` passes all 6 checks
- [ ] `POST /api/upload` with a real PDF returns a `doc_id`
- [ ] Pinecone console shows vectors in the `legal-docs` index
- [ ] Server logs show: parse → chunk → embed → upsert steps

---

## Phase 3 — LangGraph Agent Pipeline

### Files added in Phase 3

```
backend/
├── agents/
│   ├── state.py             ← GraphState TypedDict (shared between all agents)
│   ├── extraction_agent.py  ← Multi-query retrieval + GPT-4o structured extraction
│   ├── risk_agent.py        ← Risk-focused retrieval + severity classification
│   ├── summary_agent.py     ← Plain-English summary using extracted_data + chunks
│   ├── qa_agent.py          ← Per-question retrieval + grounded answer + citations
│   └── graph.py             ← StateGraph assembly: extraction → risk → summary
├── api/
│   ├── analyze.py           ← POST /api/analyze (runs the full graph)
│   └── chat.py              ← POST /api/chat (runs qa_agent per question)
```

### Agent pipeline flow

```
POST /api/analyze
        ↓
  GraphState { doc_id }
        ↓
  extraction_node   ← multi-query Pinecone retrieval → GPT-4o → extracted_data
        ↓ (conditional: skip if extraction failed)
  risk_node         ← risk-focused retrieval → GPT-4o → risks[]
        ↓
  summary_node      ← broad retrieval + extracted_data + risks → GPT-4o → summary
        ↓
  AnalysisResponse { extracted_data, risks, summary, status }

POST /api/chat
        ↓
  qa_agent          ← embed question → Pinecone top_k=5 → GPT-4o → answer + citations[]
        ↓
  ChatResponse { answer, citations[{ page_num, chunk_text, section_title }] }
```

### Test the agent pipeline

```bash
# 1. Start the backend server (from project root with venv active)
uvicorn main:app --reload --port 8000

# 2. Upload a document and grab the doc_id
curl -X POST http://localhost:8000/api/upload \
  -F "file=@/path/to/your/contract.pdf"
# → copy the doc_id from the response

# 3. Run the analysis pipeline
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<your-doc-id>"}'

# 4. Ask a question
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"doc_id": "<your-doc-id>", "question": "Who are the parties to this agreement?"}'

# 5. Run automated phase 3 tests
export TEST_DOC_ID=<your-doc-id>
python test_phase3.py
```

Expected `/api/analyze` response:
```json
{
  "doc_id": "550e8400-e29b-41d4-a716-446655440000",
  "extracted_data": {
    "parties": ["Acme Corp", "Jane Doe"],
    "effective_date": "January 1, 2024",
    "expiry_date": "December 31, 2024",
    "payment_terms": "$5,000 per month, due on the 1st",
    "obligations": ["Acme Corp shall provide...", "Jane Doe shall deliver..."],
    "termination_clauses": ["Either party may terminate with 30 days written notice"],
    "jurisdiction": "State of California",
    "governing_law": "California"
  },
  "risks": [
    {
      "severity": "HIGH",
      "clause_text": "Contractor assigns all intellectual property...",
      "page_reference": 4,
      "explanation": "Unlimited IP assignment with no carve-outs...",
      "recommendation": "Limit assignment to work product created under this agreement"
    }
  ],
  "summary": "This is a service agreement between Acme Corp and Jane Doe...",
  "status": "completed",
  "error": null
}
```

Expected `/api/chat` response:
```json
{
  "answer": "The parties to this agreement are Acme Corp and Jane Doe [Page 1].",
  "citations": [
    {
      "page_num": 1,
      "chunk_text": "This agreement is entered into between Acme Corp...",
      "section_title": "Parties"
    }
  ],
  "doc_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## Phase 3 Checkpoint ✅

- [ ] `python test_phase3.py` passes all checks (with TEST_DOC_ID set)
- [ ] `POST /api/analyze` returns extracted_data, risks, and summary
- [ ] `POST /api/chat` returns a grounded answer with page citations
- [ ] Server logs show: retrieval → LLM call → validation for each agent
- [ ] API docs at `http://localhost:8000/docs` show all 3 endpoints

---

## Phase 4 — React Frontend

### Files added in Phase 4

```
frontend/src/
├── types/
│   └── index.ts              ← TypeScript interfaces (mirrors backend Pydantic models)
├── store/
│   └── useDocumentStore.ts   ← Zustand global state (docId, analysisResult, isAnalyzing)
├── services/
│   └── api.ts                ← Axios API client (uploadDocument, analyzeDocument, sendChatMessage)
├── pages/
│   ├── UploadPage.tsx         ← Landing page: drag-and-drop upload → navigate to /analysis
│   └── AnalysisPage.tsx       ← Results page: loading → dashboard + chat
└── components/
    ├── UploadZone.tsx          ← react-dropzone with file validation + upload button
    ├── AnalysisDashboard.tsx   ← Tabbed view: Key Info | Risks | Summary
    ├── ExtractedDataCard.tsx   ← Renders ExtractedData fields (null-safe)
    ├── RiskPanel.tsx           ← Risk cards sorted HIGH → MEDIUM → LOW
    ├── ChatInterface.tsx       ← Q&A chat with typing indicator + citations
    └── CitationCard.tsx        ← Page citation with supporting excerpt
```

### Run the frontend

```bash
cd frontend
npm run dev
# → http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://localhost:8000`, so you don't need
to configure CORS or environment variables for local development.

### Full stack local dev

```bash
# Terminal 1 — backend
cd ai-legal-analyst
source backend/venv/bin/activate
uvicorn main:app --reload --port 8000

# Terminal 2 — frontend
cd ai-legal-analyst/frontend
npm run dev
```

Then open http://localhost:5173 and upload a contract PDF.

### User flow

```
http://localhost:5173 (UploadPage)
  → drag-and-drop PDF or DOCX
  → click "Upload & Analyze"
  → POST /api/upload (fast — ingests into Pinecone)
  → navigate to /analysis with loading spinner
  → POST /api/analyze (30–60s — runs LangGraph pipeline)
  → show tabs: Key Information | Risks | Summary
  → chat panel: ask questions → POST /api/chat → grounded answers with citations
```

### Page layout

```
┌─────────────────────────────────────────────────────┐
│  ⚖️ AI Legal Analyst  /  contract.pdf   + New Doc  │
├───────────────┬─────────────────────────────────────┤
│               │  [Key Info] [Risks] [Summary]       │
│  📄 filename  │  ─────────────────────────────────  │
│  12 pages     │  <tab content>                      │
│  ✅ Complete  │                                     │
│  HIGH: 2      ├─────────────────────────────────────┤
│  MEDIUM: 3    │  💬 Ask a Question                  │
│               │  [messages + citations]             │
│               │  [textarea input] [Send]            │
└───────────────┴─────────────────────────────────────┘
```

## Phase 4 Checkpoint ✅

- [ ] `npm run dev` starts without errors at http://localhost:5173
- [ ] Upload a PDF → navigates to /analysis with loading state
- [ ] Analysis completes → shows extracted data, risks, summary
- [ ] Risk panel shows severity badges (HIGH/MEDIUM/LOW) sorted correctly
- [ ] Chat panel answers questions with [Page X] citations
- [ ] "New Document" button resets state and returns to upload page
- [ ] TypeScript build passes: `npm run build`

---

## Architecture

```
User Browser (React + Vite)
        ↓ HTTP/REST
FastAPI Backend (Python)
        ↓
LangGraph Agent Pipeline
        ↓
[Pinecone Vector DB]  +  [OpenAI GPT-4o / Embeddings]
        ↓
Response streams back to UI
```

---

## Build Phases

| Phase | Focus | Status |
|---|---|---|
| Phase 1 | Setup + API connections | ✅ Complete |
| Phase 2 | RAG pipeline: ingest, chunk, embed, retrieve | ✅ Complete |
| Phase 3 | LangGraph agents: extraction, risk, Q&A | ✅ Complete |
| Phase 4 | React frontend | ✅ Complete |
| Phase 5 | Testing + deployment | ⏳ Upcoming |
