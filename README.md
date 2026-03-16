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
| Phase 1 | Setup + API connections | 🔄 In Progress |
| Phase 2 | RAG pipeline: ingest, chunk, embed, retrieve | ⏳ Upcoming |
| Phase 3 | LangGraph agents: extraction, risk, Q&A | ⏳ Upcoming |
| Phase 4 | React frontend | ⏳ Upcoming |
| Phase 5 | Testing + deployment | ⏳ Upcoming |
