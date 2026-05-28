# LexAI — Legal Document Intelligence

A RAG-powered legal document analysis tool with an agent loop, eval suite, and safety hardening.

## Architecture

```
┌─────────────────┐     PDF upload      ┌──────────────────┐
│   Browser UI    │ ──────────────────► │  Node.js Server  │
│  (HTML/CSS/JS)  │ ◄────────────────── │   (server.js)    │
└─────────────────┘   summary/answers   └────────┬─────────┘
                                                  │ HTTP
                                         ┌────────▼─────────┐
                                         │  Python RAG       │
                                         │  Microservice     │
                                         │  (rag_server.py)  │
                                         │                   │
                                         │  sentence-        │
                                         │  transformers     │
                                         │  + ChromaDB       │
                                         └──────────────────┘
```

## Setup

### 1. Node.js server

```bash
npm install
```

Create `.env`:
```
NGC_API_KEY=your_nvidia_ngc_key
MONGO_URI=your_mongodb_atlas_uri
RAG_SERVICE_URL=http://localhost:5001
```

### 2. Python RAG microservice

```bash
cd rag_service
pip install -r requirements.txt
python rag_server.py
```

### 3. Start the app

```bash
# Terminal 1 - RAG service
cd rag_service && python rag_server.py

# Terminal 2 - Node server
node server.js
```

Open http://localhost:3000

## RAG Pipeline (rag_service/rag_server.py)

- **Chunking**: splits text into 500-word overlapping windows (50-word overlap prevents context loss at boundaries)
- **Embedding**: `all-MiniLM-L6-v2` via sentence-transformers — fast, no GPU needed
- **Vector store**: ChromaDB with cosine similarity
- **Retrieval**: top-3 most relevant chunks per query
- **Safety**: regex-based prompt injection detection before any retrieval

RAG vs fine-tuning trade-off: RAG is used here because documents change (no retraining), data is private (no sending to training pipeline), and it's transparent (you can inspect which chunks were retrieved).

## Agent Loop (server.js → `runAgent()`)

Implements a ReAct-style (Reason + Act) loop:

1. **LLM receives** the user's request + tool descriptions
2. **LLM outputs** a JSON plan: `{ thought, tool, input }`
3. **Tool executes**: one of `summarize_document`, `lookup_legal_term`, `answer_question`
4. **LLM synthesizes** a final answer from the tool result

Tools available:
- `summarize_document()` — returns cached MongoDB summary
- `lookup_legal_term(term)` — returns definition from legal glossary
- `answer_question(question)` — runs full RAG retrieval pipeline

## Eval Suite (evals/eval_suite.py)

```bash
python evals/eval_suite.py --base-url http://localhost:3000
```

Test categories:
- **Summary**: checks expected parties, figures, clauses appear in output
- **RAG Q&A**: checks answers contain grounded terms from document
- **Hallucination**: checks LLM says "I cannot find" rather than inventing
- **Safety**: checks prompt injection attempts are rejected

Pass threshold: 70% accuracy. Exit code 0 = pass, 1 = fail.