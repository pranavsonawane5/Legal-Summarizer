"""
RAG Microservice for Legal Summarizer
--------------------------------------
  - RAG pipeline: chunking, embeddings, vector store, retrieval
  - LLM primitives: token-aware chunking
  - Safety: prompt injection detection before retrieval
"""

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import chromadb
import re
import hashlib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Models & Vector Store ────────────────────────────────────────────────────
# 'all-MiniLM-L6-v2' is fast, lightweight, and good for semantic similarity
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")

CHUNK_SIZE = 500        # tokens approx (we use words as proxy)
CHUNK_OVERLAP = 50      # overlap to avoid cutting mid-sentence context
TOP_K = 3               # number of chunks to retrieve per query


# ── Safety: Prompt Injection Detection ──────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore (all |previous |prior )?(instructions?|prompts?|context)",
    r"forget (everything|all|what)",
    r"you are now",
    r"act as (a |an )?(?!lawyer|attorney|paralegal)",  # allow legal personas
    r"disregard (all |previous )?",
    r"new (role|persona|instructions?)",
    r"system prompt",
    r"jailbreak",
    r"DAN mode",
]

def is_prompt_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            logger.warning(f"Prompt injection detected: pattern='{pattern}'")
            return True
    return False


# ── Chunking ─────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Splits document text into overlapping word-based chunks.
    Overlap prevents losing context at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap  # slide window with overlap
    logger.info(f"Chunked document into {len(chunks)} chunks (size={chunk_size}, overlap={overlap})")
    return chunks


# ── Collection helpers ────────────────────────────────────────────────────────
def get_or_create_collection(doc_id: str):
    safe_id = "doc_" + doc_id[:32]
    return chroma_client.get_or_create_collection(
        name=safe_id,
        metadata={"hnsw:space": "cosine"}  
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/index", methods=["POST"])
def index_document():
    """
    Receives raw document text + doc_id.
    Chunks → embeds → stores in ChromaDB.
    Called by Node server after PDF extraction.
    """
    data = request.json
    doc_id = data.get("doc_id")
    text = data.get("text", "")

    if not doc_id or not text:
        return jsonify({"error": "doc_id and text required"}), 400

    collection = get_or_create_collection(doc_id)

    if collection.count() > 0:
        logger.info(f"Document {doc_id[:8]}... already indexed, skipping")
        return jsonify({"status": "already_indexed", "chunks": collection.count()})

    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

    logger.info(f"Indexed {len(chunks)} chunks for doc {doc_id[:8]}...")
    return jsonify({"status": "indexed", "chunks": len(chunks)})


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """
    Embeds the query, retrieves top-K relevant chunks from ChromaDB.
    Includes prompt injection check before processing.
    """
    data = request.json
    doc_id = data.get("doc_id")
    question = data.get("question", "")

    if not doc_id or not question:
        return jsonify({"error": "doc_id and question required"}), 400

    # Safety check
    if is_prompt_injection(question):
        return jsonify({"error": "Unsafe input detected. Please ask a genuine legal question."}), 400

    collection = get_or_create_collection(doc_id)
    if collection.count() == 0:
        return jsonify({"error": "Document not indexed. Upload it first."}), 404

    # Embed query and retrieve
    query_embedding = embedder.encode([question]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(TOP_K, collection.count())
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    logger.info(f"Retrieved {len(chunks)} chunks for query (distances: {[round(d,3) for d in distances]})")

    return jsonify({
        "chunks": chunks,
        "distances": distances,  
        "top_k": TOP_K
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "all-MiniLM-L6-v2"})


if __name__ == "__main__":
    app.run(port=5001, debug=False)