"""
A minimal, readable RAG (Retrieval-Augmented Generation) system.

OFFLINE (run when your documents change):   Load -> Chunk -> Embed -> Store
ONLINE  (run for every question):           Question -> Embed -> Retrieve
                                            -> (Re-rank) -> Build prompt -> LLM

Usage:
    python rag.py index employee_handbook.pdf
    python rag.py ask "How many weeks of parental leave can I take?"
    python rag.py ask "How many weeks of parental leave?" --rerank
    python rag.py chat
"""

import argparse
import json
import os
from pathlib import Path

import faiss
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------- settings
EMBED_MODEL = "all-MiniLM-L6-v2"                      # turns text into vectors
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # optional re-ranker
LLM_MODEL = "claude-sonnet-5-5"                        # any LLM works; see generate()
CHUNK_SIZE = 500       # characters per chunk (roughly 100-125 tokens)
CHUNK_OVERLAP = 50     # characters repeated between neighbouring chunks
TOP_K = 3              # chunks handed to the LLM
RERANK_CANDIDATES = 20  # chunks fetched before re-ranking
STORE_DIR = Path("rag_store")                          # where the index is saved

_embedder = None


def get_embedder():
    """Load the embedding model once and reuse it."""
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ================================================================ OFFLINE
def load_pdf(path):
    """Step 1 - Load: extract text per page, keeping the page number as metadata."""
    reader = PdfReader(path)
    return [(i + 1, page.extract_text() or "") for i, page in enumerate(reader.pages)]


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Step 2 - Chunk: split text into overlapping pieces, preferring sentence ends."""
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start + size // 2:  # only cut early if it's not too early
                end = cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_index(pdf_path):
    """Steps 1-4: load, chunk, embed, store. Saves the index to disk."""
    chunks = []
    for page_no, text in load_pdf(pdf_path):
        for piece in chunk_text(text):
            chunks.append({"text": piece, "page": page_no, "source": Path(pdf_path).name})
    if not chunks:
        raise SystemExit("No text found. Is the PDF a scan? Try OCR first.")

    # Step 3 - Embed (normalized, so inner product == cosine similarity)
    vectors = get_embedder().encode(
        [c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True
    )

    # Step 4 - Store in a vector index
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(STORE_DIR / "index.faiss"))
    (STORE_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False))
    print(f"Indexed {len(chunks)} chunks from {pdf_path} -> {STORE_DIR}/")


# ================================================================ ONLINE
def load_index():
    if not (STORE_DIR / "index.faiss").exists():
        raise SystemExit("No index found. Run: python rag.py index your_file.pdf")
    index = faiss.read_index(str(STORE_DIR / "index.faiss"))
    chunks = json.loads((STORE_DIR / "chunks.json").read_text())
    return index, chunks


def retrieve(question, index, chunks, k=TOP_K, rerank=False):
    """Embed the question, find nearest chunks, optionally re-rank them."""
    q_vec = get_embedder().encode([question], normalize_embeddings=True)
    n = RERANK_CANDIDATES if rerank else k
    _, ids = index.search(q_vec, min(n, len(chunks)))
    candidates = [chunks[i] for i in ids[0] if i != -1]

    if rerank:  # slower, more precise: reads question + chunk together
        scores = CrossEncoder(RERANK_MODEL).predict([(question, c["text"]) for c in candidates])
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        candidates = [c for _, c in ranked]
    return candidates[:k]


def build_prompt(question, top_chunks):
    """Augment the question with the retrieved context."""
    context = "\n\n".join(f"[{c['source']}, page {c['page']}] {c['text']}" for c in top_chunks)
    return (
        "Answer the question using ONLY the context below. "
        "If the answer is not in the context, say you don't know. "
        "Mention the page number you used.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )


def generate(prompt):
    """Call the LLM. Uses Anthropic if ANTHROPIC_API_KEY is set; otherwise shows the prompt.

    To use another provider, replace the body of this function - the rest of the
    pipeline doesn't change (RAG never retrains the model).
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return "(No ANTHROPIC_API_KEY set - here is the augmented prompt you'd send to an LLM)\n\n" + prompt
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=LLM_MODEL, max_tokens=500, messages=[{"role": "user", "content": prompt}]
    )
    return msg.content[0].text


def ask(question, rerank=False):
    index, chunks = load_index()
    top_chunks = retrieve(question, index, chunks, rerank=rerank)
    print("\nRetrieved:", ", ".join(f"page {c['page']}" for c in top_chunks))
    return generate(build_prompt(question, top_chunks))


# ================================================================ CLI
def main():
    p = argparse.ArgumentParser(description="Minimal RAG system")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="Index a PDF (offline step)").add_argument("pdf")
    a = sub.add_parser("ask", help="Ask one question (online step)")
    a.add_argument("question")
    a.add_argument("--rerank", action="store_true", help="Re-rank candidates")
    sub.add_parser("chat", help="Ask questions in a loop")
    args = p.parse_args()

    if args.cmd == "index":
        build_index(args.pdf)
    elif args.cmd == "ask":
        print(ask(args.question, rerank=args.rerank))
    else:
        while (q := input("\nQuestion (blank to quit): ").strip()):
            print(ask(q))


if __name__ == "__main__":
    main()
