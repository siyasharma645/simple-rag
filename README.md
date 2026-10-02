# Minimal RAG System

A small, readable Retrieval-Augmented Generation project: ask questions about a PDF.

```
OFFLINE:  PDF -> Load -> Chunk -> Embed -> Vector index (FAISS)
ONLINE:   Question -> Embed -> Retrieve top-k -> (Re-rank) -> Prompt -> LLM -> Answer
```

## Setup
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key"                  # optional; without it the prompt is printed
```

## Run
```bash
python rag.py index your_document.pdf                # once per document set
python rag.py ask "What is the notice period?"
python rag.py ask "What is the notice period?" --rerank
python rag.py chat
```

## Tuning
Edit the settings at the top of `rag.py`: `CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`.
Re-run `index` after changing chunk settings.
