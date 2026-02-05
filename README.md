# Prompt Similarity & Deduplication Service

A FastAPI service for managing prompt embeddings, semantic search, and duplicate detection.

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Milvus Lite
- **Clustering**: HDBSCAN
- **Framework**: FastAPI

## Quick Start

```bash
# Install dependencies
uv sync

# Run the server
uv run uvicorn app.main:app --reload
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/prompts/embeddings/generate` | POST | Generate embeddings for prompts |
| `/prompts/search/semantic` | POST | Semantic search across prompts |
| `/prompts/{prompt_id}/similar` | GET | Find similar to specific prompt |
| `/prompts/analysis/duplicates` | GET | Detect duplicate clusters |
