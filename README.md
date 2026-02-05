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

## Case Study 1: Prompt Similarity & Deduplication Service

This project implements a **modular monolith** architecture for detecting similar and duplicate prompts using semantic embeddings.

### Architecture Overview

```
app/
├── main.py                      # FastAPI app with lifespan handler
└── modules/
    └── prompts/
        ├── schemas.py           # Pydantic request/response models
        ├── services.py          # PromptService singleton (business logic)
        └── router.py            # FastAPI endpoint definitions
```

### Key Implementation Details

#### 1. Variable Masking
Before generating embeddings, template variables are normalized to ensure structural similarity detection:

```python
"Hello {{name}}, how is your {{question}}?"
# → "Hello TOKEN_VAR, how is your TOKEN_VAR?"
```

This allows prompts with identical structure but different variable names to be correctly identified as duplicates.

#### 2. PromptService Singleton
A singleton pattern ensures a single instance manages:
- **Model Loading**: `all-MiniLM-L6-v2` from sentence-transformers (384-dim embeddings)
- **Vector Storage**: Milvus Lite with COSINE similarity metric
- **Lifecycle Management**: Graceful init/shutdown via FastAPI lifespan

#### 3. Sparse HDBSCAN Deduplication
Instead of computing a full N×N distance matrix:
1. Query top-50 neighbors for each prompt via Milvus
2. Build a sparse distance matrix from similarity scores
3. Run `HDBSCAN(metric='precomputed')` on the sparse matrix
4. Group prompts by cluster labels

This approach scales better for large prompt collections.

### Tech Stack

| Component | Technology |
|-----------|------------|
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Vector DB | Milvus Lite |
| Clustering | HDBSCAN |
| Framework | FastAPI |
| Validation | Pydantic v2 |
