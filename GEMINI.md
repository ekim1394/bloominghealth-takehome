# GEMINI.md

Project documentation for AI assistants.

## Project Overview

**Prompt Similarity & Deduplication Service** - A FastAPI microservice for semantic prompt management.

## Architecture

Modular monolith with clean separation:
- `app/main.py` - FastAPI app with lifespan lifecycle
- `app/modules/prompts/` - Prompts domain module
  - `schemas.py` - Pydantic models
  - `services.py` - Business logic (PromptService singleton)
  - `router.py` - API endpoints

## Key Patterns

### Variable Masking
Template variables (`{{var}}`) are replaced with `TOKEN_VAR` before embedding to detect structural similarity.

### Sparse HDBSCAN
Deduplication uses k-NN queries (top-50) to build a sparse distance matrix, avoiding O(n²) memory.

### Dependency Injection
`PromptService` is a singleton injected via FastAPI's `Depends()`.

## Tech Stack

- **Python 3.11+** with `uv` for package management
- **FastAPI** + **Pydantic v2**
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim)
- **Milvus Lite** (local vector DB)
- **HDBSCAN** (clustering)

## Commands

```bash
# Install dependencies
uv sync

# Run dev server
uv run uvicorn app.main:app --reload

# Import check
uv run python -c "from app.main import app"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/prompts/embeddings/generate` | Embed prompts |
| POST | `/prompts/search/semantic` | Semantic search |
| GET | `/prompts/{id}/similar` | Find similar |
| GET | `/prompts/analysis/duplicates` | Detect duplicates |
