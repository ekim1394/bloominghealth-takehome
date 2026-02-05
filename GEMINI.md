# GEMINI.md

Project documentation for AI assistants.

## Project Overview

**Blooming Health Technical Assessment** - A FastAPI modular monolith with two case studies:
1. **Prompt Similarity & Deduplication** - Semantic prompt management
2. **Call Outcome Prediction** - Real-time ML prediction for call outcomes

## Architecture

Modular monolith with clean separation:
- `app/main.py` - FastAPI app with lifespan lifecycle
- `app/modules/prompts/` - Prompts domain module
  - `schemas.py` - Pydantic models
  - `services.py` - Business logic (PromptService singleton)
  - `router.py` - API endpoints
- `app/modules/prediction/` - Call prediction module
  - `schemas.py` - CallEvent, CallMetadata, PredictionInput/Response
  - `features.py` - FeatureEngineer for temporal/interaction features
  - `model_factory.py` - Strategy pattern with 4 ML adapters
  - `router.py` - Train and predict endpoints

## Key Patterns

### Variable Masking
Template variables (`{{var}}`) are replaced with `TOKEN_VAR` before embedding to detect structural similarity.

### Sparse HDBSCAN
Deduplication uses k-NN queries (top-50) to build a sparse distance matrix, avoiding O(n²) memory.

### Model Factory (Strategy Pattern)
Supports 4 algorithms: LogisticRegression, RandomForest, CatBoost, LightGBM with lazy loading.

### Real-time Feature Engineering
`FeatureEngineer.compute_features()` calculates talk ratios, turn counts, and interruptions on-the-fly.

## Tech Stack

- **Python 3.11+** with `uv` for package management
- **FastAPI** + **Pydantic v2**
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-dim)
- **Milvus Lite** (local vector DB)
- **HDBSCAN** (clustering)
- **polars** (DataFrames)
- **scikit-learn**, **CatBoost**, **LightGBM** (ML)

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
| POST | `/prediction/train` | Train ML model |
| POST | `/prediction/predict` | Real-time prediction |
| GET | `/prediction/algorithms` | List algorithms |

