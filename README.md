# Blooming Health Technical Assessment

A FastAPI modular monolith implementing three case studies for the Blooming Health technical assessment.

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector DB**: Milvus Lite
- **Clustering**: HDBSCAN
- **ML**: scikit-learn, CatBoost, LightGBM
- **Experiment Tracking**: MLflow
- **LLM**: OpenAI
- **Framework**: FastAPI

## Quick Start

```bash
# Install dependencies
uv sync

# Set environment variables (for Case Study 3)
export OPENAI_API_KEY=your-key-here

# Run the server
uv run uvicorn app.main:app --reload
```

### MLflow UI

After running evaluations (Case Study 3), view experiment results in the MLflow dashboard:

```bash
uv run mlflow ui --backend-store-uri file:./mlruns --port 5000
```

Then open [http://localhost:5000](http://localhost:5000) and select the **llm-response-evaluation** experiment from the sidebar.

## API Endpoints

Interactive docs available at [http://localhost:8000/docs](http://localhost:8000/docs) — endpoints are grouped by case study.

### Case Study 1: Prompt Similarity & Deduplication

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/prompts/embeddings/generate` | Generate and store embeddings for prompts |
| POST | `/prompts/search/semantic` | Semantic search across all prompts |
| GET | `/prompts/{prompt_id}/similar` | Find prompts similar to a specific one |
| GET | `/prompts/analysis/duplicates` | Detect duplicate clusters via HDBSCAN |

### Case Study 2: Call Outcome Prediction

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/prediction/train` | Train an ML model (`lr`, `rf`, `catboost`, `lightgbm`) |
| POST | `/prediction/predict` | Real-time call outcome prediction from streaming events |
| GET | `/prediction/algorithms` | List available ML algorithms |

### Case Study 3: LLM Response Quality Evaluation

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/evaluate` | Score a response across 4 quality dimensions |
| POST | `/api/compare` | Compare two responses and pick a winner |
| POST | `/api/improve` | Improve a response via critique feedback loop |

---

## Case Study 1: Prompt Similarity & Deduplication Service

This project implements a **modular monolith** architecture for detecting similar and duplicate prompts using semantic embeddings.

### Architecture Overview

```
app/modules/prompts/
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

#### 2. Sparse HDBSCAN Deduplication
Instead of computing a full N×N distance matrix:
1. Query top-50 neighbors for each prompt via Milvus
2. Build a sparse distance matrix from similarity scores
3. Run `HDBSCAN(metric='precomputed')` on the sparse matrix

---

## Case Study 2: Call Outcome Prediction

Real-time ML prediction for call outcomes using multiple algorithms.

### Architecture Overview

```
app/modules/prediction/
├── schemas.py           # CallEvent, CallMetadata, PredictionInput/Response
├── features.py          # FeatureEngineer for temporal/interaction features
├── model_factory.py     # Strategy pattern with 4 ML adapters
└── router.py            # Train and predict endpoints
```

### Key Implementation Details

#### 1. Feature Engineering
`FeatureEngineer.compute_features()` calculates:
- **Temporal**: `total_duration`, `silence_ratio`, `agent_talk_ratio`, `user_talk_ratio`
- **Interaction**: `turn_count`, `interruption_count`, `tool_usage_count`

#### 2. Model Factory (Strategy Pattern)
Supports 4 algorithms with lazy loading:
| Algorithm | Adapter | Top Factors Method |
|-----------|---------|-------------------|
| `lr` | LogisticRegression | `coef_` weights |
| `rf` | RandomForest | `feature_importances_` |
| `catboost` | CatBoost | `feature_importances_` |
| `lightgbm` | LightGBM | `feature_importances_` |

#### 3. Real-time Prediction
The `/prediction/predict` endpoint featurizes streaming events on-the-fly for mid-call prediction.

---

## Case Study 3: LLM Response Quality Evaluation

Hybrid LLM evaluation framework using MLflow for experiment tracking and OpenAI for judge LLM calls.

### Architecture Overview

```
app/modules/evaluation/
├── schemas.py           # EvaluationResult, ComparisonResult, ImprovementResult
├── prompts.py           # Judge rubrics for each quality dimension
├── services.py          # EvaluationService with MLflow integration
└── router.py            # Evaluate, compare, improve endpoints
```

### Key Implementation Details

#### 1. Multi-Dimensional Evaluation
Responses are scored (1-10) across 4 dimensions:
- **Task Completion**: Did the response address the user's query?
- **Empathy**: Was the response compassionate and supportive?
- **Conciseness/Naturalness**: Was the response clear and natural?
- **Safety**: Did the response avoid harmful content?

#### 2. Comparison Mode
Evaluates two responses independently, calculates average scores, and determines a winner (A, B, or TIE if within 0.5 points).

#### 3. Improvement Feedback Loop
1. Evaluate original response
2. If score >= threshold, return as-is
3. Build critique from low-scoring dimensions
4. Generate improved response via LLM
5. Re-evaluate and log before/after metrics to MLflow

### Environment Variables

```bash
# LLM provider API keys (set whichever provider you use)
OPENAI_API_KEY=...       # For OpenAI models (default: gpt-4o-mini)
ANTHROPIC_API_KEY=...    # For Anthropic models

# Model selection (uses LiteLLM routing)
LLM_MODEL=gpt-4o-mini                    # OpenAI (default)
LLM_MODEL=claude-3-haiku-20240307        # Anthropic
LLM_MODEL=gemini/gemini-2.0-flash        # Google
```
