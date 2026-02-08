"""FastAPI router for the prediction module."""

import json
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException

from app.modules.prediction.features import FeatureEngineer
from app.modules.prediction.model_factory import ModelFactory, OutcomePredictor
from app.modules.prediction.schemas import (
    CallEvent,
    CallMetadata,
    PredictionInput,
    PredictionResponse,
    TrainRequest,
    TrainResponse,
)


router = APIRouter()

# Default model storage path
MODEL_DIR = Path("data/models")
DEFAULT_MODEL_PATH = MODEL_DIR / "outcome_model.joblib"

# Shared instances
_feature_engineer = FeatureEngineer()
_cached_predictor: OutcomePredictor | None = None


def _load_predictor(algorithm: str = "rf") -> OutcomePredictor:
    """Load or create a predictor, with caching."""
    global _cached_predictor

    if _cached_predictor is not None:
        return _cached_predictor

    predictor = ModelFactory.get_model(algorithm)
    if DEFAULT_MODEL_PATH.exists():
        predictor.load(DEFAULT_MODEL_PATH)
        _cached_predictor = predictor

    return predictor


def _parse_training_data(file_path: str) -> tuple[list[dict[str, Any]], list[str]]:
    """
    Parse training data from JSON file.

    Expected format:
    [
        {
            "call_id": "...",
            "events": [...],
            "metadata": {...},
            "outcome": "success" | "failure"
        },
        ...
    ]
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")

    with open(path) as f:
        data = json.load(f)

    # Handle both top-level list and wrapped dict {"calls": [...]}
    records = data
    if isinstance(data, dict) and "calls" in data:
        records = data["calls"]

    if not isinstance(records, list):
        raise ValueError("Training data must be a JSON array or a dictionary with a 'calls' list")

    feature_rows: list[dict[str, Any]] = []
    outcomes: list[str] = []

    for record in records:
        # Parse events
        events: list[CallEvent] = []
        for e in record.get("events", []):
            # Handle both 'timestamp' and 'ts' from different data sources
            raw_ts = e.get("timestamp") or e.get("ts")
            if raw_ts is None:
                continue

            # Convert numeric offsets (from generator) to datetimes
            if isinstance(raw_ts, (int, float)):
                import datetime
                dt_ts = datetime.datetime.fromtimestamp(raw_ts, tz=datetime.timezone.utc)
            else:
                dt_ts = raw_ts

            events.append(
                CallEvent(
                    timestamp=dt_ts,
                    type=e["type"],
                    data=e.get("data", {}),
                    duration_ms=e.get("duration_ms"),
                )
            )

        # Parse metadata
        meta_dict = record.get("metadata", {})
        metadata = CallMetadata(
            agent_id=meta_dict.get("agent_id", "unknown"),
            org_id=meta_dict.get("org_id", "unknown"),
            time_of_day=meta_dict.get("time_of_day", "unknown"),
            day_of_week=meta_dict.get("day_of_week", "unknown"),
            call_purpose=meta_dict.get("call_purpose", "unknown"),
        )

        # Compute features
        features = _feature_engineer.compute_features(events, metadata)
        feature_rows.append(features)

        # Get outcome
        outcome = record.get("outcome", "unknown")
        outcomes.append(outcome)

    return feature_rows, outcomes


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train outcome prediction model",
    description="Load training data from JSON, featurize, and train the specified model.",
)
async def train_model(request: TrainRequest) -> TrainResponse:
    """
    Train a call outcome prediction model.

    The training file should contain an array of call records with events,
    metadata, and outcome labels.
    """
    global _cached_predictor

    try:
        # Parse training data
        feature_rows, outcomes = _parse_training_data(request.file_path)

        if not feature_rows:
            raise HTTPException(
                status_code=400,
                detail="No valid training samples found in file",
            )

        # Convert to Polars
        X = pl.DataFrame(feature_rows)
        y = pl.Series("outcome", outcomes)

        # Create and train model
        predictor = ModelFactory.get_model(request.algorithm)
        predictor.train(X, y)

        # Save model
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        predictor.save(DEFAULT_MODEL_PATH)

        # Update cache
        _cached_predictor = predictor

        return TrainResponse(
            algorithm=request.algorithm,
            model_path=str(DEFAULT_MODEL_PATH),
            num_samples=len(feature_rows),
            feature_names=predictor.feature_names,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Training failed: {str(e)}",
        )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict call outcome",
    description="Predict outcome from streaming call events in real-time.",
)
async def predict_outcome(request: PredictionInput) -> PredictionResponse:
    """
    Predict call outcome from events streamed so far.

    This endpoint featurizes events on-the-fly for real-time prediction
    during an active call.
    """
    try:
        # Load trained model
        predictor = _load_predictor()
        if predictor.model is None:
            raise HTTPException(
                status_code=400,
                detail="No trained model available. Call /train first.",
            )

        # Compute features from current events
        features = _feature_engineer.compute_features(
            request.events_so_far,
            request.metadata,
        )

        # Create DataFrame with single row
        X = pl.DataFrame([features])

        # Get prediction
        labels, risk_scores = predictor.predict(X)
        predicted_outcome = labels[0]
        risk_score = risk_scores[0]

        # Calculate confidence (distance from 0.5)
        confidence = abs(risk_score - 0.5) * 2  # Scale to 0-1

        # Get top factors for explainability
        top_factors = predictor.get_top_factors(features, n=3)

        return PredictionResponse(
            predicted_outcome=predicted_outcome,
            confidence=confidence,
            risk_score=risk_score,
            top_factors=top_factors,
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


@router.get(
    "/algorithms",
    response_model=list[str],
    summary="List available algorithms",
    description="Get list of supported ML algorithms for training.",
)
async def list_algorithms() -> list[str]:
    """Return list of available algorithm identifiers."""
    return ModelFactory.list_algorithms()
