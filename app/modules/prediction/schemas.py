"""Pydantic schemas for the prediction module."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Types of call events."""

    AGENT_SPEECH = "agent_speech"
    USER_SPEECH = "user_speech"
    SILENCE = "silence"
    TOOL_CALL = "tool_call"
    CALL_START = "call_start"
    CALL_END = "call_end"


class CallEvent(BaseModel):
    """A single event within a call timeline."""

    timestamp: datetime = Field(..., description="When the event occurred")
    type: EventType = Field(..., description="Type of event")
    data: dict[str, Any] = Field(
        default_factory=dict, description="Event-specific payload"
    )
    duration_ms: int | None = Field(
        default=None, description="Duration of the event in milliseconds"
    )


class CallMetadata(BaseModel):
    """Contextual metadata about a call."""

    agent_id: str = Field(..., description="Unique identifier for the agent")
    org_id: str = Field(..., description="Organization identifier")
    time_of_day: str = Field(
        ..., description="Time bucket (e.g., 'morning', 'afternoon', 'evening')"
    )
    day_of_week: str = Field(..., description="Day of the week (e.g., 'monday')")
    call_purpose: str = Field(
        ..., description="Purpose category (e.g., 'support', 'sales', 'followup')"
    )


class PredictionInput(BaseModel):
    """Input for real-time call outcome prediction."""

    call_id: str = Field(..., description="Unique call identifier")
    events_so_far: list[CallEvent] = Field(
        ..., description="Events streamed so far in the call"
    )
    metadata: CallMetadata = Field(..., description="Call context metadata")


class FeatureImpact(BaseModel):
    """Explainability output for a single feature."""

    feature: str = Field(..., description="Name of the feature")
    impact: float = Field(
        ..., description="Impact score (positive = toward prediction, negative = against)"
    )
    value: float = Field(..., description="Current value of the feature")


class PredictionResponse(BaseModel):
    """Response from call outcome prediction."""

    predicted_outcome: str = Field(
        ..., description="Predicted call outcome (e.g., 'success', 'failure')"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Model confidence in prediction"
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of call failure"
    )
    top_factors: list[FeatureImpact] = Field(
        ..., description="Top features driving the prediction"
    )


class TrainRequest(BaseModel):
    """Request to train a prediction model."""

    file_path: str = Field(..., description="Path to JSON file containing training data")
    algorithm: str = Field(
        default="rf",
        description="Algorithm to use: 'lr' (Logistic), 'rf' (RandomForest), 'catboost', 'lightgbm'",
    )


class TrainResponse(BaseModel):
    """Response from model training."""

    algorithm: str = Field(..., description="Algorithm used")
    model_path: str = Field(..., description="Path where model was saved")
    num_samples: int = Field(..., description="Number of training samples")
    feature_names: list[str] = Field(..., description="Features used in training")
