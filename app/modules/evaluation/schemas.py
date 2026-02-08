"""Pydantic schemas for the evaluation module.

Defines the strict API contract matching the Case Study 3 specification.
All 6 dimensions scored: task_completion, empathy, conciseness,
naturalness, safety, clarity.
"""

from pydantic import BaseModel, Field


class DimensionScore(BaseModel):
    """Score and reasoning for a single evaluation dimension."""

    score: int = Field(..., ge=1, le=10, description="Score from 1-10")
    reasoning: str = Field(..., description="Explanation for the score")


class EvaluationResult(BaseModel):
    """Complete evaluation result across all 6 dimensions.

    Matches the Case Study 3 specification with overall_score,
    flags, and suggestions fields.
    """

    overall_score: float = Field(
        ..., description="Average score across all dimensions"
    )
    dimensions: dict[str, DimensionScore] = Field(
        ...,
        description="Scores per dimension: task_completion, empathy, "
        "conciseness, naturalness, safety, clarity",
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Flagged issues (dimensions scoring below 5)",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Improvement suggestions from low-scoring dimensions",
    )


class ImprovementResult(BaseModel):
    """Result from the iterative improvement process.

    Contains the original score, the improved response, and what was changed.
    """

    original_score: float = Field(
        ..., description="Average score of the original response"
    )
    improved_score: float = Field(
        ..., description="Average score of the improved response"
    )
    improved_response: str = Field(..., description="The improved response text")
    changes_made: list[str] = Field(
        ..., description="List of improvements made based on critique"
    )
    original_evaluation: EvaluationResult = Field(
        ..., description="Full evaluation of the original response"
    )
    improved_evaluation: EvaluationResult = Field(
        ..., description="Full evaluation of the improved response"
    )


class ComparisonResult(BaseModel):
    """Result from comparing two responses."""

    winner: str = Field(
        ..., description="Which response won: 'A', 'B', or 'TIE'"
    )
    score_a: float = Field(..., description="Average score for response A")
    score_b: float = Field(..., description="Average score for response B")
    evaluation_a: EvaluationResult = Field(
        ..., description="Full evaluation of response A"
    )
    evaluation_b: EvaluationResult = Field(
        ..., description="Full evaluation of response B"
    )
    reasoning: str = Field(..., description="Explanation of why one response won")


# =============================================================================
# Request Schemas
# =============================================================================


class EvaluateRequest(BaseModel):
    """Request body for evaluating a single response."""

    user_input: str = Field(..., description="The user's original query/input")
    context: str = Field(..., description="Context about the conversation/situation")
    response: str = Field(..., description="The AI response to evaluate")


class BatchEvaluateRequest(BaseModel):
    """Request body for evaluating multiple responses."""

    items: list[EvaluateRequest] = Field(
        ..., min_length=1, description="List of responses to evaluate"
    )


class BatchEvaluationResult(BaseModel):
    """Aggregated result from batch evaluation."""

    results: list[EvaluationResult] = Field(
        ..., description="Individual evaluation results"
    )
    aggregate: dict[str, float] = Field(
        ..., description="Average scores per dimension across all items"
    )
    overall_average: float = Field(
        ..., description="Overall average score across all items"
    )
    total_flags: int = Field(
        ..., description="Total number of flags across all items"
    )


class CompareRequest(BaseModel):
    """Request body for comparing two responses."""

    user_input: str = Field(..., description="The user's original query/input")
    context: str = Field(..., description="Context about the conversation/situation")
    response_a: str = Field(..., description="First response to compare")
    response_b: str = Field(..., description="Second response to compare")


class ImproveRequest(BaseModel):
    """Request body for improving a response."""

    user_input: str = Field(..., description="The user's original query/input")
    context: str = Field(..., description="Context about the conversation/situation")
    response: str = Field(..., description="The AI response to improve")
    threshold: float = Field(
        default=8.0,
        ge=1.0,
        le=10.0,
        description="Minimum average score to skip improvement (default: 8.0)",
    )
