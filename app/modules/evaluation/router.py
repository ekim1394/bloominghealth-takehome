"""FastAPI router for the evaluation module."""

from fastapi import APIRouter, Depends, HTTPException

from app.modules.evaluation.schemas import (
    CompareRequest,
    ComparisonResult,
    EvaluateRequest,
    EvaluationResult,
    ImprovementResult,
    ImproveRequest,
)
from app.modules.evaluation.services import EvaluationService, evaluation_service


router = APIRouter()


def get_evaluation_service() -> EvaluationService:
    """Dependency injection for EvaluationService."""
    if not evaluation_service._initialized:
        evaluation_service.initialize()
    return evaluation_service


@router.post(
    "/evaluate",
    response_model=EvaluationResult,
    summary="Evaluate a response",
    description="Evaluate an AI response across multiple quality dimensions: "
    "task completion, empathy, conciseness, and safety.",
)
async def evaluate_response(
    request: EvaluateRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> EvaluationResult:
    """Evaluate a single AI response.

    Returns scores and reasoning for each quality dimension.
    Results are logged to MLflow for tracking.
    """
    try:
        return await service.evaluate_response(
            user_input=request.user_input,
            context=request.context,
            response=request.response,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post(
    "/compare",
    response_model=ComparisonResult,
    summary="Compare two responses",
    description="Evaluate two AI responses and determine which one is better "
    "based on average scores across all dimensions.",
)
async def compare_responses(
    request: CompareRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> ComparisonResult:
    """Compare two AI responses and pick a winner.

    Evaluates both responses independently, calculates average scores,
    and determines a winner (A, B, or TIE if scores are within 0.5 points).
    """
    try:
        return await service.compare_responses(
            user_input=request.user_input,
            context=request.context,
            response_a=request.response_a,
            response_b=request.response_b,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Comparison failed: {str(e)}",
        )


@router.post(
    "/improve",
    response_model=ImprovementResult,
    summary="Improve a response",
    description="Evaluate a response and if it scores below the threshold, "
    "use LLM feedback to generate an improved version.",
)
async def improve_response(
    request: ImproveRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> ImprovementResult:
    """Improve an AI response using the feedback loop pattern.

    1. Evaluates the original response
    2. If score >= threshold, returns as-is
    3. Otherwise, generates critique and creates improved response
    4. Re-evaluates and logs before/after metrics to MLflow
    """
    try:
        return await service.improve_response(
            user_input=request.user_input,
            context=request.context,
            response=request.response,
            threshold=request.threshold,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Improvement failed: {str(e)}",
        )
