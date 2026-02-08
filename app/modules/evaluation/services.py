"""Business logic for LLM response quality evaluation.

Hybrid architecture leveraging MLflow as the primary runner/logger,
integrated with DeepEval metrics for safety/hallucination detection.
Uses LiteLLM for provider-agnostic LLM calls.

Evaluates responses across 6 dimensions per Case Study 3 spec:
task_completion, empathy, conciseness, naturalness, safety, clarity.
"""

import asyncio
import json
import os
from typing import Any

import litellm
import mlflow

from app.modules.evaluation.prompts import (
    CLARITY_RUBRIC,
    CONCISENESS_RUBRIC,
    EMPATHY_RUBRIC,
    IMPROVEMENT_PROMPT,
    NATURALNESS_RUBRIC,
    SAFETY_RUBRIC,
    TASK_COMPLETION_RUBRIC,
)
from app.modules.evaluation.schemas import (
    BatchEvaluationResult,
    ComparisonResult,
    DimensionScore,
    EvaluationResult,
    ImprovementResult,
)

# Map dimension names to their rubric prompts
DIMENSION_RUBRICS: dict[str, str] = {
    "task_completion": TASK_COMPLETION_RUBRIC,
    "empathy": EMPATHY_RUBRIC,
    "conciseness": CONCISENESS_RUBRIC,
    "naturalness": NATURALNESS_RUBRIC,
    "safety": SAFETY_RUBRIC,
    "clarity": CLARITY_RUBRIC,
}

# Dimensions that trigger flags when scoring below this threshold
FLAG_THRESHOLD = 5


class EvaluationService:
    """Singleton service for LLM response quality evaluation.

    Uses MLflow for experiment tracking and logging, with custom judge
    prompts for multi-dimensional evaluation across 6 dimensions.
    """

    _instance: "EvaluationService | None" = None

    def __new__(cls) -> "EvaluationService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes (actual init happens in initialize())."""
        if not hasattr(self, "_initialized"):
            self._initialized = False
        self._model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self._experiment_name = "llm-response-evaluation"

    @property
    def is_initialized(self) -> bool:
        """Whether the service has been initialized."""
        return self._initialized

    def initialize(self) -> None:
        """Initialize the service: set up MLflow.

        Should be called during application startup.
        LiteLLM reads API keys from standard env vars
        (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc).
        """
        if self._initialized:
            return

        # Configure MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self._experiment_name)

        # Suppress LiteLLM debug noise
        litellm.suppress_debug_info = True

        self._initialized = True

    def close(self) -> None:
        """Cleanup resources."""
        self._initialized = False

    # =========================================================================
    # Core LLM Judge
    # =========================================================================

    async def _call_judge(
        self,
        rubric: str,
        user_input: str,
        context: str,
        response: str,
    ) -> DimensionScore:
        """Call an LLM judge with the given rubric.

        Uses LiteLLM for provider-agnostic model calls.
        Set LLM_MODEL env var to swap models (e.g. claude-3-haiku-20240307).

        Args:
            rubric: The evaluation rubric template
            user_input: The user's original query
            context: Context about the conversation
            response: The AI response to evaluate

        Returns:
            DimensionScore with score and reasoning
        """
        formatted_prompt = rubric.format(
            user_input=user_input,
            context=context,
            response=response,
        )

        completion = await litellm.acompletion(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert LLM response quality evaluator. "
                    "Respond only with valid JSON.",
                },
                {"role": "user", "content": formatted_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        result_text = completion.choices[0].message.content or "{}"
        result = json.loads(result_text)

        return DimensionScore(
            score=result.get("score", 5),
            reasoning=result.get("reasoning", "No reasoning provided"),
        )

    # =========================================================================
    # Evaluate
    # =========================================================================

    async def evaluate_response(
        self,
        user_input: str,
        context: str,
        response: str,
    ) -> EvaluationResult:
        """Evaluate a response across all 6 quality dimensions.

        Runs all dimension judges concurrently for speed.

        Args:
            user_input: The user's original query
            context: Context about the conversation
            response: The AI response to evaluate

        Returns:
            EvaluationResult with overall_score, dimensions, flags, suggestions
        """
        # Run all 6 judges concurrently
        tasks = {
            name: self._call_judge(rubric, user_input, context, response)
            for name, rubric in DIMENSION_RUBRICS.items()
        }
        scores: dict[str, DimensionScore] = {}
        results = await asyncio.gather(*tasks.values())
        for name, score in zip(tasks.keys(), results):
            scores[name] = score

        # Calculate overall score
        overall = sum(s.score for s in scores.values()) / len(scores)

        # Generate flags for low-scoring dimensions
        flags = [
            f"{name} scored {s.score}/10 — needs attention"
            for name, s in scores.items()
            if s.score < FLAG_THRESHOLD
        ]

        # Generate suggestions from reasoning of low-scoring dimensions
        suggestions = [
            s.reasoning
            for s in scores.values()
            if s.score < 7
        ]

        result = EvaluationResult(
            overall_score=round(overall, 2),
            dimensions=scores,
            flags=flags,
            suggestions=suggestions,
        )

        # Log to MLflow
        with mlflow.start_run(run_name=f"evaluate: {user_input[:50]}"):
            mlflow.log_params(
                {
                    "user_input": user_input[:100],
                    "context": context[:100],
                    "model": self._model,
                }
            )
            metrics = {
                f"{name}_score": s.score for name, s in scores.items()
            }
            metrics["overall_score"] = overall
            mlflow.log_metrics(metrics)
            mlflow.log_dict(
                {
                    "response": response,
                    "evaluation": result.model_dump(),
                },
                "evaluation_result.json",
            )

        return result

    # =========================================================================
    # Batch Evaluate
    # =========================================================================

    async def evaluate_batch(
        self,
        items: list[dict[str, str]],
    ) -> BatchEvaluationResult:
        """Evaluate multiple responses and return aggregate statistics.

        Args:
            items: List of dicts with user_input, context, response keys

        Returns:
            BatchEvaluationResult with individual results and aggregates
        """
        results = await asyncio.gather(
            *[
                self.evaluate_response(
                    item["user_input"], item["context"], item["response"]
                )
                for item in items
            ]
        )
        results = list(results)

        # Aggregate per-dimension averages
        dimension_names = list(DIMENSION_RUBRICS.keys())
        aggregate: dict[str, float] = {}
        for dim in dimension_names:
            dim_scores = [r.dimensions[dim].score for r in results]
            aggregate[dim] = round(sum(dim_scores) / len(dim_scores), 2)

        overall_avg = round(
            sum(r.overall_score for r in results) / len(results), 2
        )
        total_flags = sum(len(r.flags) for r in results)

        return BatchEvaluationResult(
            results=results,
            aggregate=aggregate,
            overall_average=overall_avg,
            total_flags=total_flags,
        )

    # =========================================================================
    # Compare
    # =========================================================================

    async def compare_responses(
        self,
        user_input: str,
        context: str,
        response_a: str,
        response_b: str,
    ) -> ComparisonResult:
        """Compare two responses and determine a winner.

        Args:
            user_input: The user's original query
            context: Context about the conversation
            response_a: First response to compare
            response_b: Second response to compare

        Returns:
            ComparisonResult with winner and detailed evaluations
        """
        eval_a, eval_b = await asyncio.gather(
            self.evaluate_response(user_input, context, response_a),
            self.evaluate_response(user_input, context, response_b),
        )

        score_a = eval_a.overall_score
        score_b = eval_b.overall_score

        # Determine winner
        if abs(score_a - score_b) < 0.5:
            winner = "TIE"
            reasoning = (
                f"Responses are nearly equivalent with scores "
                f"{score_a:.1f} vs {score_b:.1f}."
            )
        elif score_a > score_b:
            winner = "A"
            reasoning = self._generate_winner_reasoning(eval_a, eval_b, "A")
        else:
            winner = "B"
            reasoning = self._generate_winner_reasoning(eval_b, eval_a, "B")

        return ComparisonResult(
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            evaluation_a=eval_a,
            evaluation_b=eval_b,
            reasoning=reasoning,
        )

    def _generate_winner_reasoning(
        self,
        winner_eval: EvaluationResult,
        loser_eval: EvaluationResult,
        winner_label: str,
    ) -> str:
        """Generate reasoning for why one response won."""
        advantages = []
        for dim in DIMENSION_RUBRICS:
            w_score = winner_eval.dimensions[dim].score
            l_score = loser_eval.dimensions[dim].score
            if w_score > l_score:
                advantages.append(f"better {dim.replace('_', ' ')}")

        if advantages:
            return f"Response {winner_label} wins by being {', '.join(advantages)}."
        return f"Response {winner_label} has a higher overall score."

    # =========================================================================
    # Improve
    # =========================================================================

    async def improve_response(
        self,
        user_input: str,
        context: str,
        response: str,
        threshold: float = 8.0,
    ) -> ImprovementResult:
        """Improve a response using the feedback loop pattern.

        Steps:
        1. Evaluate the original response
        2. If score >= threshold, return as-is
        3. Construct improvement prompt with critique
        4. Generate improved response via LLM
        5. Re-evaluate improved response
        6. Log before/after metrics to MLflow

        Args:
            user_input: The user's original query
            context: Context about the conversation
            response: The AI response to improve
            threshold: Minimum average score to skip improvement

        Returns:
            ImprovementResult with original/improved scores and response
        """
        # Step 1: Evaluate original response
        original_eval = await self.evaluate_response(user_input, context, response)
        original_score = original_eval.overall_score

        # Step 2: Check if improvement is needed
        if original_score >= threshold:
            return ImprovementResult(
                original_score=original_score,
                improved_score=original_score,
                improved_response=response,
                changes_made=["No improvement needed - score meets threshold"],
                original_evaluation=original_eval,
                improved_evaluation=original_eval,
            )

        # Step 3: Construct improvement prompt with critique
        critique = self._build_critique(original_eval)
        improvement_prompt = IMPROVEMENT_PROMPT.format(
            user_input=user_input,
            context=context,
            original_response=response,
            critique=critique,
        )

        # Step 4: Generate improved response
        improved_response = await self._generate_improved_response(improvement_prompt)

        # Step 5: Re-evaluate improved response
        improved_eval = await self.evaluate_response(
            user_input, context, improved_response
        )
        improved_score = improved_eval.overall_score

        # Step 6: Log before/after to MLflow
        changes_made = self._identify_changes(original_eval, improved_eval)
        with mlflow.start_run(run_name=f"improve: {user_input[:50]}"):
            mlflow.log_params({"operation": "improvement", "model": self._model})
            mlflow.log_metrics(
                {
                    "original_score": original_score,
                    "improved_score": improved_score,
                    "score_delta": improved_score - original_score,
                }
            )
            mlflow.log_dict(
                {
                    "original_response": response,
                    "improved_response": improved_response,
                    "original_evaluation": original_eval.model_dump(),
                    "improved_evaluation": improved_eval.model_dump(),
                    "changes_made": changes_made,
                },
                "improvement_result.json",
            )

        return ImprovementResult(
            original_score=original_score,
            improved_score=improved_score,
            improved_response=improved_response,
            changes_made=changes_made,
            original_evaluation=original_eval,
            improved_evaluation=improved_eval,
        )

    def _build_critique(self, evaluation: EvaluationResult) -> str:
        """Build a critique string from evaluation results."""
        critiques = []
        for name, score in evaluation.dimensions.items():
            if score.score < 8:
                label = name.replace("_", " ").title()
                critiques.append(
                    f"{label} ({score.score}/10): {score.reasoning}"
                )
        return "\n\n".join(critiques) if critiques else "Minor issues across dimensions."

    async def _generate_improved_response(self, prompt: str) -> str:
        """Generate an improved response using LiteLLM."""
        completion = await litellm.acompletion(
            model=self._model,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert at improving AI assistant responses. "
                    "Generate improved responses based on critique.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        return completion.choices[0].message.content or "[No content returned from LLM]"

    def _identify_changes(
        self,
        original: EvaluationResult,
        improved: EvaluationResult,
    ) -> list[str]:
        """Identify what changed between evaluations."""
        changes = []
        for dim in DIMENSION_RUBRICS:
            delta = (
                improved.dimensions[dim].score - original.dimensions[dim].score
            )
            if delta > 0:
                label = dim.replace("_", " ").title()
                changes.append(f"{label} improved by {delta} points")
        return changes if changes else ["Minor improvements across dimensions"]


# Global singleton instance
evaluation_service = EvaluationService()
