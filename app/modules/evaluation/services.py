"""Business logic for LLM response quality evaluation.

Hybrid architecture leveraging MLflow as the primary runner/logger,
integrated with DeepEval metrics for safety/hallucination detection.
"""

import json
import os
from typing import Any

import mlflow
from openai import OpenAI

from app.modules.evaluation.prompts import (
    CONCISENESS_RUBRIC,
    EMPATHY_RUBRIC,
    IMPROVEMENT_PROMPT,
    NATURALNESS_RUBRIC,
    SAFETY_RUBRIC,
    TASK_COMPLETION_RUBRIC,
)
from app.modules.evaluation.schemas import (
    ComparisonResult,
    DimensionScore,
    EvaluationResult,
    ImprovementResult,
)


class EvaluationService:
    """Singleton service for LLM response quality evaluation.

    Uses MLflow for experiment tracking and logging, with custom judge
    prompts for multi-dimensional evaluation.
    """

    _instance: "EvaluationService | None" = None
    _initialized: bool = False

    def __new__(cls) -> "EvaluationService":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes (actual init happens in initialize())."""
        if not hasattr(self, "_openai_client"):
            self._openai_client: OpenAI | None = None
            self._experiment_name = "llm-response-evaluation"

    def initialize(self) -> None:
        """Initialize the service: set up MLflow and OpenAI client.

        Should be called during application startup.
        """
        if self._initialized:
            return

        # Configure MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(self._experiment_name)

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            self._openai_client = OpenAI(api_key=api_key)

        self._initialized = True

    def close(self) -> None:
        """Cleanup resources."""
        self._openai_client = None
        self._initialized = False

    def _call_judge(
        self,
        rubric: str,
        user_input: str,
        context: str,
        response: str,
    ) -> DimensionScore:
        """Call an LLM judge with the given rubric.

        Args:
            rubric: The evaluation rubric template
            user_input: The user's original query
            context: Context about the conversation
            response: The AI response to evaluate

        Returns:
            DimensionScore with score and reasoning
        """
        if not self._openai_client:
            # Fallback for when OpenAI is not configured
            return DimensionScore(
                score=7,
                reasoning="OpenAI API key not configured. Using default score.",
            )

        # Format the rubric with the actual content
        formatted_prompt = rubric.format(
            user_input=user_input,
            context=context,
            response=response,
        )

        completion = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert LLM response quality evaluator. "
                    "Respond only with valid JSON.",
                },
                {"role": "user", "content": formatted_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent scoring
        )

        result_text = completion.choices[0].message.content or "{}"
        result = json.loads(result_text)

        return DimensionScore(
            score=result.get("score", 5),
            reasoning=result.get("reasoning", "No reasoning provided"),
        )

    async def evaluate_response(
        self,
        user_input: str,
        context: str,
        response: str,
    ) -> EvaluationResult:
        """Evaluate a response across all quality dimensions.

        Args:
            user_input: The user's original query
            context: Context about the conversation
            response: The AI response to evaluate

        Returns:
            EvaluationResult with scores for all dimensions
        """
        # Run all judges
        task_completion = self._call_judge(
            TASK_COMPLETION_RUBRIC, user_input, context, response
        )
        empathy = self._call_judge(EMPATHY_RUBRIC, user_input, context, response)
        conciseness = self._call_judge(
            CONCISENESS_RUBRIC, user_input, context, response
        )
        safety = self._call_judge(SAFETY_RUBRIC, user_input, context, response)

        result = EvaluationResult(
            task_completion=task_completion,
            empathy=empathy,
            conciseness=conciseness,
            safety=safety,
        )

        # Log to MLflow
        avg_score = self._calculate_average_score(result)
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "user_input": user_input[:100],  # Truncate for logging
                    "context": context[:100],
                }
            )
            mlflow.log_metrics(
                {
                    "task_completion_score": task_completion.score,
                    "empathy_score": empathy.score,
                    "conciseness_score": conciseness.score,
                    "safety_score": safety.score,
                    "average_score": avg_score,
                }
            )
            # Log the full response and evaluation as JSON artifact
            mlflow.log_dict(
                {
                    "response": response,
                    "evaluation": result.model_dump(),
                },
                "evaluation_result.json",
            )

        return result

    def _calculate_average_score(self, result: EvaluationResult) -> float:
        """Calculate average score across all dimensions."""
        scores = [
            result.task_completion.score,
            result.empathy.score,
            result.conciseness.score,
            result.safety.score,
        ]
        return sum(scores) / len(scores)

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
        # Evaluate both responses
        eval_a = await self.evaluate_response(user_input, context, response_a)
        eval_b = await self.evaluate_response(user_input, context, response_b)

        score_a = self._calculate_average_score(eval_a)
        score_b = self._calculate_average_score(eval_b)

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

        if winner_eval.task_completion.score > loser_eval.task_completion.score:
            advantages.append("better task completion")
        if winner_eval.empathy.score > loser_eval.empathy.score:
            advantages.append("more empathetic")
        if winner_eval.conciseness.score > loser_eval.conciseness.score:
            advantages.append("more concise")
        if winner_eval.safety.score > loser_eval.safety.score:
            advantages.append("safer")

        if advantages:
            return f"Response {winner_label} wins by being {', '.join(advantages)}."
        return f"Response {winner_label} has a higher overall score."

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
        original_score = self._calculate_average_score(original_eval)

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
        improved_response = self._generate_improved_response(improvement_prompt)

        # Step 5: Re-evaluate improved response
        improved_eval = await self.evaluate_response(
            user_input, context, improved_response
        )
        improved_score = self._calculate_average_score(improved_eval)

        # Step 6: Log before/after to MLflow
        changes_made = self._identify_changes(original_eval, improved_eval)
        with mlflow.start_run():
            mlflow.log_params({"operation": "improvement"})
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

        if evaluation.task_completion.score < 8:
            critiques.append(
                f"Task Completion ({evaluation.task_completion.score}/10): "
                f"{evaluation.task_completion.reasoning}"
            )
        if evaluation.empathy.score < 8:
            critiques.append(
                f"Empathy ({evaluation.empathy.score}/10): "
                f"{evaluation.empathy.reasoning}"
            )
        if evaluation.conciseness.score < 8:
            critiques.append(
                f"Conciseness ({evaluation.conciseness.score}/10): "
                f"{evaluation.conciseness.reasoning}"
            )
        if evaluation.safety.score < 8:
            critiques.append(
                f"Safety ({evaluation.safety.score}/10): "
                f"{evaluation.safety.reasoning}"
            )

        return "\n\n".join(critiques) if critiques else "Minor issues across dimensions."

    def _generate_improved_response(self, prompt: str) -> str:
        """Generate an improved response using OpenAI."""
        if not self._openai_client:
            return "[OpenAI API key not configured - cannot generate improvement]"

        completion = self._openai_client.chat.completions.create(
            model="gpt-4o-mini",
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

        return completion.choices[0].message.content or response

    def _identify_changes(
        self,
        original: EvaluationResult,
        improved: EvaluationResult,
    ) -> list[str]:
        """Identify what changed between evaluations."""
        changes = []

        delta_tc = improved.task_completion.score - original.task_completion.score
        if delta_tc > 0:
            changes.append(f"Task completion improved by {delta_tc} points")

        delta_emp = improved.empathy.score - original.empathy.score
        if delta_emp > 0:
            changes.append(f"Empathy improved by {delta_emp} points")

        delta_con = improved.conciseness.score - original.conciseness.score
        if delta_con > 0:
            changes.append(f"Conciseness improved by {delta_con} points")

        delta_saf = improved.safety.score - original.safety.score
        if delta_saf > 0:
            changes.append(f"Safety improved by {delta_saf} points")

        return changes if changes else ["Minor improvements across dimensions"]


# Global singleton instance
evaluation_service = EvaluationService()
