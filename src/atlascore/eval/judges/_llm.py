"""LLM-as-judge evaluation for research briefs."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ...llm import BaseChatCompletionClient
from ...messages import SystemMessage, UserMessage
from .._base import BaseEvalJudge, EvalScore, RunTrajectory


class CriterionScore(BaseModel):
    """Single criterion score."""

    name: str = Field(description="Criterion name")
    score: float = Field(description="Score from 0 to 10")
    reasoning: str = Field(description="Brief reasoning")


class JudgeResponse(BaseModel):
    """Structured judge response."""

    scores: List[CriterionScore] = Field(description="One score per criterion")


class LLMEvalJudge(BaseEvalJudge):
    """Uses a vision/text LLM to score a research brief on configured criteria."""

    DEFAULT_CRITERIA = ["accuracy", "citation_coverage", "hallucination", "clarity"]

    def __init__(
        self,
        client: BaseChatCompletionClient,
        name: Optional[str] = None,
        default_criteria: Optional[List[str]] = None,
        custom_instructions: Optional[str] = None,
    ) -> None:
        super().__init__(name or f"LLM-{getattr(client, 'model', 'Judge')}")
        self.client = client
        self.default_criteria = default_criteria or self.DEFAULT_CRITERIA
        self.custom_instructions = custom_instructions

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[Any] = None,
    ) -> EvalScore:
        """Score the trajectory using structured criteria."""
        eval_criteria = criteria or trajectory.task.eval_criteria or self.default_criteria
        rubric = trajectory.task.rubric or {}

        try:
            system_prompt = self._build_system_prompt(eval_criteria, rubric)
            user_prompt = self._build_user_prompt(trajectory)

            result = await self.client.create(
                messages=[
                    SystemMessage(content=system_prompt, source="system"),
                    UserMessage(content=user_prompt, source="user"),
                ],
                output_format=JudgeResponse,
            )

            if result.structured_output and isinstance(result.structured_output, JudgeResponse):
                judge_resp = result.structured_output
                dimensions = {s.name: s.score / 10.0 for s in judge_resp.scores}
                reasoning = {s.name: s.reasoning for s in judge_resp.scores}
            else:
                import json
                import re

                text = (result.message.content or "").strip()
                match = re.search(r"\{.*\}", text, re.DOTALL)
                parsed = json.loads(match.group(0) if match else text)
                dimensions = {k: float(v) / 10.0 for k, v in parsed.get("scores", {}).items()}
                reasoning = parsed.get("reasoning", {})

            for criterion in eval_criteria:
                if criterion not in dimensions:
                    dimensions[criterion] = 0.5
                    reasoning[criterion] = "Missing criterion; defaulting to 5.0/10"

            overall = sum(dimensions.values()) / len(dimensions) if dimensions else 0.0
            return EvalScore(
                overall=overall,
                dimensions=dimensions,
                reasoning=reasoning,
                trajectory=trajectory,
                metadata={
                    "judge_name": self.name,
                    "model": getattr(self.client, "model", None),
                    "criteria_used": eval_criteria,
                    "raw_response": result.message.content,
                },
            )
        except Exception as e:
            return EvalScore(
                overall=0.5,
                dimensions={c: 0.5 for c in eval_criteria},
                reasoning={c: f"Judge error: {e}" for c in eval_criteria},
                trajectory=trajectory,
                metadata={"judge_name": self.name, "error": str(e), "criteria_used": eval_criteria},
            )

    def _build_system_prompt(
        self, criteria: List[str], rubric: Optional[Dict[str, str]] = None
    ) -> str:
        default_descriptions = {
            "accuracy": "How factually correct and well-supported is the brief?",
            "citation_coverage": "Does the brief cite sources for its key claims?",
            "hallucination": "Does the brief avoid unsupported or fabricated claims?",
            "clarity": "How clear, concise, and well-structured is the brief?",
            "completeness": "Does the brief address all parts of the question?",
            "helpfulness": "How useful is the brief for the intended audience?",
        }

        details = []
        for criterion in criteria:
            description = rubric.get(criterion) if rubric else None
            if not description:
                description = default_descriptions.get(criterion, f"Quality of {criterion}")
            details.append(f"- {criterion}: {description}")

        prompt = (
            "You are an expert evaluation judge. Score the research brief on each criterion "
            "from 0 to 10 (0=poor, 5=average, 10=excellent). Return one score entry per criterion.\n\n"
            "Criteria:\n"
            + "\n".join(details)
            + "\n\nReturn strictly the JSON object matching the JudgeResponse schema."
        )

        if self.custom_instructions:
            prompt += f"\n\nAdditional guidance:\n{self.custom_instructions}"
        return prompt

    def _build_user_prompt(self, trajectory: RunTrajectory) -> str:
        task = trajectory.task
        parts = [f"Task: {task.name}", f"Query: {task.input}"]
        if task.expected_output:
            parts.append(f"Expected reference:\n{task.expected_output}")

        actual = self.extract_answer(trajectory)
        parts.append(f"Actual brief:\n{actual}")
        if not trajectory.success:
            parts.append(f"Execution failed: {trajectory.error}")
        return "\n\n".join(parts)
