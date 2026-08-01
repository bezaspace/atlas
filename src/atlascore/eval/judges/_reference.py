"""Reference-based evaluation judges for deterministic scoring."""

from __future__ import annotations

from typing import Any, List, Optional

from .._base import BaseEvalJudge, EvalScore, RunTrajectory, _extract_urls, _fuzzy_ratio


class ReferenceEvalJudge(BaseEvalJudge):
    """Scores a trajectory by comparing output to a reference answer.

    Supports exact, contains, and fuzzy matching plus citation URL overlap.
    """

    def __init__(
        self,
        name: str = "Reference",
        mode: str = "fuzzy",
        fuzzy_threshold: float = 0.8,
        case_sensitive: bool = False,
        answer_strategy: str = "last_non_empty",
    ) -> None:
        super().__init__(name, answer_strategy)
        if mode not in ("exact", "contains", "fuzzy"):
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.fuzzy_threshold = fuzzy_threshold
        self.case_sensitive = case_sensitive

    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[Any] = None,
    ) -> EvalScore:
        """Compare actual output to expected output."""
        expected = trajectory.task.expected_output
        if not expected:
            return EvalScore(
                overall=0.0,
                dimensions={"accuracy": 0.0},
                reasoning={"accuracy": "No expected_output provided for reference scoring"},
                trajectory=trajectory,
                metadata={"judge": self.name, "mode": self.mode, "error": "missing_expected"},
            )

        if not trajectory.success:
            return EvalScore(
                overall=0.0,
                dimensions={"accuracy": 0.0},
                reasoning={"accuracy": f"Execution failed: {trajectory.error}"},
                trajectory=trajectory,
                metadata={"judge": self.name, "mode": self.mode},
            )

        actual = self.extract_answer(trajectory)

        a = actual if self.case_sensitive else actual.lower()
        e = expected if self.case_sensitive else expected.lower()

        if self.mode == "exact":
            accuracy = 1.0 if a.strip() == e.strip() else 0.0
            reasoning = f"Exact match: {accuracy == 1.0}"
        elif self.mode == "contains":
            accuracy = 1.0 if e in a else 0.0
            reasoning = f"Expected contained in actual: {accuracy == 1.0}"
        else:  # fuzzy
            similarity = _fuzzy_ratio(a, e)
            accuracy = min(1.0, similarity / self.fuzzy_threshold) if similarity <= self.fuzzy_threshold else 1.0
            reasoning = f"Fuzzy similarity: {similarity:.2%} (threshold {self.fuzzy_threshold:.0%})"

        expected_urls = _extract_urls(expected)
        actual_urls = _extract_urls(actual)
        citation_match = 1.0
        if expected_urls:
            found = len(expected_urls & actual_urls)
            citation_match = found / len(expected_urls)
            reasoning += f"\nCitation overlap: {found}/{len(expected_urls)} ({citation_match:.0%})"

        dimensions = {"accuracy": accuracy, "citation_match": citation_match}
        overall = (accuracy + citation_match) / 2.0

        return EvalScore(
            overall=overall,
            dimensions=dimensions,
            reasoning={"accuracy": reasoning, "citation_match": reasoning},
            trajectory=trajectory,
            metadata={
                "judge": self.name,
                "mode": self.mode,
                "similarity": _fuzzy_ratio(a, e),
            },
        )
