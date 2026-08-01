"""Regression tests for the evaluation harness and golden dataset."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlascore.eval import (
    CallableTarget,
    Dataset,
    EvalRunner,
    JudgeResponse,
    LLMEvalJudge,
    ReferenceEvalJudge,
    Task,
)
from atlascore.eval._base import RunTrajectory
from atlascore.eval.judges._llm import CriterionScore
from atlascore.messages import AssistantMessage, UserMessage


def _golden_path() -> Path:
    return Path(__file__).parents[2] / "eval" / "golden" / "research.json"


async def _perfect_answer(task: Task) -> str:
    return task.expected_output or ""


async def _bad_answer(task: Task) -> str:
    return "I don't know."


@pytest.mark.asyncio
async def test_golden_regression_meets_baseline():
    dataset = Dataset.from_json(_golden_path())
    judge = ReferenceEvalJudge(mode="fuzzy", fuzzy_threshold=0.7)
    runner = EvalRunner(judge=judge)

    perfect_target = CallableTarget(_perfect_answer, name="perfect")
    results = await runner.run(dataset, [perfect_target])

    summaries = results.get_summaries()
    summary = summaries["perfect"]
    assert summary.avg_score >= 0.85, f"Golden regression score too low: {summary.avg_score}"

    baseline = 0.8
    comparison = results.compare_to_baseline(baseline)
    assert comparison["perfect"]["passed"]
    assert comparison["perfect"]["delta"] >= 0.0


@pytest.mark.asyncio
async def test_degraded_target_fails_baseline():
    dataset = Dataset.from_json(_golden_path())
    judge = ReferenceEvalJudge(mode="fuzzy", fuzzy_threshold=0.7)
    runner = EvalRunner(judge=judge)

    bad_target = CallableTarget(_bad_answer, name="bad")
    results = await runner.run(dataset, [bad_target])

    summaries = results.get_summaries()
    assert summaries["bad"].avg_score < 0.6

    comparison = results.compare_to_baseline(0.6)
    assert not comparison["bad"]["passed"]


@pytest.mark.asyncio
async def test_llm_judge_scores_criteria_with_fake_model():
    class FakeJudgeClient:
        model = "fake-judge"

        class FakeMessage:
            content = ""

        async def create(self, messages, output_format=None, **kwargs):
            class Result:
                structured_output = JudgeResponse(
                    scores=[
                        CriterionScore(name="accuracy", score=8.0, reasoning="Looks correct."),
                        CriterionScore(name="citation_coverage", score=7.0, reasoning="Some sources."),
                        CriterionScore(name="hallucination", score=9.0, reasoning="No hallucination."),
                        CriterionScore(name="clarity", score=8.0, reasoning="Clear."),
                    ]
                )
                message = FakeJudgeClient.FakeMessage()
                model = "fake-judge"

            return Result()

    judge = LLMEvalJudge(client=FakeJudgeClient())
    task = Task(name="test", input="What is 2+2?", expected_output="4")
    trajectory = RunTrajectory(
        task=task,
        messages=[
            UserMessage(content=task.input, source="user"),
            AssistantMessage(content="4", source="assistant"),
        ],
    )
    score = await judge.score(trajectory)
    assert 0.7 < score.overall < 0.9
    assert "accuracy" in score.dimensions


def test_dataset_loads_golden_tasks():
    dataset = Dataset.from_json(_golden_path())
    assert len(dataset.tasks) == 3
    assert {t.category for t in dataset.tasks} == {"product", "methodology"}
    assert dataset.tasks[0].rubric is not None
