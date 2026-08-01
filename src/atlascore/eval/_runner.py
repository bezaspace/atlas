"""Evaluation runner."""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Sequence, Union

from ..base_types import Usage
from ..cancellation import CancellationToken
from ._base import BaseEvalJudge, EvalScore, RunTrajectory, Target
from ._dataset import Dataset
from ._results import EvalResults, TaskResult

Runnable = Union[Target, Any]


class EvalRunner:
    """Runs evaluation tasks against targets and scores with a judge."""

    def __init__(
        self,
        judge: BaseEvalJudge,
        parallel_tasks: bool = False,
        parallel_targets: bool = False,
    ):
        self.judge = judge
        self.parallel_tasks = parallel_tasks
        self.parallel_targets = parallel_targets

    async def run(
        self,
        dataset: Dataset,
        targets: Sequence[Runnable],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> EvalResults:
        """Run the full dataset against all targets and return results."""
        resolved_targets = [self._resolve_target(t) for t in targets]
        results = EvalResults(
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            metadata={"judge": self.judge.name, "parallel_tasks": self.parallel_tasks},
        )

        if self.parallel_targets:
            target_coros = [
                self._run_target(target, dataset, cancellation_token)
                for target in resolved_targets
            ]
            target_results = await asyncio.gather(*target_coros, return_exceptions=True)
            for target_result in target_results:
                if isinstance(target_result, Exception):
                    continue
                assert isinstance(target_result, list)
                for task_result in target_result:
                    results.add_result(task_result)
        else:
            for target in resolved_targets:
                if cancellation_token and cancellation_token.is_cancelled():
                    break
                task_results = await self._run_target(target, dataset, cancellation_token)
                for task_result in task_results:
                    results.add_result(task_result)

        return results

    @staticmethod
    def _resolve_target(item: Runnable) -> Target:
        if isinstance(item, Target):
            return item
        if callable(item):
            from ._targets import CallableTarget

            return CallableTarget(item)
        raise TypeError(f"Expected Target or callable, got {type(item).__name__}")

    async def _run_target(
        self,
        target: Target,
        dataset: Dataset,
        cancellation_token: Optional[CancellationToken],
    ) -> List[TaskResult]:
        if self.parallel_tasks:
            coroutines = [
                self._run_single_task(target, task, dataset, cancellation_token)
                for task in dataset.tasks
            ]
            raw_results = await asyncio.gather(*coroutines, return_exceptions=True)
            return [r for r in raw_results if isinstance(r, TaskResult)]

        results: List[TaskResult] = []
        for task in dataset.tasks:
            if cancellation_token and cancellation_token.is_cancelled():
                break
            result = await self._run_single_task(target, task, dataset, cancellation_token)
            results.append(result)
        return results

    async def _run_single_task(
        self,
        target: Target,
        task: Any,
        dataset: Dataset,
        cancellation_token: Optional[CancellationToken],
    ) -> TaskResult:
        try:
            trajectory = await target.run(task, cancellation_token)
            criteria = task.eval_criteria or dataset.default_eval_criteria
            score = await self._score(trajectory, criteria, cancellation_token)
        except Exception as e:
            failed_trajectory = RunTrajectory(
                task=task,
                messages=[],
                success=False,
                error=str(e),
                usage=Usage(duration_ms=0),
            )
            score = EvalScore(
                overall=0.0,
                dimensions={c: 0.0 for c in (task.eval_criteria or ["accuracy"])},
                reasoning={c: f"Execution failed: {e}" for c in (task.eval_criteria or ["accuracy"])},
                trajectory=failed_trajectory,
                metadata={"judge": self.judge.name, "error": str(e)},
            )
            trajectory = failed_trajectory

        return TaskResult(
            task_id=task.id or task.name,
            target_name=target.name,
            trajectory=trajectory,
            score=score,
        )

    async def _score(
        self,
        trajectory: RunTrajectory,
        criteria: List[str],
        cancellation_token: Optional[CancellationToken],
    ) -> EvalScore:
        try:
            score = await self.judge.score(
                trajectory,
                criteria=criteria,
                cancellation_token=cancellation_token,
            )
            score.trajectory = trajectory
            return score
        except Exception as e:
            return EvalScore(
                overall=0.0,
                dimensions={c: 0.0 for c in criteria},
                reasoning={c: f"Judge error: {e}" for c in criteria},
                trajectory=trajectory,
                metadata={"judge_error": str(e)},
            )
