"""Evaluation results and aggregation."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ._base import EvalScore, RunTrajectory


@dataclass
class TaskResult:
    """Result of one task/target pair."""

    task_id: str
    target_name: str
    trajectory: RunTrajectory
    score: EvalScore
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        if self.trajectory.usage:
            return self.trajectory.usage.tokens_input + self.trajectory.usage.tokens_output
        return 0

    @property
    def duration_ms(self) -> int:
        return self.trajectory.usage.duration_ms if self.trajectory.usage else 0

    @property
    def success(self) -> bool:
        return self.trajectory.success

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target_name": self.target_name,
            "score": {
                "overall": self.score.overall,
                "dimensions": self.score.dimensions,
                "reasoning": self.score.reasoning,
            },
            "total_tokens": self.total_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.trajectory.error,
            "metrics": self.metrics,
        }


@dataclass
class TargetSummary:
    """Aggregated stats for one target across all tasks."""

    target_name: str
    task_count: int = 0
    avg_score: float = 0.0
    min_score: float = 0.0
    max_score: float = 0.0
    total_tokens: int = 0
    total_duration_ms: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_name": self.target_name,
            "task_count": self.task_count,
            "avg_score": self.avg_score,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "success_rate": self.success_rate,
        }


@dataclass
class EvalResults:
    """Complete results from an evaluation run."""

    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    dataset_name: str = ""
    dataset_version: str = ""
    target_names: List[str] = field(default_factory=list)
    task_ids: List[str] = field(default_factory=list)
    results: Dict[str, Dict[str, TaskResult]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _summaries: Optional[Dict[str, TargetSummary]] = field(default=None, repr=False)

    def add_result(self, result: TaskResult) -> None:
        if result.target_name not in self.results:
            self.results[result.target_name] = {}
            if result.target_name not in self.target_names:
                self.target_names.append(result.target_name)

        self.results[result.target_name][result.task_id] = result

        if result.task_id not in self.task_ids:
            self.task_ids.append(result.task_id)

        self._summaries = None

    def get_summaries(self) -> Dict[str, TargetSummary]:
        if self._summaries is not None:
            return self._summaries

        summaries = {}
        for target_name in self.target_names:
            target_results = list(self.results.get(target_name, {}).values())
            if not target_results:
                continue

            scores = [r.score.overall for r in target_results]
            successes = [1 if r.success else 0 for r in target_results]
            total_tokens = sum(r.total_tokens for r in target_results)
            total_duration = sum(r.duration_ms for r in target_results)

            summaries[target_name] = TargetSummary(
                target_name=target_name,
                task_count=len(target_results),
                avg_score=sum(scores) / len(scores) if scores else 0.0,
                min_score=min(scores) if scores else 0.0,
                max_score=max(scores) if scores else 0.0,
                total_tokens=total_tokens,
                total_duration_ms=total_duration,
                success_rate=sum(successes) / len(successes) if successes else 0.0,
            )

        self._summaries = summaries
        return summaries

    def compare_to_baseline(self, baseline: float) -> Dict[str, Any]:
        """Check whether each target's average score meets the baseline."""
        summaries = self.get_summaries()
        comparison = {}
        for target_name, summary in summaries.items():
            passed = summary.avg_score >= baseline
            comparison[target_name] = {
                "avg_score": summary.avg_score,
                "baseline": baseline,
                "passed": passed,
                "delta": summary.avg_score - baseline,
            }
        return comparison

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp.isoformat(),
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "target_names": self.target_names,
            "task_ids": self.task_ids,
            "results": {
                target: {task_id: r.to_dict() for task_id, r in tasks.items()}
                for target, tasks in self.results.items()
            },
            "summaries": {name: s.to_dict() for name, s in self.get_summaries().items()},
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, path: Optional[Path] = None) -> Path:
        if path is None:
            output_dir = Path.cwd() / ".atlas" / "eval"
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
            path = output_dir / f"eval_{self.run_id}_{timestamp_str}.json"

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        return path
