"""Lightweight eval harness for the Atlas research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from atlascore import OpenAIChatCompletionClient
from atlascore.eval import Dataset, EvalRunner, LLMEvalJudge, ResearchPipelineTarget
from atlascore.research import ResearchPipeline

from .models import EvalReport, EvalResult
from .models import EvalScore as BackendEvalScore


class DatasetItem(BaseModel):
    """A single legacy JSONL eval dataset item."""

    query: str = Field(...)
    expected: Optional[str] = Field(default=None)


@dataclass
class EvalConfig:
    """Dependencies for eval runs."""

    pipeline: ResearchPipeline
    judge_client: OpenAIChatCompletionClient


class EvalHarness:
    """Run the research pipeline over a dataset and score with an LLM-as-judge."""

    def __init__(self, config: EvalConfig, max_items: int = 5) -> None:
        self.config = config
        self.max_items = max_items

    async def run(
        self, dataset_path: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Yield progress events and a final `EvalReport`."""
        dataset = self._load_dataset(dataset_path)

        target = ResearchPipelineTarget(self.config.pipeline)
        judge = LLMEvalJudge(
            client=self.config.judge_client,
            default_criteria=["accuracy", "citation_coverage", "hallucination", "clarity"],
        )
        runner = EvalRunner(judge=judge)

        results = await runner.run(dataset, [target])

        eval_results: List[EvalResult] = []
        total_scores: Dict[str, float] = {
            "accuracy": 0.0,
            "citation_coverage": 0.0,
            "hallucination": 0.0,
            "overall": 0.0,
        }

        target_results = list(results.results.values())[0] if results.results else {}
        for idx, (task_id, task_result) in enumerate(target_results.items()):
            task = task_result.trajectory.task
            yield {
                "type": "eval_progress",
                "current": idx + 1,
                "total": len(target_results),
                "query": task.input,
            }

            score = self._convert_score(task_result.score)
            eval_results.append(
                EvalResult(
                    query=task.input,
                    expected=task.expected_output,
                    score=score,
                )
            )
            total_scores["accuracy"] += score.accuracy
            total_scores["citation_coverage"] += score.citation_coverage
            total_scores["hallucination"] += score.hallucination
            total_scores["overall"] += score.overall

        count = len(eval_results)
        aggregate = {k: (v / count if count else 0.0) for k, v in total_scores.items()}

        report = EvalReport(
            dataset_path=dataset_path,
            total=count,
            scores=eval_results,
            aggregate=aggregate,
        )
        yield {"type": "eval_complete", "report": report.model_dump()}

    def _load_dataset(self, dataset_path: str) -> Dataset:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        if path.suffix == ".jsonl":
            items: List[DatasetItem] = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items.append(DatasetItem.model_validate_json(line))
                    if len(items) >= self.max_items:
                        break
            name = path.stem
            tasks = [
                {
                    "id": f"{name}_{i}",
                    "name": f"{name}_{i}",
                    "input": item.query,
                    "expected_output": item.expected,
                    "eval_criteria": ["accuracy", "citation_coverage", "hallucination", "clarity"],
                }
                for i, item in enumerate(items)
            ]
            return Dataset.from_dict(
                {
                    "name": name,
                    "version": "1.0.0",
                    "description": f"Loaded from {dataset_path}",
                    "tasks": tasks,
                }
            )

        dataset = Dataset.from_json(path)
        if self.max_items:
            dataset.tasks = dataset.tasks[: self.max_items]
        return dataset

    def _convert_score(self, score: Any) -> BackendEvalScore:
        dims = getattr(score, "dimensions", {}) or {}
        reasoning = getattr(score, "reasoning", {}) or {}
        rationale = "\n".join(f"{k}: {v}" for k, v in reasoning.items()) or ""
        overall = float(getattr(score, "overall", 0.0) or 0.0)
        accuracy = float(dims.get("accuracy", overall))
        citation_coverage = float(dims.get("citation_coverage", dims.get("citation_match", overall)))
        hallucination = float(dims.get("hallucination", overall))
        return BackendEvalScore(
            accuracy=accuracy,
            citation_coverage=citation_coverage,
            hallucination=hallucination,
            overall=overall,
            rationale=rationale,
        )
