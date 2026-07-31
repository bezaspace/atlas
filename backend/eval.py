"""Lightweight eval harness for the Atlas research pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, Field

from atlascore import OpenAIChatCompletionClient
from atlascore.messages import UserMessage
from atlascore.research import ResearchPipeline
from atlascore.research_schemas import ResearchReport

from .models import EvalReport, EvalResult, EvalScore


class DatasetItem(BaseModel):
    """A single eval dataset item."""

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
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        items: List[DatasetItem] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(DatasetItem.model_validate_json(line))
                if len(items) >= self.max_items:
                    break

        results: List[EvalResult] = []
        total_accuracy = 0.0
        total_citation = 0.0
        total_hallucination = 0.0
        total_overall = 0.0

        for idx, item in enumerate(items):
            yield {
                "type": "eval_progress",
                "current": idx + 1,
                "total": len(items),
                "query": item.query,
            }

            try:
                report: ResearchReport = await self.config.pipeline.run(item.query)
                score = await self._judge(item, report)
            except Exception as e:
                score = EvalScore(
                    accuracy=0.0,
                    citation_coverage=0.0,
                    hallucination=0.0,
                    overall=0.0,
                    rationale=f"Error: {e}",
                )

            results.append(
                EvalResult(
                    query=item.query,
                    expected=item.expected,
                    score=score,
                )
            )
            total_accuracy += score.accuracy
            total_citation += score.citation_coverage
            total_hallucination += score.hallucination
            total_overall += score.overall

        aggregate = {
            "accuracy": total_accuracy / len(items) if items else 0.0,
            "citation_coverage": total_citation / len(items) if items else 0.0,
            "hallucination": total_hallucination / len(items) if items else 0.0,
            "overall": total_overall / len(items) if items else 0.0,
        }

        eval_report = EvalReport(
            dataset_path=dataset_path,
            total=len(items),
            scores=results,
            aggregate=aggregate,
        )
        yield {"type": "eval_complete", "report": eval_report.model_dump()}

    async def _judge(self, item: DatasetItem, report: ResearchReport) -> EvalScore:
        prompt = (
            f"Evaluate the following research brief for the query: {item.query}\n\n"
            f"Expected: {item.expected or 'No reference provided'}\n\n"
            f"Brief:\n{report.brief.to_markdown()}\n\n"
            "Return JSON with fields: accuracy (0-1), citation_coverage (0-1), "
            "hallucination (0-1, higher means fewer hallucinations), overall (0-1), rationale."
        )
        result = await self.config.judge_client.create(
            messages=[UserMessage(content=prompt, source="user")],
            output_format=EvalScore,
        )
        if result.structured_output:
            return result.structured_output  # type: ignore[return-value]
        content = result.message.content or "{}"
        return EvalScore.model_validate_json(content)
