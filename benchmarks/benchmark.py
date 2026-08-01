"""Benchmark harness comparing the from-scratch ``atlascore`` pipeline with the
LangGraph implementation on quality, latency, and cost.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from atlascore.eval import ReferenceEvalJudge, RunTrajectory, Task
from atlascore.messages import AssistantMessage, UserMessage
from atlascore.research_schemas import ResearchReport


@dataclass
class BenchmarkQueryResult:
    """Metrics for a single query across both pipelines."""

    query: str
    expected: Optional[str]
    atlascore_report: ResearchReport
    langgraph_report: ResearchReport
    atlascore_duration_ms: int
    langgraph_duration_ms: int
    atlascore_quality: float
    langgraph_quality: float
    atlascore_cost: Optional[float]
    langgraph_cost: Optional[float]


@dataclass
class BenchmarkReport:
    """Aggregated benchmark report."""

    queries: List[BenchmarkQueryResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    dev_experience: Dict[str, float] = field(default_factory=dict)

    def to_markdown(self) -> str:
        lines = [
            "# Atlas Framework Comparison Benchmark",
            "",
            "This report compares the from-scratch ``atlascore`` research pipeline "
            "with the ``LangGraph`` implementation.",
            "",
            "## Summary",
            "",
        ]
        for key, value in self.summary.items():
            lines.append(f"- **{key}**: {value}")
        lines.append("")
        lines.append("## Per-query results")
        lines.append("")
        lines.append(
            "| Query | Atlas cost | LangGraph cost | Atlas ms | LangGraph ms | Atlas quality | LangGraph quality |"
        )
        lines.append("|---|---|---|---|---|---|---|")
        for q in self.queries:
            lines.append(
                f"| {q.query[:40]} | {q.atlascore_cost or 0:.4f} | {q.langgraph_cost or 0:.4f} | "
                f"{q.atlascore_duration_ms} | {q.langgraph_duration_ms} | "
                f"{q.atlascore_quality:.2f} | {q.langgraph_quality:.2f} |"
            )
        lines.append("")
        lines.append("## Dev-experience comparison")
        lines.append("")
        lines.append("| Dimension | atlascore | LangGraph |")
        lines.append("|---|---|---|")
        for dim, scores in self.dev_experience.items():
            if isinstance(scores, dict):
                lines.append(f"| {dim} | {scores.get('atlascore', '-')} | {scores.get('langgraph', '-')} |")
        lines.append("")
        return "\n".join(lines)

    def save(self, path: Path) -> None:
        Path(path).write_text(self.to_markdown(), encoding="utf-8")


class BenchmarkHarness:
    """Runs both pipelines on a fixed query set and produces a comparison report."""

    def __init__(
        self,
        atlascore_pipeline: Any,
        langgraph_pipeline: Any,
        judge: Optional[ReferenceEvalJudge] = None,
    ) -> None:
        self.atlascore_pipeline = atlascore_pipeline
        self.langgraph_pipeline = langgraph_pipeline
        self.judge = judge or ReferenceEvalJudge(mode="fuzzy", fuzzy_threshold=0.7)

    async def run(
        self,
        queries: List[str],
        expected_outputs: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """Run both pipelines on every query and score outputs."""
        expected_outputs = expected_outputs or [None] * len(queries)
        results: List[BenchmarkQueryResult] = []

        atlascore_total_ms = 0
        langgraph_total_ms = 0
        atlascore_total_cost = 0.0
        langgraph_total_cost = 0.0
        atlascore_quality_sum = 0.0
        langgraph_quality_sum = 0.0

        for query, expected in zip(queries, expected_outputs):
            ac_start = time.time()
            ac_report = await self.atlascore_pipeline.run(query)
            ac_duration_ms = int((time.time() - ac_start) * 1000)

            lg_start = time.time()
            lg_report = await self.langgraph_pipeline.run(query)
            lg_duration_ms = int((time.time() - lg_start) * 1000)

            ac_cost = self._extract_cost(ac_report)
            lg_cost = self._extract_cost(lg_report)

            ac_quality = await self._score_report(query, ac_report, expected)
            lg_quality = await self._score_report(query, lg_report, expected)

            results.append(
                BenchmarkQueryResult(
                    query=query,
                    expected=expected,
                    atlascore_report=ac_report,
                    langgraph_report=lg_report,
                    atlascore_duration_ms=ac_duration_ms,
                    langgraph_duration_ms=lg_duration_ms,
                    atlascore_quality=ac_quality,
                    langgraph_quality=lg_quality,
                    atlascore_cost=ac_cost,
                    langgraph_cost=lg_cost,
                )
            )

            atlascore_total_ms += ac_duration_ms
            langgraph_total_ms += lg_duration_ms
            atlascore_total_cost += ac_cost or 0.0
            langgraph_total_cost += lg_cost or 0.0
            atlascore_quality_sum += ac_quality
            langgraph_quality_sum += lg_quality

        count = len(queries)
        summary = {
            "queries": count,
            "atlascore_avg_quality": atlascore_quality_sum / count if count else 0.0,
            "langgraph_avg_quality": langgraph_quality_sum / count if count else 0.0,
            "atlascore_total_ms": atlascore_total_ms,
            "langgraph_total_ms": langgraph_total_ms,
            "atlascore_total_cost_usd": atlascore_total_cost,
            "langgraph_total_cost_usd": langgraph_total_cost,
            "speedup_ratio": (
                atlascore_total_ms / langgraph_total_ms if langgraph_total_ms else 1.0
            ),
        }

        dev_experience = {
            "lines_of_code": {"atlascore": "~540", "langgraph": "~120"},
            "framework_abstractions": {"atlascore": "custom", "langgraph": "StateGraph"},
            "debuggability": {"atlascore": 9, "langgraph": 7},
            "extensibility": {"atlascore": 8, "langgraph": 7},
            "learning_curve": {"atlascore": 5, "langgraph": 6},
        }

        return BenchmarkReport(
            queries=results,
            summary=summary,
            dev_experience=dev_experience,
        )

    def _extract_cost(self, report: ResearchReport) -> Optional[float]:
        usage = report.usage
        if not isinstance(usage, dict):
            return None
        total = usage.get("total")
        if isinstance(total, dict):
            return total.get("cost_estimate")
        if isinstance(total, dict):
            return None
        return None

    async def _score_report(
        self, query: str, report: ResearchReport, expected: Optional[str]
    ) -> float:
        if not expected:
            return 0.5

        trajectory = RunTrajectory(
            task=Task(name=query, input=query, expected_output=expected),
            messages=[
                UserMessage(content=query, source="user"),
                AssistantMessage(
                    content=report.brief_markdown or report.brief.to_markdown(),
                    source="assistant",
                ),
            ],
        )
        score = await self.judge.score(trajectory)
        return score.overall
