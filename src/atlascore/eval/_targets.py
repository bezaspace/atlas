"""Evaluation targets for Atlas."""

from __future__ import annotations

import time
from typing import Any, Optional

from ..messages import AssistantMessage, UserMessage
from ..research import ResearchPipeline
from ..research_schemas import ResearchReport
from ._base import RunTrajectory, Target, Task


class ResearchPipelineTarget(Target):
    """Target that runs the Atlas research pipeline for an eval task."""

    def __init__(self, pipeline: ResearchPipeline, name: str = "research_pipeline"):
        super().__init__(name)
        self.pipeline = pipeline

    async def run(
        self, task: Task, cancellation_token: Optional[Any] = None
    ) -> RunTrajectory:
        start_time = time.time()
        try:
            report: ResearchReport = await self.pipeline.run(task.input)
            end_time = time.time()

            usage = self._usage_from_report(report)
            brief_text = report.brief_markdown or report.brief.to_markdown()
            messages = [
                UserMessage(content=task.input, source="user"),
                AssistantMessage(content=brief_text, source="assistant"),
            ]

            return RunTrajectory(
                task=task,
                messages=messages,
                success=True,
                error=None,
                usage=usage,
                metadata={
                    "report": report.model_dump(),
                    "sources": [s.url for s in report.sources],
                    "duration_ms": int((end_time - start_time) * 1000),
                },
            )
        except Exception as e:
            end_time = time.time()
            return RunTrajectory(
                task=task,
                messages=[UserMessage(content=task.input, source="user")],
                success=False,
                error=str(e),
                usage=None,
                metadata={"duration_ms": int((end_time - start_time) * 1000)},
            )

    @staticmethod
    def _usage_from_report(report: ResearchReport) -> Any:
        from ..base_types import Usage

        usage_data = report.usage.get("total") if isinstance(report.usage, dict) else report.usage
        if not isinstance(usage_data, dict):
            return Usage(duration_ms=0)

        return Usage(
            duration_ms=usage_data.get("duration_ms", 0),
            llm_calls=usage_data.get("llm_calls", 0),
            tokens_input=usage_data.get("tokens_input", 0),
            tokens_output=usage_data.get("tokens_output", 0),
            tool_calls=usage_data.get("tool_calls", 0),
            cost_estimate=usage_data.get("cost_estimate"),
        )


class CallableTarget(Target):
    """Target wrapping any async callable that returns a string."""

    def __init__(self, func: Any, name: str = "callable"):
        super().__init__(name)
        self.func = func

    async def run(
        self, task: Task, cancellation_token: Optional[Any] = None
    ) -> RunTrajectory:
        start_time = time.time()
        try:
            output = await self.func(task)
            end_time = time.time()
            return RunTrajectory(
                task=task,
                messages=[
                    UserMessage(content=task.input, source="user"),
                    AssistantMessage(content=str(output), source="assistant"),
                ],
                success=True,
                error=None,
                usage=None,
                metadata={"duration_ms": int((end_time - start_time) * 1000)},
            )
        except Exception as e:
            end_time = time.time()
            return RunTrajectory(
                task=task,
                messages=[UserMessage(content=task.input, source="user")],
                success=False,
                error=str(e),
                usage=None,
                metadata={"duration_ms": int((end_time - start_time) * 1000)},
            )
