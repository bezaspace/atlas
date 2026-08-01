"""Evaluation harness for Atlas research tasks."""

from ._base import BaseEvalJudge, EvalScore, RunTrajectory, Target, Task
from ._dataset import Dataset
from ._results import EvalResults, TaskResult
from ._runner import EvalRunner
from ._targets import CallableTarget, ResearchPipelineTarget
from .judges._llm import CriterionScore, JudgeResponse, LLMEvalJudge
from .judges._reference import ReferenceEvalJudge

__all__ = [
    "BaseEvalJudge",
    "CallableTarget",
    "CriterionScore",
    "Dataset",
    "EvalResults",
    "EvalRunner",
    "EvalScore",
    "JudgeResponse",
    "LLMEvalJudge",
    "ReferenceEvalJudge",
    "ResearchPipelineTarget",
    "RunTrajectory",
    "Target",
    "Task",
    "TaskResult",
]
