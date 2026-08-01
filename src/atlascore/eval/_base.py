"""Base types for the evaluation harness."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional

from ..base_types import Usage
from ..messages import AssistantMessage, Message


@dataclass
class Task:
    """A single evaluation task."""

    name: str
    input: str
    id: Optional[str] = None
    category: str = "general"
    eval_criteria: List[str] = field(
        default_factory=lambda: ["accuracy", "citation_coverage", "hallucination", "clarity"]
    )
    expected_output: Optional[str] = None
    rubric: Optional[Dict[str, str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.id is None:
            self.id = self.name


@dataclass
class RunTrajectory:
    """Execution trace for a single task."""

    task: Task
    messages: List[Message] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None
    usage: Optional[Usage] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalScore:
    """Score returned by an eval judge."""

    overall: float
    dimensions: Dict[str, float] = field(default_factory=dict)
    reasoning: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[RunTrajectory] = None

    def __post_init__(self) -> None:
        self.overall = max(0.0, min(1.0, float(self.overall)))
        self.dimensions = {k: max(0.0, min(1.0, float(v))) for k, v in self.dimensions.items()}


class Target(ABC):
    """Abstract base for something that can run an eval task."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def run(
        self, task: Task, cancellation_token: Optional[Any] = None
    ) -> RunTrajectory:
        """Execute the task and return a trajectory."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class BaseEvalJudge(ABC):
    """Abstract base for eval judges with answer extraction helpers."""

    def __init__(self, name: str, answer_strategy: str = "last_non_empty"):
        self.name = name
        self.answer_strategy = answer_strategy

    def extract_answer(self, trajectory: RunTrajectory) -> str:
        """Extract the agent's final answer from a trajectory."""
        if not trajectory.messages:
            return ""

        if self.answer_strategy == "last_non_empty":
            for msg in reversed(trajectory.messages):
                content = getattr(msg, "content", "") or ""
                if content.strip():
                    return content.strip()
            return ""

        if self.answer_strategy == "last_assistant":
            for msg in reversed(trajectory.messages):
                if isinstance(msg, AssistantMessage):
                    return (msg.content or "").strip()
            return ""

        if self.answer_strategy == "all_assistant":
            parts = []
            for msg in trajectory.messages:
                if isinstance(msg, AssistantMessage):
                    content = (msg.content or "").strip()
                    if content:
                        parts.append(content)
            return "\n".join(parts)

        raise ValueError(f"Unknown answer_strategy: {self.answer_strategy}")

    @abstractmethod
    async def score(
        self,
        trajectory: RunTrajectory,
        criteria: Optional[List[str]] = None,
        cancellation_token: Optional[Any] = None,
    ) -> EvalScore:
        """Score a trajectory."""
        pass


def _fuzzy_ratio(a: str, b: str) -> float:
    """Return normalized fuzzy similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _extract_urls(text: str) -> set[str]:
    """Naive URL extraction for citation overlap."""
    import re

    return set(re.findall(r"https?://[^\s\)\]\>\"]+", text))
