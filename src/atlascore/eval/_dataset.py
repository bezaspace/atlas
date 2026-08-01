"""Dataset definitions for evaluation runs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

from ._base import Task


@dataclass
class Dataset:
    """A collection of eval tasks."""

    name: str
    version: str = "1.0.0"
    description: str = ""
    tasks: List[Task] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    default_eval_criteria: List[str] = field(
        default_factory=lambda: ["accuracy", "citation_coverage", "hallucination", "clarity"]
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.categories:
            self.categories = list({t.category for t in self.tasks})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Dataset":
        tasks = [
            Task(
                name=t.get("name", ""),
                input=t.get("input", ""),
                id=t.get("id"),
                category=t.get("category", "general"),
                eval_criteria=t.get("eval_criteria", data.get("default_eval_criteria", [])),
                expected_output=t.get("expected_output"),
                rubric=t.get("rubric"),
                metadata=t.get("metadata", {}),
            )
            for t in data.get("tasks", [])
        ]
        return cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            tasks=tasks,
            categories=data.get("categories", []),
            default_eval_criteria=data.get("default_eval_criteria", ["accuracy"]),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: Path) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "categories": self.categories,
            "default_eval_criteria": self.default_eval_criteria,
            "tasks": [
                {
                    "id": t.id,
                    "name": t.name,
                    "input": t.input,
                    "category": t.category,
                    "eval_criteria": t.eval_criteria,
                    "expected_output": t.expected_output,
                    "rubric": t.rubric,
                    "metadata": t.metadata,
                }
                for t in self.tasks
            ],
            "metadata": self.metadata,
        }

    def to_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def filter(self, predicate: Callable[[Task], bool]) -> "Dataset":
        filtered = [t for t in self.tasks if predicate(t)]
        return Dataset(
            name=f"{self.name}_filtered",
            version=self.version,
            tasks=filtered,
            default_eval_criteria=self.default_eval_criteria,
            metadata={"filtered_from": self.name},
        )

    def __len__(self) -> int:
        return len(self.tasks)
