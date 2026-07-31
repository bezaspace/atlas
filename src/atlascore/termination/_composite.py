"""Composite termination condition."""

from typing import List, Optional, Sequence

from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class CompositeTermination(BaseTermination):
    """Combines multiple termination conditions with logical operators."""

    def __init__(self, conditions: List[BaseTermination], mode: str = "any"):
        super().__init__()
        if mode not in ("any", "all"):
            raise ValueError("Mode must be 'any' or 'all'")
        self.conditions = conditions
        self.mode = mode

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        results: List[StopMessage] = []
        for condition in self.conditions:
            result = condition.check(new_messages)
            if result:
                results.append(result)

        if self.mode == "any" and results:
            return self._set_termination(
                f"Composite (any): {results[0].content}",
                {"mode": "any", "triggered_conditions": [r.source for r in results]},
            )
        if self.mode == "all" and len(results) == len(self.conditions):
            return self._set_termination(
                f"Composite (all): {'; '.join(r.content for r in results)}",
                {"mode": "all", "triggered_conditions": [r.source for r in results]},
            )
        return None

    def reset(self) -> None:
        super().reset()
        for condition in self.conditions:
            condition.reset()

    def is_met(self) -> bool:
        met_conditions = [c.is_met() for c in self.conditions]
        if self.mode == "any":
            return any(met_conditions)
        return all(met_conditions)

    def __or__(self, other: BaseTermination) -> "CompositeTermination":
        if isinstance(other, CompositeTermination) and other.mode == "any":
            return CompositeTermination(self.conditions + other.conditions, mode="any")
        return CompositeTermination(self.conditions + [other], mode="any")

    def __and__(self, other: BaseTermination) -> "CompositeTermination":
        if isinstance(other, CompositeTermination) and other.mode == "all":
            return CompositeTermination(self.conditions + other.conditions, mode="all")
        return CompositeTermination(self.conditions + [other], mode="all")
