"""Base termination condition for agents and orchestrators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from ..messages import Message
from ..types import StopMessage

if TYPE_CHECKING:
    from ._composite import CompositeTermination


class BaseTermination(ABC):
    """Abstract base class for all termination conditions."""

    def __init__(self) -> None:
        self._met = False
        self._reason = ""
        self._metadata: Dict[str, Any] = {}

    @abstractmethod
    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        """Check termination on delta messages."""
        pass

    def is_met(self) -> bool:
        return self._met

    def reset(self) -> None:
        self._met = False
        self._reason = ""
        self._metadata = {}

    def get_reason(self) -> str:
        return self._reason

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    def _set_termination(
        self, reason: str, metadata: Optional[Dict[str, Any]] = None
    ) -> StopMessage:
        self._met = True
        self._reason = reason
        self._metadata = metadata or {}
        return StopMessage(content=reason, source=self.__class__.__name__, metadata=self._metadata)

    def __or__(self, other: "BaseTermination") -> "CompositeTermination":
        from ._composite import CompositeTermination

        return CompositeTermination([self, other], mode="any")

    def __and__(self, other: "BaseTermination") -> "CompositeTermination":
        from ._composite import CompositeTermination

        return CompositeTermination([self, other], mode="all")
