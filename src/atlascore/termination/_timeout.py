"""Timeout termination condition."""

import time
from typing import Optional, Sequence, Union

from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class TimeoutTermination(BaseTermination):
    """Terminates when a time limit is exceeded."""

    def __init__(self, max_duration_seconds: Union[int, float]):
        super().__init__()
        self.max_duration_seconds = max_duration_seconds
        self.start_time: Optional[float] = time.time()

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        if self.start_time is None:
            self.start_time = time.time()
            return None

        elapsed = time.time() - self.start_time
        if elapsed >= self.max_duration_seconds:
            return self._set_termination(
                f"Timeout reached ({elapsed:.1f}s/{self.max_duration_seconds}s)",
                {"elapsed_seconds": elapsed, "max_duration_seconds": self.max_duration_seconds},
            )
        return None

    def reset(self) -> None:
        super().reset()
        self.start_time = time.time()
