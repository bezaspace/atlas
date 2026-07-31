"""Maximum message termination condition."""

from typing import Optional, Sequence

from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class MaxMessageTermination(BaseTermination):
    """Terminates when the maximum number of messages is reached."""

    def __init__(self, max_messages: int):
        super().__init__()
        self.max_messages = max_messages
        self.message_count = 0

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        self.message_count += len(new_messages)
        if self.message_count >= self.max_messages:
            return self._set_termination(
                f"Maximum messages reached ({self.message_count}/{self.max_messages})",
                {"message_count": self.message_count, "max_messages": self.max_messages},
            )
        return None

    def reset(self) -> None:
        super().reset()
        self.message_count = 0
