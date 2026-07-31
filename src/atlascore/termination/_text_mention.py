"""Text mention termination condition."""

from typing import Optional, Sequence

from ..messages import Message
from ..types import StopMessage
from ._base import BaseTermination


class TextMentionTermination(BaseTermination):
    """Terminates when specific text is mentioned in a message."""

    def __init__(self, text: str, case_sensitive: bool = False):
        super().__init__()
        self.text = text
        self.case_sensitive = case_sensitive
        self.search_text = text if case_sensitive else text.lower()

    def check(self, new_messages: Sequence[Message]) -> Optional[StopMessage]:
        for message in new_messages:
            content = message.content if self.case_sensitive else message.content.lower()
            if self.search_text in content:
                return self._set_termination(
                    f"Text mention found: '{self.text}'",
                    {"text": self.text, "case_sensitive": self.case_sensitive},
                )
        return None

    def reset(self) -> None:
        super().reset()
