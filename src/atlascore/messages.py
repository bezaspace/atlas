"""Core message types for agent communication."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .base_types import Usage


class BaseMessage(BaseModel):
    """Base class for all message types."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(..., description="The message content")
    source: str = Field(..., description="Source of the message")
    timestamp: datetime = Field(default_factory=datetime.now)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | {self.content}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(source='{self.source}', "
            f"content='{self.content[:50]}...', timestamp='{self.timestamp}')"
        )


class SystemMessage(BaseMessage):
    """System message with instructions/role definition."""

    role: Literal["system"] = Field(default="system")


class UserMessage(BaseMessage):
    """User message containing input."""

    role: Literal["user"] = Field(default="user")
    name: Optional[str] = Field(default=None)


class ToolCallRequest(BaseModel):
    """Structured representation of an LLM tool call request."""

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(..., description="Name of the tool to call")
    parameters: Dict[str, Any] = Field(..., description="Arguments for the tool")
    call_id: str = Field(..., description="Unique identifier for this call")


class AssistantMessage(BaseMessage):
    """Assistant message from the agent/LLM."""

    role: Literal["assistant"] = Field(default="assistant")
    tool_calls: Optional[List[ToolCallRequest]] = Field(default=None)
    structured_content: Optional[BaseModel] = Field(default=None)
    usage: Optional[Usage] = Field(default=None)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        if self.tool_calls:
            tool_info = ", ".join(
                [
                    f"{tc.tool_name}({', '.join(f'{k}={v}' for k, v in tc.parameters.items())})"
                    for tc in self.tool_calls
                ]
            )
            if self.content and self.content.strip():
                return f"[{self.source}] {time_str} | {self.content} [tools: {tool_info}]"
            return f"[{self.source}] {time_str} | [calling tools: {tool_info}]"
        return f"[{self.source}] {time_str} | {self.content}"


class ToolMessage(BaseMessage):
    """Tool message containing execution result."""

    role: Literal["tool"] = Field(default="tool")
    tool_call_id: str = Field(...)
    tool_name: str = Field(...)
    success: bool = Field(...)
    error: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MultiModalMessage(BaseMessage):
    """Message supporting multiple content types (text, images)."""

    role: Literal["user", "assistant"] = Field(...)
    mime_type: str = Field(...)
    data: Optional[Union[bytes, str]] = Field(default=None)
    media_url: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_media_data(self):
        if self.data is None and self.media_url is None:
            raise ValueError("Either 'data' or 'media_url' must be provided")
        if self.data is not None and self.media_url is not None:
            raise ValueError("Only one of 'data' or 'media_url' should be provided")
        return self

    def is_text(self) -> bool:
        return self.mime_type.startswith("text/")

    def is_image(self) -> bool:
        return self.mime_type.startswith("image/")

    def to_base64(self) -> Optional[str]:
        if self.data is None:
            return None
        if isinstance(self.data, str):
            return self.data
        import base64

        return base64.b64encode(self.data).decode("utf-8")


Message = Union[SystemMessage, UserMessage, AssistantMessage, ToolMessage, MultiModalMessage]
