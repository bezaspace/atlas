"""Core data types and models for atlascore."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from .base_types import ToolResult, Usage
from .context import AgentContext, ToolApprovalRequest
from .messages import AssistantMessage, Message


class AgentResponse(BaseModel):
    """Final result from agent.run()."""

    model_config = ConfigDict(frozen=False)

    context: Optional[AgentContext] = Field(default=None)
    source: str = Field(...)
    usage: Usage = Field(default_factory=lambda: Usage(duration_ms=0))
    timestamp: datetime = Field(default_factory=datetime.now)
    finish_reason: str = Field(...)

    @property
    def messages(self) -> List[Message]:
        return self.context.messages if self.context else []

    @property
    def needs_approval(self) -> bool:
        return self.context.waiting_for_approval if self.context else False

    @property
    def final_content(self) -> str:
        if self.messages:
            content = self.messages[-1].content
            return content[:50] + "..." if len(content) > 50 else content
        return "No messages"

    def __str__(self) -> str:
        messages_str = "\n".join(str(msg) for msg in self.messages)
        duration_s = self.usage.duration_ms / 1000
        tokens_in = (
            f"{self.usage.tokens_input / 1000:.1f}k"
            if self.usage.tokens_input >= 1000
            else str(self.usage.tokens_input)
        )
        tokens_out = (
            f"{self.usage.tokens_output / 1000:.1f}k"
            if self.usage.tokens_output >= 1000
            else str(self.usage.tokens_output)
        )
        cost_str = f", cost: ${self.usage.cost_estimate:.4f}" if self.usage.cost_estimate else ""
        usage_line = (
            f"[usage] duration: {duration_s:.1f}s, tokens: in:{tokens_in}, "
            f"out:{tokens_out}{cost_str} | finish: {self.finish_reason}"
        )
        return f"{messages_str}\n\n{usage_line}"


class ChatCompletionResult(BaseModel):
    """Standardized LLM response from model client."""

    model_config = ConfigDict(frozen=True)

    message: AssistantMessage = Field(...)
    usage: Usage = Field(...)
    model: str = Field(...)
    finish_reason: str = Field(...)
    structured_output: Optional[BaseModel] = Field(default=None)


class ChatCompletionChunk(BaseModel):
    """Streaming response chunk."""

    model_config = ConfigDict(frozen=True)

    content: str = Field(...)
    is_complete: bool = Field(...)
    tool_call_chunk: Optional[Dict[str, Any]] = Field(default=None)
    usage: Optional[Usage] = Field(default=None)


# Events
class BaseEvent(BaseModel):
    """Abstract base class for all agent events."""

    model_config = ConfigDict(frozen=True)

    timestamp: datetime = Field(default_factory=datetime.now)
    source: str = Field(...)
    event_type: str = Field(...)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{self.source}] {time_str} | {self.event_type}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(event_type='{self.event_type}', "
            f"source='{self.source}', timestamp='{self.timestamp}')"
        )


class TaskStartEvent(BaseEvent):
    event_type: str = Field(default="task_start")
    task: str = Field(...)


class TaskCompleteEvent(BaseEvent):
    event_type: str = Field(default="task_complete")
    result: str = Field(...)


class ModelCallEvent(BaseEvent):
    event_type: str = Field(default="model_call")
    input_messages: Sequence[Message] = Field(...)
    model: str = Field(...)


class ModelResponseEvent(BaseEvent):
    event_type: str = Field(default="model_response")
    response: str = Field(...)
    has_tool_calls: bool = Field(default=False)


class ModelStreamChunkEvent(BaseEvent):
    event_type: str = Field(default="model_stream_chunk")
    chunk: str = Field(...)
    is_final: bool = Field(default=False)


class ToolCallEvent(BaseEvent):
    event_type: str = Field(default="tool_call")
    tool_name: str = Field(...)
    parameters: Dict[str, Any] = Field(...)
    call_id: str = Field(...)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        params_str = ", ".join([f"{k}={v}" for k, v in self.parameters.items()])
        return f"[{self.source}] {time_str} | tool_call: {self.tool_name}({params_str})"


class ToolCallResponseEvent(BaseEvent):
    event_type: str = Field(default="tool_call_response")
    call_id: str = Field(...)
    result: Optional[ToolResult] = Field(default=None)

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        if self.result:
            status = "✓" if self.result.success else "✗"
            preview = str(self.result.result)[:50] + "..."
            return f"[{self.source}] {time_str} | tool_response: {status} {preview}"
        return f"[{self.source}] {time_str} | tool_response: (no result)"


class ToolApprovalEvent(BaseEvent):
    event_type: str = Field(default="tool_approval")
    approval_request: ToolApprovalRequest = Field(...)


class ToolValidationEvent(BaseEvent):
    event_type: str = Field(default="tool_validation")
    tool_name: str = Field(...)
    is_valid: bool = Field(...)
    errors: Optional[List[str]] = Field(default=None)


class MemoryUpdateEvent(BaseEvent):
    event_type: str = Field(default="memory_update")
    operation: str = Field(...)
    content_summary: str = Field(...)


class MemoryRetrievalEvent(BaseEvent):
    event_type: str = Field(default="memory_retrieval")
    query: str = Field(...)
    results_count: int = Field(...)


class ErrorEvent(BaseEvent):
    event_type: str = Field(default="error")
    error_message: str = Field(...)
    error_type: str = Field(...)
    is_recoverable: bool = Field(default=True)


AgentEvent = Union[
    TaskStartEvent,
    TaskCompleteEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ToolApprovalEvent,
    ToolValidationEvent,
    MemoryUpdateEvent,
    MemoryRetrievalEvent,
    ErrorEvent,
]


class StopMessage(BaseModel):
    content: str = Field(...)
    source: str = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class OrchestrationResponse(BaseModel):
    messages: Sequence[Message] = Field(...)
    final_result: str = Field(...)
    usage: Usage = Field(...)
    stop_message: StopMessage = Field(...)
    pattern_metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
