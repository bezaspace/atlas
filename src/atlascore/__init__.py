"""Atlascore: from-scratch multi-agent core for Atlas."""

from .agents._agent import Agent
from .base_types import ToolResult, Usage
from .cancellation import CancellationToken
from .context import AgentContext, ToolApprovalRequest, ToolApprovalResponse
from .llm._base import (
    AuthenticationError,
    BaseChatCompletionClient,
    BaseChatCompletionError,
    InvalidRequestError,
    RateLimitError,
)
from .llm._openai import OpenAIChatCompletionClient
from .messages import (
    AssistantMessage,
    BaseMessage,
    Message,
    MultiModalMessage,
    SystemMessage,
    ToolCallRequest,
    ToolMessage,
    UserMessage,
)
from .tools._base import ApprovalMode, BaseTool, FunctionTool
from .tools._core_tools import (
    CalculatorTool,
    DateTimeTool,
    JSONParserTool,
    RegexTool,
    TaskStatusTool,
    ThinkTool,
    create_core_tools,
)
from .types import (
    AgentEvent,
    AgentResponse,
    ChatCompletionChunk,
    ChatCompletionResult,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    OrchestrationResponse,
    StopMessage,
    TaskCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Usage",
    "ToolResult",
    "CancellationToken",
    "AgentContext",
    "ToolApprovalRequest",
    "ToolApprovalResponse",
    "BaseChatCompletionClient",
    "BaseChatCompletionError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "OpenAIChatCompletionClient",
    "BaseMessage",
    "Message",
    "SystemMessage",
    "UserMessage",
    "AssistantMessage",
    "ToolMessage",
    "MultiModalMessage",
    "ToolCallRequest",
    "BaseTool",
    "FunctionTool",
    "ApprovalMode",
    "ThinkTool",
    "CalculatorTool",
    "DateTimeTool",
    "JSONParserTool",
    "RegexTool",
    "TaskStatusTool",
    "create_core_tools",
    "AgentResponse",
    "AgentEvent",
    "ChatCompletionResult",
    "ChatCompletionChunk",
    "TaskStartEvent",
    "TaskCompleteEvent",
    "ModelCallEvent",
    "ModelResponseEvent",
    "ToolCallEvent",
    "ToolCallResponseEvent",
    "ErrorEvent",
    "OrchestrationResponse",
    "StopMessage",
]
