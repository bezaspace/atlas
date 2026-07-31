"""Base types shared across atlascore modules to avoid import cycles."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class Usage(BaseModel):
    """Structured execution statistics and resource consumption."""

    model_config = ConfigDict(frozen=True)

    duration_ms: int = Field(default=0, description="Total execution time in milliseconds")
    llm_calls: int = Field(default=0, description="Number of LLM API calls made")
    tokens_input: int = Field(default=0, description="Total input tokens consumed")
    tokens_output: int = Field(default=0, description="Total output tokens generated")
    tool_calls: int = Field(default=0, description="Number of tool executions")
    memory_operations: int = Field(default=0, description="Number of memory read/write operations")
    cost_estimate: Optional[float] = Field(default=None, description="Estimated cost in USD")

    def __add__(self, other: "Usage") -> "Usage":
        """Aggregate usage statistics from multiple sources."""
        return Usage(
            duration_ms=max(self.duration_ms, other.duration_ms),
            llm_calls=self.llm_calls + other.llm_calls,
            tokens_input=self.tokens_input + other.tokens_input,
            tokens_output=self.tokens_output + other.tokens_output,
            tool_calls=self.tool_calls + other.tool_calls,
            memory_operations=self.memory_operations + other.memory_operations,
            cost_estimate=(self.cost_estimate or 0) + (other.cost_estimate or 0) or None,
        )


class ToolResult(BaseModel):
    """Standardized tool execution result."""

    model_config = ConfigDict(frozen=True)

    success: bool = Field(...)
    result: Any = Field(...)
    error: Optional[str] = Field(default=None)
    metadata: Dict[str, Any] = Field(default_factory=dict)
