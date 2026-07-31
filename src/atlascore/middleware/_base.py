"""Middleware base classes for atlascore."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict

from pydantic import BaseModel, Field

from ..context import AgentContext


class MiddlewareContext(BaseModel):
    """Context passed through middleware chain."""

    model_config = {"frozen": False}

    operation: str = Field(...)
    agent_name: str = Field(...)
    agent_context: AgentContext = Field(...)
    data: Any = Field(...)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseMiddleware(ABC):
    """Abstract base for middleware intercepting agent operations."""

    @abstractmethod
    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Any, None]:
        """Process before operation; final yield is usually the context."""
        yield context

    @abstractmethod
    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Any, None]:
        """Process after operation; final yield is usually the result."""
        yield result

    @abstractmethod
    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Any, None]:
        """Process errors; re-raise by default."""
        if False:
            yield
        raise error
