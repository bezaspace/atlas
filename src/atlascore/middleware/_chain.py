"""Middleware chain execution."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional

from ..context import AgentContext
from ..types import (
    ErrorEvent,
    MemoryRetrievalEvent,
    MemoryUpdateEvent,
    ModelCallEvent,
    ModelResponseEvent,
    ModelStreamChunkEvent,
    TaskCompleteEvent,
    TaskStartEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
    ToolValidationEvent,
)
from ._base import BaseMiddleware, MiddlewareContext

_EVENT_TYPES = (
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
)


class MiddlewareChain:
    """Executes a chain of middleware as an async generator pipeline."""

    def __init__(self, middlewares: Optional[List[BaseMiddleware]] = None):
        self.middlewares = middlewares or []

    def add(self, middleware: BaseMiddleware) -> None:
        self.middlewares.append(middleware)

    def remove(self, middleware: BaseMiddleware) -> None:
        if middleware in self.middlewares:
            self.middlewares.remove(middleware)

    async def execute_stream(
        self,
        operation: str,
        agent_name: str,
        agent_context: AgentContext,
        data: Any,
        func: Callable[[Any], Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[Any, None]:
        """Execute middleware chain around an operation."""
        ctx = MiddlewareContext(
            operation=operation,
            agent_name=agent_name,
            agent_context=agent_context,
            data=data,
            metadata=metadata or {},
        )

        # Pre-process
        for middleware in self.middlewares:
            final_ctx = None
            try:
                async for item in middleware.process_request(ctx):
                    if isinstance(item, MiddlewareContext):
                        final_ctx = item
                    elif isinstance(item, _EVENT_TYPES):
                        yield item
                        if isinstance(item, ToolApprovalEvent):
                            return
            except Exception as e:
                recovered = False
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            if isinstance(item, _EVENT_TYPES):
                                yield item
                            else:
                                yield item
                                recovered = True
                                return
                    except Exception:
                        continue
                if not recovered:
                    raise
            if final_ctx is None:
                return
            ctx = final_ctx

        # Execute operation
        result: Any
        try:
            coro = func(ctx.data)
            if asyncio.iscoroutine(coro):
                result = await coro
            else:
                result = coro
        except Exception as e:
            recovered = False
            for middleware in reversed(self.middlewares):
                try:
                    async for item in middleware.process_error(ctx, e):
                        if isinstance(item, _EVENT_TYPES):
                            yield item
                        else:
                            yield item
                            recovered = True
                            return
                except Exception:
                    continue
            if not recovered:
                raise
            raise RuntimeError("Middleware recovery logic error")  # pragma: no cover

        # Post-process
        for middleware in reversed(self.middlewares):
            final_result = None
            try:
                async for item in middleware.process_response(ctx, result):
                    if isinstance(item, _EVENT_TYPES):
                        yield item
                    else:
                        final_result = item
            except Exception as e:
                for error_mw in reversed(self.middlewares):
                    try:
                        async for item in error_mw.process_error(ctx, e):
                            if isinstance(item, _EVENT_TYPES):
                                yield item
                            else:
                                yield item
                                return
                    except Exception:
                        continue
                raise
            result = final_result if final_result is not None else result

        yield result
