"""Logging middleware for atlascore."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Optional

from ._base import BaseMiddleware, MiddlewareContext


class LoggingMiddleware(BaseMiddleware):
    """Logs agent operations with timing."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Any, None]:
        self.logger.info(
            "[%s] Starting %s",
            context.agent_name,
            context.operation,
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "session_id": context.agent_context.session_id,
            },
        )
        context.metadata["start_time"] = time.time()
        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Any, None]:
        duration = time.time() - context.metadata.get("start_time", 0)
        self.logger.info(
            "[%s] Completed %s in %.2fs",
            context.agent_name,
            context.operation,
            duration,
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "duration": duration,
                "session_id": context.agent_context.session_id,
            },
        )
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Any, None]:
        self.logger.error(
            "[%s] Error in %s: %s",
            context.agent_name,
            context.operation,
            error,
            extra={
                "agent": context.agent_name,
                "operation": context.operation,
                "error_type": type(error).__name__,
                "session_id": context.agent_context.session_id,
            },
        )
        if False:
            yield
        raise error
