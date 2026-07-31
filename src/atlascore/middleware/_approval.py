"""Approval middleware for tool calls."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, List, Optional

from ..tools._base import BaseTool
from ..types import ToolApprovalEvent
from ._base import BaseMiddleware, MiddlewareContext


class ApprovalMiddleware(BaseMiddleware):
    """Pauses execution when a tool requires human approval."""

    def __init__(self, tools: Optional[List[BaseTool]] = None):
        self.tools = tools or []
        self._tool_map = {tool.name: tool for tool in self.tools}

    async def process_request(
        self, context: MiddlewareContext
    ) -> AsyncGenerator[Any, None]:
        if context.operation != "tool_call":
            yield context
            return

        tool_call = context.data
        tool = self._tool_map.get(tool_call.tool_name)
        if tool is None or tool.approval_mode.value != "always_require":
            yield context
            return

        approval = context.agent_context.get_approval_response(tool_call.call_id)
        if approval is None:
            request = context.agent_context.add_approval_request(tool_call, tool_call.tool_name)
            yield ToolApprovalEvent(source=context.agent_name, approval_request=request)
            return

        if not approval.approved:
            from ..base_types import ToolResult

            context.data = ToolResult(
                success=False,
                result=None,
                error=f"Approval denied: {approval.reason or 'User declined'}",
                metadata={"tool_name": tool_call.tool_name, "call_id": tool_call.call_id},
            )
            yield context
            return

        yield context

    async def process_response(
        self, context: MiddlewareContext, result: Any
    ) -> AsyncGenerator[Any, None]:
        yield result

    async def process_error(
        self, context: MiddlewareContext, error: Exception
    ) -> AsyncGenerator[Any, None]:
        if False:
            yield
        raise error
