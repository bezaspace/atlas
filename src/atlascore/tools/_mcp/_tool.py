"""MCPTool - Bridge between MCP server tools and atlascore tools."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

from ...base_types import ToolResult
from .._base import ApprovalMode, BaseTool

if TYPE_CHECKING:
    from ._client import MCPClientManager


def _is_destructive_tool(name: str, description: str = "") -> bool:
    """Heuristic to classify MCP tools that write or delete data."""
    text = f"{name} {description}".lower()
    write_keywords = (
        "write", "create", "delete", "remove", "update", "modify",
        "patch", "insert", "append", "post", "put", "destroy", "drop",
        "submit", "send", "publish", "deploy", "exec", "execute",
    )
    pattern = re.compile(r"\b(" + "|".join(re.escape(kw) for kw in write_keywords) + r")\b")
    return bool(pattern.search(text))


class MCPTool(BaseTool):
    """Wraps an MCP server tool as an atlascore BaseTool."""

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_tool_description: str,
        mcp_tool_schema: Dict[str, Any],
        client_manager: "MCPClientManager",
        server_id: str,
        version: str = "1.0.0",
        approval_mode: Optional[ApprovalMode] = None,
    ):
        tool_name = f"mcp_{server_id}_{mcp_tool_name}"
        if approval_mode is None:
            approval_mode = (
                ApprovalMode.ALWAYS
                if _is_destructive_tool(mcp_tool_name, mcp_tool_description)
                else ApprovalMode.NEVER
            )
        super().__init__(
            name=tool_name,
            description=mcp_tool_description,
            version=version,
            approval_mode=approval_mode,
        )

        self.mcp_tool_name = mcp_tool_name
        self._parameters_schema = mcp_tool_schema
        self.client_manager = client_manager
        self.server_id = server_id

    @property
    def parameters(self) -> Dict[str, Any]:
        """Return the MCP tool's parameter schema."""
        return self._parameters_schema

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the MCP tool via the client manager."""
        try:
            client = await self.client_manager.get_client(self.server_id)
            result = await client.call_tool(self.mcp_tool_name, parameters)

            output = self._extract_result_content(result)

            return ToolResult(
                success=not result.is_error,
                result=output,
                error=None if not result.is_error else "MCP tool execution failed",
                metadata={
                    "tool_name": self.name,
                    "mcp_server": self.server_id,
                    "mcp_tool": self.mcp_tool_name,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                metadata={
                    "tool_name": self.name,
                    "exception_type": type(e).__name__,
                },
            )

    def _extract_result_content(self, result: Any) -> Any:
        """Extract text/structured content from an MCP CallToolResult."""
        if getattr(result, "structured_content", None):
            return result.structured_content

        content = getattr(result, "content", None)
        if not content:
            return None

        try:
            from mcp.types import TextContent

            text_parts = []
            for block in content:
                if isinstance(block, TextContent):
                    text_parts.append(block.text)

            return "\n".join(text_parts) if text_parts else None
        except Exception:
            return str(content)

    def __repr__(self) -> str:
        return (
            f"MCPTool(name='{self.name}', "
            f"server='{self.server_id}', "
            f"mcp_tool='{self.mcp_tool_name}')"
        )
