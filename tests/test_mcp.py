"""Tests for MCP integration."""

from __future__ import annotations

from typing import Any

import pytest

from atlascore.tools import MCP_AVAILABLE
from atlascore.tools._base import ApprovalMode

pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="mcp not installed")


class FakeMCPResult:
    def __init__(self, content: Any, is_error: bool = False):
        self.content = content
        self.is_error = is_error
        self.structured_content = None


class FakeMCPClient:
    def __init__(self, result: Any):
        self._result = result

    async def call_tool(self, name: str, arguments: dict) -> Any:
        return self._result


class FakeMCPManager:
    def __init__(self, client: Any):
        self._client = client

    async def get_client(self, server_id: str) -> Any:
        return self._client


def test_destructive_tool_approval():
    from atlascore.tools._mcp import MCPTool

    tool = MCPTool(
        mcp_tool_name="delete_book",
        mcp_tool_description="Delete a book from the catalog.",
        mcp_tool_schema={"type": "object", "properties": {}},
        client_manager=FakeMCPManager(FakeMCPClient(FakeMCPResult([]))),  # type: ignore[arg-type]
        server_id="test",
    )
    assert tool.approval_mode == ApprovalMode.ALWAYS


def test_readonly_tool_default_approval():
    from atlascore.tools._mcp import MCPTool

    tool = MCPTool(
        mcp_tool_name="search_books",
        mcp_tool_description="Search the catalog by title or author.",
        mcp_tool_schema={"type": "object", "properties": {}},
        client_manager=FakeMCPManager(FakeMCPClient(FakeMCPResult([]))),  # type: ignore[arg-type]
        server_id="test",
    )
    assert tool.approval_mode == ApprovalMode.NEVER


@pytest.mark.asyncio
async def test_mcp_tool_execute_success():
    from mcp.types import TextContent

    from atlascore.tools._mcp import MCPTool

    result = FakeMCPResult([TextContent(text="hello world")])
    tool = MCPTool(
        mcp_tool_name="echo",
        mcp_tool_description="Echo tool.",
        mcp_tool_schema={"type": "object", "properties": {}},
        client_manager=FakeMCPManager(FakeMCPClient(result)),  # type: ignore[arg-type]
        server_id="test",
    )

    response = await tool.execute({})
    assert response.success
    assert "hello world" in str(response.result)
    assert response.metadata["mcp_server"] == "test"


@pytest.mark.asyncio
async def test_mcp_tool_execute_error():
    from atlascore.tools._mcp import MCPTool

    result = FakeMCPResult([], is_error=True)
    tool = MCPTool(
        mcp_tool_name="broken",
        mcp_tool_description="Broken tool.",
        mcp_tool_schema={"type": "object", "properties": {}},
        client_manager=FakeMCPManager(FakeMCPClient(result)),  # type: ignore[arg-type]
        server_id="test",
    )

    response = await tool.execute({})
    assert not response.success
    assert response.error == "MCP tool execution failed"


@pytest.mark.asyncio
async def test_create_mcp_tools_skips_offline_server():
    from atlascore.tools._mcp import HTTPServerConfig, create_mcp_tools

    manager, tools = await create_mcp_tools(
        server_configs=[
            HTTPServerConfig(
                server_id="offline",
                url="http://localhost:59999/mcp",
                transport="streamable-http",
            )
        ],
        auto_connect=True,
        timeout=1.0,
    )

    assert manager is not None
    assert tools == []
    assert not manager.is_connected("offline")
