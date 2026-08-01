"""MCP (Model Context Protocol) integration for atlascore."""

from __future__ import annotations

try:
    from ._client import MCPClientManager
    from ._config import HTTPServerConfig, MCPServerConfig, StdioServerConfig, TransportType
    from ._integration import create_mcp_tools
    from ._tool import MCPTool

    MCP_AVAILABLE = True
except ImportError:
    MCPClientManager = None  # type: ignore[assignment, misc]
    HTTPServerConfig = None  # type: ignore[assignment, misc]
    MCPServerConfig = None  # type: ignore[assignment, misc]
    StdioServerConfig = None  # type: ignore[assignment, misc]
    TransportType = None  # type: ignore[assignment, misc]
    create_mcp_tools = None  # type: ignore[assignment, misc]
    MCPTool = None  # type: ignore[assignment, misc]
    MCP_AVAILABLE = False

__all__ = [
    "MCPClientManager",
    "MCPServerConfig",
    "StdioServerConfig",
    "HTTPServerConfig",
    "TransportType",
    "MCPTool",
    "create_mcp_tools",
    "MCP_AVAILABLE",
]
