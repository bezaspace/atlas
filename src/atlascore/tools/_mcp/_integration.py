"""Helper functions for easy MCP integration with atlascore."""

from __future__ import annotations

from typing import List, Tuple

from .._base import BaseTool
from ._client import MCPClientManager
from ._config import MCPServerConfig


async def create_mcp_tools(
    server_configs: List[MCPServerConfig],
    auto_connect: bool = True,
    timeout: float = 10.0,
) -> Tuple[MCPClientManager, List[BaseTool]]:
    """Create an MCP manager, register servers, and optionally connect/discover tools.

    Servers that fail to connect are skipped so the rest of the application keeps
    working when an MCP server is offline or unavailable.
    """
    manager = MCPClientManager()

    for config in server_configs:
        manager.add_server(config)

    if auto_connect:
        for config in server_configs:
            try:
                await manager.connect(config.server_id, timeout=timeout)
            except Exception:
                pass  # Best-effort: missing servers must not block startup

    all_tools = manager.get_tools()
    return manager, all_tools
