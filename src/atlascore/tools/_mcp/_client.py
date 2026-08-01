"""MCPClientManager - Manages connections to MCP servers."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from .._base import ApprovalMode, BaseTool
from ._config import HTTPServerConfig, MCPServerConfig, StdioServerConfig
from ._tool import MCPTool, _is_destructive_tool


class MCPClientManager:
    """Manages connections to MCP servers and provides discovered tools."""

    def __init__(self, default_approval_mode: Optional[ApprovalMode] = None):
        self._servers: Dict[str, MCPServerConfig] = {}
        self._clients: Dict[str, Any] = {}
        self._tools: Dict[str, List[MCPTool]] = {}
        self._default_approval_mode = default_approval_mode

    def add_server(self, config: MCPServerConfig) -> None:
        """Register an MCP server configuration."""
        if config.server_id in self._servers:
            raise ValueError(f"Server '{config.server_id}' is already registered")
        self._servers[config.server_id] = config

    async def connect(self, server_id: str, timeout: Optional[float] = 10.0) -> None:
        """Connect to an MCP server and discover tools."""
        if server_id not in self._servers:
            raise ValueError(f"Unknown server: {server_id}")

        if server_id in self._clients:
            return

        config = self._servers[server_id]
        client = self._create_client(config)

        try:
            entered_client = await asyncio.wait_for(
                client.__aenter__(), timeout=timeout
            )
        except Exception as e:
            try:
                await client.__aexit__(type(e), e, None)
            except Exception:
                pass
            raise ConnectionError(
                f"Failed to connect to MCP server '{server_id}': {e}"
            ) from e

        self._clients[server_id] = entered_client
        await self._discover_tools(server_id)

    def _create_client(self, config: MCPServerConfig) -> Any:
        """Create an MCP Client for the given transport."""
        try:
            from mcp import Client
            from mcp.client.sse import sse_client
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as e:
            raise ImportError("mcp>=2.0.0 is required for MCP support") from e

        if isinstance(config, StdioServerConfig):
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
            )
            transport = stdio_client(server_params)
            return Client(transport)

        if isinstance(config, HTTPServerConfig):
            if config.transport == "sse":
                transport = sse_client(
                    url=config.url,
                    headers=config.headers or {},
                )
                return Client(transport)

            # streamable-http
            if config.headers:
                try:
                    import httpx2
                except ImportError as e:
                    raise ImportError(
                        "httpx2 is required for streamable-http MCP transports"
                    ) from e
                http_client = httpx2.AsyncClient(headers=config.headers or {})
                transport = streamable_http_client(
                    config.url, http_client=http_client
                )
                return Client(transport)

            return Client(config.url)

        raise ValueError(f"Unsupported MCP transport: {config.transport}")

    async def _discover_tools(self, server_id: str) -> None:
        """Discover tools from a connected MCP server."""
        client = self._clients[server_id]
        tools_response = await client.list_tools()

        mcp_tools = []
        for tool in tools_response.tools:
            approval_mode = self._default_approval_mode

            if approval_mode is None:
                annotations = getattr(tool, "annotations", None)
                if annotations:
                    if getattr(annotations, "read_only_hint", None):
                        approval_mode = ApprovalMode.NEVER
                    elif getattr(annotations, "destructive_hint", None):
                        approval_mode = ApprovalMode.ALWAYS

            if approval_mode is None and _is_destructive_tool(
                tool.name, tool.description or ""
            ):
                approval_mode = ApprovalMode.ALWAYS

            mcp_tool = MCPTool(
                mcp_tool_name=tool.name,
                mcp_tool_description=tool.description or "",
                mcp_tool_schema=tool.input_schema,
                client_manager=self,
                server_id=server_id,
                approval_mode=approval_mode,
            )
            mcp_tools.append(mcp_tool)

        self._tools[server_id] = mcp_tools

    async def get_client(self, server_id: str) -> Any:
        """Get or connect the MCP client for a server."""
        if server_id not in self._clients:
            await self.connect(server_id)
        return self._clients[server_id]

    def get_tools(self, server_id: Optional[str] = None) -> List[BaseTool]:
        """Return tools from one or all connected servers."""
        if server_id:
            return list(self._tools.get(server_id, []))
        all_tools: List[BaseTool] = []
        for tools in self._tools.values():
            all_tools.extend(tools)
        return all_tools

    def list_servers(self) -> List[str]:
        """List all registered server IDs."""
        return list(self._servers.keys())

    def is_connected(self, server_id: str) -> bool:
        """Check if a server is currently connected."""
        return server_id in self._clients

    async def disconnect(self, server_id: str) -> None:
        """Disconnect from an MCP server."""
        if server_id in self._clients:
            try:
                client = self._clients.pop(server_id)
                await client.__aexit__(None, None, None)
            except Exception:
                pass
        self._tools.pop(server_id, None)

    async def connect_all(self, timeout: Optional[float] = 10.0) -> None:
        """Best-effort connect to all registered servers; failures are ignored."""
        await asyncio.gather(
            *(
                self.connect(sid, timeout=timeout)
                for sid in self._servers.keys()
                if not self.is_connected(sid)
            ),
            return_exceptions=True,
        )

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        for server_id in list(self._clients.keys()):
            await self.disconnect(server_id)

    @asynccontextmanager
    async def managed_connection(self, server_id: str):
        """Context manager for automatic connection/disconnection."""
        await self.connect(server_id)
        try:
            yield self
        finally:
            await self.disconnect(server_id)

    def __repr__(self) -> str:
        connected = [sid for sid in self._servers if self.is_connected(sid)]
        return (
            f"MCPClientManager("
            f"servers={len(self._servers)}, "
            f"connected={len(connected)}, "
            f"tools={sum(len(t) for t in self._tools.values())})"
        )
