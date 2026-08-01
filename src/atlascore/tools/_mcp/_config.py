"""Configuration classes for MCP server connections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

TransportType = Literal["stdio", "sse", "streamable-http"]


@dataclass
class MCPServerConfig:
    """Base configuration for an MCP server connection."""

    server_id: str
    transport: TransportType
    env: Optional[Dict[str, str]] = None


@dataclass
class StdioServerConfig(MCPServerConfig):
    """Configuration for a stdio MCP server subprocess."""

    command: str = ""
    args: List[str] = field(default_factory=list)

    def __init__(
        self,
        server_id: str,
        command: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
    ):
        super().__init__(server_id, "stdio", env)
        self.command = command
        self.args = args


@dataclass
class HTTPServerConfig(MCPServerConfig):
    """Configuration for an HTTP/SSE MCP server."""

    url: str = ""
    headers: Optional[Dict[str, str]] = None

    def __init__(
        self,
        server_id: str,
        url: str,
        transport: Literal["sse", "streamable-http"] = "streamable-http",
        headers: Optional[Dict[str, str]] = None,
        env: Optional[Dict[str, str]] = None,
    ):
        super().__init__(server_id, transport, env)
        self.url = url
        self.headers = headers
