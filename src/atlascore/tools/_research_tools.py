"""Research tools for web search and content fetching.

Follows the patterns in victordibia/designing-multiagent-systems
(PicoAgents) for secure, filtered information retrieval.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import httpx

from ..base_types import ToolResult
from ._base import BaseTool

BeautifulSoup: Any = None
BS4_AVAILABLE = False
try:
    from bs4 import BeautifulSoup  # type: ignore[import]

    BS4_AVAILABLE = True
except ImportError:
    pass

html2text: Any = None  # noqa: F811
HTML2TEXT_AVAILABLE = False
try:
    import html2text  # type: ignore[import]

    HTML2TEXT_AVAILABLE = True
except ImportError:
    pass


class WebSearchTool(BaseTool):
    """Search the web using a configurable provider (Tavily or Google CSE)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "tavily",
        cse_id: Optional[str] = None,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            name="web_search",
            description=(
                "Search the web for information. Returns titles, URLs, and snippets. "
                "Supports Tavily and Google Custom Search Engine backends."
            ),
        )
        self.api_key = api_key
        self.provider = provider.lower()
        self.cse_id = cse_id
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query string",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 5, max: 10)",
                },
            },
            "required": ["query"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        query = parameters["query"]
        max_results = min(max(1, parameters.get("max_results", 5)), 10)

        if self.provider == "tavily":
            return await self._search_tavily(query, max_results)
        if self.provider == "google":
            return await self._search_google(query, max_results)

        return ToolResult(
            success=False,
            result=None,
            error=f"Unknown search provider: {self.provider}. Use 'tavily' or 'google'.",
            metadata={"query": query},
        )

    async def _search_tavily(self, query: str, max_results: int) -> ToolResult:
        api_key = self.api_key
        if not api_key:
            return ToolResult(
                success=False,
                result=None,
                error=(
                    "Tavily API key not provided. "
                    "Pass api_key to WebSearchTool or set TAVILY_API_KEY."
                ),
                metadata={"query": query},
            )

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": api_key,
                        "query": query,
                        "max_results": max_results,
                    },
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("results", []):
                url = item.get("url", "")
                if self._is_domain_allowed(url):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("content", ""),
                        }
                    )

            return ToolResult(
                success=True,
                result=results,
                error=None,
                metadata={
                    "query": query,
                    "count": len(results),
                    "provider": "tavily",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Tavily search failed: {e}",
                metadata={"query": query},
            )

    async def _search_google(self, query: str, max_results: int) -> ToolResult:
        if not self.api_key or not self.cse_id:
            return ToolResult(
                success=False,
                result=None,
                error=(
                    "Google CSE requires api_key and cse_id. "
                    "Set GOOGLE_API_KEY and GOOGLE_CSE_ID or pass them to WebSearchTool."
                ),
                metadata={"query": query},
            )

        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": max_results,
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://www.googleapis.com/customsearch/v1", params=params
                )
                response.raise_for_status()
                data = response.json()

            results = []
            for item in data.get("items", []):
                url = item.get("link", "")
                if self._is_domain_allowed(url):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("snippet", ""),
                        }
                    )

            return ToolResult(
                success=True,
                result=results,
                error=None,
                metadata={
                    "query": query,
                    "count": len(results),
                    "provider": "google",
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Google search failed: {e}",
                metadata={"query": query},
            )

    def _is_domain_allowed(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            if not domain:
                return False

            for blocked in self.blocked_domains:
                blocked_lower = blocked.lower()
                if domain == blocked_lower or domain.endswith("." + blocked_lower):
                    return False

            if self.allowed_domains:
                for allowed in self.allowed_domains:
                    allowed_lower = allowed.lower()
                    if domain == allowed_lower or domain.endswith("." + allowed_lower):
                        return True
                return False

            return True
        except Exception:
            return False


class WebFetchTool(BaseTool):
    """Fetch content from a URL with optional text/markdown extraction."""

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        blocked_domains: Optional[List[str]] = None,
        max_content_length: int = 100_000,
    ) -> None:
        super().__init__(
            name="web_fetch",
            description=(
                "Fetch content from a URL. Output as raw HTML, plain text, or markdown. "
                "Text/markdown extraction requires beautifulsoup4/html2text."
            ),
        )
        self.allowed_domains = allowed_domains or []
        self.blocked_domains = blocked_domains or []
        self.max_content_length = max_content_length

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to fetch",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["html", "text", "markdown"],
                    "description": "Output format: 'html', 'text', or 'markdown' (default: 'html')",
                },
            },
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        url = parameters["url"]
        output_format = parameters.get("output_format", "html")

        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"Invalid URL: {url}",
                    metadata={"url": url},
                )

            if not self._is_domain_allowed(url):
                return ToolResult(
                    success=False,
                    result=None,
                    error=f"URL domain blocked or not in allowed list: {parsed.netloc}",
                    metadata={"url": url, "domain": parsed.netloc},
                )

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                )
            }

            async with httpx.AsyncClient(
                follow_redirects=True, headers=headers, timeout=30.0
            ) as client:
                response = await client.get(url)
                response.raise_for_status()

            content = response.text
            original_length = len(content)

            if output_format == "markdown":
                content = self._extract_markdown(content)
            elif output_format == "text":
                content = self._extract_text(content)

            was_truncated = False
            if len(content) > self.max_content_length:
                content = content[: self.max_content_length]
                was_truncated = True

            return ToolResult(
                success=True,
                result=content,
                error=None,
                metadata={
                    "url": url,
                    "output_format": output_format,
                    "content_length": len(content),
                    "original_length": original_length,
                    "truncated": was_truncated,
                    "status_code": response.status_code,
                },
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to fetch URL: {e}",
                metadata={"url": url},
            )

    def _extract_text(self, html: str) -> str:
        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            return "\n".join(line for line in lines if line)

        # Fallback: strip tags with a regex and collapse whitespace
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _extract_markdown(self, html: str) -> str:
        if HTML2TEXT_AVAILABLE:
            h = html2text.HTML2Text()
            h.body_width = 0
            h.ignore_images = False
            h.ignore_emphasis = False
            h.ignore_links = False
            h.ignore_tables = False
            return h.handle(html)

        if BS4_AVAILABLE:
            soup = BeautifulSoup(html, "html.parser")
            for script in soup(["script", "style"]):
                script.decompose()
            return soup.get_text(separator="\n")

        return html

    def _is_domain_allowed(self, url: str) -> bool:
        try:
            domain = urlparse(url).netloc.lower()
            if not domain:
                return False

            for blocked in self.blocked_domains:
                blocked_lower = blocked.lower()
                if domain == blocked_lower or domain.endswith("." + blocked_lower):
                    return False

            if self.allowed_domains:
                for allowed in self.allowed_domains:
                    allowed_lower = allowed.lower()
                    if domain == allowed_lower or domain.endswith("." + allowed_lower):
                        return True
                return False

            return True
        except Exception:
            return False


def create_research_tools(
    api_key: Optional[str] = None,
    provider: str = "tavily",
    cse_id: Optional[str] = None,
) -> List[BaseTool]:
    """Create a default pair of research tools."""
    return [
        WebSearchTool(api_key=api_key, provider=provider, cse_id=cse_id),
        WebFetchTool(),
    ]
