"""Tests for Phase 12 — Computer-use fallback."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Type

import pytest
from pydantic import BaseModel

from atlascore.agents import ComputerUseAgent, PlaywrightWebClient, create_playwright_tools
from atlascore.agents._computer_use import Action, ActionType
from atlascore.base_types import ToolResult, Usage
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import AssistantMessage, Message, ToolCallRequest
from atlascore.research import ResearcherAgent
from atlascore.research_schemas import ResearchPlan, SearchOutput, SearchResult
from atlascore.tools import BaseTool
from atlascore.types import ChatCompletionChunk, ChatCompletionResult


class FakeVisionModelClient(BaseChatCompletionClient):
    """Model client that drives a ComputerUseAgent through a canned tool sequence."""

    def __init__(self, sequence: List[Dict[str, Any]]) -> None:
        super().__init__(model="fake-vision")
        self.sequence = sequence
        self.index = 0

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        if output_format is SearchOutput:
            output = SearchOutput(
                query="test query",
                evidence=[
                    SearchResult(
                        title="Example",
                        url="https://example.com/page",
                        snippet="snippet",
                        content="Browser retrieved the answer.",
                        relevance="relevant",
                        source_index=0,
                    )
                ],
                summary="Browser fallback succeeded.",
            )
            return ChatCompletionResult(
                message=AssistantMessage(content=output.model_dump_json(), source="llm"),
                usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3),
                model=self.model,
                finish_reason="stop",
                structured_output=output,
            )

        item = self.sequence[self.index]
        self.index += 1
        tool_calls = None
        if "tool" in item:
            tool_calls = [
                ToolCallRequest(
                    tool_name=item["tool"],
                    parameters=item["parameters"],
                    call_id=f"call_{self.index}",
                )
            ]
        content = item.get("content", "")
        return ChatCompletionResult(
            message=AssistantMessage(content=content, source="llm", tool_calls=tool_calls),
            usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3),
            model=self.model,
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        if False:
            yield ChatCompletionChunk(content="", is_complete=True)
        raise NotImplementedError


class FakeSearchTool(BaseTool):
    """Canned search tool."""

    def __init__(self, results: List[Dict[str, str]]) -> None:
        super().__init__(name="fake_search", description="Returns canned results.")
        self.results = results

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, result=self.results, metadata={})


class FakeFetchTool(BaseTool):
    """Canned fetch tool that can succeed or fail."""

    def __init__(self, content: Optional[str] = None, fail: bool = False) -> None:
        super().__init__(name="fake_fetch", description="Returns canned page content.")
        self.content = content
        self.fail = fail

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}, "output_format": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        if self.fail:
            return ToolResult(success=False, result=None, error="Failed to fetch")
        return ToolResult(success=True, result=self.content or "", metadata={})


class FakeBrowserAgent:
    """Minimal stand-in for ComputerUseAgent in fallback tests."""

    def __init__(self, answer: str = "Browser fallback content.") -> None:
        self.answer = answer
        self.usage = Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3)

    async def run(self, **kwargs: Any) -> Any:
        from atlascore.context import AgentContext
        from atlascore.messages import AssistantMessage
        from atlascore.types import AgentResponse

        ctx = AgentContext()
        ctx.add_message(AssistantMessage(content=self.answer, source="computer_use"))
        return AgentResponse(context=ctx, source="computer_use", usage=self.usage, finish_reason="stop")


@pytest.mark.asyncio
async def test_playwright_client_state_and_screenshot():
    data_url = "data:text/html,<html><head><title>Test Page</title></head><body><h1>Hello</h1></body></html>"
    client = PlaywrightWebClient(start_url=data_url, headless=True)
    await client.initialize()
    try:
        state = await client.get_state("hybrid")
        assert state.url == data_url
        assert state.title == "Test Page"
        assert "Hello" in state.content
        assert state.screenshot is not None
        assert len(state.screenshot) > 0

        result = await client.execute_action(Action(action_type=ActionType.CLICK, selector="h1"))
        assert result.success
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_playwright_tools_navigate_and_observe():
    client = PlaywrightWebClient(headless=True)
    tools = {t.name: t for t in create_playwright_tools(client)}
    await client.initialize()
    try:
        navigate = await tools["navigate"].execute({"url": "https://example.com"})
        assert navigate.success

        observe = await tools["observe_page"].execute({})
        assert observe.success
        assert "https://example.com" in str(observe.result)
        assert "screenshot" in observe.metadata
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_computer_use_agent_runs_tool_sequence():
    sequence = [
        {
            "tool": "navigate",
            "parameters": {"url": "https://example.com"},
        },
        {"content": "The page says 'Example Domain'"},
    ]
    model = FakeVisionModelClient(sequence)
    agent = ComputerUseAgent(model_client=model, headless=True, max_actions=5)
    response = await agent.run("What is the title of example.com?")
    assert response is not None
    assert response.finish_reason in ("stop", "termination")
    final = response.messages[-1].content if response.messages else ""
    assert "Example Domain" in final
    await agent.close()


@pytest.mark.asyncio
async def test_researcher_uses_browser_fallback_on_fetch_failure():
    search_tool = FakeSearchTool(
        results=[{"title": "Example", "url": "https://example.com/page", "snippet": "snippet"}]
    )
    fetch_tool = FakeFetchTool(fail=True)
    browser_agent = FakeBrowserAgent(answer="Browser retrieved the answer.")
    researcher = ResearcherAgent(
        model_client=FakeVisionModelClient([{"content": "ignored"}]),
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        computer_use_agent=browser_agent,  # type: ignore[arg-type]
    )
    plan = ResearchPlan(query="test query", search_queries=["test query"], required_sources=1)
    output = await researcher.run("test query", plan)
    contents = [r.content or "" for r in output.evidence]
    assert any("Browser retrieved" in c for c in contents)


@pytest.mark.asyncio
async def test_researcher_uses_browser_fallback_on_sparse_content():
    search_tool = FakeSearchTool(
        results=[{"title": "Example", "url": "https://example.com/page", "snippet": "snippet"}]
    )
    fetch_tool = FakeFetchTool(content="short")
    browser_agent = FakeBrowserAgent(answer="Browser retrieved the answer.")
    researcher = ResearcherAgent(
        model_client=FakeVisionModelClient([{"content": "ignored"}]),
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        computer_use_agent=browser_agent,  # type: ignore[arg-type]
    )
    plan = ResearchPlan(query="test query", search_queries=["test query"], required_sources=1)
    output = await researcher.run("test query", plan)
    contents = [r.content or "" for r in output.evidence]
    assert any("Browser retrieved" in c for c in contents)
