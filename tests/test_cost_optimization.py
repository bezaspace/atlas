"""Tests for Phase 13 — cost optimization (two-stage filtering + cost tracking)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Type

import pytest
from pydantic import BaseModel

from atlascore.base_types import ToolResult
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import AssistantMessage, Message
from atlascore.research import ResearcherAgent, TriageAgent
from atlascore.research_schemas import (
    ResearchPlan,
    SearchOutput,
    SearchResult,
    TriageClassification,
    TriageResult,
)
from atlascore.tools import BaseTool
from atlascore.types import ChatCompletionChunk, ChatCompletionResult, Usage


class CostedFakeModelClient(BaseChatCompletionClient):
    """Model client that tracks a configurable per-token price for cost estimates."""

    def __init__(self, input_price: float, output_price: float, model: str = "fake") -> None:
        super().__init__(model=model)
        self.input_price = input_price
        self.output_price = output_price

    def _token_count(self, text: str) -> int:
        return max(1, len(text) // 4)

    def _urls_in_messages(self, messages: List[Message]) -> List[str]:
        urls: List[str] = []
        for msg in messages:
            text = getattr(msg, "content", "") or ""
            if isinstance(text, str):
                urls.extend(re.findall(r"https?://\S+", text))
        return urls

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        prompt_text = "\n".join(str(getattr(m, "content", "") or "") for m in messages)
        tokens_input = self._token_count(prompt_text)

        output: Optional[BaseModel] = None
        if output_format is TriageResult:
            classifications = []
            for url in self._urls_in_messages(messages):
                if "good.example.com" in url:
                    classifications.append(
                        TriageClassification(
                            url=url,
                            relevance="relevant",
                            reason="Directly matches the query.",
                            confidence=0.95,
                        )
                    )
                else:
                    classifications.append(
                        TriageClassification(
                            url=url,
                            relevance="irrelevant",
                            reason="Off topic.",
                            confidence=0.95,
                        )
                    )
            output = TriageResult(classifications=classifications)
        elif output_format is SearchOutput:
            urls = self._urls_in_messages(messages)
            if not urls:
                urls = ["https://good.example.com/page"]
            evidence = [
                SearchResult(
                    title="Result",
                    url=url,
                    snippet="snippet",
                    content="Extracted content.",
                    relevance="relevant",
                    source_index=i,
                )
                for i, url in enumerate(urls[:1])
            ]
            output = SearchOutput(query="test", evidence=evidence, summary="Extracted.")
        else:
            output = None

        content = output.model_dump_json() if output else "ok"
        tokens_output = self._token_count(content)
        cost = (tokens_input * self.input_price + tokens_output * self.output_price) / 1_000_000

        return ChatCompletionResult(
            message=AssistantMessage(content=content, source="llm"),
            usage=Usage(
                duration_ms=1,
                llm_calls=1,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost_estimate=cost,
            ),
            model=self.model,
            finish_reason="stop",
            structured_output=output,
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        if False:
            yield ChatCompletionChunk(content="", is_complete=True)
        raise NotImplementedError


class FakeSearchTool(BaseTool):
    """Returns a fixed list of search results for cost benchmarks."""

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
    """Returns a long content string so fetched text dominates prompt size."""

    def __init__(self, length: int = 4000) -> None:
        super().__init__(name="fake_fetch", description="Returns canned page content.")
        self.content = "x" * length

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}, "output_format": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, result=self.content, metadata={})


def _make_results(count: int = 10) -> List[Dict[str, str]]:
    results = [{"title": "Good result", "url": "https://good.example.com/page", "snippet": "y" * 500}]
    for i in range(1, count):
        results.append(
            {
                "title": f"Bad result {i}",
                "url": f"https://bad-{i}.example.com/page",
                "snippet": "n" * 500,
            }
        )
    return results


@pytest.mark.asyncio
async def test_triage_filters_irrelevant_sources():
    cheap_client = CostedFakeModelClient(input_price=0.15, output_price=0.60, model="cheap")
    strong_client = CostedFakeModelClient(input_price=30.0, output_price=60.0, model="strong")
    search_tool = FakeSearchTool(_make_results(10))
    fetch_tool = FakeFetchTool()
    triage_agent = TriageAgent(model_client=cheap_client)
    researcher = ResearcherAgent(
        model_client=strong_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        triage_agent=triage_agent,
        max_sources=10,
    )
    plan = ResearchPlan(query="test", search_queries=["test"], required_sources=10)
    output = await researcher.run("test", plan)
    assert len(output.evidence) == 1
    assert output.evidence[0].url == "https://good.example.com/page"


@pytest.mark.asyncio
async def test_two_stage_cost_reduction_versus_naive():
    """Two-stage (cheap triage + strong extraction) should be ~90% cheaper than naive."""
    cheap_client = CostedFakeModelClient(input_price=0.15, output_price=0.60, model="cheap")
    strong_client = CostedFakeModelClient(input_price=30.0, output_price=60.0, model="strong")
    search_tool = FakeSearchTool(_make_results(10))
    fetch_tool = FakeFetchTool()
    plan = ResearchPlan(query="test", search_queries=["test"], required_sources=10)

    # Optimized run
    optimized_researcher = ResearcherAgent(
        model_client=strong_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        triage_agent=TriageAgent(model_client=cheap_client),
        max_sources=10,
    )
    await optimized_researcher.run("test", plan)
    optimized_cost = optimized_researcher.last_usage.cost_estimate or 0.0

    # Naive run: no triage, strong model processes all fetched sources.
    naive_researcher = ResearcherAgent(
        model_client=strong_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        triage_agent=None,
        max_sources=10,
    )
    await naive_researcher.run("test", plan)
    naive_cost = naive_researcher.last_usage.cost_estimate or 0.0

    assert naive_cost > 0
    assert optimized_cost < naive_cost * 0.20, (
        f"Optimized cost {optimized_cost:.6f} should be < 20% of naive {naive_cost:.6f}"
    )


@pytest.mark.asyncio
async def test_cost_estimate_returned_by_openai_client_for_free_and_known_models():
    from openai.types.completion_usage import CompletionUsage

    from atlascore.llm import OpenAIChatCompletionClient

    client = OpenAIChatCompletionClient(model="inclusionai/ling-3.0-flash:free", api_key="x")
    assert (
        client._estimate_cost(
            CompletionUsage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200)
        )
        == 0.0
    )

    client_gpt4o = OpenAIChatCompletionClient(model="openai/gpt-4o", api_key="x")
    cost = client_gpt4o._estimate_cost(
        CompletionUsage(prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
    )
    assert cost is not None
    assert 10.0 < cost < 15.0
