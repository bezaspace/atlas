"""Fake model client and tools for benchmark demonstrations."""

from __future__ import annotations

from typing import Any, Dict, Type

from pydantic import BaseModel

from atlascore.base_types import ToolResult, Usage
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import AssistantMessage
from atlascore.research_schemas import (
    Citation,
    CriticReview,
    Evidence,
    ResearchBrief,
    ResearchPlan,
    SearchOutput,
    SearchResult,
    TriageClassification,
    TriageResult,
    VerificationResult,
)
from atlascore.tools import BaseTool
from atlascore.types import ChatCompletionResult


class FakeBenchmarkModelClient(BaseChatCompletionClient):
    """Fake model client returning pre-configured structured research outputs."""

    def __init__(self, outputs: Dict[Type[BaseModel], BaseModel]) -> None:
        super().__init__(model="fake")
        self.outputs = outputs

    async def create(
        self,
        messages,
        tools=None,
        output_format: Type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        output = self.outputs.get(output_format)
        if output is not None:
            content = output.model_dump_json()
            return ChatCompletionResult(
                message=AssistantMessage(content=content, source="assistant"),
                usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3, cost_estimate=0.0001),
                model=self.model,
                finish_reason="stop",
                structured_output=output,
            )
        return ChatCompletionResult(
            message=AssistantMessage(content="OK", source="assistant"),
            usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3, cost_estimate=0.0001),
            model=self.model,
            finish_reason="stop",
        )

    async def create_stream(self, messages, **kwargs):
        raise NotImplementedError


class FakeBenchmarkSearchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="fake_search", description="Canned search")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
            "required": ["query"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            result=[
                {
                    "title": "Example Source",
                    "url": "https://example.com/article",
                    "snippet": "Atlas is a multi-agent research platform.",
                }
            ],
            metadata={"query": parameters.get("query", "")},
        )


class FakeBenchmarkFetchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="fake_fetch", description="Canned fetch")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}, "output_format": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            result="Atlas is a multi-agent research platform built for autonomous deep research.",
            metadata={"url": parameters.get("url", "")},
        )


def make_fake_outputs() -> Dict[Type[BaseModel], BaseModel]:
    """Return a canned set of structured outputs for the research pipeline."""
    return {
        ResearchPlan: ResearchPlan(
            query="What is Atlas?",
            sub_questions=["What is Atlas?"],
            search_queries=["Atlas multi-agent platform"],
            required_sources=1,
            reasoning="Direct search is enough.",
        ),
        TriageResult: TriageResult(
            classifications=[
                TriageClassification(
                    url="https://example.com/article",
                    relevance="relevant",
                    reason="Directly answers the query.",
                    confidence=0.9,
                )
            ]
        ),
        SearchOutput: SearchOutput(
            query="What is Atlas?",
            evidence=[
                SearchResult(
                    title="Example Source",
                    url="https://example.com/article",
                    snippet="Atlas is a multi-agent research platform.",
                    content="Atlas is a multi-agent research platform built for autonomous deep research.",
                    relevance="relevant",
                    source_index=0,
                )
            ],
            summary="Atlas is a multi-agent research platform.",
        ),
        VerificationResult: VerificationResult(
            overall_confidence=0.9,
            evidence=[
                Evidence(
                    claim="Atlas is a multi-agent research platform",
                    assessment="supported",
                    confidence=0.9,
                    citations=[
                        Citation(
                            source_url="https://example.com/article",
                            source_title="Example Source",
                            quote="Atlas is a multi-agent research platform.",
                            index=0,
                        )
                    ],
                )
            ],
        ),
        ResearchBrief: ResearchBrief(
            title="Atlas",
            summary="Atlas is a multi-agent research platform.",
            sections=[{"heading": "Overview", "content": "Atlas is a multi-agent research platform."}],
            citations=[
                Citation(
                    source_url="https://example.com/article",
                    source_title="Example Source",
                    quote="Atlas is a multi-agent research platform.",
                    index=0,
                )
            ],
            confidence=0.9,
        ),
        CriticReview: CriticReview(
            revisions_required=False,
            feedback="The brief is well-supported.",
            suggested_changes=[],
            confidence=0.9,
        ),
    }
