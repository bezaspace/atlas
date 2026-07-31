"""Tests for Phase 7 — research agents, critic panel, and pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pytest
from pydantic import BaseModel

from atlascore.base_types import ToolResult, Usage
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import AssistantMessage
from atlascore.research import (
    CriticPanel,
    PlannerAgent,
    ResearcherAgent,
    ResearchPipeline,
    SynthesizerAgent,
    TriageAgent,
    VerifierAgent,
)
from atlascore.research_schemas import (
    Citation,
    CriticReview,
    Evidence,
    ResearchBrief,
    ResearchPlan,
    ResearchReport,
    SearchOutput,
    SearchResult,
    TriageClassification,
    TriageResult,
    VerificationResult,
)
from atlascore.tools import BaseTool
from atlascore.types import ChatCompletionResult


class FakeResearchModelClient(BaseChatCompletionClient):
    """A model client that returns pre-configured structured outputs by class."""

    def __init__(
        self,
        outputs: Optional[Dict[Type[BaseModel], BaseModel]] = None,
        default_text: str = "This looks good. NO_REVISIONS",
    ) -> None:
        super().__init__(model="fake")
        self.outputs = outputs or {}
        self.default_text = default_text

    async def create(
        self,
        messages: List[Any],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        output = self.outputs.get(output_format)
        if output is not None:
            content = output.model_dump_json()
            return ChatCompletionResult(
                message=AssistantMessage(content=content, source="assistant"),
                usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3),
                model=self.model,
                finish_reason="stop",
                structured_output=output,
            )

        return ChatCompletionResult(
            message=AssistantMessage(content=self.default_text, source="assistant"),
            usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3),
            model=self.model,
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        raise NotImplementedError


class FakeSearchTool(BaseTool):
    """ deterministic web search tool for tests."""

    def __init__(self, results: Optional[List[Dict[str, str]]] = None) -> None:
        super().__init__(
            name="fake_search",
            description="Returns canned search results.",
        )
        self.results = results or [
            {
                "title": "Example Source",
                "url": "https://example.com/article",
                "snippet": "A useful snippet.",
            }
        ]

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer"},
            },
            "required": ["query"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            result=self.results,
            metadata={"query": parameters.get("query", "")},
        )


class FakeFetchTool(BaseTool):
    """Deterministic web fetch tool for tests."""

    def __init__(self, content: str = "Full article content.") -> None:
        super().__init__(
            name="fake_fetch",
            description="Returns canned page content.",
        )
        self.content = content

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "output_format": {"type": "string"},
            },
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            result=self.content,
            metadata={"url": parameters.get("url", "")},
        )


def _make_outputs() -> Dict[Type[BaseModel], BaseModel]:
    plan = ResearchPlan(
        query="What is Atlas?",
        sub_questions=["What is Atlas?"],
        search_queries=["Atlas multi-agent platform"],
        required_sources=1,
        reasoning="Direct search is enough.",
    )
    triage = TriageResult(
        classifications=[
            TriageClassification(
                url="https://example.com/article",
                relevance="relevant",
                reason="Directly answers the query.",
                confidence=0.9,
            )
        ]
    )
    evidence = SearchOutput(
        query="What is Atlas?",
        evidence=[
            SearchResult(
                title="Example Source",
                url="https://example.com/article",
                snippet="A useful snippet.",
                content="Full article content.",
                relevance="relevant",
                source_index=0,
            )
        ],
        summary="Atlas is a multi-agent platform.",
    )
    verification = VerificationResult(
        overall_confidence=0.9,
        evidence=[
            Evidence(
                claim="Atlas is a multi-agent platform",
                assessment="supported",
                confidence=0.9,
                citations=[
                    Citation(
                        source_url="https://example.com/article",
                        source_title="Example Source",
                        quote="A useful snippet.",
                        index=0,
                    )
                ],
            )
        ],
    )
    brief = ResearchBrief(
        title="Atlas",
        summary="Atlas is a multi-agent platform.",
        sections=[{"heading": "Overview", "content": "Atlas is a multi-agent platform."}],
        citations=[
            Citation(
                source_url="https://example.com/article",
                source_title="Example Source",
                quote="A useful snippet.",
                index=0,
            )
        ],
        confidence=0.9,
    )
    critic = CriticReview(
        revisions_required=False,
        feedback="The brief is well-supported.",
        suggested_changes=[],
        confidence=0.9,
    )
    return {
        ResearchPlan: plan,
        TriageResult: triage,
        SearchOutput: evidence,
        VerificationResult: verification,
        ResearchBrief: brief,
        CriticReview: critic,
    }


@pytest.fixture
def model_client():
    return FakeResearchModelClient(outputs=_make_outputs())


@pytest.fixture
def search_tool():
    return FakeSearchTool()


@pytest.fixture
def fetch_tool():
    return FakeFetchTool()


@pytest.mark.asyncio
async def test_planner_agent(model_client):
    planner = PlannerAgent(model_client)
    plan = await planner.run("What is Atlas?")
    assert isinstance(plan, ResearchPlan)
    assert plan.query == "What is Atlas?"
    assert plan.search_queries


@pytest.mark.asyncio
async def test_triage_agent(model_client):
    triage_agent = TriageAgent(model_client)
    results = [
        SearchResult(title="A", url="https://example.com/article", snippet="..."),
        SearchResult(title="B", url="https://other.com/page", snippet="..."),
    ]
    triage = await triage_agent.run("What is Atlas?", results)
    assert isinstance(triage, TriageResult)
    assert triage.classifications
    assert "https://example.com/article" in triage.relevant_urls()


@pytest.mark.asyncio
async def test_researcher_agent(model_client, search_tool, fetch_tool):
    triage = TriageAgent(model_client)
    researcher = ResearcherAgent(
        model_client=model_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        triage_agent=triage,
        max_sources=2,
    )
    plan = ResearchPlan(
        query="What is Atlas?",
        sub_questions=["What is Atlas?"],
        search_queries=["Atlas multi-agent platform"],
        required_sources=1,
    )
    output = await researcher.run("What is Atlas?", plan)
    assert isinstance(output, SearchOutput)
    assert output.evidence
    assert output.evidence[0].url == "https://example.com/article"


@pytest.mark.asyncio
async def test_verifier_agent(model_client, search_tool, fetch_tool):
    verifier = VerifierAgent(model_client)
    evidence = [
        SearchResult(
            title="Example Source",
            url="https://example.com/article",
            snippet="...",
            content="Full content.",
        )
    ]
    verification = await verifier.run("What is Atlas?", evidence)
    assert isinstance(verification, VerificationResult)
    assert verification.overall_confidence >= 0


@pytest.mark.asyncio
async def test_synthesizer_agent(model_client):
    synthesizer = SynthesizerAgent(model_client)
    verification = VerificationResult(overall_confidence=0.9)
    brief = await synthesizer.run("What is Atlas?", verification)
    assert isinstance(brief, ResearchBrief)
    assert brief.title
    assert brief.to_markdown()


@pytest.mark.asyncio
async def test_critic_panel(model_client):
    critic_panel = CriticPanel(model_client)
    brief = ResearchBrief(
        title="Atlas",
        summary="Atlas is a multi-agent platform.",
        sections=[],
        citations=[],
        confidence=0.9,
    )
    review = await critic_panel.review(brief, "What is Atlas?")
    assert isinstance(review, CriticReview)
    assert not review.revisions_required


@pytest.mark.asyncio
async def test_research_pipeline(model_client, search_tool, fetch_tool, tmp_path):
    pipeline = ResearchPipeline(
        model_client=model_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        persist_dir=str(tmp_path / "research"),
    )
    report = await pipeline.run("What is Atlas?")
    assert isinstance(report, ResearchReport)
    assert report.query == "What is Atlas?"
    assert report.brief.title
    assert report.critic_review is not None
    assert not report.critic_review.revisions_required
    assert report.paths
    assert all(Path(p).exists() for p in report.paths)
