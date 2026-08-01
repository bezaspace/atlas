"""Tests for Phase 8 — FastAPI backend, sessions, SSE, approval, and eval."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional, Type

import pytest
from httpx import ASGITransport, AsyncClient
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
from backend.execution import EngineConfig
from backend.main import app, get_engine_config


class FakeBackendModelClient(BaseChatCompletionClient):
    """Model client returning deterministic structured outputs for backend tests."""

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
            return ChatCompletionResult(
                message=AssistantMessage(content=output.model_dump_json(), source="assistant"),
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
    def __init__(self) -> None:
        super().__init__(name="fake_search", description="Canned search.")

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
            result=[{"title": "Source", "url": "https://example.com", "snippet": "..."}],
            metadata={"query": parameters.get("query", "")},
        )


class FakeFetchTool(BaseTool):
    def __init__(self) -> None:
        super().__init__(name="fake_fetch", description="Canned fetch.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}, "output_format": {"type": "string"}},
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        return ToolResult(success=True, result="Content.", metadata={"url": parameters.get("url", "")})


class EvalScore(BaseModel):
    accuracy: float = 0.8
    citation_coverage: float = 0.8
    hallucination: float = 0.9
    overall: float = 0.8
    rationale: str = "Good."


def _make_outputs() -> Dict[Type[BaseModel], BaseModel]:
    return {
        ResearchPlan: ResearchPlan(
            query="What is Atlas?",
            sub_questions=["What is Atlas?"],
            search_queries=["Atlas multi-agent platform"],
            required_sources=1,
            reasoning="Direct search.",
        ),
        TriageResult: TriageResult(
            classifications=[
                TriageClassification(
                    url="https://example.com",
                    relevance="relevant",
                    reason="Relevant.",
                    confidence=0.9,
                )
            ]
        ),
        SearchOutput: SearchOutput(
            query="What is Atlas?",
            evidence=[
                SearchResult(
                    title="Source",
                    url="https://example.com",
                    snippet="...",
                    content="Content.",
                    relevance="relevant",
                    source_index=0,
                )
            ],
            summary="Summary.",
        ),
        VerificationResult: VerificationResult(
            overall_confidence=0.9,
            evidence=[
                Evidence(
                    claim="Atlas is a platform",
                    assessment="supported",
                    confidence=0.9,
                    citations=[
                        Citation(
                            source_url="https://example.com",
                            source_title="Source",
                            quote="...",
                            index=0,
                        )
                    ],
                )
            ],
        ),
        ResearchBrief: ResearchBrief(
            title="Atlas",
            summary="Atlas is a multi-agent platform.",
            sections=[{"heading": "Overview", "content": "Atlas is a multi-agent platform."}],
            citations=[
                Citation(
                    source_url="https://example.com",
                    source_title="Source",
                    quote="...",
                    index=0,
                )
            ],
            confidence=0.9,
        ),
        CriticReview: CriticReview(
            revisions_required=False,
            feedback="Good.",
            suggested_changes=[],
            confidence=0.9,
        ),
        EvalScore: EvalScore(),
    }


@pytest.fixture
def engine_config(tmp_path):
    model_client = FakeBackendModelClient(outputs=_make_outputs())
    return EngineConfig(
        model_client=model_client,
        search_tool=FakeSearchTool(),
        fetch_tool=FakeFetchTool(),
        persist_dir=str(tmp_path / "research"),
    )


@pytest.fixture
def test_client(engine_config):
    app.dependency_overrides[get_engine_config] = lambda: engine_config
    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health(test_client):
    response = await test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "triage_model" in data


@pytest.mark.asyncio
async def test_create_and_list_sessions(test_client):
    response = await test_client.post("/sessions", json={"entity_id": "research_pipeline"})
    assert response.status_code == 200
    session_id = response.json()["id"]

    response = await test_client.get("/sessions")
    assert response.status_code == 200
    sessions = response.json()
    assert any(s["id"] == session_id for s in sessions)


@pytest.mark.asyncio
async def test_run_and_stream_research(test_client, tmp_path):
    response = await test_client.post("/sessions", json={"entity_id": "research_pipeline"})
    session_id = response.json()["id"]

    response = await test_client.post(
        f"/sessions/{session_id}/run",
        json={"query": "What is Atlas?"},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "running"

    events = []
    async with test_client.stream(
        "GET", f"/sessions/{session_id}/stream"
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append(payload)
                if payload["event"].get("event_type") == "workflow_completed":
                    break

    assert events
    workflow_events = [e for e in events if e["event"].get("event_type")]
    assert any(e["event"]["event_type"] == "workflow_completed" for e in workflow_events)


@pytest.mark.asyncio
async def test_approve_resumes_paused_research(test_client, tmp_path):
    response = await test_client.post("/sessions", json={"entity_id": "research_pipeline"})
    session_id = response.json()["id"]

    response = await test_client.post(
        f"/sessions/{session_id}/run",
        json={"query": "What is Atlas?", "require_human_approval": True},
    )
    assert response.status_code == 200

    async def approve_after_delay():
        await asyncio.sleep(0.3)
        await test_client.post(
            f"/sessions/{session_id}/approve",
            json={"approved": True, "approvals": []},
        )

    approve_task = asyncio.create_task(approve_after_delay())

    events = []
    async with test_client.stream(
        "GET", f"/sessions/{session_id}/stream"
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append(payload)
                if payload["event"].get("event_type") == "workflow_completed":
                    break

    await approve_task
    assert any(e["event"].get("event_type") == "workflow_completed" for e in events)


@pytest.mark.asyncio
async def test_eval_endpoint(test_client, tmp_path):
    dataset_path = tmp_path / "eval.jsonl"
    dataset_path.write_text(
        json.dumps({"query": "What is Atlas?", "expected": "A multi-agent platform."}) + "\n",
        encoding="utf-8",
    )

    response = await test_client.post("/eval", json={"dataset_path": str(dataset_path), "max_items": 1})
    assert response.status_code == 200
    session_id = response.json()["session_id"]

    events = []
    async with test_client.stream(
        "GET", f"/sessions/{session_id}/stream"
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                events.append(payload)
                if payload["event"].get("type") == "eval_complete":
                    break

    assert any(e["event"].get("type") == "eval_complete" for e in events)
    complete = [e for e in events if e["event"].get("type") == "eval_complete"][0]
    assert complete["event"]["report"]["total"] == 1
