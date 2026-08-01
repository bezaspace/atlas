"""Pydantic request/response models for the Atlas backend API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from atlascore.context import ToolApprovalResponse


class CreateSessionRequest(BaseModel):
    """Request to create a new session."""

    entity_id: str = Field(default="research_pipeline")
    entity_type: str = Field(default="research")


class RunRequest(BaseModel):
    """Request to start a research run."""

    query: str = Field(..., description="Research question")
    context: Optional[str] = Field(default=None, description="Optional prior context")
    require_human_approval: bool = Field(
        default=False,
        description="Pause before persistence and wait for human approval",
    )


class ApprovalRequest(BaseModel):
    """Request to approve pending tool calls or a human approval gate."""

    approvals: List[ToolApprovalResponse] = Field(default_factory=list)
    approved: bool = Field(default=True, description="Human approval gate decision")
    feedback: Optional[str] = Field(default=None, description="Optional feedback")


class EvalRequest(BaseModel):
    """Request to run an eval harness over a dataset."""

    dataset_path: str = Field(..., description="Path to JSONL dataset")
    max_items: int = Field(default=5, ge=1, le=50)


class HealthResponse(BaseModel):
    """Response from the /health endpoint."""

    status: str = "healthy"
    model: Optional[str] = None
    search_provider: Optional[str] = None
    embedding_provider: Optional[str] = None
    version: str = "0.1.0"


class SessionInfo(BaseModel):
    """Minimal session metadata."""

    id: str
    entity_id: str
    entity_type: str
    status: str
    created_at: str
    last_activity: str


class EvalScore(BaseModel):
    """Score for a single eval item."""

    accuracy: float = Field(..., ge=0, le=1)
    citation_coverage: float = Field(..., ge=0, le=1)
    hallucination: float = Field(..., ge=0, le=1)
    overall: float = Field(..., ge=0, le=1)
    rationale: str = ""


class EvalResult(BaseModel):
    """Result for one eval item."""

    query: str
    expected: Optional[str]
    score: EvalScore


class EvalReport(BaseModel):
    """Aggregated eval report."""

    dataset_path: str
    total: int
    scores: List[EvalResult]
    aggregate: Dict[str, float]
