"""Structured research output schemas for atlascore."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    """A citation linking a claim to a source."""

    model_config = ConfigDict(frozen=True)

    source_url: Optional[str] = Field(default=None, description="URL of the source")
    source_title: Optional[str] = Field(default=None, description="Title of the source")
    quote: str = Field(..., description="Exact quote or excerpt from the source")
    index: int = Field(default=0, description="Citation index for referencing in the brief")


class Evidence(BaseModel):
    """A piece of evidence supporting or refuting a claim."""

    model_config = ConfigDict(frozen=True)

    claim: str = Field(..., description="The claim being evaluated")
    assessment: str = Field(..., description="Assessment of the claim (supported/refuted/unclear)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score between 0 and 1")
    citations: List[Citation] = Field(default_factory=list, description="Supporting citations")


class VerificationResult(BaseModel):
    """Result of fact-checking a set of claims."""

    model_config = ConfigDict(frozen=True)

    overall_confidence: float = Field(..., ge=0, le=1)
    evidence: List[Evidence] = Field(default_factory=list)


class ResearchBrief(BaseModel):
    """Final structured research brief with citations."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., description="Title of the research brief")
    summary: str = Field(..., description="Concise summary of findings")
    sections: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed sections with headings and content"
    )
    citations: List[Citation] = Field(default_factory=list, description="All citations used")
    confidence: float = Field(..., ge=0, le=1, description="Overall confidence in the brief")

    def to_markdown(self) -> str:
        """Render the brief as markdown."""
        lines = [f"# {self.title}", "", self.summary, ""]
        for section in self.sections:
            heading = section.get("heading", "Section")
            content = section.get("content", "")
            lines.extend([f"## {heading}", "", content, ""])
        if self.citations:
            lines.append("## Citations")
            for citation in self.citations:
                ref = f"[{citation.index}]"
                source = citation.source_title or citation.source_url or "unknown source"
                lines.append(f"{ref} {source}: {citation.quote}")
        lines.append(f"\nOverall confidence: {self.confidence:.0%}")
        return "\n".join(lines)


class ResearchPlan(BaseModel):
    """Plan produced by the Planner agent."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Original research question")
    sub_questions: List[str] = Field(
        default_factory=list, description="Sub-questions that need answering"
    )
    search_queries: List[str] = Field(
        default_factory=list, description="Concrete search queries to run"
    )
    required_sources: int = Field(
        default=3, ge=1, description="Minimum number of distinct sources desired"
    )
    reasoning: str = Field(default="", description="Why this plan was chosen")


class SearchResult(BaseModel):
    """A single web search result with optional fetched content."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(default="", description="Source title")
    url: str = Field(..., description="Source URL")
    snippet: str = Field(default="", description="Search snippet or short preview")
    content: Optional[str] = Field(default=None, description="Full or fetched content")
    relevance: Optional[str] = Field(
        default=None, description="relevant, partial, or irrelevant"
    )
    source_index: int = Field(default=0, description="Index for citation referencing")


class SearchOutput(BaseModel):
    """Raw evidence gathered by the Researcher."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Research query")
    evidence: List[SearchResult] = Field(
        default_factory=list, description="Collected search results and content"
    )
    summary: str = Field(default="", description="Short summary of the evidence")


class TriageClassification(BaseModel):
    """Relevance classification for a single search result."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(..., description="URL of the result")
    relevance: str = Field(
        ..., description="One of: relevant, partial, irrelevant"
    )
    reason: str = Field(default="", description="Why it was classified this way")
    confidence: float = Field(default=0.0, ge=0, le=1)


class TriageResult(BaseModel):
    """Structured output from the TriageAgent."""

    model_config = ConfigDict(frozen=True)

    classifications: List[TriageClassification] = Field(default_factory=list)

    def relevant_urls(self) -> List[str]:
        """URLs rated as relevant or partial."""
        return [
            c.url
            for c in self.classifications
            if c.relevance in ("relevant", "partial")
        ]


class CriticReview(BaseModel):
    """Structured output from the CriticPanel."""

    model_config = ConfigDict(frozen=True)

    revisions_required: bool = Field(..., description="Whether the brief needs revisions")
    feedback: str = Field(default="", description="Detailed feedback")
    suggested_changes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0, le=1)


class ResearchReport(BaseModel):
    """Final artifact from the full research pipeline."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(..., description="Research question")
    brief: ResearchBrief = Field(..., description="Final research brief")
    sources: List[SearchResult] = Field(
        default_factory=list, description="Sources used"
    )
    critic_review: Optional[CriticReview] = Field(
        default=None, description="Review from the critic panel"
    )
    paths: List[str] = Field(default_factory=list, description="Files persisted")
    approved: bool = Field(default=True, description="Whether the human approval gate approved")
    human_feedback: str = Field(default="", description="Feedback from human approval gate")
    usage: Dict[str, Any] = Field(
        default_factory=dict, description="Aggregated usage statistics"
    )
