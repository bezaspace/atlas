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
