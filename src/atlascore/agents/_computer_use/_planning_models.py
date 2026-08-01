"""Structured planning/output models for computer-use tasks."""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field

from ._interface_clients import ActionType


class InterfaceRepresentation(str, Enum):
    """How to represent the interface to the LLM."""

    TEXT = "text"
    HTML = "html"
    VISUAL = "visual"
    HYBRID = "hybrid"


class PlanningStrategy(str, Enum):
    """Planning strategy for the computer-use agent."""

    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    AUTO = "auto"


class PageObservation(BaseModel):
    """Structured observation of the current page state."""

    url: str = Field(description="Current page URL")
    title: str = Field(description="Page title")
    summary: str = Field(description="Brief summary of what's visible")
    key_elements: List[str] = Field(description="Important interactive elements or content")
    is_task_complete: bool = Field(description="Whether the task appears to be complete")
    confidence: float = Field(ge=0, le=1, description="Confidence in this observation")


class NextActionPlan(BaseModel):
    """Structured plan for the next action."""

    action_type: ActionType = Field(description="Type of action to perform")
    selector: Optional[str] = Field(default=None, description="CSS selector or visible text")
    value: Optional[str] = Field(
        default=None, description="Value to input (type/select) or URL (navigate)"
    )
    reasoning: str = Field(description="Why this action is being taken")
    expected_outcome: str = Field(description="What should happen after this action")
    confidence: float = Field(ge=0, le=1, description="Confidence in this action")


class TaskCompletion(BaseModel):
    """Assessment of whether the task is complete."""

    is_complete: bool = Field(description="Whether the task has been completed")
    completion_confidence: float = Field(ge=0, le=1, description="Confidence in completion")
    summary: str = Field(description="Summary of what was accomplished")


class DOMFilter(BaseModel):
    """Configuration for filtering DOM content."""

    max_text_length: int = Field(default=2000, description="Maximum text content length")
    include_hidden: bool = Field(default=False, description="Include hidden elements")
    interactive_only: bool = Field(default=False, description="Only include interactive elements")
    exclude_tags: List[str] = Field(
        default_factory=lambda: ["script", "style", "meta", "link"],
        description="Tags to exclude from DOM",
    )


class InterfaceConfig(BaseModel):
    """Configuration for interface representation."""

    representation: InterfaceRepresentation = Field(
        default=InterfaceRepresentation.HYBRID, description="How to represent the interface"
    )
    dom_filter: DOMFilter = Field(
        default_factory=DOMFilter, description="How to filter DOM content"
    )
    include_screenshot: bool = Field(default=True, description="Whether to include screenshots")
