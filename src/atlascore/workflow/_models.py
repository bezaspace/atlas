"""Core data models for the atlascore workflow engine."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

# Generic type variables for step inputs and outputs
InputType = TypeVar("InputType", bound=BaseModel)
OutputType = TypeVar("OutputType", bound=BaseModel)


class StepStatus(str, Enum):
    """Status of a step in workflow execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"


class WorkflowStatus(str, Enum):
    """Status of workflow execution."""

    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EdgeCondition(BaseModel):
    """Defines conditions for workflow edges."""

    type: str = Field(
        default="always",
        description="Type of condition: always, output_based, state_based",
    )
    expression: Optional[str] = Field(default=None, description="Python expression to evaluate")
    field: Optional[str] = Field(default=None, description="Field to check in output or state")
    value: Optional[Any] = Field(default=None, description="Expected value")
    operator: Optional[str] = Field(default=None, description="Comparison operator: ==, !=, >, <, in, etc.")

    model_config = ConfigDict(extra="forbid")


class Edge(BaseModel):
    """Represents a connection between workflow steps."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    from_step: str = Field(description="Source step ID")
    to_step: str = Field(description="Target step ID")
    condition: EdgeCondition = Field(default_factory=EdgeCondition)

    model_config = ConfigDict(extra="forbid")


class StepExecution(BaseModel):
    """Tracks execution details of a step."""

    step_id: str
    status: StepStatus = StepStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: int = 0

    model_config = ConfigDict(extra="forbid")


class WorkflowExecution(BaseModel):
    """Tracks execution of an entire workflow."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.CREATED
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    state: Dict[str, Any] = Field(default_factory=dict)
    step_executions: Dict[str, StepExecution] = Field(default_factory=dict)
    error: Optional[str] = None

    model_config = ConfigDict(extra="forbid")


class StepMetadata(BaseModel):
    """Metadata for workflow steps."""

    name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    max_retries: int = 0
    timeout_seconds: Optional[int] = None

    model_config = ConfigDict(extra="forbid")


class WorkflowMetadata(BaseModel):
    """Metadata for workflows."""

    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)
    author: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = ConfigDict(extra="forbid")


class Context:
    """Shared, mutable workflow state with a progress callback."""

    def __init__(self, state: Optional[Dict[str, Any]] = None) -> None:
        self.state: Dict[str, Any] = state if state is not None else {}
        self._progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None

    @classmethod
    def from_state_ref(cls, state: Dict[str, Any]) -> "Context":
        """Create a Context that directly references an existing state dict."""
        return cls(state=state)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from workflow state."""
        return self.state.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in workflow state."""
        self.state[key] = value

    def emit_progress(
        self,
        message: str,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Emit a progress update from within a workflow step."""
        if self._progress_callback:
            self._progress_callback(
                {
                    "message": message,
                    "completed": completed,
                    "total": total,
                    "metadata": metadata or {},
                }
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary that the runner can pass to step.run()."""
        return {
            "workflow_state": self.state,
            "_context_obj": self,
            **self.state,
        }


class WorkflowValidationResult(BaseModel):
    """Result of workflow validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    has_cycles: bool = False
    unreachable_steps: List[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")


class WorkflowEventType(str, Enum):
    """Types of workflow events."""

    WORKFLOW_STARTED = "workflow_started"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"
    WORKFLOW_CANCELLED = "workflow_cancelled"
    WORKFLOW_RESUMED = "workflow_resumed"
    CHECKPOINT_SAVED = "checkpoint_saved"
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    STEP_PROGRESS = "step_progress"
    EDGE_ACTIVATED = "edge_activated"


class WorkflowEvent(BaseModel):
    """Base class for workflow events."""

    event_type: WorkflowEventType
    timestamp: datetime
    workflow_id: str

    model_config = ConfigDict(extra="forbid")

    def __str__(self) -> str:
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.event_type.value}"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"event_type='{self.event_type.value}', "
            f"workflow_id='{self.workflow_id[:8]}...', "
            f"timestamp='{self.timestamp}')"
        )


class WorkflowStartedEvent(WorkflowEvent):
    """Workflow execution started."""

    event_type: WorkflowEventType = WorkflowEventType.WORKFLOW_STARTED
    initial_input: Dict[str, Any]


class WorkflowCompletedEvent(WorkflowEvent):
    """Workflow execution completed successfully."""

    event_type: WorkflowEventType = WorkflowEventType.WORKFLOW_COMPLETED
    execution: WorkflowExecution


class WorkflowFailedEvent(WorkflowEvent):
    """Workflow execution failed."""

    event_type: WorkflowEventType = WorkflowEventType.WORKFLOW_FAILED
    error: str
    execution: Optional[WorkflowExecution] = None


class WorkflowCancelledEvent(WorkflowEvent):
    """Workflow execution was cancelled."""

    event_type: WorkflowEventType = WorkflowEventType.WORKFLOW_CANCELLED
    execution: WorkflowExecution
    reason: str


class WorkflowResumedEvent(WorkflowEvent):
    """Workflow resumed from checkpoint."""

    event_type: WorkflowEventType = WorkflowEventType.WORKFLOW_RESUMED
    checkpoint_id: str
    completed_steps: List[str]
    pending_steps: List[str]


class StepStartedEvent(WorkflowEvent):
    """Step execution started."""

    event_type: WorkflowEventType = WorkflowEventType.STEP_STARTED
    step_id: str
    input_data: Dict[str, Any]


class StepCompletedEvent(WorkflowEvent):
    """Step execution completed successfully."""

    event_type: WorkflowEventType = WorkflowEventType.STEP_COMPLETED
    step_id: str
    output_data: Dict[str, Any]
    duration_seconds: float


class StepFailedEvent(WorkflowEvent):
    """Step execution failed."""

    event_type: WorkflowEventType = WorkflowEventType.STEP_FAILED
    step_id: str
    error: str
    duration_seconds: float


class StepProgressEvent(WorkflowEvent):
    """Progress update from within a step execution."""

    event_type: WorkflowEventType = WorkflowEventType.STEP_PROGRESS
    step_id: str
    message: str
    completed: Optional[int] = None
    total: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EdgeActivatedEvent(WorkflowEvent):
    """Edge between steps activated (data flowing)."""

    event_type: WorkflowEventType = WorkflowEventType.EDGE_ACTIVATED
    from_step: str
    to_step: str
    data: Dict[str, Any]


class CheckpointSavedEvent(WorkflowEvent):
    """Checkpoint saved during execution."""

    event_type: WorkflowEventType = WorkflowEventType.CHECKPOINT_SAVED
    checkpoint_id: str
    completed_steps: int
    total_steps: int
