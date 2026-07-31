"""Checkpoint storage backends for the atlascore workflow engine."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ._models import Edge, StepStatus, WorkflowExecution

logger = logging.getLogger(__name__)

T = TypeVar("T", bound="WorkflowCheckpoint")


class WorkflowCheckpoint(BaseModel):
    """Checkpoint containing workflow execution state."""

    workflow_id: str = Field(description="Workflow ID this checkpoint belongs to")
    workflow_version: str = Field(default="1.0.0", description="Workflow version")
    workflow_structure_hash: str = Field(
        description="Hash of workflow steps+edges for compatibility check"
    )

    checkpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = Field(default_factory=datetime.now)
    checkpoint_type: str = Field(default="manual", description="manual | auto | on_step | on_error")

    execution: WorkflowExecution = Field(description="Complete workflow execution state")

    completed_step_ids: List[str] = Field(default_factory=list, description="Quick lookup for completed steps")
    pending_step_ids: List[str] = Field(default_factory=list, description="Steps not yet started or still pending")

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_execution(
        cls,
        execution: WorkflowExecution,
        workflow_id: str,
        workflow_version: str,
        workflow_structure_hash: str,
        all_step_ids: List[str],
        checkpoint_type: str = "manual",
    ) -> "WorkflowCheckpoint":
        """Create a checkpoint from workflow execution state."""
        completed_step_ids = [
            step_id
            for step_id, step_exec in execution.step_executions.items()
            if step_exec.status == StepStatus.COMPLETED
        ]

        pending_step_ids = [
            step_id
            for step_id in all_step_ids
            if step_id not in execution.step_executions
            or execution.step_executions[step_id].status == StepStatus.PENDING
        ]

        return cls(
            workflow_id=workflow_id,
            workflow_version=workflow_version,
            workflow_structure_hash=workflow_structure_hash,
            checkpoint_type=checkpoint_type,
            execution=execution,
            completed_step_ids=completed_step_ids,
            pending_step_ids=pending_step_ids,
        )


class CheckpointMetadata(BaseModel):
    """Lightweight checkpoint metadata (without full execution state)."""

    checkpoint_id: str
    workflow_id: str
    workflow_version: str
    created_at: datetime
    checkpoint_type: str
    completed_steps: int
    pending_steps: int
    total_steps: int
    size_bytes: Optional[int] = None

    @classmethod
    def from_checkpoint(cls, checkpoint: WorkflowCheckpoint, size_bytes: Optional[int] = None) -> "CheckpointMetadata":
        """Extract metadata from a full checkpoint."""
        return cls(
            checkpoint_id=checkpoint.checkpoint_id,
            workflow_id=checkpoint.workflow_id,
            workflow_version=checkpoint.workflow_version,
            created_at=checkpoint.created_at,
            checkpoint_type=checkpoint.checkpoint_type,
            completed_steps=len(checkpoint.completed_step_ids),
            pending_steps=len(checkpoint.pending_step_ids),
            total_steps=len(checkpoint.completed_step_ids) + len(checkpoint.pending_step_ids),
            size_bytes=size_bytes,
        )


class CheckpointValidationResult(BaseModel):
    """Result of checkpoint validation."""

    is_valid: bool
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    can_resume: bool = Field(default=False)
    checkpoint_info: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class CheckpointStore(ABC, Generic[T]):
    """Abstract base class for checkpoint storage backends."""

    @abstractmethod
    async def save(self, checkpoint: T) -> None:
        """Save checkpoint to storage."""

    @abstractmethod
    async def load(self, checkpoint_id: str) -> Optional[T]:
        """Load checkpoint by ID."""

    @abstractmethod
    async def load_latest(self, workflow_id: str) -> Optional[T]:
        """Load the most recent checkpoint for a workflow."""

    @abstractmethod
    async def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint by ID."""

    @abstractmethod
    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        """List checkpoint metadata without loading full data."""

    @abstractmethod
    async def cleanup_old(self, workflow_id: str, keep_last_n: int = 5) -> int:
        """Remove old checkpoints, keeping only the N most recent."""


class FileCheckpointStore(CheckpointStore[WorkflowCheckpoint]):
    """File-based checkpoint storage using JSON files."""

    def __init__(self, base_path: Path) -> None:
        """Initialize the store with a root directory."""
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _get_workflow_dir(self, workflow_id: str) -> Path:
        workflow_dir = self.base_path / workflow_id
        workflow_dir.mkdir(parents=True, exist_ok=True)
        return workflow_dir

    def _get_checkpoint_path(self, workflow_id: str, checkpoint_id: str) -> Path:
        return self._get_workflow_dir(workflow_id) / f"{checkpoint_id}.json"

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        checkpoint_path = self._get_checkpoint_path(
            checkpoint.workflow_id, checkpoint.checkpoint_id
        )
        json_data = checkpoint.model_dump_json(indent=2)
        await asyncio.to_thread(checkpoint_path.write_text, json_data)
        logger.debug(f"Saved checkpoint {checkpoint.checkpoint_id} to {checkpoint_path}")

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        for workflow_dir in self.base_path.iterdir():
            if workflow_dir.is_dir():
                checkpoint_path = workflow_dir / f"{checkpoint_id}.json"
                if checkpoint_path.exists():
                    json_data = await asyncio.to_thread(checkpoint_path.read_text)
                    return WorkflowCheckpoint.model_validate_json(json_data)
        return None

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        workflow_dir = self._get_workflow_dir(workflow_id)
        checkpoint_files = list(workflow_dir.glob("*.json"))
        if not checkpoint_files:
            return None

        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        json_data = await asyncio.to_thread(checkpoint_files[0].read_text)
        return WorkflowCheckpoint.model_validate_json(json_data)

    async def delete(self, checkpoint_id: str) -> bool:
        for workflow_dir in self.base_path.iterdir():
            if workflow_dir.is_dir():
                checkpoint_path = workflow_dir / f"{checkpoint_id}.json"
                if checkpoint_path.exists():
                    await asyncio.to_thread(checkpoint_path.unlink)
                    logger.debug(f"Deleted checkpoint {checkpoint_id}")
                    return True
        return False

    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        metadata_list: List[CheckpointMetadata] = []
        search_dirs = [self._get_workflow_dir(workflow_id)] if workflow_id else [
            d for d in self.base_path.iterdir() if d.is_dir()
        ]

        for workflow_dir in search_dirs:
            for checkpoint_file in workflow_dir.glob("*.json"):
                json_data = await asyncio.to_thread(checkpoint_file.read_text)
                checkpoint = WorkflowCheckpoint.model_validate_json(json_data)
                metadata = CheckpointMetadata.from_checkpoint(
                    checkpoint, size_bytes=checkpoint_file.stat().st_size
                )
                metadata_list.append(metadata)

        metadata_list.sort(key=lambda m: m.created_at, reverse=True)
        return metadata_list[:limit]

    async def cleanup_old(self, workflow_id: str, keep_last_n: int = 5) -> int:
        workflow_dir = self._get_workflow_dir(workflow_id)
        checkpoint_files = list(workflow_dir.glob("*.json"))
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        files_to_delete = checkpoint_files[keep_last_n:]
        for file_path in files_to_delete:
            await asyncio.to_thread(file_path.unlink)

        logger.debug(f"Cleaned up {len(files_to_delete)} old checkpoints for {workflow_id}")
        return len(files_to_delete)


class InMemoryCheckpointStore(CheckpointStore[WorkflowCheckpoint]):
    """In-memory checkpoint storage (useful for testing)."""

    def __init__(self) -> None:
        self._checkpoints: Dict[str, WorkflowCheckpoint] = {}
        self._by_workflow: Dict[str, List[str]] = {}

    async def save(self, checkpoint: WorkflowCheckpoint) -> None:
        self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        if checkpoint.workflow_id not in self._by_workflow:
            self._by_workflow[checkpoint.workflow_id] = []
        self._by_workflow[checkpoint.workflow_id].append(checkpoint.checkpoint_id)

    async def load(self, checkpoint_id: str) -> Optional[WorkflowCheckpoint]:
        return self._checkpoints.get(checkpoint_id)

    async def load_latest(self, workflow_id: str) -> Optional[WorkflowCheckpoint]:
        checkpoint_ids = self._by_workflow.get(workflow_id, [])
        checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids if cid in self._checkpoints]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda c: c.created_at, reverse=True)
        return checkpoints[0]

    async def delete(self, checkpoint_id: str) -> bool:
        if checkpoint_id in self._checkpoints:
            checkpoint = self._checkpoints.pop(checkpoint_id)
            if checkpoint.workflow_id in self._by_workflow:
                self._by_workflow[checkpoint.workflow_id].remove(checkpoint_id)
            return True
        return False

    async def list_metadata(
        self,
        workflow_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[CheckpointMetadata]:
        if workflow_id:
            checkpoint_ids = self._by_workflow.get(workflow_id, [])
            checkpoints = [self._checkpoints[cid] for cid in checkpoint_ids if cid in self._checkpoints]
        else:
            checkpoints = list(self._checkpoints.values())

        metadata_list = [CheckpointMetadata.from_checkpoint(cp) for cp in checkpoints]
        metadata_list.sort(key=lambda m: m.created_at, reverse=True)
        return metadata_list[:limit]

    async def cleanup_old(self, workflow_id: str, keep_last_n: int = 5) -> int:
        checkpoint_ids = self._by_workflow.get(workflow_id, [])
        checkpoints = [
            (cid, self._checkpoints[cid])
            for cid in checkpoint_ids
            if cid in self._checkpoints
        ]
        checkpoints.sort(key=lambda x: x[1].created_at, reverse=True)

        to_delete = checkpoints[keep_last_n:]
        for checkpoint_id, _ in to_delete:
            await self.delete(checkpoint_id)

        return len(to_delete)

    def clear(self) -> None:
        """Clear all checkpoints (useful for testing)."""
        self._checkpoints.clear()
        self._by_workflow.clear()


class CheckpointConfig(BaseModel):
    """Configuration for checkpoint behavior with reasonable defaults."""

    store: CheckpointStore[WorkflowCheckpoint] = Field(
        default_factory=lambda: InMemoryCheckpointStore(),
        description="Checkpoint storage backend",
    )
    auto_save: bool = Field(default=True, description="Automatically save checkpoint after each step")
    save_interval_steps: int = Field(default=1, description="Save checkpoint every N steps")
    auto_cleanup: bool = Field(default=False, description="Automatically cleanup old checkpoints")
    keep_last_n: int = Field(default=5, description="Number of recent checkpoints to keep")

    model_config = ConfigDict(arbitrary_types_allowed=True)


def compute_workflow_structure_hash(
    steps: Dict[str, Any],
    edges: List[Edge],
    start_step_id: Optional[str],
    end_step_ids: List[str],
) -> str:
    """Compute a deterministic hash of workflow structure for checkpoint compatibility."""
    structure = {
        "steps": {
            step_id: {
                "type": step.__class__.__name__,
                "input_type": step.input_type.__name__,
                "output_type": step.output_type.__name__,
            }
            for step_id, step in sorted(steps.items())
        },
        "edges": [
            {
                "from": edge.from_step,
                "to": edge.to_step,
                "condition_type": edge.condition.type,
            }
            for edge in sorted(edges, key=lambda e: (e.from_step, e.to_step))
        ],
        "start_step": start_step_id,
        "end_steps": sorted(end_step_ids),
    }
    json_str = json.dumps(structure, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()[:16]
