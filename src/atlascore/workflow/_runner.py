"""Workflow runner with streaming events and checkpointing for atlascore."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, Optional, Set

from ..cancellation import CancellationToken
from ._checkpoint import CheckpointConfig, CheckpointValidationResult, WorkflowCheckpoint
from ._models import (
    CheckpointSavedEvent,
    Context,
    EdgeActivatedEvent,
    StepCompletedEvent,
    StepExecution,
    StepFailedEvent,
    StepProgressEvent,
    StepStartedEvent,
    StepStatus,
    WorkflowCancelledEvent,
    WorkflowCompletedEvent,
    WorkflowEvent,
    WorkflowExecution,
    WorkflowFailedEvent,
    WorkflowResumedEvent,
    WorkflowStartedEvent,
    WorkflowStatus,
)
from ._step import BaseStep
from ._workflow import Workflow

logger = logging.getLogger(__name__)


class WorkflowRunner:
    """Executes workflows with support for parallel steps, streaming, and checkpointing."""

    def __init__(self, max_concurrent_steps: int = 5) -> None:
        self.max_concurrent_steps = max_concurrent_steps
        self._execution_semaphore = asyncio.Semaphore(max_concurrent_steps)
        self._cancellation_tokens: Dict[str, CancellationToken] = {}

    async def run(
        self,
        workflow: Workflow,
        initial_input: Optional[Dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> WorkflowExecution:
        """Run a workflow and return the final execution state."""
        final_execution: Optional[WorkflowExecution] = None
        async for event in self.run_stream(workflow, initial_input, cancellation_token):
            if event.event_type.value == "workflow_completed":
                final_execution = getattr(event, "execution", None)
            elif event.event_type.value == "workflow_failed":
                final_execution = getattr(event, "execution", None)
                error = getattr(event, "error", "Unknown workflow error")
                raise RuntimeError(error)
            elif event.event_type.value == "workflow_cancelled":
                final_execution = getattr(event, "execution", None)
                reason = getattr(event, "reason", "Workflow cancelled")
                raise RuntimeError(reason)

        if final_execution is None:
            raise RuntimeError("Workflow completed but no final execution received")

        return final_execution

    async def run_stream(
        self,
        workflow: Workflow,
        initial_input: Optional[Dict[str, Any]] = None,
        cancellation_token: Optional[CancellationToken] = None,
        checkpoint: Optional[WorkflowCheckpoint] = None,
        checkpoint_config: Optional[CheckpointConfig] = None,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Run a workflow and yield real-time events."""
        logger.info(f"Starting workflow execution: {workflow.id}")

        if cancellation_token:
            self._cancellation_tokens[workflow.id] = cancellation_token

        if checkpoint:
            validation = self.validate_checkpoint(workflow, checkpoint)
            if not validation.can_resume:
                error_msg = f"Checkpoint validation failed: {validation.errors}"
                logger.error(error_msg)
                yield WorkflowFailedEvent(
                    timestamp=datetime.now(), workflow_id=workflow.id, error=error_msg
                )
                return

            if validation.warnings:
                logger.warning(f"Checkpoint warnings: {validation.warnings}")

            execution = checkpoint.execution
            execution.status = WorkflowStatus.RUNNING
            execution.end_time = None

            logger.info(
                f"Resuming from checkpoint: "
                f"{len(checkpoint.completed_step_ids)} steps completed, "
                f"{len(checkpoint.pending_step_ids)} pending"
            )

            yield WorkflowResumedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                checkpoint_id=checkpoint.checkpoint_id,
                completed_steps=checkpoint.completed_step_ids,
                pending_steps=checkpoint.pending_step_ids,
            )
        else:
            yield WorkflowStartedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                initial_input=initial_input or {},
            )

            validation = workflow.validate_workflow()
            if not validation.is_valid:
                error_msg = f"Workflow validation failed: {validation.errors}"
                logger.error(error_msg)
                yield WorkflowFailedEvent(
                    timestamp=datetime.now(), workflow_id=workflow.id, error=error_msg
                )
                return

            if initial_input and workflow.start_step_id:
                start_step = workflow.steps.get(workflow.start_step_id)
                if start_step:
                    try:
                        start_step.input_type(**initial_input)
                    except Exception as e:
                        error_msg = (
                            f"Initial input validation failed: Input does not match start step "
                            f"'{workflow.start_step_id}' input type {start_step.input_type.__name__}: {e}"
                        )
                        logger.error(error_msg)
                        yield WorkflowFailedEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            error=error_msg,
                        )
                        return

            execution = WorkflowExecution(
                workflow_id=workflow.id,
                status=WorkflowStatus.RUNNING,
                start_time=datetime.now(),
                state=workflow.initial_state.copy(),
            )
            if initial_input:
                execution.state.update(initial_input)

        config = checkpoint_config or CheckpointConfig()

        try:
            async for event in self._execute_workflow_stream(
                workflow=workflow,
                execution=execution,
                initial_input=initial_input or {},
                checkpoint_config=config,
            ):
                yield event

            if execution.status == WorkflowStatus.CANCELLED:
                return

            if execution.status == WorkflowStatus.COMPLETED:
                if execution.end_time is None:
                    execution.end_time = datetime.now()
                logger.info(f"Workflow {workflow.id} completed successfully")

                yield WorkflowCompletedEvent(
                    timestamp=datetime.now(),
                    workflow_id=workflow.id,
                    execution=execution,
                )
            elif all(
                step_exec.status == StepStatus.COMPLETED
                for step_exec in execution.step_executions.values()
            ) and len(execution.step_executions) == len(workflow.steps):
                execution.status = WorkflowStatus.COMPLETED
                execution.end_time = datetime.now()
                logger.info(f"Workflow {workflow.id} completed successfully")

                yield WorkflowCompletedEvent(
                    timestamp=datetime.now(),
                    workflow_id=workflow.id,
                    execution=execution,
                )
            else:
                execution.status = WorkflowStatus.FAILED
                execution.end_time = datetime.now()
                error_msg = f"Workflow {workflow.id} did not complete all steps"
                logger.error(error_msg)

                yield WorkflowFailedEvent(
                    timestamp=datetime.now(),
                    workflow_id=workflow.id,
                    error=error_msg,
                    execution=execution,
                )

        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            logger.error(f"Workflow {workflow.id} failed with error: {e}")

            yield WorkflowFailedEvent(
                timestamp=datetime.now(),
                workflow_id=workflow.id,
                error=str(e),
                execution=execution,
            )
        finally:
            self._cancellation_tokens.pop(workflow.id, None)

    async def _execute_workflow_stream(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        initial_input: Dict[str, Any],
        checkpoint_config: CheckpointConfig,
    ) -> AsyncGenerator[WorkflowEvent, None]:
        """Execute workflow steps and yield events."""
        completed_steps: Set[str] = {
            step_id
            for step_id, step_exec in execution.step_executions.items()
            if step_exec.status == StepStatus.COMPLETED
        }

        if completed_steps:
            logger.info(f"Skipping {len(completed_steps)} completed steps: {completed_steps}")

        running_tasks: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        steps_since_last_checkpoint = 0
        progress_queue: asyncio.Queue[tuple[str, Dict[str, Any]]] = asyncio.Queue()

        while len(completed_steps) < len(workflow.steps):
            cancellation_token = self._cancellation_tokens.get(workflow.id)
            if cancellation_token and cancellation_token.is_cancelled():
                for step_id, task in list(running_tasks.items()):
                    step_execution = execution.step_executions.get(step_id)
                    if step_execution and step_execution.status == StepStatus.RUNNING:
                        step_execution.status = StepStatus.CANCELLED
                        step_execution.end_time = datetime.now()
                        step_execution.error = "Step cancelled due to workflow cancellation"
                        yield StepFailedEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            step_id=step_id,
                            error="Step cancelled due to workflow cancellation",
                            duration_seconds=0.0,
                        )
                    task.cancel()

                if not running_tasks:
                    execution.status = WorkflowStatus.CANCELLED
                    execution.end_time = datetime.now()
                    yield WorkflowCancelledEvent(
                        timestamp=datetime.now(),
                        workflow_id=workflow.id,
                        execution=execution,
                        reason="Cancelled by user",
                    )
                    return

            ready_steps = workflow.get_ready_steps(execution)
            ready_steps = [
                s for s in ready_steps if s not in completed_steps and s not in running_tasks
            ]

            if not ready_steps and not running_tasks:
                remaining_steps = set(workflow.steps.keys()) - completed_steps
                if remaining_steps:
                    error_msg = f"Workflow stuck: remaining steps {remaining_steps} cannot be executed"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                break

            cancellation_token = self._cancellation_tokens.get(workflow.id)
            if not (cancellation_token and cancellation_token.is_cancelled()):
                for step_id in ready_steps:
                    if len(running_tasks) >= self.max_concurrent_steps:
                        break

                    step = workflow.steps[step_id]
                    input_data = self._prepare_step_input(
                        step_id, workflow, execution, initial_input
                    )

                    step_execution = StepExecution(
                        step_id=step_id,
                        status=StepStatus.RUNNING,
                        start_time=datetime.now(),
                        input_data=input_data,
                    )
                    execution.step_executions[step_id] = step_execution

                    yield StepStartedEvent(
                        timestamp=datetime.now(),
                        workflow_id=workflow.id,
                        step_id=step_id,
                        input_data=input_data,
                    )

                    task = asyncio.create_task(
                        self._run_step_with_semaphore(
                            step,
                            step_id,
                            input_data,
                            execution.state,
                            progress_queue,
                            workflow.id,
                            cancellation_token,
                        )
                    )
                    if cancellation_token:
                        cancellation_token.link_future(task)
                    running_tasks[step_id] = task
                    logger.info(f"Started step {step_id} in workflow {workflow.id}")

            if running_tasks:
                done, _ = await asyncio.wait(
                    set(running_tasks.values()),
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=0.01,
                )

                while not progress_queue.empty():
                    try:
                        step_id_prog, progress_data = progress_queue.get_nowait()
                        yield StepProgressEvent(
                            timestamp=datetime.now(),
                            workflow_id=workflow.id,
                            step_id=step_id_prog,
                            message=progress_data["message"],
                            completed=progress_data.get("completed"),
                            total=progress_data.get("total"),
                            metadata=progress_data.get("metadata", {}),
                        )
                    except asyncio.QueueEmpty:
                        break

                if not done:
                    continue

                for task in done:
                    step_id: Optional[str] = None
                    for sid, t in running_tasks.items():
                        if t is task:
                            step_id = sid
                            break

                    if step_id:
                        step_execution = execution.step_executions[step_id]

                        try:
                            result = await task
                            step_execution.status = StepStatus.COMPLETED
                            step_execution.output_data = result
                            step_execution.end_time = datetime.now()

                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            execution.state[f"{step_id}_output"] = result
                            completed_steps.add(step_id)
                            logger.info(f"Step {step_id} completed successfully")

                            yield StepCompletedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                output_data=result,
                                duration_seconds=duration,
                            )

                            for edge in workflow.edges:
                                if edge.from_step == step_id:
                                    yield EdgeActivatedEvent(
                                        timestamp=datetime.now(),
                                        workflow_id=workflow.id,
                                        from_step=step_id,
                                        to_step=edge.to_step,
                                        data=result,
                                    )

                            steps_since_last_checkpoint += 1
                            if checkpoint_config and checkpoint_config.auto_save:
                                if steps_since_last_checkpoint >= checkpoint_config.save_interval_steps:
                                    checkpoint = self._create_checkpoint(
                                        workflow=workflow,
                                        execution=execution,
                                        checkpoint_type="auto",
                                    )
                                    await checkpoint_config.store.save(checkpoint)
                                    yield CheckpointSavedEvent(
                                        timestamp=datetime.now(),
                                        workflow_id=workflow.id,
                                        checkpoint_id=checkpoint.checkpoint_id,
                                        completed_steps=len(completed_steps),
                                        total_steps=len(workflow.steps),
                                    )
                                    if checkpoint_config.auto_cleanup:
                                        await checkpoint_config.store.cleanup_old(
                                            workflow_id=workflow.id,
                                            keep_last_n=checkpoint_config.keep_last_n,
                                        )
                                    steps_since_last_checkpoint = 0

                            if step_id in workflow.end_step_ids:
                                logger.info(f"Reached end step {step_id} in workflow {workflow.id}")
                                execution.status = WorkflowStatus.COMPLETED
                                execution.end_time = datetime.now()
                                return

                        except asyncio.CancelledError:
                            step_execution.status = StepStatus.CANCELLED
                            step_execution.error = "Step was cancelled"
                            step_execution.end_time = datetime.now()
                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            logger.info(f"Step {step_id} was cancelled")
                            yield StepFailedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                error="Step was cancelled",
                                duration_seconds=duration,
                            )

                        except Exception as e:
                            step_execution.status = StepStatus.FAILED
                            step_execution.error = str(e)
                            step_execution.end_time = datetime.now()
                            duration = 0.0
                            if step_execution.end_time and step_execution.start_time:
                                duration = (
                                    step_execution.end_time - step_execution.start_time
                                ).total_seconds()

                            logger.error(f"Step {step_id} failed: {e}")
                            yield StepFailedEvent(
                                timestamp=datetime.now(),
                                workflow_id=workflow.id,
                                step_id=step_id,
                                error=str(e),
                                duration_seconds=duration,
                            )
                            raise

                        finally:
                            running_tasks.pop(step_id, None)

    async def _run_step_with_semaphore(
        self,
        step: "BaseStep[Any, Any]",
        step_id: str,
        input_data: Dict[str, Any],
        workflow_state: Dict[str, Any],
        progress_queue: asyncio.Queue[tuple[str, Dict[str, Any]]],
        workflow_id: str,
        cancellation_token: Optional[CancellationToken],
    ) -> Dict[str, Any]:
        """Run a single step with concurrency control and progress tracking."""
        async with self._execution_semaphore:
            typed_context = Context.from_state_ref(workflow_state)

            def progress_callback(progress_data: Dict[str, Any]) -> None:
                try:
                    progress_queue.put_nowait((step_id, progress_data))
                except asyncio.QueueFull:
                    logger.warning(f"Progress queue full, dropping progress update for step {step_id}")

            typed_context._progress_callback = progress_callback
            context = typed_context.to_dict()
            context["cancellation_token"] = cancellation_token
            return await step.run(input_data, context)

    def _prepare_step_input(
        self,
        step_id: str,
        workflow: Workflow,
        execution: WorkflowExecution,
        initial_input: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare input data for a step from its dependencies."""
        if step_id == workflow.start_step_id:
            return initial_input.copy()

        incoming = [edge for edge in workflow.edges if edge.to_step == step_id]
        if not incoming:
            return initial_input.copy()

        merged: Dict[str, Any] = {}
        for edge in incoming:
            dep_exec = execution.step_executions.get(edge.from_step)
            if dep_exec and dep_exec.output_data:
                merged.update(dep_exec.output_data)

        return merged

    def _create_checkpoint(
        self,
        workflow: Workflow,
        execution: WorkflowExecution,
        checkpoint_type: str = "manual",
    ) -> WorkflowCheckpoint:
        """Create a checkpoint from current execution state."""
        return WorkflowCheckpoint.from_execution(
            execution=execution,
            workflow_id=workflow.id,
            workflow_version=workflow.metadata.version,
            workflow_structure_hash=workflow.compute_structure_hash(),
            all_step_ids=list(workflow.steps.keys()),
            checkpoint_type=checkpoint_type,
        )

    async def run_step(
        self,
        step: "BaseStep[Any, Any]",
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run a single step independently."""
        return await step.run(input_data, context or {})

    def get_execution_status(self, execution: WorkflowExecution) -> Dict[str, Any]:
        """Get detailed status of a workflow execution."""
        total_steps = len(execution.step_executions)
        completed_steps = sum(
            1 for step_exec in execution.step_executions.values() if step_exec.status == StepStatus.COMPLETED
        )
        failed_steps = sum(
            1 for step_exec in execution.step_executions.values() if step_exec.status == StepStatus.FAILED
        )
        running_steps = sum(
            1 for step_exec in execution.step_executions.values() if step_exec.status == StepStatus.RUNNING
        )

        duration = None
        if execution.start_time and execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()

        return {
            "execution_id": execution.id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "progress": {
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "running_steps": running_steps,
                "percentage": (completed_steps / total_steps * 100) if total_steps > 0 else 0,
            },
            "timing": {
                "start_time": execution.start_time,
                "end_time": execution.end_time,
                "duration_seconds": duration,
            },
            "error": execution.error,
        }

    def validate_checkpoint(
        self,
        workflow: Workflow,
        checkpoint: WorkflowCheckpoint,
    ) -> CheckpointValidationResult:
        """Validate whether a checkpoint can be resumed with the given workflow."""
        result = CheckpointValidationResult(is_valid=True, can_resume=True)

        if checkpoint.workflow_id != workflow.id:
            result.warnings.append(
                f"Checkpoint workflow_id '{checkpoint.workflow_id}' differs from "
                f"current workflow '{workflow.id}'. OK if you renamed the workflow."
            )

        current_hash = workflow.compute_structure_hash()
        if checkpoint.workflow_structure_hash != current_hash:
            result.errors.append(
                f"Workflow structure has changed since checkpoint was created. Cannot safely resume. "
                f"Checkpoint hash: {checkpoint.workflow_structure_hash}, Current hash: {current_hash}"
            )
            result.is_valid = False
            result.can_resume = False
            return result

        for step_id in checkpoint.completed_step_ids:
            if step_id not in workflow.steps:
                result.errors.append(
                    f"Checkpoint references completed step '{step_id}' that no longer exists in workflow"
                )
                result.is_valid = False
                result.can_resume = False

        result.checkpoint_info = {
            "created_at": checkpoint.created_at.isoformat(),
            "completed_steps": len(checkpoint.completed_step_ids),
            "pending_steps": len(checkpoint.pending_step_ids),
            "checkpoint_type": checkpoint.checkpoint_type,
        }

        return result

    async def cancel_workflow(self, workflow_id: str, reason: str = "Cancelled by user") -> bool:
        """Cancel a running workflow."""
        cancellation_token = self._cancellation_tokens.get(workflow_id)
        if cancellation_token:
            cancellation_token.cancel()
            logger.info(f"Workflow {workflow_id} cancellation requested: {reason}")
            return True

        logger.warning(f"Workflow {workflow_id} not found or already completed")
        return False
