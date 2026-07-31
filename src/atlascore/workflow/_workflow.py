"""Workflow DAG builder and validator for atlascore."""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel
from typing_extensions import Self

from ._checkpoint import compute_workflow_structure_hash
from ._models import (
    Edge,
    EdgeCondition,
    StepStatus,
    WorkflowExecution,
    WorkflowMetadata,
    WorkflowValidationResult,
)
from ._step import BaseStep

logger = logging.getLogger(__name__)


class Workflow:
    """A typed DAG workflow builder and validator."""

    def __init__(
        self,
        metadata: WorkflowMetadata,
        initial_state: Optional[Dict[str, Any]] = None,
        workflow_id: Optional[str] = None,
    ) -> None:
        """Initialize a workflow.

        Args:
            metadata: Workflow metadata.
            initial_state: Initial shared workflow state.
            workflow_id: Optional workflow ID; generated if not provided.
        """
        self.id = workflow_id or str(uuid.uuid4())
        self.metadata = metadata
        self.initial_state = initial_state or {}
        self.steps: Dict[str, BaseStep] = {}
        self.edges: List[Edge] = []
        self.start_step_id: Optional[str] = None
        self.end_step_ids: List[str] = []

    def add_step(self, step: BaseStep) -> Self:
        """Add a step to the workflow."""
        self.steps[step.step_id] = step
        logger.debug(f"Added step {step.step_id} to workflow {self.id}")
        return self

    def add_edge(
        self,
        from_step: Union[str, BaseStep],
        to_step: Union[str, BaseStep],
        condition: Optional[Union[EdgeCondition, Dict[str, Any]]] = None,
    ) -> Self:
        """Add an edge between steps, optionally with a condition."""
        if isinstance(from_step, BaseStep):
            self.steps[from_step.step_id] = from_step
            from_step_id = from_step.step_id
        else:
            from_step_id = str(from_step)

        if isinstance(to_step, BaseStep):
            self.steps[to_step.step_id] = to_step
            to_step_id = to_step.step_id
        else:
            to_step_id = str(to_step)

        edge_condition = (
            condition
            if isinstance(condition, EdgeCondition)
            else EdgeCondition(**condition) if condition else EdgeCondition()
        )
        edge = Edge(
            from_step=str(from_step_id),
            to_step=str(to_step_id),
            condition=edge_condition,
        )
        self.edges.append(edge)
        logger.debug(f"Added edge {from_step_id} -> {to_step_id} to workflow {self.id}")
        return self

    def set_start_step(self, step: Union[str, BaseStep]) -> Self:
        """Set the starting step for the workflow."""
        if isinstance(step, BaseStep):
            step_id = step.step_id
        else:
            step_id = str(step)
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} not found in workflow")
        self.start_step_id = step_id
        logger.debug(f"Set start step to {step_id} for workflow {self.id}")
        return self

    def add_end_step(self, step: Union[str, BaseStep]) -> Self:
        """Add an end step to the workflow."""
        if isinstance(step, BaseStep):
            step_id = step.step_id
        else:
            step_id = str(step)
        if step_id not in self.steps:
            raise ValueError(f"Step {step_id} not found in workflow")
        if step_id not in self.end_step_ids:
            self.end_step_ids.append(step_id)
        logger.debug(f"Added end step {step_id} to workflow {self.id}")
        return self

    def chain(self, *steps: BaseStep) -> Self:
        """Chain steps sequentially and configure start/end steps."""
        if len(steps) < 2:
            raise ValueError("chain() requires at least 2 steps")

        for i in range(len(steps) - 1):
            self.add_edge(steps[i], steps[i + 1])

        self.set_start_step(steps[0])
        self.add_end_step(steps[-1])
        logger.debug(f"Chained {len(steps)} steps in workflow {self.id}")
        return self

    def get_step_dependencies(self, step_id: str) -> List[str]:
        """Get all step IDs that must complete before this step can run."""
        return [edge.from_step for edge in self.edges if edge.to_step == step_id]

    def get_step_dependents(self, step_id: str) -> List[str]:
        """Get all step IDs that depend on this step."""
        return [edge.to_step for edge in self.edges if edge.from_step == step_id]

    def get_ready_steps(self, execution: WorkflowExecution) -> List[str]:
        """Get steps that are ready to run (all dependencies completed and conditions met)."""
        ready_steps: List[str] = []

        for step_id in self.steps:
            step_exec = execution.step_executions.get(step_id)

            if step_exec and step_exec.status != StepStatus.PENDING:
                continue

            incoming = [edge for edge in self.edges if edge.to_step == step_id]
            if not incoming:
                if step_id == self.start_step_id:
                    ready_steps.append(step_id)
                continue

            all_satisfied = True
            for edge in incoming:
                dep_exec = execution.step_executions.get(edge.from_step)
                if not dep_exec or dep_exec.status != StepStatus.COMPLETED:
                    all_satisfied = False
                    break
                if not self._evaluate_edge_condition(edge, execution):
                    all_satisfied = False
                    break

            if all_satisfied:
                ready_steps.append(step_id)

        return ready_steps

    def _evaluate_edge_condition(self, edge: Edge, execution: WorkflowExecution) -> bool:
        """Evaluate if an edge condition is met."""
        condition = edge.condition

        if condition.type == "always":
            return True

        if condition.type == "output_based":
            from_step_exec = execution.step_executions.get(edge.from_step)
            if not from_step_exec or not from_step_exec.output_data:
                return False
            if condition.field and condition.operator and condition.value is not None:
                field_value = from_step_exec.output_data.get(condition.field)
                return self._compare_values(field_value, condition.operator, condition.value)
            return True

        if condition.type == "state_based":
            if condition.field and condition.operator and condition.value is not None:
                field_value = execution.state.get(condition.field)
                return self._compare_values(field_value, condition.operator, condition.value)
            return True

        return True

    def _compare_values(self, left: Any, operator: str, right: Any) -> bool:
        """Compare two values using the given operator."""
        try:
            if operator == "==":
                return left == right
            if operator == "!=":
                return left != right
            if operator == ">":
                return bool(left is not None and right is not None and left > right)
            if operator == "<":
                return bool(left is not None and right is not None and left < right)
            if operator == ">=":
                return bool(left is not None and right is not None and left >= right)
            if operator == "<=":
                return bool(left is not None and right is not None and left <= right)
            if operator == "in":
                return bool(left in right)
            if operator == "not_in":
                return bool(left not in right)
        except Exception:
            return False

        logger.warning(f"Unknown operator: {operator}")
        return True

    def validate_workflow(self) -> WorkflowValidationResult:
        """Validate the workflow DAG structure."""
        result = WorkflowValidationResult(is_valid=True)

        if not self.steps:
            result.errors.append("Workflow has no steps")
            result.is_valid = False

        if not self.start_step_id:
            result.errors.append("No start step specified")
            result.is_valid = False
        elif self.start_step_id not in self.steps:
            result.errors.append(f"Start step {self.start_step_id} not found in workflow")
            result.is_valid = False

        if not self.end_step_ids:
            result.warnings.append("No end steps specified - workflow may run indefinitely")
        else:
            for end_step_id in self.end_step_ids:
                if end_step_id not in self.steps:
                    result.errors.append(f"End step {end_step_id} not found in workflow")
                    result.is_valid = False

        for edge in self.edges:
            if edge.from_step not in self.steps:
                result.errors.append(f"Edge references non-existent step: {edge.from_step}")
                result.is_valid = False
            if edge.to_step not in self.steps:
                result.errors.append(f"Edge references non-existent step: {edge.to_step}")
                result.is_valid = False

        result.has_cycles, cycle_info = self._detect_cycles()
        if result.has_cycles:
            result.errors.append(f"Workflow contains cycles: {cycle_info}")
            result.is_valid = False

        result.unreachable_steps = self._find_unreachable_steps()
        if result.unreachable_steps:
            result.warnings.append(f"Unreachable steps found: {result.unreachable_steps}")

        conditional_issues = self._validate_conditional_edges()
        result.errors.extend(conditional_issues["errors"])
        result.warnings.extend(conditional_issues["warnings"])

        # Validate type compatibility across incoming edges for each target step.
        edges_by_target: Dict[str, List[Edge]] = {}
        for edge in self.edges:
            edges_by_target.setdefault(edge.to_step, []).append(edge)

        for step_id, incoming in edges_by_target.items():
            if step_id not in self.steps:
                continue
            target_step = self.steps[step_id]
            if not (isinstance(target_step.input_type, type) and issubclass(target_step.input_type, BaseModel)):
                continue

            target_schema = target_step.input_type.model_json_schema()
            target_required = set(target_schema.get("required", []))
            target_props = target_schema.get("properties", {})

            union_fields: Dict[str, Any] = {}
            edge_messages: List[str] = []

            for edge in incoming:
                if edge.from_step not in self.steps:
                    continue
                source_step = self.steps[edge.from_step]
                if not (isinstance(source_step.output_type, type) and issubclass(source_step.output_type, BaseModel)):
                    continue

                source_schema = source_step.output_type.model_json_schema()
                source_props = source_schema.get("properties", {})

                for field_name, field_schema in source_props.items():
                    if field_name in union_fields and union_fields[field_name] != field_schema:
                        edge_messages.append(
                            f"Field '{field_name}' has conflicting types across incoming edges to step '{step_id}'"
                        )
                    else:
                        union_fields[field_name] = field_schema

            missing = target_required - set(union_fields)
            if missing:
                result.errors.append(
                    f"Step '{step_id}' input {target_step.input_type.__name__} is missing required "
                    f"field(s) {sorted(missing)} from its incoming edge outputs"
                )
                result.is_valid = False

            for field_name, field_schema in target_props.items():
                if field_name in union_fields and union_fields[field_name] != field_schema:
                    result.warnings.append(
                        f"Type warning for step '{step_id}': field '{field_name}' schema differs between "
                        f"incoming edges and {target_step.input_type.__name__}"
                    )

            result.warnings.extend(edge_messages)

        return result

    def _are_types_compatible(
        self, source: type, target: type
    ) -> tuple[bool, Optional[str]]:
        """Check whether a source Pydantic model can satisfy a target input model."""
        if not (isinstance(source, type) and issubclass(source, BaseModel)):
            return True, None
        if not (isinstance(target, type) and issubclass(target, BaseModel)):
            return True, None

        if source is target:
            return True, None
        if (
            source.__name__ == target.__name__
            and source.model_json_schema() == target.model_json_schema()
        ):
            return True, None

        source_schema = source.model_json_schema()
        target_schema = target.model_json_schema()
        source_props = source_schema.get("properties", {})
        target_props = target_schema.get("properties", {})
        target_required = set(target_schema.get("required", []))

        missing_required: List[str] = []
        type_mismatches: List[str] = []

        for field_name in target_required:
            if field_name not in source_props:
                missing_required.append(field_name)
                continue
            if source_props[field_name] != target_props.get(field_name):
                type_mismatches.append(field_name)

        if missing_required:
            return False, (
                f"Type mismatch: step output {source.__name__} is missing "
                f"required field(s) {missing_required} needed by input {target.__name__}"
            )

        if type_mismatches:
            return True, (
                f"Type warning: field(s) {type_mismatches} differ between "
                f"{source.__name__} and {target.__name__}"
            )

        return True, None

    def _detect_cycles(self) -> tuple[bool, Optional[str]]:
        """Detect cycles in the workflow graph using DFS."""
        if not self.start_step_id:
            return False, None

        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs(step_id: str, path: List[str]) -> tuple[bool, Optional[str]]:
            if step_id in rec_stack:
                cycle_start = path.index(step_id)
                cycle = " -> ".join(path[cycle_start:] + [step_id])
                return True, cycle

            if step_id in visited:
                return False, None

            visited.add(step_id)
            rec_stack.add(step_id)

            for next_step in self.get_step_dependents(step_id):
                has_cycle, cycle_info = dfs(next_step, path + [step_id])
                if has_cycle:
                    return True, cycle_info

            rec_stack.remove(step_id)
            return False, None

        return dfs(self.start_step_id, [])

    def _find_unreachable_steps(self) -> List[str]:
        """Find steps that cannot be reached from the start step."""
        if not self.start_step_id:
            return list(self.steps.keys())

        reachable: Set[str] = set()
        to_visit = [self.start_step_id]

        while to_visit:
            current = to_visit.pop()
            if current in reachable:
                continue
            reachable.add(current)
            to_visit.extend(self.get_step_dependents(current))

        return [step_id for step_id in self.steps if step_id not in reachable]

    def _validate_conditional_edges(self) -> Dict[str, List[str]]:
        """Validate conditional edge logic for common issues."""
        errors: List[str] = []
        warnings: List[str] = []

        edges_by_target: Dict[str, List[Edge]] = {}
        for edge in self.edges:
            edges_by_target.setdefault(edge.to_step, []).append(edge)

        for step_id, incoming in edges_by_target.items():
            if len(incoming) > 1:
                field_conditions: Dict[str, List[Any]] = {}
                for edge in incoming:
                    condition = edge.condition
                    if condition.type in ("output_based", "state_based") and condition.field:
                        key = f"{condition.type}:{condition.field}"
                        field_conditions.setdefault(key, []).append((edge.from_step, condition))

                for key, conditions in field_conditions.items():
                    if len(conditions) > 1:
                        true_values = [
                            c for _, c in conditions if c.operator == "==" and c.value is True
                        ]
                        false_values = [
                            c for _, c in conditions if c.operator == "==" and c.value is False
                        ]
                        if true_values and false_values:
                            from_steps = [fs for fs, _ in conditions]
                            warnings.append(
                                f"Step '{step_id}' has contradictory boolean conditions from {from_steps}"
                            )

        for step_id in self.steps:
            if step_id not in self.end_step_ids:
                outgoing = [e for e in self.edges if e.from_step == step_id]
                if not outgoing:
                    warnings.append(
                        f"Step '{step_id}' has no outgoing edges but is not marked as an end step"
                    )

        return {"errors": errors, "warnings": warnings}

    def get_execution_plan(self) -> Dict[str, Any]:
        """Get a JSON-serializable representation of the workflow plan."""
        return {
            "workflow_id": self.id,
            "metadata": self.metadata.model_dump(),
            "steps": {step_id: step.get_schema() for step_id, step in self.steps.items()},
            "edges": [edge.model_dump() for edge in self.edges],
            "start_step": self.start_step_id,
            "end_steps": self.end_step_ids,
            "validation": self.validate_workflow().model_dump(),
        }

    def compute_structure_hash(self) -> str:
        """Compute a hash of the workflow structure for checkpoint compatibility."""
        return compute_workflow_structure_hash(
            steps=self.steps,
            edges=self.edges,
            start_step_id=self.start_step_id,
            end_step_ids=self.end_step_ids,
        )
