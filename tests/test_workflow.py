"""Tests for the atlascore workflow engine."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from atlascore import (
    AgentContext,
    AgentResponse,
    AssistantMessage,
    Usage,
)
from atlascore.workflow import (
    AgentStep,
    CheckpointConfig,
    FileCheckpointStore,
    FunctionStep,
    InMemoryCheckpointStore,
    StepExecution,
    StepMetadata,
    StepStatus,
    Workflow,
    WorkflowExecution,
    WorkflowMetadata,
    WorkflowRunner,
    WorkflowStatus,
)


# ---------------------------------------------------------------------------
# Test data models
# ---------------------------------------------------------------------------
class NumberInput(BaseModel):
    value: int


class NumberOutput(BaseModel):
    result: int


class AOutput(BaseModel):
    a: int


class BOutput(BaseModel):
    b: int


class SumInput(BaseModel):
    a: int
    b: int


class SumOutput(BaseModel):
    result: int


class TextInput(BaseModel):
    text: str


class TextOutput(BaseModel):
    text: str


async def double_number(input_data: NumberInput, context) -> NumberOutput:
    return NumberOutput(result=input_data.value * 2)


async def add_ten(input_data: NumberOutput, context) -> NumberOutput:
    return NumberOutput(result=input_data.result + 10)


async def square(input_data: NumberOutput, context) -> NumberOutput:
    return NumberOutput(result=input_data.result ** 2)


async def make_a(input_data: NumberOutput, context) -> AOutput:
    return AOutput(a=input_data.result)


async def make_b(input_data: NumberOutput, context) -> BOutput:
    return BOutput(b=input_data.result)


async def sum_inputs(input_data: SumInput, context) -> SumOutput:
    return SumOutput(result=input_data.a + input_data.b)


async def pass_through(input_data: NumberInput, context) -> NumberOutput:
    return NumberOutput(result=input_data.value)


async def echo_text(input_data: TextInput, context) -> TextOutput:
    return TextOutput(text=input_data.text)


def _function_step(step_id: str, func, input_type, output_type):
    return FunctionStep(
        step_id=step_id,
        metadata=StepMetadata(name=step_id.replace("_", " ").title()),
        input_type=input_type,
        output_type=output_type,
        func=func,
    )


# ---------------------------------------------------------------------------
# Builder / validation tests
# ---------------------------------------------------------------------------
def test_workflow_validation_passes_for_simple_chain():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Math")).chain(
        double_step, add_step
    )

    result = workflow.validate_workflow()
    assert result.is_valid
    assert not result.has_cycles
    assert not result.unreachable_steps


def test_workflow_validation_detects_missing_start():
    workflow = Workflow(metadata=WorkflowMetadata(name="No Start"))
    step = _function_step("double", double_number, NumberInput, NumberOutput)
    workflow.add_step(step).add_end_step("double")
    result = workflow.validate_workflow()
    assert not result.is_valid
    assert "No start step" in result.errors[0]


def test_workflow_validation_detects_cycle():
    a = _function_step("a", double_number, NumberInput, NumberOutput)
    b = _function_step("b", add_ten, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Cycle"))
    workflow.add_step(a).add_step(b)
    workflow.add_edge("a", "b").add_edge("b", "a")
    workflow.set_start_step("a")
    workflow.add_end_step("b")

    result = workflow.validate_workflow()
    assert not result.is_valid
    assert result.has_cycles


def test_workflow_validation_detects_unreachable_step():
    a = _function_step("a", double_number, NumberInput, NumberOutput)
    b = _function_step("b", add_ten, NumberOutput, NumberOutput)
    c = _function_step("c", square, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Unreachable"))
    workflow.chain(a, b)
    workflow.add_step(c)
    # c is not connected

    result = workflow.validate_workflow()
    assert "c" in result.unreachable_steps
    assert any("c" in w for w in result.warnings)


def test_workflow_validation_detects_type_mismatch():
    source = _function_step("source", double_number, NumberInput, NumberOutput)
    target = _function_step("target", echo_text, TextInput, TextOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Mismatch"))
    workflow.add_step(source).add_step(target)
    workflow.add_edge("source", "target")
    workflow.set_start_step("source").add_end_step("target")

    result = workflow.validate_workflow()
    assert not result.is_valid
    assert any("text" in e.lower() for e in result.errors)


def test_workflow_structure_hash_stable():
    def build():
        double_step = _function_step("double", double_number, NumberInput, NumberOutput)
        add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)
        return Workflow(metadata=WorkflowMetadata(name="Hash")).chain(double_step, add_step)

    assert build().compute_structure_hash() == build().compute_structure_hash()
    assert len(build().compute_structure_hash()) == 16


# ---------------------------------------------------------------------------
# Runner tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_workflow_runner_chain():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)
    square_step = _function_step("square", square, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Math Pipeline")).chain(
        double_step, add_step, square_step
    )

    runner = WorkflowRunner()
    execution = await runner.run(workflow, initial_input={"value": 3})

    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.state["square_output"]["result"] == (3 * 2 + 10) ** 2


@pytest.mark.asyncio
async def test_workflow_runner_fan_in():
    start = _function_step("start", pass_through, NumberInput, NumberOutput)
    a = _function_step("a", make_a, NumberOutput, AOutput)
    b = _function_step("b", make_b, NumberOutput, BOutput)
    total = _function_step("sum", sum_inputs, SumInput, SumOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Fan In"))
    workflow.add_step(start).add_step(a).add_step(b).add_step(total)
    workflow.add_edge("start", "a")
    workflow.add_edge("start", "b")
    workflow.add_edge("a", "sum")
    workflow.add_edge("b", "sum")
    workflow.set_start_step("start").add_end_step("sum")

    runner = WorkflowRunner()
    execution = await runner.run(workflow, initial_input={"value": 5})

    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.state["sum_output"]["result"] == 5 + 5


@pytest.mark.asyncio
async def test_workflow_runner_stream_events():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Stream")).chain(double_step, add_step)
    runner = WorkflowRunner()

    events = []
    async for event in runner.run_stream(workflow, initial_input={"value": 4}):
        events.append(event)

    assert events[0].event_type == "workflow_started"
    assert events[-1].event_type == "workflow_completed"
    step_started = [e for e in events if e.event_type == "step_started"]
    step_completed = [e for e in events if e.event_type == "step_completed"]
    assert len(step_started) == 2
    assert len(step_completed) == 2


@pytest.mark.asyncio
async def test_conditional_edge_true():
    class GateOutput(BaseModel):
        proceed: bool

    async def gate(input_data: NumberInput, context) -> GateOutput:
        return GateOutput(proceed=input_data.value > 0)

    async def success(input_data: GateOutput, context) -> TextOutput:
        return TextOutput(text="yes")

    async def failure(input_data: GateOutput, context) -> TextOutput:
        return TextOutput(text="no")

    gate_step = _function_step("gate", gate, NumberInput, GateOutput)
    success_step = _function_step("success", success, GateOutput, TextOutput)
    failure_step = _function_step("failure", failure, GateOutput, TextOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Conditional"))
    workflow.add_step(gate_step).add_step(success_step).add_step(failure_step)
    workflow.add_edge(
        "gate",
        "success",
        condition={"type": "output_based", "field": "proceed", "operator": "==", "value": True},
    )
    workflow.add_edge(
        "gate",
        "failure",
        condition={"type": "output_based", "field": "proceed", "operator": "==", "value": False},
    )
    workflow.set_start_step("gate")
    workflow.add_end_step("success").add_end_step("failure")

    runner = WorkflowRunner()
    execution = await runner.run(workflow, initial_input={"value": 1})
    assert execution.state["success_output"]["text"] == "yes"

    execution = await runner.run(workflow, initial_input={"value": -1})
    assert execution.state["failure_output"]["text"] == "no"


@pytest.mark.asyncio
async def test_multi_edge_waits_for_all_dependencies():
    """A step with two incoming conditional edges should wait for both predecessors."""
    class PassOutput(BaseModel):
        text: str

    async def make_text(input_data: NumberInput, context) -> PassOutput:
        return PassOutput(text=input_data.text)

    a = _function_step("a", make_text, TextInput, PassOutput)
    b = _function_step("b", make_text, TextInput, PassOutput)
    c = _function_step("c", make_text, PassOutput, PassOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Multi-Edge"))
    workflow.add_step(a).add_step(b).add_step(c)
    workflow.add_edge(
        "a",
        "c",
        condition={"type": "output_based", "field": "text", "operator": "in", "value": "pass"},
    )
    workflow.add_edge(
        "b",
        "c",
        condition={"type": "output_based", "field": "text", "operator": "in", "value": "go"},
    )
    workflow.set_start_step("a").add_end_step("c")
    # b has no incoming edges but is not the start, so it should not run in this config

    execution = WorkflowExecution(workflow_id=workflow.id)
    execution.step_executions["a"] = StepExecution(
        step_id="a",
        status=StepStatus.COMPLETED,
        output_data={"text": "a: pass"},
    )

    ready = workflow.get_ready_steps(execution)
    assert "c" not in ready


# ---------------------------------------------------------------------------
# Checkpoint tests
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_checkpoint_save_and_resume():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)
    square_step = _function_step("square", square, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Checkpoint")).chain(
        double_step, add_step, square_step
    )

    store = InMemoryCheckpointStore()
    config = CheckpointConfig(store=store, auto_save=True, save_interval_steps=1)
    runner = WorkflowRunner()

    events = []
    async for event in runner.run_stream(
        workflow=workflow, initial_input={"value": 3}, checkpoint_config=config
    ):
        events.append(event)

    assert events[-1].event_type == "workflow_completed"
    assert events[-1].execution.state["square_output"]["result"] == (3 * 2 + 10) ** 2

    metadata = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata) == 3

    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert len(checkpoint.completed_step_ids) == 3
    assert len(checkpoint.pending_step_ids) == 0


@pytest.mark.asyncio
async def test_checkpoint_resume_from_failure():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Resume")).chain(double_step, add_step)
    runner = WorkflowRunner()
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(store=store, auto_save=True, save_interval_steps=1)

    checkpoint_saved = False
    async for event in runner.run_stream(
        workflow=workflow, initial_input={"value": 5}, checkpoint_config=config
    ):
        if event.event_type == "checkpoint_saved":
            checkpoint_saved = True
            break

    assert checkpoint_saved
    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert "double" in checkpoint.completed_step_ids

    resume_events = []
    async for event in runner.run_stream(
        workflow=workflow,
        initial_input={"value": 5},
        checkpoint=checkpoint,
        checkpoint_config=config,
    ):
        resume_events.append(event)

    assert any(e.event_type == "workflow_resumed" for e in resume_events)
    completed = [e for e in resume_events if e.event_type == "step_completed"]
    assert len(completed) == 1
    assert completed[0].step_id == "add_ten"
    assert resume_events[-1].execution.state["add_ten_output"]["result"] == 20


@pytest.mark.asyncio
async def test_file_checkpoint_store(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    store = FileCheckpointStore(base_path=checkpoint_dir)
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="File")).add_step(double_step)
    workflow.set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    config = CheckpointConfig(store=store, auto_save=True)

    async for _ in runner.run_stream(
        workflow=workflow, initial_input={"value": 7}, checkpoint_config=config
    ):
        pass

    workflow_dir = checkpoint_dir / workflow.id
    assert workflow_dir.exists()
    assert len(list(workflow_dir.glob("*.json"))) == 1

    checkpoint = await store.load_latest(workflow.id)
    assert checkpoint is not None
    assert checkpoint.workflow_id == workflow.id
    assert len(checkpoint.completed_step_ids) == 1


@pytest.mark.asyncio
async def test_checkpoint_validation_detects_structure_change():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    workflow_v1 = Workflow(metadata=WorkflowMetadata(name="V1")).add_step(double_step)
    workflow_v1.set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    store = InMemoryCheckpointStore()
    config = CheckpointConfig(store=store, auto_save=True)

    async for _ in runner.run_stream(
        workflow=workflow_v1, initial_input={"value": 3}, checkpoint_config=config
    ):
        pass

    checkpoint = await store.load_latest(workflow_v1.id)
    assert checkpoint is not None

    add_step = _function_step("add_ten", add_ten, NumberOutput, NumberOutput)
    workflow_v2 = Workflow(metadata=WorkflowMetadata(name="V2")).chain(double_step, add_step)

    validation = runner.validate_checkpoint(workflow_v2, checkpoint)
    assert not validation.can_resume
    assert any("structure" in e.lower() for e in validation.errors)


@pytest.mark.asyncio
async def test_checkpoint_cleanup():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    workflow = Workflow(metadata=WorkflowMetadata(name="Cleanup")).add_step(double_step)
    workflow.set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    store = InMemoryCheckpointStore()

    for i in range(10):
        config = CheckpointConfig(store=store, auto_save=True)
        async for _ in runner.run_stream(
            workflow=workflow, initial_input={"value": i}, checkpoint_config=config
        ):
            pass

    metadata_before = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata_before) == 10

    deleted = await store.cleanup_old(workflow_id=workflow.id, keep_last_n=3)
    assert deleted == 7

    metadata_after = await store.list_metadata(workflow_id=workflow.id)
    assert len(metadata_after) == 3


@pytest.mark.asyncio
async def test_checkpoint_with_default_config():
    double_step = _function_step("double", double_number, NumberInput, NumberOutput)
    workflow = Workflow(metadata=WorkflowMetadata(name="Default")).add_step(double_step)
    workflow.set_start_step("double").add_end_step("double")

    runner = WorkflowRunner()
    config = CheckpointConfig()

    checkpoint_events = []
    async for event in runner.run_stream(
        workflow=workflow, initial_input={"value": 4}, checkpoint_config=config
    ):
        if event.event_type == "checkpoint_saved":
            checkpoint_events.append(event)

    assert len(checkpoint_events) == 1
    assert checkpoint_events[0].completed_steps == 1


# ---------------------------------------------------------------------------
# Agent step tests
# ---------------------------------------------------------------------------
class FakeAgent:
    """Minimal stand-in for an atlascore Agent."""

    def __init__(self, name: str):
        self.name = name
        self.description = "fake"
        self.instructions = "fake"

    async def run(self, task: str) -> AgentResponse:
        return AgentResponse(
            context=AgentContext(
                messages=[AssistantMessage(content=f"answer: {task}", source="fake")]
            ),
            source="fake",
            usage=Usage(duration_ms=1),
            finish_reason="stop",
        )


@pytest.mark.asyncio
async def test_agent_step():
    step = AgentStep(
        step_id="agent",
        metadata=StepMetadata(name="Agent"),
        agent=FakeAgent("planner"),
    )

    workflow = Workflow(metadata=WorkflowMetadata(name="Agent Workflow"))
    workflow.add_step(step).set_start_step("agent").add_end_step("agent")

    runner = WorkflowRunner()
    execution = await runner.run(workflow, initial_input={"task": "hello"})

    assert execution.status == WorkflowStatus.COMPLETED
    assert execution.state["agent_output"]["response"] == "answer: hello"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_workflow_stuck_raises():
    a = _function_step("a", double_number, NumberInput, NumberOutput)
    b = _function_step("b", add_ten, NumberOutput, NumberOutput)

    workflow = Workflow(metadata=WorkflowMetadata(name="Stuck"))
    workflow.add_step(a).add_step(b)
    # a has no outgoing edge to b, so b will never be ready
    workflow.set_start_step("a").add_end_step("b")

    runner = WorkflowRunner()
    with pytest.raises(RuntimeError):
        await runner.run(workflow, initial_input={"value": 1})
