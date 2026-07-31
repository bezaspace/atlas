"""Tests for atlascore orchestration patterns."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from atlascore import (
    Agent,
    AssistantMessage,
    ChatCompletionResult,
    Usage,
)
from atlascore.cancellation import CancellationToken
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import Message
from atlascore.orchestration import (
    AgentSelection,
    AIOrchestrator,
    ExecutionPlan,
    PlanBasedOrchestrator,
    PlanStep,
    RoundRobinOrchestrator,
    StepProgressEvaluation,
)
from atlascore.termination import MaxMessageTermination, TextMentionTermination
from atlascore.types import OrchestrationResponse


def _assistant_result(content: str, structured: Any = None) -> ChatCompletionResult:
    return ChatCompletionResult(
        message=AssistantMessage(content=content, source="llm"),
        usage=Usage(duration_ms=1, llm_calls=1, tokens_input=5, tokens_output=3),
        model="fake",
        finish_reason="stop",
        structured_output=structured,
    )


class FakeResponseClient(BaseChatCompletionClient):
    """A fake model client that always returns the same response."""

    def __init__(self, response: ChatCompletionResult):
        super().__init__("fake")
        self.response = response

    async def create(
        self,
        messages: list[Message],
        tools: Any = None,
        output_format: Any = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        return self.response

    async def create_stream(
        self,
        messages: list[Message],
        tools: Any = None,
        output_format: Any = None,
        **kwargs: Any,
    ):
        raise NotImplementedError


class FakeListClient(BaseChatCompletionClient):
    """A fake model client that returns queued responses in order."""

    def __init__(self, responses: list[ChatCompletionResult]):
        super().__init__("fake")
        self.responses = list(responses)
        self.index = 0

    async def create(
        self,
        messages: list[Message],
        tools: Any = None,
        output_format: Any = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        if self.index >= len(self.responses):
            raise RuntimeError("FakeListClient ran out of queued responses")
        response = self.responses[self.index]
        self.index += 1
        return response

    async def create_stream(
        self,
        messages: list[Message],
        tools: Any = None,
        output_format: Any = None,
        **kwargs: Any,
    ):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_round_robin_orchestrator_stops_on_text_mention():
    poet = Agent(
        name="poet",
        instructions="You are a haiku poet.",
        model_client=FakeResponseClient(
            _assistant_result(
                "Cherry blossoms bloom,\nPetals fall in springtime breeze,\nNature's short-lived art."
            )
        ),
    )
    critic = Agent(
        name="critic",
        instructions="You are a poetry critic. Respond with APPROVED if satisfied.",
        model_client=FakeResponseClient(_assistant_result("APPROVED")),
    )

    termination = MaxMessageTermination(max_messages=6) | TextMentionTermination(
        text="APPROVED"
    )
    orchestrator = RoundRobinOrchestrator(
        agents=[poet, critic], termination=termination, max_iterations=4
    )

    result = await orchestrator.run("Write a haiku about cherry blossoms")

    assert isinstance(result, OrchestrationResponse)
    assert "APPROVED" in result.stop_message.content
    assert "APPROVED" in result.final_result
    assert len(result.messages) == 3  # user, poet, critic

    assistant_contents = [
        m.content for m in result.messages if getattr(m, "role", None) == "assistant"
    ]
    assert any("blossoms" in content for content in assistant_contents)


@pytest.mark.asyncio
async def test_ai_orchestrator_selects_agents():
    writer = Agent(
        name="writer",
        instructions="You are a creative writer.",
        model_client=FakeResponseClient(
            _assistant_result("Remote work boosts focus and eliminates commutes.")
        ),
    )
    editor = Agent(
        name="editor",
        instructions="You are an editor. Respond with APPROVED if satisfied.",
        model_client=FakeResponseClient(_assistant_result("APPROVED")),
    )

    selections = [
        AgentSelection(selected_agent="writer", reasoning="Need a draft", confidence=0.9),
        AgentSelection(selected_agent="editor", reasoning="Review the draft", confidence=0.9),
    ]
    model_client = FakeListClient(
        [
            _assistant_result(sel.selected_agent, structured=sel)
            for sel in selections
        ]
    )

    termination = MaxMessageTermination(max_messages=5) | TextMentionTermination(
        text="APPROVED"
    )
    orchestrator = AIOrchestrator(
        agents=[writer, editor],
        termination=termination,
        model_client=model_client,
        max_iterations=4,
    )

    result = await orchestrator.run("Write a note about remote work")

    assert isinstance(result, OrchestrationResponse)
    assert "APPROVED" in result.stop_message.content
    assert model_client.index == 2
    assert result.pattern_metadata.get("unique_agents_selected") == 2


@pytest.mark.asyncio
async def test_plan_based_orchestrator_executes_steps():
    researcher = Agent(
        name="researcher",
        instructions="You are a research specialist.",
        model_client=FakeResponseClient(
            _assistant_result("Solar panels convert sunlight into electricity efficiently.")
        ),
    )
    writer = Agent(
        name="writer",
        instructions="You are a technical writer.",
        model_client=FakeResponseClient(
            _assistant_result("Renewable energy guide final brief.")
        ),
    )

    plan = ExecutionPlan(
        steps=[
            PlanStep(
                task="Research renewable energy benefits",
                agent_name="researcher",
                reasoning="Researcher gathers facts",
            ),
            PlanStep(
                task="Write a guide",
                agent_name="writer",
                reasoning="Writer synthesizes research",
            ),
        ]
    )
    eval_pass = StepProgressEvaluation(
        step_completed=True,
        failure_reason="None",
        confidence_score=0.9,
        suggested_improvements=[],
    )

    model_client = FakeListClient(
        [
            _assistant_result("plan", structured=plan),
            _assistant_result("eval1", structured=eval_pass),
            _assistant_result("eval2", structured=eval_pass),
        ]
    )

    termination = MaxMessageTermination(max_messages=3)
    orchestrator = PlanBasedOrchestrator(
        agents=[researcher, writer],
        termination=termination,
        model_client=model_client,
        max_iterations=6,
    )

    result = await orchestrator.run("Research and write about renewable energy")

    assert isinstance(result, OrchestrationResponse)
    assert result.stop_message.source == "MaxMessageTermination"
    assert model_client.index == 3
    assert result.pattern_metadata.get("steps_completed") == 2
    assert any(
        "guide" in m.content
        for m in result.messages
        if getattr(m, "role", None) == "assistant"
    )


@pytest.mark.asyncio
async def test_orchestrator_stream_emits_events():
    agent = Agent(
        name="echo",
        instructions="Echo the user.",
        model_client=FakeResponseClient(_assistant_result("hello back")),
    )
    termination = MaxMessageTermination(max_messages=2)
    orchestrator = RoundRobinOrchestrator(
        agents=[agent], termination=termination, max_iterations=2
    )

    events = []
    async for item in orchestrator.run_stream("hello"):
        events.append(item)

    assert events[0].event_type == "orchestration_start"
    assert events[-2].event_type == "orchestration_complete"
    assert any(e.event_type == "agent_selection" for e in events if hasattr(e, "event_type"))

    response = events[-1]
    assert isinstance(response, OrchestrationResponse)
    assert response.stop_message.source == "MaxMessageTermination"


@pytest.mark.asyncio
async def test_orchestrator_cancellation():
    class SlowFakeClient(BaseChatCompletionClient):
        async def create(
            self,
            messages: list[Message],
            tools: Any = None,
            output_format: Any = None,
            **kwargs: Any,
        ) -> ChatCompletionResult:
            import asyncio

            await asyncio.sleep(10)
            return _assistant_result("never")

        async def create_stream(
            self,
            messages: list[Message],
            tools: Any = None,
            output_format: Any = None,
            **kwargs: Any,
        ):
            raise NotImplementedError

    agent = Agent(
        name="slow", instructions="Slow agent.", model_client=SlowFakeClient("slow")
    )
    token = CancellationToken()
    token.cancel()

    orchestrator = RoundRobinOrchestrator(
        agents=[agent],
        termination=MaxMessageTermination(max_messages=2),
        max_iterations=2,
    )

    with pytest.raises(asyncio.CancelledError):
        await orchestrator.run("hello", cancellation_token=token)


def test_orchestrator_requires_unique_agent_names():
    a = Agent(
        name="agent",
        instructions="First.",
        model_client=FakeResponseClient(_assistant_result("first")),
    )
    b = Agent(
        name="agent",
        instructions="Second.",
        model_client=FakeResponseClient(_assistant_result("second")),
    )

    with pytest.raises(ValueError, match="unique"):
        RoundRobinOrchestrator(
            agents=[a, b], termination=MaxMessageTermination(max_messages=2)
        )
