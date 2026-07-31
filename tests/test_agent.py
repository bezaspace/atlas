import pytest

from atlascore import Agent
from atlascore.base_types import Usage
from atlascore.cancellation import CancellationToken
from atlascore.context import AgentContext
from atlascore.llm import BaseChatCompletionClient
from atlascore.messages import AssistantMessage, ToolCallRequest
from atlascore.tools import CalculatorTool
from atlascore.types import ChatCompletionResult, ToolCallEvent, ToolCallResponseEvent


class FakeClient(BaseChatCompletionClient):
    """A deterministic model client for unit tests."""

    def __init__(self):
        super().__init__("fake")
        self.calls = []

    async def create(self, messages, tools=None, output_format=None, **kwargs):
        self.calls.append([m.content[:30] for m in messages])
        if len(self.calls) == 1:
            return ChatCompletionResult(
                message=AssistantMessage(
                    content="",
                    source="llm",
                    tool_calls=[
                        ToolCallRequest(
                            tool_name="calculator",
                            parameters={"expression": "2+2"},
                            call_id="call_1",
                        )
                    ],
                ),
                usage=Usage(duration_ms=10, llm_calls=1, tokens_input=10, tokens_output=5),
                model="fake",
                finish_reason="tool_calls",
            )
        return ChatCompletionResult(
            message=AssistantMessage(content="The answer is 4.", source="llm"),
            usage=Usage(duration_ms=10, llm_calls=1, tokens_input=20, tokens_output=3),
            model="fake",
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        # Not used by default in agent tests
        raise NotImplementedError("Streaming not implemented in fake client")


@pytest.fixture
def calculator_agent():
    client = FakeClient()
    agent = Agent(
        name="math_agent",
        instructions="You are a math assistant. Use the calculator tool when needed.",
        model_client=client,
        tools=[CalculatorTool()],
        max_iterations=5,
    )
    return agent, client


@pytest.mark.asyncio
async def test_agent_run_with_tool(calculator_agent):
    agent, client = calculator_agent
    response = await agent.run("What is 2 + 2?")
    assert response.finish_reason == "stop"
    assert "4" in response.final_content
    assert len(client.calls) == 2
    assert response.usage.tool_calls >= 1
    assert response.usage.llm_calls == 2


@pytest.mark.asyncio
async def test_agent_run_stream_emits_events(calculator_agent):
    agent, client = calculator_agent
    events = []
    async for item in agent.run_stream("What is 2 + 2?"):
        events.append(item)

    response = events[-1]
    assert isinstance(response, ChatCompletionResult) is False
    from atlascore.types import AgentResponse

    assert isinstance(response, AgentResponse)
    assert response.finish_reason == "stop"

    tool_call_events = [e for e in events if isinstance(e, ToolCallEvent)]
    tool_response_events = [e for e in events if isinstance(e, ToolCallResponseEvent)]
    assert len(tool_call_events) == 1
    assert len(tool_response_events) == 1
    assert tool_call_events[0].tool_name == "calculator"


@pytest.mark.asyncio
async def test_agent_cancellation():
    class SlowFakeClient(BaseChatCompletionClient):
        def __init__(self):
            super().__init__("fake")

        async def create(self, messages, tools=None, output_format=None, **kwargs):
            import asyncio

            await asyncio.sleep(10)
            return ChatCompletionResult(
                message=AssistantMessage(content="done", source="llm"),
                usage=Usage(duration_ms=0),
                model="fake",
                finish_reason="stop",
            )

        async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
            raise NotImplementedError

    agent = Agent(
        name="slow_agent",
        instructions="wait",
        model_client=SlowFakeClient(),
        max_iterations=1,
    )
    token = CancellationToken()
    token.cancel()
    response = await agent.run("hello", cancellation_token=token)
    assert response.finish_reason == "cancelled"


@pytest.mark.asyncio
async def test_agent_context_isolation(calculator_agent):
    agent, client = calculator_agent
    ctx1 = AgentContext()
    ctx2 = AgentContext()
    await agent.run("What is 2 + 2?", context=ctx1)
    await agent.run("What is 3 + 3?", context=ctx2)
    # The agent's own context should be empty because run copies context
    assert agent.context.message_count == 0
