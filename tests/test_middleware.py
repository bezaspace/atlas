import pytest

from atlascore import Agent
from atlascore.base_types import ToolResult, Usage
from atlascore.llm._base import BaseChatCompletionClient
from atlascore.messages import AssistantMessage, ToolCallRequest
from atlascore.middleware import ApprovalMiddleware, LoggingMiddleware, MiddlewareChain
from atlascore.tools import ApprovalMode, BaseTool
from atlascore.types import ChatCompletionResult, ToolApprovalEvent


class FakeTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="fake_tool",
            description="does nothing",
            approval_mode=ApprovalMode.ALWAYS,
        )

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, parameters):
        return ToolResult(success=True, result="ok")


class FakeMiddlewareClient(BaseChatCompletionClient):
    def __init__(self):
        super().__init__("fake")
        self.calls = 0

    async def create(self, messages, tools=None, output_format=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return ChatCompletionResult(
                message=AssistantMessage(
                    content="",
                    source="llm",
                    tool_calls=[
                        ToolCallRequest(tool_name="fake_tool", parameters={}, call_id="c1")
                    ],
                ),
                usage=Usage(duration_ms=10, llm_calls=1),
                model="fake",
                finish_reason="tool_calls",
            )
        return ChatCompletionResult(
            message=AssistantMessage(content="done", source="llm"),
            usage=Usage(duration_ms=10, llm_calls=1),
            model="fake",
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_middleware_chain_logs():
    chain = MiddlewareChain([LoggingMiddleware()])
    async for item in chain.execute_stream(
        operation="model_call",
        agent_name="agent",
        agent_context=__import__("atlascore").AgentContext(),
        data=[],
        func=lambda data: "result",
    ):
        assert item == "result"


@pytest.mark.asyncio
async def test_approval_middleware_pauses():
    tool = FakeTool()
    approval = ApprovalMiddleware(tools=[tool])
    chain = MiddlewareChain([approval])
    tc = ToolCallRequest(tool_name="fake_tool", parameters={}, call_id="c1")

    events = []
    async for item in chain.execute_stream(
        operation="tool_call",
        agent_name="agent",
        agent_context=__import__("atlascore").AgentContext(),
        data=tc,
        func=lambda data: ToolResult(success=True, result="ok"),
    ):
        events.append(item)

    assert len(events) == 1
    assert isinstance(events[0], ToolApprovalEvent)


@pytest.mark.asyncio
async def test_approval_middleware_denied():
    from atlascore import AgentContext

    tool = FakeTool()
    context = AgentContext()
    approval = ApprovalMiddleware(tools=[tool])
    chain = MiddlewareChain([approval])
    tc = ToolCallRequest(tool_name="fake_tool", parameters={}, call_id="c1")

    request = context.add_approval_request(tc, "fake_tool")
    context.add_approval_response(request.create_response(approved=False))

    events = []
    async for item in chain.execute_stream(
        operation="tool_call",
        agent_name="agent",
        agent_context=context,
        data=tc,
        func=lambda data: data if isinstance(data, ToolResult) else ToolResult(success=True, result="ok"),
    ):
        events.append(item)

    assert len(events) == 1
    assert isinstance(events[0], ToolResult)
    assert events[0].success is False


@pytest.mark.asyncio
async def test_agent_approval_flow():
    client = FakeMiddlewareClient()
    tool = FakeTool()
    context = __import__("atlascore").AgentContext()
    agent = Agent(
        name="approval_agent",
        instructions="use tool",
        model_client=client,
        tools=[tool],
        middlewares=[ApprovalMiddleware(tools=[tool])],
        max_iterations=2,
    )

    response = await agent.run("call tool", context=context)
    assert response.finish_reason == "approval_needed"
    assert response.context.waiting_for_approval

    request = response.context.pending_approval_requests[0]
    response.context.add_approval_response(request.create_response(approved=True))
    response2 = await agent.run(context=response.context)
    assert response2.finish_reason == "stop"
    assert response2.final_content == "done"
