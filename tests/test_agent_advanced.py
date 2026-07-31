import pytest

from atlascore import Agent, ListMemory, MaxMessageTermination, MemoryContent
from atlascore.base_types import Usage
from atlascore.llm._base import BaseChatCompletionClient
from atlascore.messages import AssistantMessage
from atlascore.types import ChatCompletionResult


class CapturingClient(BaseChatCompletionClient):
    """A fake client that captures every call's messages."""

    def __init__(self):
        super().__init__("fake")
        self.captured_messages = []

    async def create(self, messages, tools=None, output_format=None, **kwargs):
        self.captured_messages.append(messages)
        return ChatCompletionResult(
            message=AssistantMessage(content="ok", source="llm"),
            usage=Usage(duration_ms=10, llm_calls=1),
            model="fake",
            finish_reason="stop",
        )

    async def create_stream(self, messages, tools=None, output_format=None, **kwargs):
        raise NotImplementedError


@pytest.mark.asyncio
async def test_agent_memory_includes_context():
    memory = ListMemory(max_memories=10)
    await memory.add(MemoryContent(content="User name is Alice."))

    client = CapturingClient()
    agent = Agent(
        name="memory_agent",
        instructions="You are helpful.",
        model_client=client,
        memory=memory,
    )

    await agent.run("Greet me.")
    assert client.captured_messages
    system_message = client.captured_messages[0][0]
    assert "Alice" in system_message.content


@pytest.mark.asyncio
async def test_agent_termination_stops():
    client = CapturingClient()
    agent = Agent(
        name="term_agent",
        instructions="You are helpful.",
        model_client=client,
        termination=MaxMessageTermination(max_messages=2),
    )

    response = await agent.run("hi")
    assert response.finish_reason == "termination"
