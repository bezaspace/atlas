import json

from atlascore.base_types import Usage
from atlascore.messages import (
    AssistantMessage,
    SystemMessage,
    ToolCallRequest,
    ToolMessage,
    UserMessage,
)


def test_message_creation():
    system = SystemMessage(content="You are a helpful assistant", source="system")
    user = UserMessage(content="hello", source="user")
    assert system.role == "system"
    assert user.role == "user"


def test_assistant_message_with_tool_call():
    tc = ToolCallRequest(tool_name="calculator", parameters={"expression": "2+2"}, call_id="call_1")
    msg = AssistantMessage(content="", source="assistant", tool_calls=[tc])
    assert msg.tool_calls[0].tool_name == "calculator"
    data = msg.model_dump()
    assert data["tool_calls"][0]["parameters"]["expression"] == "2+2"


def test_tool_message():
    tm = ToolMessage(
        content="4",
        source="calculator",
        tool_call_id="call_1",
        tool_name="calculator",
        success=True,
    )
    assert tm.role == "tool"
    assert tm.success is True


def test_usage_serialization():
    usage = Usage(duration_ms=100, llm_calls=1, tokens_input=10, tokens_output=5)
    dumped = json.loads(usage.model_dump_json())
    assert dumped["duration_ms"] == 100
    assert dumped["tokens_input"] == 10


def test_usage_addition():
    u1 = Usage(duration_ms=100, llm_calls=1, tokens_input=10, tokens_output=5)
    u2 = Usage(duration_ms=50, llm_calls=1, tokens_input=20, tokens_output=10)
    total = u1 + u2
    assert total.duration_ms == 100
    assert total.llm_calls == 2
    assert total.tokens_input == 30
    assert total.tokens_output == 15
