import pytest

from atlascore import (
    MaxMessageTermination,
    TextMentionTermination,
    TimeoutTermination,
    TokenUsageTermination,
)
from atlascore.messages import AssistantMessage, UserMessage


@pytest.mark.asyncio
async def test_max_message_termination():
    term = MaxMessageTermination(max_messages=2)
    assert term.check([UserMessage(content="hi", source="user")]) is None
    assert term.check([AssistantMessage(content="hello", source="agent")]) is not None
    assert term.is_met()


@pytest.mark.asyncio
async def test_text_mention_termination():
    term = TextMentionTermination(text="DONE")
    assert term.check([AssistantMessage(content="working", source="agent")]) is None
    result = term.check([AssistantMessage(content="I am DONE", source="agent")])
    assert result is not None
    assert "DONE" in result.content


@pytest.mark.asyncio
async def test_token_usage_termination():
    term = TokenUsageTermination(max_tokens=10)
    assert term.check([UserMessage(content="hi", source="user")]) is None
    result = term.check([AssistantMessage(content="1234567890123456789012345678901234567890", source="agent")])
    assert result is not None
    assert term.total_tokens > 0


@pytest.mark.asyncio
async def test_timeout_termination():
    import time

    term = TimeoutTermination(max_duration_seconds=0.01)
    time.sleep(0.02)
    result = term.check([UserMessage(content="hi", source="user")])
    assert result is not None
    assert "Timeout" in result.content


@pytest.mark.asyncio
async def test_composite_termination_or():
    term = MaxMessageTermination(max_messages=10) | TextMentionTermination(text="stop")
    assert term.check([AssistantMessage(content="stop", source="agent")]) is not None
    assert term.is_met()


@pytest.mark.asyncio
async def test_composite_termination_and():
    term = MaxMessageTermination(max_messages=1) & TextMentionTermination(text="stop")
    assert term.check([AssistantMessage(content="go", source="agent")]) is None
    assert term.check([AssistantMessage(content="stop", source="agent")]) is not None
