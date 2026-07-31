"""Smoke script for atlascore orchestration patterns.

This demo runs a poet/critic round-robin on a real model. Set LLM_API_KEY and
optionally LLM_BASE_URL/LLM_MODEL in the environment before running.
"""

import asyncio
import os

from atlascore import (
    Agent,
    MaxMessageTermination,
    OpenAIChatCompletionClient,
    RoundRobinOrchestrator,
    TextMentionTermination,
)


async def main():
    client = OpenAIChatCompletionClient(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
    )

    poet = Agent(
        name="poet",
        instructions="You are a haiku poet. Write a haiku on request.",
        model_client=client,
    )
    critic = Agent(
        name="critic",
        instructions=(
            "You are a poetry critic. Provide brief feedback. "
            "If the haiku is good, respond with APPROVED."
        ),
        model_client=client,
    )

    termination = MaxMessageTermination(max_messages=6) | TextMentionTermination(
        text="APPROVED"
    )
    orchestrator = RoundRobinOrchestrator(
        agents=[poet, critic], termination=termination, max_iterations=4
    )

    result = await orchestrator.run("Write a haiku about cherry blossoms")
    print(result)
    for msg in result.messages:
        print(msg)


if __name__ == "__main__":
    asyncio.run(main())
