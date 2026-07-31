import asyncio
import os

from atlascore import Agent, CalculatorTool, DateTimeTool, OpenAIChatCompletionClient


async def main():
    client = OpenAIChatCompletionClient(
        model=os.environ.get("LLM_MODEL", "gpt-4o-mini"),
        api_key=os.environ.get("LLM_API_KEY"),
        base_url=os.environ.get("LLM_BASE_URL"),
    )

    agent = Agent(
        name="math_assistant",
        instructions=(
            "You are a helpful assistant. When the user asks a math question, "
            "use the calculator tool. When they ask for the current time, use the datetime tool. "
            "Then summarize the result briefly."
        ),
        model_client=client,
        tools=[CalculatorTool(), DateTimeTool()],
        max_iterations=5,
    )

    queries = [
        "What is 135 * 42?",
        "What is the current UTC time?",
    ]

    for query in queries:
        print(f"\nUser: {query}")
        response = await agent.run(query)
        print(response)


if __name__ == "__main__":
    asyncio.run(main())
