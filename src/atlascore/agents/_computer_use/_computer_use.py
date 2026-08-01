"""Vision-driven computer-use agent for browser automation."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any, List, Optional, Union

from ...cancellation import CancellationToken
from ...context import AgentContext
from ...llm import BaseChatCompletionClient
from ...messages import Message, MultiModalMessage, ToolMessage, UserMessage
from ...types import AgentEvent, AgentResponse, ChatCompletionChunk
from .._agent import Agent
from ._interface_clients import BaseInterfaceClient, PlaywrightWebClient
from ._playwright_tools import BaseTool, create_playwright_tools


class ComputerUseAgent(Agent):
    """Browser automation agent that sends screenshots to a vision-capable LLM."""

    def __init__(
        self,
        model_client: BaseChatCompletionClient,
        interface_client: Optional[BaseInterfaceClient] = None,
        name: str = "computer_use",
        description: str = "Agent that operates a web browser using vision and tool calling",
        start_url: str = "about:blank",
        headless: bool = True,
        use_screenshots: bool = True,
        max_actions: int = 20,
        **kwargs: Any,
    ) -> None:
        self.interface_client = interface_client or PlaywrightWebClient(
            start_url=start_url, headless=headless
        )
        self.use_screenshots = use_screenshots
        self.is_initialized = False
        self._pending_screenshot: Optional[MultiModalMessage] = None

        tools: List[BaseTool] = create_playwright_tools(self.interface_client)
        instructions = kwargs.pop("instructions", None) or self._default_instructions(tools)

        super().__init__(
            name=name,
            description=description,
            instructions=instructions,
            model_client=model_client,
            tools=tools,  # type: ignore[arg-type]
            max_iterations=max_actions,
            **kwargs,
        )

    def _default_instructions(self, tools: List[Any]) -> str:
        tool_list = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
        return f"""You are an efficient computer-use agent that controls a web browser.

GOAL:
- Complete the user's task using the fewest browser actions possible.
- After each action, ask: "Do I already have the information needed to answer?"
- If YES, provide the final answer and stop.
- If NO, choose the single most useful next action.

AVAILABLE TOOLS:
{tool_list}

GUIDELINES:
- Use `navigate` to open a URL.
- Use `observe_page` to understand the current page (you will receive a screenshot).
- Use `click`, `type`, `scroll`, and `press` to interact with the page.
- Prefer visible text or simple CSS selectors when clicking or typing.
- If a page is long, scroll to find the answer.
- When you have the answer, reply with a concise final summary and stop.

EFFICIENCY:
- Don't click or scroll more than necessary.
- If the answer is visible after `observe_page`, state it immediately."""

    async def run_stream(
        self,
        task: Optional[Union[str, UserMessage, List[Message]]] = None,
        context: Optional[AgentContext] = None,
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = False,
        stream_tokens: bool = False,
    ) -> AsyncGenerator[Union[Message, AgentEvent, AgentResponse, ChatCompletionChunk], None]:
        if not self.is_initialized:
            await self.interface_client.initialize()
            self.is_initialized = True

        if self.use_screenshots:
            try:
                initial_state = await self.interface_client.get_state("hybrid")
                if initial_state.screenshot:
                    msg = MultiModalMessage(
                        content=f"Initial page - URL: {initial_state.url or 'N/A'}",
                        source=self.name,
                        role="user",
                        mime_type="image/png",
                        data=initial_state.screenshot,
                    )
                    self._pending_screenshot = msg
                    yield msg
            except Exception:
                pass

        async for item in super().run_stream(
            task=task,
            context=context,
            cancellation_token=cancellation_token,
            verbose=verbose,
            stream_tokens=stream_tokens,
        ):
            yield item

            if isinstance(item, ToolMessage) and self.use_screenshots:
                try:
                    state = await self.interface_client.get_state("hybrid")
                    if state.screenshot:
                        caption = f"After {item.tool_name} - URL: {state.url or 'N/A'}"
                        msg = MultiModalMessage(
                            content=caption,
                            source=self.name,
                            role="user",
                            mime_type="image/png",
                            data=state.screenshot,
                        )
                        yield msg
                        if item.tool_name == "observe_page":
                            self._pending_screenshot = msg
                except Exception:
                    pass

    async def _prepare_llm_messages(self, working_context: AgentContext) -> List[Message]:
        if self._pending_screenshot:
            working_context.add_message(self._pending_screenshot)
            self._pending_screenshot = None
        return await super()._prepare_llm_messages(working_context)

    async def close(self) -> None:
        """Close the browser and reset initialization state."""
        if self.is_initialized:
            await self.interface_client.close()
            self.is_initialized = False

    async def reset(self) -> None:
        await self.close()
        await super().reset()

    async def __aenter__(self) -> "ComputerUseAgent":
        if not self.is_initialized:
            await self.interface_client.initialize()
            self.is_initialized = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        await self.close()
        return False
