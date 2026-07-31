"""Base orchestrator implementation.

Provides the foundational BaseOrchestrator class following the patterns in
victordibia/designing-multiagent-systems (PicoAgents) Chapter 7.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from typing import Any, Dict, List, Optional, Sequence, Union

from ..agents import Agent
from ..cancellation import CancellationToken
from ..context import AgentContext
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import (
    AgentExecutionCompleteEvent,
    AgentExecutionStartEvent,
    AgentResponse,
    AgentSelectionEvent,
    OrchestrationCompleteEvent,
    OrchestrationEvent,
    OrchestrationResponse,
    OrchestrationStartEvent,
    StopMessage,
    Usage,
)


class BaseOrchestrator(ABC):
    """Abstract base class for all orchestration patterns."""

    def __init__(
        self,
        agents: Sequence[Agent],
        termination: BaseTermination,
        max_iterations: int = 50,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        if not agents:
            raise ValueError("At least one agent is required")

        self.agents = list(agents)
        self.termination = termination
        self.max_iterations = max_iterations
        self.name = name or self.__class__.__name__
        self.description = description or ""

        # Runtime state
        self.shared_messages: List[Message] = []
        self.iteration_count = 0
        self.start_time: Optional[float] = None

        names = [agent.name for agent in agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")

    async def run(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> OrchestrationResponse:
        """Execute the orchestration pattern and return the final response."""
        self._reset_for_run()

        final_result: Optional[OrchestrationResponse] = None
        try:
            async for item in self.run_stream(task, cancellation_token=cancellation_token):
                if isinstance(item, OrchestrationResponse):
                    final_result = item
            return final_result or self._create_fallback_result("No result produced")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            return self._create_fallback_result(f"Orchestration failed: {e}")

    async def run_stream(
        self,
        task: Union[str, UserMessage, List[Message]],
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = True,
    ) -> AsyncGenerator[Union[Message, OrchestrationEvent, OrchestrationResponse], None]:
        """Execute orchestration with streaming output."""
        self._reset_for_run()
        self.start_time = time.time()

        stop_message: Optional[StopMessage] = None
        streamed_messages: List[Message] = []
        agent_usage_stats: List[Usage] = []

        try:
            if verbose:
                yield OrchestrationStartEvent(
                    source="orchestrator",
                    task=str(task),
                    pattern=self.__class__.__name__,
                )

            initial_messages = self._normalize_task_to_messages(task)
            self.shared_messages.extend(initial_messages)
            for message in initial_messages:
                yield message
                streamed_messages.append(message)

            # Prime termination state with the initial user messages.
            self.termination.check(initial_messages)
            last_termination_check_count = len(streamed_messages)

            while self.iteration_count < self.max_iterations:
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()

                if self.iteration_count > 0 and self.termination.is_met():
                    stop_message = StopMessage(
                        content=self.termination.get_reason(),
                        source=self.termination.__class__.__name__,
                        metadata=self.termination.get_metadata(),
                    )
                    break

                next_agent = await self.select_next_agent()

                if verbose:
                    yield AgentSelectionEvent(
                        source="orchestrator",
                        selected_agent=next_agent.name,
                        selection_reason=f"Iteration {self.iteration_count + 1}",
                    )

                prepared_context = await self.prepare_context_for_agent(next_agent)
                sent_context = self._normalize_task_to_messages(prepared_context)
                sent_context_ids = {id(m) for m in sent_context}

                if verbose:
                    yield AgentExecutionStartEvent(
                        source="orchestrator",
                        executing_agent=next_agent.name,
                        context_size=len(sent_context),
                    )

                agent_messages: List[Message] = []
                result: Optional[AgentResponse] = None

                try:
                    async for item in next_agent.run_stream(
                        task=sent_context,
                        cancellation_token=cancellation_token,
                        verbose=verbose,
                        stream_tokens=False,
                    ):
                        if isinstance(item, Message):
                            if not isinstance(item, UserMessage) and id(item) not in sent_context_ids:
                                yield item
                                streamed_messages.append(item)
                            agent_messages.append(item)
                        elif isinstance(item, AgentResponse):
                            result = item
                except asyncio.CancelledError:
                    if verbose:
                        yield AgentExecutionCompleteEvent(
                            source="orchestrator",
                            executing_agent=next_agent.name,
                            success=False,
                            message_count=len(agent_messages),
                        )
                    raise

                if result is None:
                    # Agent did not emit a final response; synthesize one from observed messages.
                    result = AgentResponse(
                        context=AgentContext(messages=agent_messages),
                        source=next_agent.name,
                        usage=Usage(duration_ms=0, llm_calls=0),
                        finish_reason="completed_without_response",
                    )

                if verbose:
                    yield AgentExecutionCompleteEvent(
                        source="orchestrator",
                        executing_agent=next_agent.name,
                        success=True,
                        message_count=len(result.messages),
                    )

                agent_usage_stats.append(result.usage)

                new_messages = self._extract_new_messages(result.messages, sent_context)
                await self.update_shared_state(result, new_messages)

                new_streamed_messages = streamed_messages[last_termination_check_count:]
                last_termination_check_count = len(streamed_messages)
                stop_message = self.termination.check(new_streamed_messages)
                if stop_message:
                    break

                self.iteration_count += 1

            if self.iteration_count >= self.max_iterations and stop_message is None:
                stop_message = StopMessage(
                    content=f"Maximum iterations reached ({self.max_iterations})",
                    source="MaxIterations",
                )

            if stop_message is None:
                stop_message = StopMessage(
                    content="Orchestration completed normally",
                    source="Completion",
                )

            final_result = self._generate_final_result()
            if verbose:
                yield OrchestrationCompleteEvent(
                    source="orchestrator",
                    result=final_result,
                    stop_reason=stop_message.content,
                )

            elapsed_time = int((time.time() - self.start_time) * 1000)
            total_usage = Usage(duration_ms=elapsed_time)
            for agent_usage in agent_usage_stats:
                total_usage = total_usage + agent_usage

            yield OrchestrationResponse(
                messages=self.shared_messages,
                final_result=final_result,
                usage=total_usage,
                stop_message=stop_message,
                pattern_metadata=self._get_pattern_metadata(),
            )

        except asyncio.CancelledError:
            elapsed_time = int((time.time() - (self.start_time or time.time())) * 1000)

            if verbose:
                yield OrchestrationCompleteEvent(
                    source="orchestrator",
                    result="Orchestration cancelled",
                    stop_reason="Cancellation",
                )

            total_usage = Usage(duration_ms=elapsed_time)
            for agent_usage in agent_usage_stats:
                total_usage = total_usage + agent_usage

            yield OrchestrationResponse(
                messages=self.shared_messages,
                final_result="Orchestration was cancelled",
                usage=total_usage,
                stop_message=StopMessage(
                    content="Orchestration cancelled", source="CancellationToken"
                ),
                pattern_metadata=self._get_pattern_metadata(),
            )
            raise

    @abstractmethod
    async def select_next_agent(self) -> Agent:
        """Pattern-specific agent selection logic."""
        pass

    @abstractmethod
    async def prepare_context_for_agent(
        self, agent: Agent
    ) -> Union[str, UserMessage, List[Message]]:
        """Pattern-specific context preparation."""
        pass

    async def update_shared_state(
        self, result: AgentResponse, new_messages: List[Message]
    ) -> None:
        """Default shared-state update: append new messages to the shared history."""
        self.shared_messages.extend(new_messages)

    def _normalize_task_to_messages(
        self, task: Union[str, UserMessage, List[Message]]
    ) -> List[Message]:
        """Convert a task argument into a list of messages."""
        if isinstance(task, str):
            return [UserMessage(content=task, source="user")]
        if isinstance(task, UserMessage):
            return [task]
        if isinstance(task, list):
            return list(task)
        if hasattr(task, "content") and hasattr(task, "source"):
            return [task]  # type: ignore[return-value]
        return [UserMessage(content=str(task), source="user")]

    def _extract_new_messages(
        self, agent_messages: List[Message], sent_context: List[Message]
    ) -> List[Message]:
        """Return only messages produced beyond the context we sent to the agent."""
        context_len = len(sent_context)
        if len(agent_messages) >= context_len:
            return agent_messages[context_len:]
        return []

    def _reset_for_run(self) -> None:
        """Reset orchestrator state for a new run."""
        self.shared_messages = []
        self.iteration_count = 0
        self.start_time = None
        self.termination.reset()

    def _generate_final_result(self) -> str:
        """Generate a final result summary from shared messages."""
        if not self.shared_messages:
            return "No messages generated"

        for message in reversed(self.shared_messages):
            if hasattr(message, "role") and getattr(message, "role") == "assistant":
                return message.content

        return "Task completed"

    def get_agent_capabilities_summary(self) -> str:
        """Build a summary of agent capabilities for LLM consumption."""
        summary_lines = []
        for agent in self.agents:
            line = f"- {agent.name}: {agent.description}"

            if hasattr(agent, "tools") and agent.tools:
                tool_names = []
                for tool in agent.tools:
                    if hasattr(tool, "name"):
                        tool_names.append(tool.name)
                    elif hasattr(tool, "__name__"):
                        tool_names.append(tool.__name__)
                    else:
                        tool_names.append(str(tool)[:20])

                if tool_names:
                    line += f" | Tools: {', '.join(tool_names)}"

            summary_lines.append(line)

        return "\n".join(summary_lines)

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Pattern-agnostic metadata for the final response."""
        return {
            "pattern": self.__class__.__name__,
            "iterations_completed": self.iteration_count,
            "agents_count": len(self.agents),
            "message_count": len(self.shared_messages),
        }

    def _create_fallback_result(self, reason: str) -> OrchestrationResponse:
        """Create a fallback result for error cases."""
        elapsed_time = int((time.time() - (self.start_time or time.time())) * 1000)

        return OrchestrationResponse(
            messages=self.shared_messages,
            final_result=reason,
            usage=Usage(duration_ms=elapsed_time),
            stop_message=StopMessage(content=reason, source="Fallback"),
            pattern_metadata=self._get_pattern_metadata(),
        )
