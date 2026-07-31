"""Concrete Agent implementation for atlascore."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..base_types import Usage
from ..cancellation import CancellationToken
from ..context import AgentContext
from ..llm._base import BaseChatCompletionClient
from ..messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallRequest,
    ToolMessage,
    UserMessage,
)
from ..tools._base import BaseTool, FunctionTool
from ..types import (
    AgentEvent,
    AgentResponse,
    ChatCompletionChunk,
    ModelCallEvent,
    ModelResponseEvent,
    TaskCompleteEvent,
    TaskStartEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)


class Agent:
    """A concrete agent with reasoning/action loop and streaming support."""

    def __init__(
        self,
        name: str,
        instructions: str,
        model_client: BaseChatCompletionClient,
        description: str = "",
        tools: Optional[List[Union[BaseTool, Callable]]] = None,
        context: Optional[AgentContext] = None,
        max_iterations: int = 10,
        output_format: Optional[Type[BaseModel]] = None,
    ):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model_client = model_client
        self.tools = self._process_tools(tools or [])
        self.context = context or AgentContext()
        self.max_iterations = max_iterations
        self.output_format = output_format

    def _process_tools(self, tools: List[Union[BaseTool, Callable]]) -> List[BaseTool]:
        processed = []
        for tool in tools:
            if isinstance(tool, BaseTool):
                processed.append(tool)
            elif callable(tool):
                processed.append(FunctionTool(tool))
            else:
                raise ValueError(f"Invalid tool type: {type(tool)}")
        return processed

    def _find_tool(self, name: str) -> Optional[BaseTool]:
        return next((tool for tool in self.tools if tool.name == name), None)

    def _get_tools_for_llm(self) -> List[Dict[str, Any]]:
        return [tool.to_llm_format() for tool in self.tools]

    async def run(
        self,
        task: Optional[Union[str, UserMessage, List[Message]]] = None,
        context: Optional[AgentContext] = None,
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AgentResponse:
        """Execute the agent's main loop and return the final response."""
        final_response = None
        async for item in self.run_stream(task, context, cancellation_token):
            if isinstance(item, AgentResponse):
                final_response = item
        return final_response or AgentResponse(
            source=self.name,
            finish_reason="no_response",
            usage=Usage(duration_ms=0),
        )

    async def run_stream(
        self,
        task: Optional[Union[str, UserMessage, List[Message]]] = None,
        context: Optional[AgentContext] = None,
        cancellation_token: Optional[CancellationToken] = None,
        verbose: bool = True,
        stream_tokens: bool = False,
    ) -> AsyncGenerator[Union[Message, AgentEvent, AgentResponse, ChatCompletionChunk], None]:
        """Execute the agent with streaming output."""
        start_time = time.time()
        working_context = (
            context.model_copy(deep=True) if context else self.context.model_copy(deep=True)
        )

        try:
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            if task:
                task_messages = self._convert_task_to_messages(task)
                for msg in task_messages:
                    working_context.add_message(msg)
                user_message = task_messages[0]
                yield user_message
                if verbose:
                    yield TaskStartEvent(source=self.name, task=user_message.content)

            total_usage = Usage(duration_ms=0)
            finish_reason = "stop"

            for iteration in range(self.max_iterations):
                if cancellation_token and cancellation_token.is_cancelled():
                    raise asyncio.CancelledError()

                llm_messages = await self._prepare_llm_messages(working_context)
                tools = self._get_tools_for_llm() if self.tools else None

                if verbose:
                    yield ModelCallEvent(
                        source=self.name,
                        input_messages=llm_messages,
                        model=getattr(self.model_client, "model", "unknown"),
                    )

                if stream_tokens:
                    accumulated_content, tool_calls, usage = await self._stream_llm_response(
                        llm_messages, tools, cancellation_token
                    )
                    total_usage = total_usage + usage
                    assistant_message = AssistantMessage(
                        content=accumulated_content,
                        source=self.name,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                else:
                    result = await self.model_client.create(
                        llm_messages, tools=tools, output_format=self.output_format
                    )
                    total_usage = total_usage + result.usage
                    assistant_message = result.message.model_copy(
                        update={"source": self.name, "usage": result.usage}
                    )

                working_context.add_message(assistant_message)
                yield assistant_message
                if verbose:
                    yield ModelResponseEvent(
                        source=self.name,
                        response=assistant_message.content,
                        has_tool_calls=bool(assistant_message.tool_calls),
                    )

                if not assistant_message.tool_calls:
                    finish_reason = "stop"
                    break

                # Execute tool calls sequentially and emit events.
                for tc in assistant_message.tool_calls:
                    if cancellation_token and cancellation_token.is_cancelled():
                        raise asyncio.CancelledError()

                    if verbose:
                        yield ToolCallEvent(
                            source=self.name,
                            tool_name=tc.tool_name,
                            parameters=tc.parameters,
                            call_id=tc.call_id,
                        )

                    tool_msg = await self._execute_tool_call(tc)
                    working_context.add_message(tool_msg)
                    total_usage = total_usage + Usage(duration_ms=0, tool_calls=1)
                    yield tool_msg

                    if verbose:
                        # Reconstruct ToolResult for the response event.
                        from ..base_types import ToolResult

                        result_obj = ToolResult(
                            success=tool_msg.success,
                            result=tool_msg.content,
                            error=tool_msg.error,
                            metadata=tool_msg.metadata,
                        )
                        yield ToolCallResponseEvent(
                            source=self.name,
                            call_id=tc.call_id,
                            result=result_obj,
                        )

                finish_reason = "tool_calls"
            else:
                finish_reason = "max_iterations"

            duration_ms = int((time.time() - start_time) * 1000)
            total_usage = total_usage + Usage(duration_ms=duration_ms)

            response = AgentResponse(
                context=working_context,
                source=self.name,
                usage=total_usage,
                finish_reason=finish_reason,
            )
            if verbose:
                yield TaskCompleteEvent(source=self.name, result=response.final_content)
            yield response

        except asyncio.CancelledError:
            duration_ms = int((time.time() - start_time) * 1000)
            response = AgentResponse(
                context=working_context,
                source=self.name,
                usage=Usage(duration_ms=duration_ms),
                finish_reason="cancelled",
            )
            yield response
        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)
            error_message = AssistantMessage(content=f"Error: {str(e)}", source=self.name)
            working_context.add_message(error_message)
            response = AgentResponse(
                context=working_context,
                source=self.name,
                usage=Usage(duration_ms=duration_ms),
                finish_reason="error",
            )
            yield response

    async def _execute_tool_call(self, tc: ToolCallRequest) -> ToolMessage:
        """Execute a single tool call and return a ToolMessage."""
        tool = self._find_tool(tc.tool_name)
        if tool is None:
            return ToolMessage(
                content=f"Tool '{tc.tool_name}' not found",
                source=self.name,
                tool_call_id=tc.call_id,
                tool_name=tc.tool_name,
                success=False,
                error=f"Tool '{tc.tool_name}' not found",
            )

        tool_result = await tool.execute(tc.parameters)
        return ToolMessage(
            content=str(tool_result.result) if tool_result.success else (tool_result.error or ""),
            source=tool.name,
            tool_call_id=tc.call_id,
            tool_name=tc.tool_name,
            success=tool_result.success,
            error=tool_result.error,
            metadata=tool_result.metadata,
        )

    async def _stream_llm_response(
        self,
        llm_messages: List[Message],
        tools: Optional[List[Dict[str, Any]]],
        cancellation_token: Optional[CancellationToken],
    ) -> tuple[str, Optional[List[ToolCallRequest]], Usage]:
        """Stream an LLM response and accumulate content/tool calls."""
        accumulated_content = ""
        accumulated_tool_calls: Dict[str, Dict[str, Any]] = {}
        last_call_id: Optional[str] = None
        usage = Usage(duration_ms=0, llm_calls=1)

        async for chunk in self.model_client.create_stream(
            llm_messages, tools=tools, output_format=self.output_format
        ):
            if cancellation_token and cancellation_token.is_cancelled():
                raise asyncio.CancelledError()

            if chunk.is_complete:
                if chunk.usage:
                    usage = usage + chunk.usage
                continue

            if chunk.content:
                accumulated_content += chunk.content

            if chunk.tool_call_chunk:
                call_id = chunk.tool_call_chunk.get("id")
                chunk_func = chunk.tool_call_chunk.get("function", {}) or {}
                if call_id:
                    last_call_id = call_id
                    if call_id not in accumulated_tool_calls:
                        accumulated_tool_calls[call_id] = {
                            "id": call_id,
                            "function": {"name": "", "arguments": ""},
                        }
                    existing_func = accumulated_tool_calls[call_id].get("function", {})
                    if chunk_func.get("name"):
                        existing_func["name"] = chunk_func["name"]
                    if chunk_func.get("arguments"):
                        existing_func["arguments"] = (
                            existing_func.get("arguments", "") + chunk_func["arguments"]
                        )
                    accumulated_tool_calls[call_id]["function"] = existing_func
                elif last_call_id:
                    existing = accumulated_tool_calls.get(last_call_id, {})
                    existing_func = existing.get("function", {})
                    if chunk_func.get("arguments"):
                        existing_func["arguments"] = (
                            existing_func.get("arguments", "") + chunk_func["arguments"]
                        )
                    existing["function"] = existing_func
                    accumulated_tool_calls[last_call_id] = existing

        tool_calls = []
        import json

        for call_id, tc_data in accumulated_tool_calls.items():
            func = tc_data.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "")
            if name and arguments:
                try:
                    params = json.loads(arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append(
                    ToolCallRequest(tool_name=name, parameters=params, call_id=call_id)
                )

        return accumulated_content, tool_calls if tool_calls else None, usage

    async def _prepare_llm_messages(self, working_context: AgentContext) -> List[Message]:
        """Prepare messages for the LLM call including system instructions and history."""
        messages: List[Message] = [SystemMessage(content=self.instructions, source="system")]
        messages.extend(working_context.messages)
        return messages

    def _convert_task_to_messages(
        self, task: Union[str, UserMessage, List[Message]]
    ) -> List[Message]:
        if isinstance(task, str):
            return [UserMessage(content=task, source="user")]
        if isinstance(task, UserMessage):
            return [task]
        if isinstance(task, list):
            return task
        raise ValueError(f"Unsupported task type: {type(task)}")

    async def reset(self) -> None:
        """Reset the agent context."""
        self.context.reset()

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "model": getattr(self.model_client, "model", "unknown"),
            "tools_count": len(self.tools),
            "message_history_length": self.context.message_count,
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
