"""Concrete Agent implementation for atlascore."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Callable, Dict, List, Optional, Type, Union

from pydantic import BaseModel

from ..base_types import ToolResult, Usage
from ..cancellation import CancellationToken
from ..context import AgentContext
from ..llm._base import BaseChatCompletionClient
from ..memory import BaseMemory
from ..messages import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCallRequest,
    ToolMessage,
    UserMessage,
)
from ..middleware import BaseMiddleware, MiddlewareChain
from ..termination import BaseTermination
from ..tools._base import ApprovalMode, BaseTool, FunctionTool
from ..types import (
    AgentEvent,
    AgentResponse,
    ChatCompletionChunk,
    ChatCompletionResult,
    ErrorEvent,
    ModelCallEvent,
    ModelResponseEvent,
    TaskCompleteEvent,
    TaskStartEvent,
    ToolApprovalEvent,
    ToolCallEvent,
    ToolCallResponseEvent,
)

logger = logging.getLogger(__name__)


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
        memory: Optional[BaseMemory] = None,
        termination: Optional[BaseTermination] = None,
        middlewares: Optional[List[BaseMiddleware]] = None,
        summarize_tool_result: bool = True,
    ):
        self.name = name
        self.description = description
        self.instructions = instructions
        self.model_client = model_client
        self.tools = self._process_tools(tools or [])
        self.context = context or AgentContext()
        self.max_iterations = max_iterations
        self.output_format = output_format
        self.memory = memory
        self.termination = termination
        self.middleware_chain = MiddlewareChain(middlewares or [])
        self.summarize_tool_result = summarize_tool_result

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

                if self.termination:
                    stop_message = self.termination.check(working_context.messages[-1:])
                    if stop_message:
                        working_context.add_message(
                            UserMessage(content=stop_message.content, source="system")
                        )
                        finish_reason = "termination"
                        break

                llm_messages = await self._prepare_llm_messages(working_context)
                tools = self._get_tools_for_llm() if self.tools else None

                if verbose:
                    yield ModelCallEvent(
                        source=self.name,
                        input_messages=llm_messages,
                        model=getattr(self.model_client, "model", "unknown"),
                    )

                if stream_tokens and self.middleware_chain.middlewares:
                    logger.warning("Token streaming disabled: middlewares require full requests")
                    stream_tokens = False

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
                    result = await self._call_model(llm_messages, tools, cancellation_token)
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
                    if self.termination:
                        stop_message = self.termination.check([assistant_message])
                        if stop_message:
                            working_context.add_message(
                                UserMessage(content=stop_message.content, source="system")
                            )
                            finish_reason = "termination"
                            break
                    finish_reason = "stop"
                    break

                approval_needed = False
                for tc in assistant_message.tool_calls:
                    if cancellation_token and cancellation_token.is_cancelled():
                        raise asyncio.CancelledError()

                    async for item in self._execute_tool_call_with_events(
                        tc, working_context, cancellation_token, verbose
                    ):
                        if isinstance(item, ToolApprovalEvent):
                            approval_needed = True
                        if isinstance(item, ToolMessage):
                            working_context.add_message(item)
                            total_usage = total_usage + Usage(duration_ms=0, tool_calls=1)
                        yield item

                    if approval_needed:
                        break

                    if self.termination:
                        stop_message = self.termination.check(working_context.messages[-1:])
                        if stop_message:
                            working_context.add_message(
                                UserMessage(content=stop_message.content, source="system")
                            )
                            finish_reason = "termination"
                            break

                if approval_needed:
                    finish_reason = "approval_needed"
                    break

                if finish_reason == "termination":
                    break

                if not self.summarize_tool_result:
                    finish_reason = "tool_calls"
                    break

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

    async def _call_model(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]],
        cancellation_token: Optional[CancellationToken],
    ) -> ChatCompletionResult:
        """Call the model, optionally routed through middleware."""
        if not self.middleware_chain.middlewares:
            return await self.model_client.create(
                messages, tools=tools, output_format=self.output_format
            )

        async def _model_call(data):
            return await self.model_client.create(
                data, tools=tools, output_format=self.output_format
            )

        final_result = None
        async for item in self.middleware_chain.execute_stream(
            operation="model_call",
            agent_name=self.name,
            agent_context=self.context,
            data=messages,
            func=_model_call,
            metadata={"model": getattr(self.model_client, "model", "unknown")},
        ):
            if isinstance(item, ChatCompletionResult):
                final_result = item
            elif isinstance(item, ErrorEvent):
                raise RuntimeError(item.error_message)
            elif isinstance(item, (ToolApprovalEvent,)):
                continue
        if final_result is None:
            raise RuntimeError("Model call did not produce a result")
        return final_result

    async def _execute_tool_call_with_events(
        self,
        tc: ToolCallRequest,
        working_context: AgentContext,
        cancellation_token: Optional[CancellationToken],
        verbose: bool,
    ) -> AsyncGenerator[Union[ToolMessage, ToolCallEvent, ToolCallResponseEvent, ToolApprovalEvent], None]:
        if verbose:
            yield ToolCallEvent(source=self.name, tool_name=tc.tool_name, parameters=tc.parameters, call_id=tc.call_id)

        if self.middleware_chain.middlewares:
            async def _tool_call(data):
                if isinstance(data, ToolResult):
                    return data
                tool = self._find_tool(data.tool_name)
                if tool is None:
                    raise RuntimeError(f"Tool '{data.tool_name}' not found")
                return await tool.execute(data.parameters)

            final_result = None
            async for item in self.middleware_chain.execute_stream(
                operation="tool_call",
                agent_name=self.name,
                agent_context=working_context,
                data=tc,
                func=_tool_call,
            ):
                if isinstance(item, ToolResult):
                    final_result = item
                elif isinstance(item, ToolApprovalEvent):
                    yield item
                    return
                elif isinstance(item, ToolCallResponseEvent):
                    yield item
                elif isinstance(item, ErrorEvent):
                    raise RuntimeError(item.error_message)

            if final_result is None:
                raise RuntimeError("Tool call did not produce a result")

            tool_msg = self._tool_result_to_message(final_result, tc)
            if verbose:
                yield ToolCallResponseEvent(source=self.name, call_id=tc.call_id, result=final_result)
            yield tool_msg
            return

        # Direct execution
        tool = self._find_tool(tc.tool_name)
        if tool is None:
            result = ToolResult(
                success=False,
                result=None,
                error=f"Tool '{tc.tool_name}' not found",
                metadata={"tool_name": tc.tool_name},
            )
            tool_msg = self._tool_result_to_message(result, tc)
            if verbose:
                yield ToolCallResponseEvent(source=self.name, call_id=tc.call_id, result=result)
            yield tool_msg
            return

        if tool.approval_mode == ApprovalMode.ALWAYS:
            approval = working_context.get_approval_response(tc.call_id)
            if approval is None:
                request = working_context.add_approval_request(tc, tc.tool_name)
                yield ToolApprovalEvent(source=self.name, approval_request=request)
                return
            if not approval.approved:
                result = ToolResult(
                    success=False,
                    result=None,
                    error=f"Approval denied: {approval.reason or 'User declined'}",
                    metadata={"tool_name": tc.tool_name, "call_id": tc.call_id},
                )
                tool_msg = self._tool_result_to_message(result, tc)
                if verbose:
                    yield ToolCallResponseEvent(source=self.name, call_id=tc.call_id, result=result)
                yield tool_msg
                return

        tool_result = await tool.execute(tc.parameters)
        tool_msg = self._tool_result_to_message(tool_result, tc)
        if verbose:
            yield ToolCallResponseEvent(source=self.name, call_id=tc.call_id, result=tool_result)
        yield tool_msg

    def _tool_result_to_message(self, tool_result: ToolResult, tc: ToolCallRequest) -> ToolMessage:
        return ToolMessage(
            content=str(tool_result.result) if tool_result.success else (tool_result.error or ""),
            source=tc.tool_name,
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
                        existing_func["arguments"] = existing_func.get("arguments", "") + chunk_func["arguments"]
                    accumulated_tool_calls[call_id]["function"] = existing_func
                elif last_call_id:
                    existing = accumulated_tool_calls.get(last_call_id, {})
                    existing_func = existing.get("function", {})
                    if chunk_func.get("arguments"):
                        existing_func["arguments"] = existing_func.get("arguments", "") + chunk_func["arguments"]
                    existing["function"] = existing_func
                    accumulated_tool_calls[last_call_id] = existing

        tool_calls = []
        for call_id, tc_data in accumulated_tool_calls.items():
            func = tc_data.get("function", {})
            name = func.get("name", "")
            arguments = func.get("arguments", "")
            if name and arguments:
                try:
                    params = json.loads(arguments)
                except json.JSONDecodeError:
                    params = {}
                tool_calls.append(ToolCallRequest(tool_name=name, parameters=params, call_id=call_id))

        return accumulated_content, tool_calls if tool_calls else None, usage

    async def _prepare_llm_messages(self, working_context: AgentContext) -> List[Message]:
        system_content = self.instructions

        if self.memory:
            try:
                memory_context = await self.memory.get_context(max_items=10)
                if memory_context.results:
                    memory_text = "\n".join(
                        f"- {m.content}" for m in memory_context.results
                    )
                    system_content += f"\n\nRelevant context from memory:\n{memory_text}"
            except Exception as e:
                logger.warning("Memory retrieval failed: %s", e)

        return [SystemMessage(content=system_content, source="system"), *working_context.messages]

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
        self.context.reset()
        if self.memory:
            await self.memory.clear()
        if self.termination:
            self.termination.reset()

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "type": self.__class__.__name__,
            "model": getattr(self.model_client, "model", "unknown"),
            "tools_count": len(self.tools),
            "message_history_length": self.context.message_count,
            "has_memory": self.memory is not None,
            "has_termination": self.termination is not None,
            "middleware_count": len(self.middleware_chain.middlewares),
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
