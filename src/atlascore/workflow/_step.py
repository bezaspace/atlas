"""Workflow step implementations for atlascore."""

from __future__ import annotations

import asyncio
import inspect
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

from ..agents._agent import Agent
from ..base_types import Usage
from ..messages import AssistantMessage
from ..types import AgentResponse
from ._models import (
    Context,
    InputType,
    OutputType,
    StepMetadata,
    StepStatus,
)

logger = logging.getLogger(__name__)

T_in = TypeVar("T_in", bound=BaseModel)
T_out = TypeVar("T_out", bound=BaseModel)


class BaseStep(ABC, Generic[InputType, OutputType]):
    """Base class for workflow steps with typed input/output validation."""

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[InputType],
        output_type: Type[OutputType],
    ) -> None:
        self.step_id = step_id
        self.metadata = metadata
        self.input_type = input_type
        self.output_type = output_type
        self._status = StepStatus.PENDING
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
        self._error: Optional[str] = None

    @property
    def status(self) -> StepStatus:
        return self._status

    @property
    def start_time(self) -> Optional[datetime]:
        return self._start_time

    @property
    def end_time(self) -> Optional[datetime]:
        return self._end_time

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def duration(self) -> Optional[float]:
        if self._start_time and self._end_time:
            return (self._end_time - self._start_time).total_seconds()
        return None

    @abstractmethod
    async def execute(self, input_data: InputType, context: Context) -> OutputType:
        """Execute the step logic."""

    async def run(self, input_data: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Run the step with input/output validation, retries, and timeout."""
        logger.info(f"Starting step {self.step_id} ({self.metadata.name})")

        self._status = StepStatus.RUNNING
        self._start_time = datetime.now()
        self._error = None

        retry_count = 0
        max_retries = self.metadata.max_retries

        # Resolve a typed Context object from the dictionary the runner passes in.
        if isinstance(context, dict):
            typed_context = context.get("_context_obj")
            if typed_context is None:
                typed_context = Context.from_state_ref(context.get("workflow_state", {}))
        else:
            typed_context = context

        while retry_count <= max_retries:
            try:
                validated_input = self.input_type(**input_data)

                if self.metadata.timeout_seconds:
                    output = await asyncio.wait_for(
                        self.execute(validated_input, typed_context),
                        timeout=self.metadata.timeout_seconds,
                    )
                else:
                    output = await self.execute(validated_input, typed_context)

                if not isinstance(output, self.output_type):
                    if isinstance(output, BaseModel):
                        output = self.output_type(**output.model_dump())
                    elif isinstance(output, dict):
                        output = self.output_type(**output)
                    else:
                        output = self.output_type(result=output)

                self._status = StepStatus.COMPLETED
                self._end_time = datetime.now()

                logger.info(f"Step {self.step_id} completed in {self.duration:.2f}s")
                return output.model_dump()

            except asyncio.TimeoutError:
                error_msg = f"Step {self.step_id} timed out after {self.metadata.timeout_seconds}s"
                logger.error(error_msg)
                self._error = error_msg
                self._status = StepStatus.FAILED
                self._end_time = datetime.now()
                raise Exception(error_msg)

            except Exception as e:
                retry_count += 1
                error_msg = f"Step {self.step_id} failed (attempt {retry_count}/{max_retries + 1}): {e}"
                logger.error(error_msg)

                if retry_count <= max_retries:
                    logger.info(f"Retrying step {self.step_id} in 1 second...")
                    await asyncio.sleep(1)
                    continue

                self._error = str(e)
                self._status = StepStatus.FAILED
                self._end_time = datetime.now()
                raise

        raise Exception(f"Unexpected error in step {self.step_id}")

    def validate_input(self, data: Dict[str, Any]) -> bool:
        try:
            self.input_type(**data)
            return True
        except Exception:
            return False

    def validate_output(self, data: Dict[str, Any]) -> bool:
        try:
            self.output_type(**data)
            return True
        except Exception:
            return False

    def get_schema(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "metadata": self.metadata.model_dump(),
            "input_type": self.input_type.__name__,
            "output_type": self.output_type.__name__,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type.model_json_schema(),
        }

    def _serialize_types(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "metadata": self.metadata.model_dump(),
            "input_type_name": self.input_type.__name__,
            "output_type_name": self.output_type.__name__,
            "input_schema": self.input_type.model_json_schema(),
            "output_schema": self.output_type.model_json_schema(),
        }


class FunctionStep(BaseStep[T_in, T_out]):
    """A step that executes a Python function (sync or async)."""

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        input_type: Type[T_in],
        output_type: Type[T_out],
        func: Callable[..., Any],
    ) -> None:
        super().__init__(step_id, metadata, input_type, output_type)
        self.func = func
        self._accepts_context = self._function_accepts_context(func)

    @staticmethod
    def _function_accepts_context(func: Callable[..., Any]) -> bool:
        try:
            sig = inspect.signature(func)
            return len(sig.parameters) >= 2
        except (ValueError, TypeError):
            return False

    async def execute(self, input_data: T_in, context: Context) -> T_out:
        if asyncio.iscoroutinefunction(self.func):
            if self._accepts_context:
                result = await self.func(input_data, context)
            else:
                result = await self.func(input_data)
        else:
            if self._accepts_context:
                result = self.func(input_data, context)
            else:
                result = self.func(input_data)

        if isinstance(result, dict):
            return self.output_type(**result)
        if isinstance(result, BaseModel):
            if isinstance(result, self.output_type):
                return result
            return self.output_type(**result.model_dump())

        return self.output_type(result=result)


class AgentStepInput(BaseModel):
    """Input schema for an Agent workflow step."""

    task: str = Field(..., description="The task or question to send to the agent")
    additional_context: Optional[Dict[str, Any]] = Field(
        default=None, description="Additional context or parameters"
    )


class AgentStepOutput(BaseModel):
    """Output schema for an Agent workflow step."""

    response: str = Field(..., description="The agent's final response")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation messages")
    usage: Dict[str, Any] = Field(default_factory=dict, description="Usage statistics")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class AgentStep(BaseStep[AgentStepInput, AgentStepOutput]):
    """Workflow step that wraps an atlascore Agent."""

    def __init__(
        self,
        step_id: str,
        metadata: StepMetadata,
        agent: Agent,
    ) -> None:
        super().__init__(
            step_id=step_id,
            metadata=metadata,
            input_type=AgentStepInput,
            output_type=AgentStepOutput,
        )
        self.agent = agent

    async def execute(self, input_data: AgentStepInput, context: Context) -> AgentStepOutput:
        context.set(
            f"{self.step_id}_request_info",
            {
                "agent_name": self.agent.name,
                "task": input_data.task,
                "timestamp": datetime.now().isoformat(),
                "additional_context": input_data.additional_context,
            },
        )

        agent_result: AgentResponse = await self.agent.run(input_data.task)

        final_response = ""
        if agent_result.messages:
            for message in reversed(agent_result.messages):
                if isinstance(message, AssistantMessage):
                    final_response = message.content
                    break
            if not final_response:
                final_response = agent_result.messages[-1].content

        if not final_response:
            final_response = "No response generated"

        serializable_messages = []
        for msg in agent_result.messages:
            msg_dict = {
                "role": msg.role,
                "content": msg.content,
                "source": msg.source,
                "timestamp": msg.timestamp.isoformat() if msg.timestamp else None,
            }
            serializable_messages.append(msg_dict)

        usage = agent_result.usage
        usage_dict: Dict[str, Any] = (
            usage.model_dump() if isinstance(usage, Usage) else dict(usage)
        )

        execution_metadata = {
            "agent_name": self.agent.name,
            "message_count": len(agent_result.messages),
            "elapsed_time": agent_result.usage.duration_ms / 1000.0,
            "llm_calls": agent_result.usage.llm_calls,
            "tokens_total": agent_result.usage.tokens_input + agent_result.usage.tokens_output,
            "execution_timestamp": datetime.now().isoformat(),
        }
        if input_data.additional_context:
            execution_metadata["additional_context"] = input_data.additional_context

        context.set(
            f"{self.step_id}_output",
            {
                "response": final_response,
                "messages": serializable_messages,
                "usage": usage_dict,
                "metadata": execution_metadata,
            },
        )

        return AgentStepOutput(
            response=final_response,
            messages=serializable_messages,
            usage=usage_dict,
            metadata=execution_metadata,
        )
