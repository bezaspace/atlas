"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import BaseModel

try:
    from openai import APIError, AsyncOpenAI
    from openai import AuthenticationError as OpenAIAuthError
    from openai import RateLimitError as OpenAIRateLimitError
    from openai.types.chat import ChatCompletion
    from openai.types.chat.chat_completion import Choice
    from openai.types.chat.chat_completion_chunk import Choice as ChunkChoice
    from openai.types.completion_usage import CompletionUsage
except ImportError:
    raise ImportError("OpenAI library not installed. Please install with: pip install openai")

from ..messages import AssistantMessage, Message, ToolCallRequest
from ..types import ChatCompletionChunk, ChatCompletionResult, Usage
from ._base import (
    AuthenticationError,
    BaseChatCompletionClient,
    BaseChatCompletionError,
    RateLimitError,
)

logger = logging.getLogger(__name__)
_unknown_cost_warned: set[str] = set()


class OpenAIChatCompletionClient(BaseChatCompletionClient):
    """Generic OpenAI-compatible chat completion client.

    Works with OpenAI, Groq, OpenRouter, Together, local Ollama/vLLM, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(model, api_key, **kwargs)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            **kwargs,
        )

    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        try:
            start_time = time.time()
            api_messages = self._convert_messages_to_api_format(messages)
            request_params = {"model": self.model, "messages": api_messages, **kwargs}

            if tools:
                request_params["tools"] = tools
                request_params["tool_choice"] = "auto"

            if output_format:
                try:
                    schema = output_format.model_json_schema()
                    schema = self._make_schema_compatible(schema)
                    request_params["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.get("title", output_format.__name__),
                            "description": schema.get(
                                "description",
                                f"Structured output for {output_format.__name__}",
                            ),
                            "strict": True,
                            "schema": schema,
                        },
                    }
                except Exception as e:
                    print(
                        f"Warning: Failed to convert {output_format.__name__} to JSON schema: {e}"
                    )

            response: ChatCompletion = await self.client.chat.completions.create(**request_params)
            duration_ms = int((time.time() - start_time) * 1000)

            choice: Choice = response.choices[0]
            assistant_content = choice.message.content or ""

            tool_calls = []
            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    if tc.type == "function":
                        function_call = tc.function
                        tool_calls.append(
                            ToolCallRequest(
                                tool_name=function_call.name,
                                parameters=json.loads(function_call.arguments)
                                if function_call.arguments
                                else {},
                                call_id=tc.id,
                            )
                        )

            assistant_message = AssistantMessage(
                content=assistant_content,
                source="llm",
                tool_calls=tool_calls if tool_calls else None,
            )

            structured_output = None
            if output_format and assistant_content:
                try:
                    structured_output = output_format.model_validate_json(assistant_content)
                except Exception as e:
                    print(f"Warning: Failed to parse structured output: {e}")

            usage_data = response.usage
            usage = Usage(
                duration_ms=duration_ms,
                llm_calls=1,
                tokens_input=usage_data.prompt_tokens if usage_data else 0,
                tokens_output=usage_data.completion_tokens if usage_data else 0,
                cost_estimate=self._estimate_cost(usage_data) if usage_data else None,
            )

            return ChatCompletionResult(
                message=assistant_message,
                usage=usage,
                model=response.model,
                finish_reason=choice.finish_reason or "stop",
                structured_output=structured_output,
            )

        except OpenAIAuthError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {str(e)}")
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except APIError as e:
            raise BaseChatCompletionError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}")

    async def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        if output_format:
            print("Warning: Structured output is not yet supported in streaming mode")

        api_messages = self._convert_messages_to_api_format(messages)
        request_params = {
            "model": self.model,
            "messages": api_messages,
            "stream": True,
            **kwargs,
        }
        if stream_options is None:
            stream_options = {"include_usage": True}
        if stream_options:
            request_params["stream_options"] = stream_options
        if tools:
            request_params["tools"] = tools
            request_params["tool_choice"] = "auto"

        try:
            stream = await self.client.chat.completions.create(**request_params)
            accumulated_content = ""
            tool_call_chunks = {}

            async for chunk in stream:
                if (
                    hasattr(chunk, "usage")
                    and chunk.usage
                    and (not chunk.choices or len(chunk.choices) == 0)
                ):
                    usage_data = Usage(
                        duration_ms=0,
                        llm_calls=1,
                        tokens_input=chunk.usage.prompt_tokens,
                        tokens_output=chunk.usage.completion_tokens,
                        tool_calls=0,
                        cost_estimate=self._estimate_cost(chunk.usage),
                    )
                    yield ChatCompletionChunk(
                        content="",
                        is_complete=True,
                        tool_call_chunk=None,
                        usage=usage_data,
                    )
                    break

                if not chunk.choices:
                    continue

                chunk_choice: ChunkChoice = chunk.choices[0]
                delta = chunk_choice.delta

                if delta.content:
                    accumulated_content += delta.content
                    yield ChatCompletionChunk(
                        content=delta.content, is_complete=False, tool_call_chunk=None
                    )

                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        index = getattr(tc_delta, "index", None)
                        call_id = tc_delta.id
                        tracking_key = index if index is not None else call_id

                        if tracking_key not in tool_call_chunks:
                            tool_call_chunks[tracking_key] = {
                                "id": call_id,
                                "function": {"name": "", "arguments": ""},
                            }

                        if call_id:
                            tool_call_chunks[tracking_key]["id"] = call_id

                        if tc_delta.function:
                            if tc_delta.function.name:
                                tool_call_chunks[tracking_key]["function"]["name"] = (
                                    tc_delta.function.name
                                )
                            if tc_delta.function.arguments:
                                tool_call_chunks[tracking_key]["function"]["arguments"] += (
                                    tc_delta.function.arguments
                                )

                        yield ChatCompletionChunk(
                            content="",
                            is_complete=False,
                            tool_call_chunk=tool_call_chunks[tracking_key],
                        )

        except OpenAIAuthError as e:
            raise AuthenticationError(f"OpenAI authentication failed: {str(e)}")
        except OpenAIRateLimitError as e:
            raise RateLimitError(f"OpenAI rate limit exceeded: {str(e)}")
        except APIError as e:
            raise BaseChatCompletionError(f"OpenAI API error: {str(e)}")
        except Exception as e:
            raise BaseChatCompletionError(f"Unexpected error: {str(e)}")

    def _estimate_cost(self, usage: CompletionUsage) -> Optional[float]:
        """Estimate the cost of the API call based on token usage.

        Prices are in USD per 1M tokens and are matched by substring on the
        model identifier. Free providers and unknown models return 0 or None.
        """
        model = self.model.lower()

        # Free aliases carry zero cost.
        if ":free" in model or model.endswith("/free") or model == "kilo-auto/free":
            return 0.0

        # Prices: USD per 1M tokens (input, output). Order matters: more specific first.
        pricing: list[tuple[str, tuple[float, float]]] = [
            # OpenAI
            ("gpt-4o-mini", (0.15, 0.60)),
            ("gpt-4o", (2.50, 10.00)),
            ("gpt-4-turbo", (10.00, 30.00)),
            ("gpt-4-32k", (60.00, 120.00)),
            ("gpt-4", (30.00, 60.00)),
            ("gpt-3.5-turbo", (0.50, 1.50)),
            # Anthropic / OpenRouter
            ("claude-3-5-sonnet", (3.00, 15.00)),
            ("claude-3.5-sonnet", (3.00, 15.00)),
            ("claude-3-opus", (15.00, 75.00)),
            ("claude-3-haiku", (0.25, 1.25)),
            ("claude-sonnet", (3.00, 15.00)),
            ("claude-opus", (15.00, 75.00)),
            ("claude-haiku", (0.25, 1.25)),
            ("claude-4", (5.00, 25.00)),
            # Google / Gemini
            ("gemini-2.5-pro", (1.25, 10.00)),
            ("gemini-2.5-flash-lite", (0.10, 0.40)),
            ("gemini-2.5-flash", (0.10, 0.40)),
            ("gemini-1.5-pro", (1.25, 5.00)),
            ("gemini-1.5-flash", (0.075, 0.30)),
            # Kilo / StepFun / inclusionAI paid
            ("stepfun/step-3.7-flash", (0.20, 1.15)),
            ("stepfun/step-3.5-flash", (0.10, 0.30)),
            ("inclusionai/ling-3.0-flash", (0.30, 2.50)),
        ]

        for key, (input_rate, output_rate) in pricing:
            if key in model:
                return (usage.prompt_tokens * input_rate + usage.completion_tokens * output_rate) / 1_000_000

        if self.model not in _unknown_cost_warned:
            _unknown_cost_warned.add(self.model)
            logger.warning("Cost estimate unavailable for model %s", self.model)
        return None

    def _make_schema_compatible(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Make a JSON schema compatible with OpenAI strict mode."""
        compatible_schema = schema.copy()

        if "$defs" in compatible_schema:
            compatible_schema["$defs"] = {
                name: self._make_schema_compatible(defn)
                for name, defn in compatible_schema["$defs"].items()
            }

        if compatible_schema.get("type") == "object":
            compatible_schema["additionalProperties"] = False
            properties = compatible_schema.get("properties", {})
            compatible_schema["required"] = list(properties.keys())
            for prop_name, prop_schema in properties.items():
                if isinstance(prop_schema, dict):
                    compatible_schema["properties"][prop_name] = self._make_schema_compatible(
                        prop_schema
                    )

        if compatible_schema.get("type") == "array":
            items = compatible_schema.get("items")
            if isinstance(items, dict):
                compatible_schema["items"] = self._make_schema_compatible(items)

        return compatible_schema
