"""Abstract base class for chat completion clients."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional, Type

from pydantic import BaseModel

from ..messages import AssistantMessage, Message, MultiModalMessage, ToolMessage
from ..types import ChatCompletionChunk, ChatCompletionResult


class BaseChatCompletionClient(ABC):
    """Abstract base class for LLM provider implementations."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs: Any):
        self.model = model
        self.api_key = api_key
        self.config = kwargs

    @abstractmethod
    async def create(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> ChatCompletionResult:
        pass

    @abstractmethod
    def create_stream(
        self,
        messages: List[Message],
        tools: Optional[List[Dict[str, Any]]] = None,
        output_format: Optional[Type[BaseModel]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[ChatCompletionChunk, None]:
        pass

    def _convert_messages_to_api_format(self, messages: List[Message]) -> List[Dict[str, Any]]:
        """Convert internal Message objects to OpenAI-compatible API format."""
        api_messages = []
        for msg in messages:
            if isinstance(msg, MultiModalMessage):
                api_msg: Dict[str, Any] = {"role": msg.role}
                if msg.is_text():
                    api_msg["content"] = msg.content
                else:
                    content_parts = []
                    if msg.content and msg.content.strip():
                        content_parts.append({"type": "text", "text": msg.content})
                    if msg.is_image():
                        if msg.data:
                            base64_data = msg.to_base64()
                            data_url = f"data:{msg.mime_type};base64,{base64_data}"
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        elif msg.media_url:
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": msg.media_url}}
                            )
                    api_msg["content"] = content_parts
            else:
                api_msg = {"role": msg.role, "content": msg.content}
                if isinstance(msg, AssistantMessage) and msg.tool_calls:
                    api_msg["tool_calls"] = [
                        {
                            "id": tc.call_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.parameters)
                                if isinstance(tc.parameters, dict)
                                else tc.parameters,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                if isinstance(msg, ToolMessage):
                    api_msg["tool_call_id"] = msg.tool_call_id
            api_messages.append(api_msg)
        return api_messages


class BaseChatCompletionError(Exception):
    """Raised when a chat completion API call fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response_data: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.response_data = response_data
        super().__init__(message)


class RateLimitError(BaseChatCompletionError):
    pass


class AuthenticationError(BaseChatCompletionError):
    pass


class InvalidRequestError(BaseChatCompletionError):
    pass
