"""Base tool classes and interfaces for atlascore."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union, get_type_hints

from ..base_types import ToolResult
from ..cancellation import CancellationToken
from ..types import AgentEvent, Message


class ApprovalMode(Enum):
    """Tool approval requirements."""

    NEVER = "never_require"
    ALWAYS = "always_require"


class BaseTool(ABC):
    """Abstract base class that all tools must implement."""

    def __init__(
        self,
        name: str,
        description: str,
        version: str = "1.0.0",
        approval_mode: ApprovalMode = ApprovalMode.NEVER,
    ):
        self.name = name
        self.description = description
        self.version = version
        self.approval_mode = approval_mode

    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """JSON schema defining expected inputs for this tool."""
        pass

    @abstractmethod
    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the tool with the given parameters."""
        pass

    async def execute_stream(
        self,
        parameters: Dict[str, Any],
        cancellation_token: Optional[CancellationToken] = None,
    ) -> AsyncGenerator[Union[Message, AgentEvent, ToolResult], None]:
        """Execute the tool with streaming output support."""
        result = await self.execute(parameters)
        yield result

    def supports_streaming(self) -> bool:
        return type(self).execute_stream is not BaseTool.execute_stream

    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate that the provided parameters match the tool's schema."""
        try:
            schema = self.parameters
            required_fields = schema.get("required", [])
            for field in required_fields:
                if field not in params:
                    return False

            properties = schema.get("properties", {})
            for param_name, param_value in params.items():
                if param_name in properties:
                    expected_type = properties[param_name].get("type")
                    if expected_type and not self._check_type(param_value, expected_type):
                        return False
            return True
        except Exception:
            return False

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON schema type."""
        type_mapping = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected_python_type = type_mapping.get(expected_type)
        if expected_python_type is None:
            return True
        return isinstance(value, expected_python_type)

    def to_llm_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function calling format."""
        versioned_name = (
            f"{self.name}_v{self.version}" if self.version != "1.0.0" else self.name
        )
        return {
            "type": "function",
            "function": {
                "name": versioned_name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', description='{self.description}')"


class FunctionTool(BaseTool):
    """Tool that wraps a Python function for use by agents."""

    def __init__(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        version: str = "1.0.0",
        approval_mode: ApprovalMode = ApprovalMode.NEVER,
    ):
        self.func = func
        tool_name = name or func.__name__
        tool_description = (
            description or func.__doc__ or f"Execute {func.__name__} function"
        )
        super().__init__(tool_name, tool_description, version, approval_mode)
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)
        self._parameters_schema = self._build_parameters_schema()

    @property
    def parameters(self) -> Dict[str, Any]:
        return self._parameters_schema

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        """Execute the wrapped function with validated parameters."""
        try:
            if not self.validate_parameters(parameters):
                return ToolResult(
                    success=False,
                    result=None,
                    error="Invalid parameters provided",
                    metadata={"tool_name": self.name},
                )

            if inspect.iscoroutinefunction(self.func):
                result = await self.func(**parameters)
            else:
                result = self.func(**parameters)

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"tool_name": self.name},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=str(e),
                metadata={"tool_name": self.name, "exception_type": type(e).__name__},
            )

    def _build_parameters_schema(self) -> Dict[str, Any]:
        """Build JSON schema from function signature and type hints."""
        schema: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}

        for param_name, param in self.signature.parameters.items():
            if param_name == "self":
                continue

            param_type = self.type_hints.get(param_name)
            json_type = self._python_type_to_json_type(param_type)
            property_schema: Dict[str, Any] = {"type": json_type}

            enum_values = self._extract_enum_values(param_type)
            if enum_values:
                property_schema["enum"] = enum_values

            schema["properties"][param_name] = property_schema

            if param.default == inspect.Parameter.empty:
                schema["required"].append(param_name)

        return schema

    def _extract_enum_values(self, param_type: Any) -> Optional[List[Any]]:
        """Extract enum values from Literal types or Enum classes."""
        from typing import Literal, get_args, get_origin

        try:
            if get_origin(param_type) is Literal:
                return list(get_args(param_type))
        except Exception:
            pass

        try:
            if isinstance(param_type, type) and issubclass(param_type, Enum):
                return [e.value for e in param_type]
        except (TypeError, AttributeError):
            pass

        return None

    def _python_type_to_json_type(self, python_type: Any) -> str:
        """Convert Python type hints to JSON schema types."""
        if python_type is None or python_type is type(None):
            return "null"
        if python_type is str:
            return "string"
        if python_type is int:
            return "integer"
        if python_type is float:
            return "number"
        if python_type is bool:
            return "boolean"
        if python_type is list or (
            hasattr(python_type, "__origin__") and python_type.__origin__ is list
        ):
            return "array"
        if python_type is dict or (
            hasattr(python_type, "__origin__") and python_type.__origin__ is dict
        ):
            return "object"
        return "string"
