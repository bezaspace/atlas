"""Core deterministic tools for atlascore."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

from ..base_types import ToolResult
from ._base import BaseTool


class ThinkTool(BaseTool):
    """Allow the agent to pause and reason about current state."""

    def __init__(self) -> None:
        super().__init__(
            name="think",
            description=(
                "Use this tool when you need to pause and think carefully about a complex problem, "
                "analyze tool results, plan your next steps, or reason through a difficult decision. "
                "Provide your reasoning in the 'thought' parameter."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "thought": {
                    "type": "string",
                    "description": (
                        "Your detailed reasoning about the current situation. "
                        "Include what you have learned, options, risks, and planned approach."
                    ),
                }
            },
            "required": ["thought"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        thought = parameters["thought"]
        return ToolResult(
            success=True,
            result=f"Reasoning recorded: {thought[:100]}..."
            if len(thought) > 100
            else f"Reasoning recorded: {thought}",
            error=None,
            metadata={"thought_length": len(thought), "tool_name": "think"},
        )


class CalculatorTool(BaseTool):
    """Evaluate mathematical expressions safely."""

    def __init__(self) -> None:
        super().__init__(
            name="calculator",
            description=(
                "Evaluate mathematical expressions and perform calculations. "
                "Use this when you need to compute numeric results or arithmetic operations. "
                "Supports: basic operations (+, -, *, /, **), math functions (sin, cos, tan, sqrt, log, exp), "
                "and constants (pi, e). Returns the numerical result as a string."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '2 + 2', 'sqrt(16)', 'pi * 2')",
                }
            },
            "required": ["expression"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        expression = parameters["expression"]
        try:
            allowed_names = {
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
                "sum": sum,
                "pow": pow,
                "sqrt": math.sqrt,
                "sin": math.sin,
                "cos": math.cos,
                "tan": math.tan,
                "log": math.log,
                "log10": math.log10,
                "exp": math.exp,
                "ceil": math.ceil,
                "floor": math.floor,
                "pi": math.pi,
                "e": math.e,
            }
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return ToolResult(
                success=True,
                result=str(result),
                error=None,
                metadata={"expression": expression, "result": result},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Failed to evaluate expression: {str(e)}",
                metadata={"expression": expression},
            )


class DateTimeTool(BaseTool):
    """Get current date/time or parse/format datetime strings."""

    def __init__(self) -> None:
        super().__init__(
            name="datetime",
            description=(
                "Work with dates and times - get current time, parse datetime strings, or format timestamps. "
                "All operations use ISO 8601 format by default. Returns datetime strings in ISO format."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["now", "parse", "format"],
                    "description": (
                        "Operation: 'now' (current time), 'parse' (string to datetime), "
                        "'format' (datetime to string)"
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "ISO format datetime string (for parse/format operations)",
                },
                "format": {
                    "type": "string",
                    "description": "Output format string (for format operation, e.g., '%Y-%m-%d %H:%M:%S')",
                },
            },
            "required": ["operation"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        operation = parameters["operation"]
        try:
            if operation == "now":
                now = datetime.now(timezone.utc)
                result = now.isoformat()
            elif operation == "parse":
                value = parameters.get("value")
                if not value:
                    raise ValueError("'value' parameter required for parse operation")
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                result = dt.isoformat()
            elif operation == "format":
                value = parameters.get("value")
                fmt = parameters.get("format", "%Y-%m-%d %H:%M:%S")
                if not value:
                    raise ValueError("'value' parameter required for format operation")
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                result = dt.strftime(fmt)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"operation": operation},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"DateTime operation failed: {str(e)}",
                metadata={"operation": operation},
            )


class JSONParserTool(BaseTool):
    """Parse JSON strings or validate JSON structure."""

    def __init__(self) -> None:
        super().__init__(
            name="json_parser",
            description="Parse JSON strings, validate JSON structure, or extract values from JSON.",
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "json_string": {
                    "type": "string",
                    "description": "JSON string to parse",
                },
                "path": {
                    "type": "string",
                    "description": "Optional JSON path to extract (dot notation, e.g., 'user.name')",
                },
            },
            "required": ["json_string"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        json_string = parameters["json_string"]
        path = parameters.get("path")
        try:
            parsed = json.loads(json_string)
            if path:
                result = parsed
                for key in path.split("."):
                    if isinstance(result, dict):
                        result = result.get(key)
                    elif isinstance(result, list) and key.isdigit():
                        result = result[int(key)]
                    else:
                        raise KeyError(f"Path '{path}' not found in JSON")
            else:
                result = parsed

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"path": path},
            )
        except json.JSONDecodeError as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid JSON: {str(e)}",
                metadata={},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"JSON parsing failed: {str(e)}",
                metadata={},
            )


class RegexTool(BaseTool):
    """Match patterns using regular expressions."""

    def __init__(self) -> None:
        super().__init__(
            name="regex",
            description=(
                "Find patterns in text using regular expressions. "
                "Supports search, match, findall, and replace operations."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["search", "match", "findall", "replace"],
                    "description": "Regex operation",
                },
                "pattern": {
                    "type": "string",
                    "description": "Regular expression pattern",
                },
                "text": {"type": "string", "description": "Text to search"},
                "replacement": {
                    "type": "string",
                    "description": "Replacement text (for replace operation)",
                },
                "flags": {
                    "type": "string",
                    "description": "Regex flags: 'i' (ignore case), 'm' (multiline), 's' (dotall)",
                },
            },
            "required": ["operation", "pattern", "text"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        operation = parameters["operation"]
        pattern = parameters["pattern"]
        text = parameters["text"]
        replacement = parameters.get("replacement", "")
        flags_str = parameters.get("flags", "")
        try:
            flags = 0
            if "i" in flags_str:
                flags |= re.IGNORECASE
            if "m" in flags_str:
                flags |= re.MULTILINE
            if "s" in flags_str:
                flags |= re.DOTALL

            if operation == "search":
                match = re.search(pattern, text, flags)
                result = match.group(0) if match else None
            elif operation == "match":
                match = re.match(pattern, text, flags)
                result = match.group(0) if match else None
            elif operation == "findall":
                result = re.findall(pattern, text, flags)
            elif operation == "replace":
                result = re.sub(pattern, replacement, text, flags=flags)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            return ToolResult(
                success=True,
                result=result,
                error=None,
                metadata={"operation": operation, "pattern": pattern},
            )
        except re.error as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Invalid regex pattern: {str(e)}",
                metadata={"pattern": pattern},
            )
        except Exception as e:
            return ToolResult(
                success=False,
                result=None,
                error=f"Regex operation failed: {str(e)}",
                metadata={"operation": operation},
            )


class TaskStatusTool(BaseTool):
    """Explicit task completion evaluation tool."""

    def __init__(self) -> None:
        super().__init__(
            name="task_status",
            description=(
                "Evaluate current task status with explicit rationale. "
                "Use this to formally assess whether a task is complete or incomplete. "
                "Provide status ('complete' or 'incomplete') and a detailed rationale."
            ),
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["complete", "incomplete"],
                    "description": "'complete' if all task requirements are satisfied, 'incomplete' otherwise",
                },
                "rationale": {
                    "type": "string",
                    "description": "Detailed explanation of your assessment",
                },
                "requirements_met": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: requirements that have been satisfied",
                },
                "requirements_pending": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional: requirements not yet satisfied",
                },
            },
            "required": ["status", "rationale"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        status = parameters["status"]
        rationale = parameters["rationale"]
        result_lines = [
            f"Task Status: {status.upper()}",
            "",
            "Rationale:",
            rationale,
        ]
        if "requirements_met" in parameters:
            result_lines.extend(["", "Requirements Met:"])
            for req in parameters["requirements_met"]:
                result_lines.append(f"  ✓ {req}")
        if "requirements_pending" in parameters:
            result_lines.extend(["", "Requirements Pending:"])
            for req in parameters["requirements_pending"]:
                result_lines.append(f"  ⧗ {req}")

        return ToolResult(
            success=True,
            result="\n".join(result_lines),
            error=None,
            metadata={
                "status": status,
                "rationale": rationale,
                "requirements_met": parameters.get("requirements_met", []),
                "requirements_pending": parameters.get("requirements_pending", []),
            },
        )


def create_core_tools() -> Sequence[BaseTool]:
    """Create a list of core utility tools."""
    return [
        ThinkTool(),
        TaskStatusTool(),
        CalculatorTool(),
        DateTimeTool(),
        JSONParserTool(),
        RegexTool(),
    ]
