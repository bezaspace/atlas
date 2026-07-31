import pytest

from atlascore.tools import CalculatorTool, DateTimeTool, FunctionTool, JSONParserTool, RegexTool


@pytest.mark.asyncio
async def test_calculator_success():
    tool = CalculatorTool()
    result = await tool.execute({"expression": "sqrt(16) + 2 * pi"})
    assert result.success is True
    assert float(result.result) > 10.2


@pytest.mark.asyncio
async def test_calculator_invalid_expression():
    tool = CalculatorTool()
    result = await tool.execute({"expression": "__import__('os')"})
    assert result.success is False
    assert "Failed to evaluate" in result.error


@pytest.mark.asyncio
async def test_datetime_now():
    tool = DateTimeTool()
    result = await tool.execute({"operation": "now"})
    assert result.success is True
    assert result.result.startswith("20")


@pytest.mark.asyncio
async def test_json_parser():
    tool = JSONParserTool()
    result = await tool.execute({"json_string": '{"user": {"name": "Alice"}}', "path": "user.name"})
    assert result.success is True
    assert result.result == "Alice"


@pytest.mark.asyncio
async def test_regex_findall():
    tool = RegexTool()
    result = await tool.execute(
        {
            "operation": "findall",
            "pattern": r"\b\w+@\w+\.\w+",
            "text": "Contact a@b.com and c@d.org",
        }
    )
    assert result.success is True
    assert result.result == ["a@b.com", "c@d.org"]


@pytest.mark.asyncio
async def test_function_tool():
    def greet(name: str, greeting: str = "hello") -> str:
        return f"{greeting} {name}"

    tool = FunctionTool(greet)
    result = await tool.execute({"name": "Alice", "greeting": "hi"})
    assert result.success is True
    assert result.result == "hi Alice"
    assert "name" in tool.parameters["required"]
    assert "greeting" not in tool.parameters["required"]


def test_tool_schema_generation():
    tool = CalculatorTool()
    schema = tool.parameters
    assert schema["type"] == "object"
    assert "expression" in schema["required"]
