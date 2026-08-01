"""Playwright browser actions as BaseTool implementations."""

import base64
from typing import Any, Dict, List

from ...base_types import ToolResult
from ...tools import BaseTool
from ._interface_clients import Action, ActionType, BaseInterfaceClient


class NavigateTool(BaseTool):
    """Navigate to a URL."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(name="navigate", description="Navigate to a specific URL")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to navigate to"}
            },
            "required": ["url"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.NAVIGATE, value=parameters["url"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success,
            result=result.description,
            error=result.error,
            metadata={"url": parameters.get("url")},
        )


class ClickTool(BaseTool):
    """Click on an element."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(
            name="click",
            description="Click on an element using a CSS selector or visible text",
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector, element ID, or visible text to click",
                }
            },
            "required": ["selector"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.CLICK, selector=parameters["selector"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class TypeTool(BaseTool):
    """Type text into an input element."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(name="type", description="Type text into an input element")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector or label of the input",
                },
                "text": {"type": "string", "description": "Text to type"},
            },
            "required": ["selector", "text"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.TYPE,
            selector=parameters["selector"],
            value=parameters["text"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class SelectTool(BaseTool):
    """Select an option from a dropdown."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(
            name="select", description="Select an option from a dropdown element"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the dropdown",
                },
                "value": {
                    "type": "string",
                    "description": "Option value or label to select",
                },
            },
            "required": ["selector", "value"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.SELECT,
            selector=parameters["selector"],
            value=parameters["value"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class PressTool(BaseTool):
    """Press a key or key combination."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(
            name="press", description="Press a key or key combination on an element"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector of the element to focus",
                },
                "key": {
                    "type": "string",
                    "description": "Key or key combination (e.g., 'Enter', 'Control+a')",
                },
            },
            "required": ["selector", "key"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.PRESS,
            selector=parameters["selector"],
            value=parameters["key"],
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class HoverTool(BaseTool):
    """Hover over an element."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(
            name="hover", description="Hover over an element to reveal hidden UI"
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector or text of the element",
                }
            },
            "required": ["selector"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(action_type=ActionType.HOVER, selector=parameters["selector"])
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class ScrollTool(BaseTool):
    """Scroll the page or an element."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(name="scroll", description="Scroll the page or an element")
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down", "left", "right"],
                    "description": "Direction to scroll",
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (default: 500)",
                    "default": 500,
                },
                "selector": {
                    "type": "string",
                    "description": "Optional CSS selector of an element to scroll",
                },
            },
            "required": ["direction"],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        action = Action(
            action_type=ActionType.SCROLL,
            value=parameters["direction"],
            selector=parameters.get("selector"),
            metadata={"amount": parameters.get("amount", 500)},
        )
        result = await self.interface_client.execute_action(action)
        return ToolResult(
            success=result.success, result=result.description, error=result.error
        )


class ObservePageTool(BaseTool):
    """Observe the current page state, returning text and a screenshot."""

    def __init__(self, interface_client: BaseInterfaceClient):
        super().__init__(
            name="observe_page",
            description="Get the current URL, title, visible text, interactive elements, and screenshot",
        )
        self.interface_client = interface_client

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": [],
        }

    async def execute(self, parameters: Dict[str, Any]) -> ToolResult:
        state = await self.interface_client.get_state("hybrid")

        lines = [f"URL: {state.url or 'N/A'}", f"Title: {state.title or 'N/A'}"]
        if state.content:
            lines.append("Page content:")
            lines.append(state.content[:2000])

        elements = state.interactive_elements or []
        if elements:
            lines.append(f"\nInteractive elements ({len(elements)}):")
            for el in elements[:15]:
                text = el.get("text", "") or el.get("placeholder", "")
                tag = el.get("tag", "")
                if text:
                    lines.append(f"  - <{tag}> {text}")

        description = "\n".join(lines)
        metadata: Dict[str, Any] = {
            "url": state.url,
            "title": state.title,
            "element_count": len(elements),
        }
        if state.screenshot:
            metadata["screenshot"] = base64.b64encode(state.screenshot).decode("utf-8")

        return ToolResult(success=True, result=description, metadata=metadata)


def create_playwright_tools(interface_client: BaseInterfaceClient) -> List[BaseTool]:
    """Create the default set of Playwright browser tools."""
    return [
        NavigateTool(interface_client),
        ClickTool(interface_client),
        TypeTool(interface_client),
        SelectTool(interface_client),
        PressTool(interface_client),
        HoverTool(interface_client),
        ScrollTool(interface_client),
        ObservePageTool(interface_client),
    ]
