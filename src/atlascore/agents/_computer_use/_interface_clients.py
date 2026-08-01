"""Interface clients for computer-use automation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Supported action types for interface automation."""

    CLICK = "click"
    TYPE = "type"
    SELECT = "select"
    NAVIGATE = "navigate"
    SCREENSHOT = "screenshot"
    SCROLL = "scroll"
    PRESS = "press"
    HOVER = "hover"


class Action(BaseModel):
    """Represents an action to be executed on an interface."""

    action_type: ActionType
    selector: Optional[str] = None
    value: Optional[str] = None
    coordinates: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ActionResult(BaseModel):
    """Result of executing an action on an interface."""

    success: bool
    description: str
    error: Optional[str] = None
    screenshot: Optional[bytes] = None
    task_complete: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class InterfaceState(BaseModel):
    """Represents the current state of an interface."""

    url: Optional[str] = None
    title: Optional[str] = None
    content: str = ""
    interactive_elements: List[Dict[str, Any]] = Field(default_factory=list)
    screenshot: Optional[bytes] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseInterfaceClient(ABC):
    """Abstract base class for interface automation clients."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the interface client."""
        pass

    @abstractmethod
    async def get_state(self, format: str = "hybrid") -> InterfaceState:
        """Get current state of the interface."""
        pass

    @abstractmethod
    async def execute_action(self, action: Action) -> ActionResult:
        """Execute an action on the interface."""
        pass

    @abstractmethod
    async def get_screenshot(self) -> bytes:
        """Get current screenshot as PNG bytes."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up resources."""
        pass


class PlaywrightWebClient(BaseInterfaceClient):
    """Web browser automation using Playwright."""

    def __init__(
        self,
        start_url: str = "about:blank",
        headless: bool = True,
        browser_type: str = "chromium",
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
    ) -> None:
        self.start_url = start_url
        self.headless = headless
        self.browser_type = browser_type
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.user_agent = user_agent
        self.playwright: Any = None
        self.browser: Any = None
        self.context: Any = None
        self.page: Any = None
        self._action_history: List[Action] = []

    async def initialize(self) -> None:
        """Launch the browser and navigate to the start URL."""
        try:
            from playwright.async_api import async_playwright
        except ImportError as e:
            raise ImportError(
                "Playwright is not installed. Install with: pip install 'atlascore[computer-use]'"
            ) from e

        self.playwright = await async_playwright().start()
        browser_cls = getattr(self.playwright, self.browser_type, None)
        if browser_cls is None:
            raise ValueError(f"Unsupported browser_type: {self.browser_type}")
        self.browser = await browser_cls.launch(headless=self.headless)
        context_kwargs: Dict[str, Any] = {"viewport": self.viewport}
        if self.user_agent:
            context_kwargs["user_agent"] = self.user_agent
        self.context = await self.browser.new_context(**context_kwargs)
        self.page = await self.context.new_page()
        await self.page.goto(self.start_url)

    async def get_state(self, format: str = "hybrid") -> InterfaceState:
        """Capture the current page state."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        state = InterfaceState(url=self.page.url, title=await self.page.title())

        if format in ("text", "hybrid"):
            try:
                content = await self.page.evaluate(
                    """() => {
                        const body = document.body;
                        return body ? body.innerText : "";
                    }"""
                )
                state.content = (content or "")[:4000]
            except Exception:
                state.content = ""
            state.interactive_elements = await self._get_interactive_elements()

        if format in ("visual", "hybrid"):
            try:
                state.screenshot = await self.get_screenshot()
            except Exception:
                state.screenshot = None

        return state

    async def _get_interactive_elements(self) -> List[Dict[str, Any]]:
        """Extract visible, interactive elements from the page."""
        if not self.page:
            return []
        return await self.page.evaluate(
            """() => {
                const selectors = [
                    'button', 'a', 'input', 'select', 'textarea',
                    '[role="button"]', '[role="link"]', '[onclick]'
                ];
                const elements = [];
                selectors.forEach(selector => {
                    document.querySelectorAll(selector).forEach(el => {
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 0 && rect.height > 0) {
                            const text = (el.innerText || el.value ||
                                el.getAttribute('aria-label') || '').trim();
                            const aria = el.getAttribute('aria-label');
                            let sel = el.tagName.toLowerCase();
                            if (el.id) sel = '#' + el.id;
                            else if (aria) sel = `[aria-label="${aria}"]`;
                            elements.push({
                                tag: el.tagName.toLowerCase(),
                                type: el.type || '',
                                text: text.slice(0, 100),
                                placeholder: el.placeholder || '',
                                href: el.href || '',
                                id: el.id || '',
                                selector: sel
                            });
                        }
                    });
                });
                return elements.slice(0, 30);
            }"""
        )

    async def execute_action(self, action: Action) -> ActionResult:
        """Execute a browser action."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")

        try:
            if action.action_type == ActionType.NAVIGATE:
                if not action.value:
                    raise ValueError("Navigate action requires a URL value")
                try:
                    await self.page.goto(action.value, wait_until="domcontentloaded", timeout=15000)
                except Exception:
                    await self.page.goto(action.value, timeout=30000)
                result = ActionResult(
                    success=True,
                    description=f"Navigated to {self.page.url}",
                    metadata={"url": self.page.url},
                )

            elif action.action_type == ActionType.CLICK:
                if not action.selector:
                    raise ValueError("Click action requires a selector")
                selector = await self._resolve_selector(action.selector)
                await self.page.click(selector, timeout=5000)
                result = ActionResult(
                    success=True,
                    description=f"Clicked on {action.selector}",
                )

            elif action.action_type == ActionType.TYPE:
                if not action.selector or action.value is None:
                    raise ValueError("Type action requires both selector and value")
                selector = await self._resolve_selector(action.selector)
                await self.page.fill(selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Typed into {action.selector}",
                )

            elif action.action_type == ActionType.SELECT:
                if not action.selector or action.value is None:
                    raise ValueError("Select action requires both selector and value")
                await self.page.select_option(action.selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Selected {action.value} in {action.selector}",
                )

            elif action.action_type == ActionType.PRESS:
                if not action.selector or action.value is None:
                    raise ValueError("Press action requires both selector and key")
                await self.page.press(action.selector, action.value)
                result = ActionResult(
                    success=True,
                    description=f"Pressed {action.value} on {action.selector}",
                )

            elif action.action_type == ActionType.HOVER:
                if not action.selector:
                    raise ValueError("Hover action requires a selector")
                await self.page.hover(action.selector)
                result = ActionResult(success=True, description=f"Hovered over {action.selector}")

            elif action.action_type == ActionType.SCROLL:
                direction = action.value or "down"
                amount = action.metadata.get("amount", 500)
                if isinstance(direction, str) and direction.lower() in ("up", "down", "left", "right"):
                    dx, dy = 0, 0
                    if direction == "down":
                        dy = amount
                    elif direction == "up":
                        dy = -amount
                    elif direction == "right":
                        dx = amount
                    elif direction == "left":
                        dx = -amount
                else:
                    try:
                        dy = int(direction)
                    except (ValueError, TypeError):
                        dy = amount
                    dx = 0

                if action.selector:
                    await self.page.evaluate(
                        f"(el) => el.scrollBy({{top: {dy}, left: {dx}, behavior: 'smooth'}})",
                        await self.page.locator(action.selector).first,
                    )
                else:
                    await self.page.evaluate(f"() => window.scrollBy({{top: {dy}, left: {dx}, behavior: 'smooth'}})")
                result = ActionResult(
                    success=True,
                    description=f"Scrolled {direction} by {amount}px",
                )

            elif action.action_type == ActionType.SCREENSHOT:
                screenshot = await self.get_screenshot()
                result = ActionResult(
                    success=True,
                    description="Captured screenshot",
                    screenshot=screenshot,
                )

            else:
                result = ActionResult(
                    success=False,
                    description="",
                    error=f"Unsupported action type: {action.action_type}",
                )

            self._action_history.append(action)
            return result

        except Exception as e:
            return ActionResult(success=False, description="", error=str(e))

    async def _resolve_selector(self, selector: str) -> str:
        """Try to resolve a human-like selector to a Playwright selector."""
        # If it looks like a CSS selector, prefer it directly.
        if re.match(r"^[.#\[a-zA-Z]", selector.strip()):
            try:
                await self.page.locator(selector).first.wait_for(timeout=1000)
                return selector
            except Exception:
                pass
        # Try text-based selection.
        try:
            text = selector.strip().strip('"').strip("'")
            await self.page.get_by_text(text, exact=False).first.wait_for(timeout=1000)
            return f"text={text}"
        except Exception:
            pass
        # Fallback: label/aria.
        try:
            await self.page.get_by_label(selector).first.wait_for(timeout=1000)
            return f"[aria-label='{selector}']"
        except Exception:
            pass
        return selector

    async def get_screenshot(self) -> bytes:
        """Capture the current viewport as PNG."""
        if not self.page:
            raise RuntimeError("Browser not initialized. Call initialize() first.")
        return await self.page.screenshot(type="png")

    async def close(self) -> None:
        """Close the browser and release resources."""
        if self.page:
            try:
                await self.page.close()
            except Exception:
                pass
            self.page = None
        if self.context:
            try:
                await self.context.close()
            except Exception:
                pass
            self.context = None
        if self.browser:
            try:
                await self.browser.close()
            except Exception:
                pass
            self.browser = None
        if self.playwright:
            try:
                await self.playwright.stop()
            except Exception:
                pass
            self.playwright = None
