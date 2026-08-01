"""Execution engine for streaming research workflows over SSE."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional

from atlascore import OpenAIChatCompletionClient
from atlascore.cancellation import CancellationToken
from atlascore.memory import BaseMemory
from atlascore.research import ResearchPipeline
from atlascore.tools import MCPClientManager, WebFetchTool, WebSearchTool
from atlascore.workflow import WorkflowCompletedEvent, WorkflowRunner

from .session_store import Session, SessionManager


@dataclass
class EngineConfig:
    """Configuration for the research execution engine."""

    model_client: OpenAIChatCompletionClient
    search_tool: WebSearchTool
    fetch_tool: WebFetchTool
    triage_model_client: Optional[OpenAIChatCompletionClient] = None
    vision_model_client: Optional[OpenAIChatCompletionClient] = None
    memory: Optional[BaseMemory] = None
    mcp_manager: Optional[MCPClientManager] = None
    persist_dir: str = "data/research"


class ResearchExecutionEngine:
    """Runs research pipelines and streams workflow events to sessions."""

    def __init__(self, session_manager: SessionManager, config: EngineConfig) -> None:
        self.session_manager = session_manager
        self.config = config

    async def run_pipeline(
        self,
        session: Session,
        query: str,
        context: Optional[str] = None,
        require_human_approval: bool = False,
    ) -> None:
        """Run the research pipeline and put SSE events into the session queue."""
        session.status = "running"
        session.metadata["query"] = query
        session.metadata["require_human_approval"] = require_human_approval
        await self.session_manager.update(session.session_id, session)

        cancellation_token = CancellationToken()
        session.cancellation_token = cancellation_token

        def approval_event_factory(session_id: str) -> asyncio.Event:
            # Synchronous factory; the event lives on the session
            return session.approval_event

        try:
            pipeline = ResearchPipeline(
                model_client=self.config.model_client,
                search_tool=self.config.search_tool,
                fetch_tool=self.config.fetch_tool,
                triage_model_client=self.config.triage_model_client,
                vision_model_client=self.config.vision_model_client,
                memory=self.config.memory,
                mcp_manager=self.config.mcp_manager,
                persist_dir=self.config.persist_dir,
                approval_event_factory=approval_event_factory if require_human_approval else None,
            )
            workflow = pipeline.build_workflow()
            runner = WorkflowRunner()

            initial_input = {
                "query": query,
                "context": context,
                "session_id": session.session_id,
            }

            async def event_stream() -> AsyncGenerator[Dict[str, Any], None]:
                async for event in runner.run_stream(
                    workflow,
                    initial_input=initial_input,
                    cancellation_token=cancellation_token,
                ):
                    payload: Dict[str, Any] = {
                        "session_id": session.session_id,
                        "event": event.model_dump(mode="json"),
                    }
                    if isinstance(event, WorkflowCompletedEvent):
                        report = event.execution.state.get("persist_output")
                        if report:
                            payload["event"]["report"] = report
                    yield payload

            async for payload in event_stream():
                if cancellation_token.is_cancelled():
                    break
                session.events.put_nowait(
                    f"data: {json.dumps(payload)}\n\n"
                )

            session.status = "completed"
        except asyncio.CancelledError:
            session.status = "cancelled"
            session.events.put_nowait(
                f"data: {json.dumps({'session_id': session.session_id, 'event': {'type': 'cancelled'}})}\n\n"
            )
        except Exception as e:
            session.status = "failed"
            error_payload = {
                "session_id": session.session_id,
                "event": {"type": "error", "message": str(e)},
            }
            session.events.put_nowait(f"data: {json.dumps(error_payload)}\n\n")
        finally:
            session.events.put_nowait(None)  # Sentinel
            await self.session_manager.update(session.session_id, session)
