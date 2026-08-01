"""Atlas FastAPI backend.

Run:
    export LLM_API_KEY="..."
    export LLM_BASE_URL="https://api.openai.com/v1"  # optional
    export LLM_MODEL="gpt-4o-mini"
    export TAVILY_API_KEY="..."  # or GOOGLE_API_KEY + GOOGLE_CSE_ID
    uvicorn backend.main:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlascore import OpenAIChatCompletionClient
from atlascore.memory import BaseMemory
from atlascore.tools import WebFetchTool, WebSearchTool

from .eval import EvalConfig, EvalHarness
from .execution import EngineConfig, ResearchExecutionEngine
from .models import (
    ApprovalRequest,
    CreateSessionRequest,
    EvalRequest,
    HealthResponse,
    RunRequest,
    SessionInfo,
)
from .session_store import Session, SessionManager


def _default_model_client() -> Optional[OpenAIChatCompletionClient]:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    return OpenAIChatCompletionClient(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def _default_search_tool() -> Optional[WebSearchTool]:
    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key:
        return WebSearchTool(api_key=tavily_key, provider="tavily")
    google_key = os.getenv("GOOGLE_API_KEY")
    cse_id = os.getenv("GOOGLE_CSE_ID")
    if google_key and cse_id:
        return WebSearchTool(api_key=google_key, provider="google", cse_id=cse_id)
    return None


def _default_memory() -> Optional[BaseMemory]:
    """Initialize Qdrant-backed memory with Gemini embeddings, or fall back."""
    try:
        from atlascore.embeddings import GeminiEmbeddingClient
        from atlascore.memory import QdrantMemory
    except ImportError:
        return None

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    embedding_fn = None
    if gemini_api_key:
        try:
            embedding_fn = GeminiEmbeddingClient(
                api_key=gemini_api_key,
                model=os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
            )
        except Exception:
            embedding_fn = None

    qdrant_path = os.getenv("QDRANT_PATH", "data/qdrant")
    try:
        return QdrantMemory(
            collection_name="atlas_rag",
            path=qdrant_path,
            embedding_fn=embedding_fn,
        )
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize shared state."""
    app.state.session_manager = SessionManager()
    app.state.model_client = _default_model_client()
    app.state.search_tool = _default_search_tool()
    app.state.fetch_tool = WebFetchTool()
    app.state.memory = _default_memory()
    app.state.persist_dir = os.getenv("ATLAS_PERSIST_DIR", "data/research")
    yield


app = FastAPI(title="Atlas", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_session_manager(request: Request) -> SessionManager:
    if not hasattr(request.app.state, "session_manager"):
        request.app.state.session_manager = SessionManager()
    return request.app.state.session_manager


def get_engine_config(request: Request) -> EngineConfig:
    """Return the runtime engine config (may be partially configured)."""
    model_client = request.app.state.model_client
    search_tool = request.app.state.search_tool
    fetch_tool = request.app.state.fetch_tool
    memory = getattr(request.app.state, "memory", None)
    if model_client is None or search_tool is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Backend not fully configured. Set LLM_API_KEY and "
                "TAVILY_API_KEY or GOOGLE_API_KEY + GOOGLE_CSE_ID."
            ),
        )
    return EngineConfig(
        model_client=model_client,
        search_tool=search_tool,
        fetch_tool=fetch_tool,
        memory=memory,
        persist_dir=request.app.state.persist_dir,
    )


def _search_provider_name() -> Optional[str]:
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        return "google"
    return None


def _embedding_provider_name() -> Optional[str]:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return None


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check with configured model, search, and embedding provider."""
    return HealthResponse(
        status="healthy",
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        search_provider=_search_provider_name(),
        embedding_provider=_embedding_provider_name(),
        version="0.1.0",
    )


@app.post("/sessions", response_model=SessionInfo)
async def create_session(
    request: CreateSessionRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Create a new execution session."""
    session_id = session_manager.create_session_id()
    session = await session_manager.get_or_create(
        session_id, request.entity_id, request.entity_type
    )
    return SessionInfo(
        id=session_id,
        entity_id=session.entity_id,
        entity_type=session.entity_type,
        status=session.status,
        created_at=session.created_at.isoformat(),
        last_activity=session.created_at.isoformat(),
    )


@app.get("/sessions")
async def list_sessions(
    entity_id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """List sessions with metadata."""
    return await session_manager.list(entity_id)


@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Get session context."""
    session = await session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session.context.model_dump()


@app.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Delete a session and cancel any running task."""
    success = await session_manager.delete(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@app.post("/sessions/{session_id}/run")
async def run_session(
    session_id: str,
    request: RunRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    config: EngineConfig = Depends(get_engine_config),
):
    """Start a research workflow for the session."""
    session = await session_manager.get_or_create(
        session_id, "research_pipeline", "research"
    )
    if session.status == "running":
        raise HTTPException(status_code=409, detail="Session already running")

    engine = ResearchExecutionEngine(session_manager, config)
    session.metadata["run_request"] = request.model_dump()
    session.task = asyncio.create_task(
        engine.run_pipeline(
            session,
            request.query,
            context=request.context,
            require_human_approval=request.require_human_approval,
        )
    )
    return {"session_id": session_id, "status": "running"}


@app.get("/sessions/{session_id}/stream")
async def stream_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Stream session events as Server-Sent Events."""
    session = await session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(session.events.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # Allow periodic checks for cancellation/disconnect
                    continue
                if event is None:
                    break
                yield event
        except asyncio.CancelledError:
            if session.cancellation_token:
                session.cancellation_token.cancel()
            if session.task and not session.task.done():
                session.task.cancel()
            raise

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.post("/sessions/{session_id}/approve")
async def approve_session(
    session_id: str,
    request: ApprovalRequest,
    session_manager: SessionManager = Depends(get_session_manager),
):
    """Inject tool approvals and/or signal human approval for a paused session."""
    session = await session_manager.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    for approval in request.approvals:
        session.context.add_approval_response(approval)

    session.metadata["human_approved"] = request.approved
    session.metadata["human_feedback"] = request.feedback
    session.approval_event.set()

    await session_manager.update(session_id, session)
    return {"session_id": session_id, "status": "approved", "approved": request.approved}


@app.post("/eval")
async def run_eval(
    request: EvalRequest,
    session_manager: SessionManager = Depends(get_session_manager),
    config: EngineConfig = Depends(get_engine_config),
):
    """Start an eval run for a dataset."""
    session_id = session_manager.create_session_id()
    session = await session_manager.get_or_create(session_id, "eval", "eval")
    session.status = "running"

    harness = EvalHarness(
        EvalConfig(pipeline=_make_pipeline(config, session), judge_client=config.model_client),
        max_items=request.max_items,
    )

    async def eval_task() -> None:
        try:
            async for payload in harness.run(request.dataset_path):
                session.events.put_nowait(
                    f"data: {json.dumps({'session_id': session_id, 'event': payload})}\n\n"
                )
        except Exception as e:
            session.events.put_nowait(
                f"data: {json.dumps({'session_id': session_id, 'event': {'type': 'error', 'message': str(e)}})}\n\n"
            )
        finally:
            session.events.put_nowait(None)
            session.status = "completed"
            await session_manager.update(session_id, session)

    session.task = asyncio.create_task(eval_task())
    return {"session_id": session_id, "status": "running"}


def _make_pipeline(config: EngineConfig, session: Session) -> Any:
    """Build a ResearchPipeline wired to the session's approval event."""
    from atlascore.research import ResearchPipeline

    def approval_event_factory(session_id: str) -> asyncio.Event:
        return session.approval_event

    return ResearchPipeline(
        model_client=config.model_client,
        search_tool=config.search_tool,
        fetch_tool=config.fetch_tool,
        triage_model_client=config.model_client,
        memory=config.memory,
        persist_dir=config.persist_dir,
        approval_event_factory=approval_event_factory,
    )


# Serve the built React dashboard when a dist folder is present.
_frontend_dist = Path("frontend/dist")
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
