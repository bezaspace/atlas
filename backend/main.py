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
from typing import Any, AsyncGenerator, List, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from atlascore import OpenAIChatCompletionClient
from atlascore.memory import BaseMemory
from atlascore.tools import MCPClientManager, WebFetchTool, WebSearchTool
from atlascore.tools._mcp import (
    MCP_AVAILABLE,
    HTTPServerConfig,
    StdioServerConfig,
)

from .eval import EvalConfig, EvalHarness
from .execution import EngineConfig, ResearchExecutionEngine
from .models import (
    ApprovalRequest,
    CreateSessionRequest,
    EvalRequest,
    HealthResponse,
    MCPServerInfo,
    MCPToolInfo,
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


def _default_triage_model_client() -> Optional[OpenAIChatCompletionClient]:
    """Create a cheap triage model client if LLM_CHEAP_MODEL is configured."""
    api_key = os.getenv("LLM_API_KEY")
    cheap_model = os.getenv("LLM_CHEAP_MODEL")
    if not api_key or not cheap_model:
        return None
    return OpenAIChatCompletionClient(
        model=cheap_model,
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL") or None,
    )


def _default_vision_model_client() -> Optional[OpenAIChatCompletionClient]:
    """Create a vision-capable model client if a model/API key is configured."""
    api_key = os.getenv("LLM_VISION_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    vision_model = os.getenv("LLM_VISION_MODEL")
    if not vision_model:
        return None
    return OpenAIChatCompletionClient(
        model=vision_model,
        api_key=api_key,
        base_url=os.getenv("LLM_VISION_BASE_URL") or os.getenv("LLM_BASE_URL") or None,
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


def _mcp_server_configs() -> List[Any]:
    """Build default MCP server configs from the environment."""
    raw = os.getenv("MCP_SERVERS")
    if raw:
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                return [_parse_mcp_server_config(item) for item in data]
        except Exception:
            pass

    configs: List[Any] = []
    exa_url = os.getenv("MCP_EXA_URL", "https://mcp.exa.ai/mcp")
    configs.append(
        HTTPServerConfig(server_id="exa", url=exa_url, transport="streamable-http")
    )

    if os.getenv("MCP_ARXIV_ENABLED", "").lower() in ("1", "true", "yes"):
        configs.append(
            StdioServerConfig(
                server_id="arxiv",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-arxiv"],
            )
        )

    if os.getenv("MCP_GITHUB_ENABLED", "").lower() in ("1", "true", "yes"):
        configs.append(
            StdioServerConfig(
                server_id="github",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-github"],
                env={"GITHUB_TOKEN": os.getenv("GITHUB_TOKEN", "")},
            )
        )

    if os.getenv("MCP_NEWS_ENABLED", "").lower() in ("1", "true", "yes"):
        configs.append(
            StdioServerConfig(
                server_id="news",
                command="npx",
                args=["-y", "@modelcontextprotocol/server-news"],
            )
        )

    return configs


def _parse_mcp_server_config(item: Any) -> Any:
    """Parse a single MCP server config from JSON/env representation."""
    if isinstance(item, str):
        return HTTPServerConfig(server_id=item, url=item, transport="streamable-http")

    server_id = item.get("server_id") or item.get("id")
    transport = item.get("transport", "streamable-http")
    url = item.get("url", "")
    command = item.get("command", "")
    args = item.get("args", [])
    env = item.get("env")
    headers = item.get("headers")

    if transport == "stdio":
        return StdioServerConfig(
            server_id=server_id, command=command, args=args, env=env
        )

    return HTTPServerConfig(
        server_id=server_id,
        url=url,
        transport=transport,
        headers=headers,
        env=env,
    )


def _default_mcp_manager() -> Optional[MCPClientManager]:
    """Create an MCP manager with default server configs but do not connect yet."""
    if not MCP_AVAILABLE:
        return None

    configs = _mcp_server_configs()
    if not configs:
        return None

    manager = MCPClientManager()
    for config in configs:
        try:
            manager.add_server(config)
        except Exception:
            pass
    return manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: initialize shared state."""
    app.state.session_manager = SessionManager()
    app.state.model_client = _default_model_client()
    app.state.triage_model_client = _default_triage_model_client()
    app.state.vision_model_client = _default_vision_model_client()
    app.state.search_tool = _default_search_tool()
    app.state.fetch_tool = WebFetchTool()
    app.state.memory = _default_memory()
    app.state.mcp_manager = _default_mcp_manager()
    app.state.persist_dir = os.getenv("ATLAS_PERSIST_DIR", "data/research")

    # Best-effort background MCP connection; failures do not block startup.
    mcp_manager = app.state.mcp_manager
    if mcp_manager is not None:
        asyncio.create_task(
            asyncio.wait_for(
                mcp_manager.connect_all(),
                timeout=float(os.getenv("MCP_CONNECT_TIMEOUT", "30")),
            )
        )

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
    triage_model_client = getattr(request.app.state, "triage_model_client", None)
    vision_model_client = getattr(request.app.state, "vision_model_client", None)
    memory = getattr(request.app.state, "memory", None)
    mcp_manager = getattr(request.app.state, "mcp_manager", None)
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
        triage_model_client=triage_model_client,
        vision_model_client=vision_model_client,
        memory=memory,
        mcp_manager=mcp_manager,
        persist_dir=request.app.state.persist_dir,
    )


def _search_provider_name() -> Optional[str]:
    if os.getenv("TAVILY_API_KEY"):
        return "tavily"
    if os.getenv("GOOGLE_API_KEY") and os.getenv("GOOGLE_CSE_ID"):
        return "google"
    return None


def _triage_model_name() -> Optional[str]:
    return os.getenv("LLM_CHEAP_MODEL") or None


def _vision_model_name() -> Optional[str]:
    return os.getenv("LLM_VISION_MODEL") or None


def _embedding_provider_name() -> Optional[str]:
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    return None


def _get_mcp_manager(request: Request) -> Optional[MCPClientManager]:
    return getattr(request.app.state, "mcp_manager", None)


@app.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check with configured model, search, embedding, and MCP servers."""
    mcp_manager = _get_mcp_manager(request)
    mcp_servers = mcp_manager.list_servers() if mcp_manager else []
    return HealthResponse(
        status="healthy",
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        triage_model=_triage_model_name(),
        vision_model=_vision_model_name(),
        search_provider=_search_provider_name(),
        embedding_provider=_embedding_provider_name(),
        mcp_servers=mcp_servers,
        version="0.1.0",
    )


@app.get("/mcp/servers", response_model=List[MCPServerInfo])
async def list_mcp_servers(request: Request):
    """List registered MCP servers and their connection status."""
    mcp_manager = _get_mcp_manager(request)
    if not mcp_manager:
        return []

    result = []
    for sid in mcp_manager.list_servers():
        config = getattr(mcp_manager, "_servers", {}).get(sid)
        result.append(
            MCPServerInfo(
                server_id=sid,
                transport=config.transport if config else "unknown",
                connected=mcp_manager.is_connected(sid),
                tool_count=len(mcp_manager.get_tools(sid)),
            )
        )
    return result


@app.get("/mcp/tools", response_model=List[MCPToolInfo])
async def list_mcp_tools(request: Request):
    """List tools discovered from connected MCP servers."""
    mcp_manager = _get_mcp_manager(request)
    if not mcp_manager:
        return []

    tools = mcp_manager.get_tools()
    return [
        MCPToolInfo(
            name=tool.name,
            server_id=getattr(tool, "server_id", ""),
            description=tool.description,
            approval_mode=tool.approval_mode.value,
        )
        for tool in tools
    ]


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
        triage_model_client=config.triage_model_client,
        vision_model_client=config.vision_model_client,
        memory=config.memory,
        mcp_manager=config.mcp_manager,
        persist_dir=config.persist_dir,
        approval_event_factory=approval_event_factory,
    )


# Serve the built React dashboard when a dist folder is present.
_frontend_dist = Path("frontend/dist")
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="static")
