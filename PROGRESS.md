# Atlas — Implementation Progress

This file tracks what has been implemented and what is next for the Atlas project.

## Latest state

- **Current branch target:** `master`
- **Last milestone completed:** Phase 10 (RAG knowledge base with Gemini embeddings)
- **Package:** `atlascore` is installable and tested

## Milestones

| Phase | Goal | Status | Notes |
|-------|------|--------|-------|
| 0 | Repo, toolchain, environment | Done | `pyproject.toml`, `README.md`, `.env.example`, `ruff`, `pyright`, `pytest` |
| 1 | atlascore primitives | Done | Messages, events, context, usage, response types, OpenAI-compatible LLM client |
| 2 | Tool system | Done | `BaseTool`, `FunctionTool`, schema generation, core tools (`calculator`, `datetime`, `think`, `json_parser`, `regex`, `task_status`) |
| 3.1 | Agent reasoning loop (`run()`) | Done | Sequential tool-call loop, `AgentResponse`, `max_iterations` |
| 3.2 | Agent streaming (`run_stream()`) | Done | Yields events and final response, cancellation token support |
| 3.3 | Memory injection | Done | `BaseMemory`, `ListMemory`, `FileMemory`, `QdrantMemory` (optional); Agent prepends memory context |
| 3.4 | Structured output | Done | `ResearchBrief`, `Citation`, `Evidence`, `VerificationResult`; `Agent.output_format` |
| 3.5 | Termination conditions | Done | `MaxMessageTermination`, `TokenUsageTermination`, `TimeoutTermination`, `TextMentionTermination`, `ExternalTermination`, `CompositeTermination` |
| 3.6 | Middleware / approval | Done | `MiddlewareChain`, `BaseMiddleware`, `LoggingMiddleware`, `ApprovalMiddleware`, `ToolApprovalEvent`/`ToolApprovalRequest` pause/resume |
| 3.7 | OpenTelemetry | Done | `OTelMiddleware` + `auto_instrument()` with Gen-AI semantic conventions, gated by `ATLAS_ENABLE_OTEL` |
| 4 | Memory | Done | `ListMemory`, `FileMemory`, `QdrantMemory` (with `:memory:`, local path, and Qdrant Cloud support) |
| 5 | Workflow engine | Done | `Workflow`, `FunctionStep`, `AgentStep`, `WorkflowRunner` with streaming, `FileCheckpointStore`/`InMemoryCheckpointStore`, DAG validation |
| 6 | Orchestration | Done | `BaseOrchestrator`, `RoundRobinOrchestrator`, `AIOrchestrator`, `PlanBasedOrchestrator` with streaming and usage aggregation |
| 7 | Research product | Done | `ResearchPlan`, `SearchResult`, `TriageResult`, `CriticReview`, `ResearchReport`; `WebSearchTool`/`WebFetchTool`; `PlannerAgent`, `TriageAgent`, `ResearcherAgent`, `VerifierAgent`, `SynthesizerAgent`; `CriticPanel`; full typed research pipeline DAG |
| 8 | Backend API | Done | FastAPI + lifespan + CORS; `/health`; in-memory `SessionManager`/`SessionStore`; `POST /sessions/{id}/run` + `GET /sessions/{id}/stream` (SSE); `POST /sessions/{id}/approve` human-in-the-loop gate; `POST /eval` dataset eval harness with LLM-as-judge; tests |
| 9 | Frontend dashboard | Done | React 19 + TypeScript + Vite + Tailwind CSS v4; live SSE activity feed; research query input; markdown brief with citations; sources panel; human approval gate UI; sessions/eval viewers; cost/trace inspector |
| 10 | RAG knowledge base | Done | Persistent Qdrant collection for briefs/sources; Gemini `gemini-embedding-001` free-tier embeddings with local sentence-transformers fallback; retrieval at plan/research/synthesis time; source deduplication and `data/sources/<url_hash>.md` re-hydration files |
| 11+ | MCP, computer-use, deployment | Not started | Later phases |

## Verification of completed work

- `pytest tests/` — 77 passed, 1 skipped (cloud integration tests skipped by default)
- `tests/test_qdrant_memory.py` covers add/query/get_context/clear for `:memory:` and local path; cloud test verified against Qdrant Cloud
- `tests/test_workflow.py` covers DAG builder/validation, runner, fan-in, conditional edges, checkpoint save/resume, file store, and `AgentStep`
- `tests/test_orchestration.py` covers round-robin, AI-driven, and plan-based orchestrators with mocked LLM clients
- `tests/test_research.py` covers Planner, Triage, Researcher, Verifier, Synthesizer, CriticPanel, and full `ResearchPipeline`
- `tests/test_backend.py` covers `/health`, `/sessions`, `POST /sessions/{id}/run`, `GET /sessions/{id}/stream` (SSE), `POST /sessions/{id}/approve` resume, and `POST /eval`
- Frontend `npm run build` outputs `frontend/dist` that the FastAPI backend can serve
- `npm run lint` (TypeScript) and `npm run build` pass in `frontend/`
- `ruff check src backend tests examples` — clean
- `pyright` (includes `src` and `backend`) — clean (one pre-existing `__all__` warning)
- `uvicorn backend.main:app --port 8000` starts and `/health` reports the configured embedding provider
- Real API smoke test (`examples/calculator_agent.py`) used `calculator` and `datetime` tools correctly.
- New example `examples/orchestration_demo.py` provides a poet/critic round-robin smoke script.
- New example `examples/research_pipeline.py` demonstrates the full research pipeline with live web search/fetch.

## Running the app

### Backend

```bash
source .venv/bin/activate
export LLM_API_KEY="..."
export LLM_MODEL="gpt-4o-mini"
export TAVILY_API_KEY="..."  # or GOOGLE_API_KEY + GOOGLE_CSE_ID
export GEMINI_API_KEY="..."   # free-tier Gemini embeddings for RAG
uvicorn backend.main:app --port 8000
```

### Frontend (dev)

```bash
cd frontend
npm install
npm run dev
```

### Frontend (production build served by FastAPI)

```bash
cd frontend
npm run build
cd ..
source .venv/bin/activate
uvicorn backend.main:app --port 8000
```

## Next recommended step

Phase 10 is complete. Next is **Phase 11+** (MCP, computer-use, deployment).

## References

- `TASK_BACKLOG.md` — full ordered task list
- `PROJECT_PLAN.md` — architecture and feature overview
- `src/atlascore/` — implemented core framework
- `backend/` — FastAPI backend and session management
- `frontend/` — React dashboard
