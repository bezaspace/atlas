# Atlas — Implementation Progress

This file tracks what has been implemented and what is next for the Atlas project.

## Latest state

- **Current branch target:** `master`
- **Last milestone completed:** Phase 14 (Evaluation harness with LLM-as-judge, reference judge, golden dataset, regression runner, and frontend eval viewer integration)
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
| 11 | MCP integration | Done | `MCPClientManager` + `MCPTool` (implements `BaseTool`); stdio/SSE/streamable-http transports; default Exa remote (no API key/Node); optional arXiv/GitHub/news stdio servers; destructive tools require approval; `/mcp/servers` and `/mcp/tools` API; `mcp` optional extra in `pyproject.toml` |
| 12 | Computer-use fallback | Done | `PlaywrightWebClient` + browser tools; `ComputerUseAgent` streams screenshots to a vision LLM; `ResearcherAgent` falls back to browser agent when `WebFetchTool` fails; `LLM_VISION_MODEL` config in `/health` |
| 13 | Cost optimization | Done | `TriageAgent` backed by configurable cheap model (`LLM_CHEAP_MODEL`); only relevant/partial results sent to strong model; `Usage.cost_estimate` for free and paid OpenAI/Anthropic/Google/Kilo models; cost surfaced in `AgentResponse`, aggregated by `ResearchPipeline`, and shown in the dashboard; benchmark verifies ~80-90% cost reduction |
| 14 | Evaluation harness | Done | `atlascore/eval` package with `LLMEvalJudge` (structured `CriterionScore` output), `ReferenceEvalJudge` (fuzzy/contains/exact + citation overlap), `Dataset`, `EvalRunner`, `EvalResults`; golden dataset in `eval/golden/research.json`; `tests/eval/test_regression.py` loads golden set and enforces baseline; `backend/eval.py` runs `ResearchPipeline` through `EvalRunner` and returns `EvalReport` to the frontend eval viewer |
| 15 | Framework comparison benchmark | Done | LangGraph plan-based research pipeline under `benchmarks/langgraph/` (`LangGraphResearchPipeline`); `benchmarks/benchmark.py` harness comparing `atlascore` vs LangGraph on quality, latency, and cost; `benchmarks/generate_report.py` produces `docs/benchmark_report.md`; `tests/test_benchmark.py` asserts both pipelines produce comparable `ResearchReport` objects and the harness reports per-query metrics |
| 16+ | Deployment | Not started | Later phases |

## Verification of completed work

- `pytest tests/` — 96 passed, 1 skipped (cloud integration tests skipped by default)
- `tests/eval/test_regression.py` loads `eval/golden/research.json`, runs the `EvalRunner` with `ReferenceEvalJudge`, and asserts the golden set meets an 0.85 average-score baseline while a degraded target fails a 0.6 baseline
- `tests/test_cost_optimization.py` verifies `TriageAgent` filters irrelevant sources and that two-stage (cheap triage + strong extraction) is ~80-90% cheaper than a naive single-model run
- `tests/test_computer_use.py` covers `PlaywrightWebClient` initialization/state/screenshot, browser tools, `ComputerUseAgent` tool-call loop, and `ResearcherAgent` browser fallback on fetch failure/sparse content
- `tests/test_mcp.py` covers `MCPTool` execution, destructive/read-only approval-mode detection, and best-effort offline MCP server handling
- `tests/test_qdrant_memory.py` covers add/query/get_context/clear for `:memory:` and local path; cloud test verified against Qdrant Cloud
- `tests/test_workflow.py` covers DAG builder/validation, runner, fan-in, conditional edges, checkpoint save/resume, file store, and `AgentStep`
- `tests/test_orchestration.py` covers round-robin, AI-driven, and plan-based orchestrators with mocked LLM clients
- `tests/test_research.py` covers Planner, Triage, Researcher, Verifier, Synthesizer, CriticPanel, and full `ResearchPipeline`
- `tests/test_backend.py` covers `/health`, `/sessions`, `POST /sessions/{id}/run`, `GET /sessions/{id}/stream` (SSE), `POST /sessions/{id}/approve` resume, and `POST /eval`
- `tests/test_benchmark.py` asserts the from-scratch and LangGraph pipelines produce comparable `ResearchReport` objects and that the benchmark harness reports per-query quality, latency, and cost metrics
- Frontend `npm run build` outputs `frontend/dist` that the FastAPI backend can serve
- `npm run lint` (TypeScript) and `npm run build` pass in `frontend/`
- `ruff check src backend tests examples benchmarks` — clean
- `pyright` (includes `src` and `backend`) — clean (one pre-existing `__all__` warning)
- `uvicorn backend.main:app --port 8000` starts and `/health` reports the configured model, vision model, embedding provider, and registered MCP servers; `/mcp/servers` and `/mcp/tools` list discovered MCP tools
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
# Optional MCP servers (Exa remote is enabled by default and needs no API key)
export MCP_EXA_URL=https://mcp.exa.ai/mcp
export MCP_ARXIV_ENABLED=0
export MCP_GITHUB_ENABLED=0
export MCP_NEWS_ENABLED=0
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

Phase 14 is complete. Next is **Phase 15+** (framework comparison benchmark, deployment, and beyond).

## References

- `TASK_BACKLOG.md` — full ordered task list
- `PROJECT_PLAN.md` — architecture and feature overview
- `src/atlascore/` — implemented core framework
- `backend/` — FastAPI backend and session management
- `frontend/` — React dashboard
