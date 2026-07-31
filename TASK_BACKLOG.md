# Atlas — Bounded Task Backlog (Draft)

This is a **planning-only** artifact. No implementation has been started. It translates the `PROJECT_PLAN.md` vision into ordered, testable tasks and calls out the dependencies, risks, and forgotten-to-mention items that usually bite a project like this.

## Current state

- `bezaspace/atlas` currently contains only `PROJECT_PLAN.md`.
- `bezaspace/designing-multiagent-systems` is the canonical reference (PicoAgents).
- Atlas should build its own from-scratch framework (`atlascore`) and then the product on top, mirroring the pedagogy of the reference repo rather than wrapping it.

## Tech stack

- **Backend:** Python 3.10+, FastAPI, Uvicorn, Pydantic v2, pytest-asyncio, Qdrant client, httpx, beautifulsoup4/html2text, Playwright (computer-use), OpenTelemetry SDK/exporters.
- **Frontend:** React 19, TypeScript, Vite, Tailwind CSS v4.
- **Vector store:** Qdrant (local path for single-session dev; Qdrant Cloud free cluster for cross-session persistence; source documents also stored as committed markdown/JSON so the index can be rebuilt).
- **Search providers:** Exa MCP (`https://mcp.exa.ai/mcp`, no API key, ~50 calls/day, 2 QPS) and Tavily (keyless or 1k credits/mo with API key). Google Custom Search / Perplexity as optional fallbacks.
- **MCP servers:** Exa MCP, arXiv MCP, GitHub MCP, news APIs via MCP.
- **Tracing:** Jaeger/Tempo via OTLP HTTP; `ATLAS_ENABLE_OTEL=true` style env flag.
- **Deployment:** Docker Compose (backend, frontend, qdrant, jaeger).

---

## Phase 0 — Repo, toolchain, and environment

### 0.1 Project skeleton and package structure
- **Goal:** Create the `pyproject.toml`, package directories, and `.env.example`.
- **Acceptance criteria:**
  - `pip install -e ".[dev,web,rag,computer-use,otel]"` installs all groups.
  - `python -c "import atlascore"` succeeds.
  - `.env.example` documents every required secret and config var.
- **Dependencies:** None.
- **Risks:** Naming collisions with the reference repo; keep `atlascore` distinct.

### 0.2 Test, lint, and type-check harness
- **Goal:** Wire `pytest`, `pytest-asyncio`, `ruff`/`black`, `pyright`/`mypy`, and a CI skeleton.
- **Acceptance criteria:**
  - `pytest` runs and passes with zero tests (or a smoke test).
  - `ruff check .` and `pyright` run cleanly on a `backend/atlascore` smoke file.
- **Dependencies:** 0.1.
- **Risks:** Pyright in strict mode is unforgiving; start with standard mode and tighten later.

### 0.3 Secrets and environment configuration
- **Goal:** Define how API keys, model endpoints, search credentials, and MCP tokens are loaded.
- **Acceptance criteria:**
  - Single `.env` file is validated at startup with Pydantic `BaseSettings`.
  - No secrets are hard-coded; failing validation produces a clear error message.
- **Dependencies:** 0.1.
- **Risks:** Missing keys must not crash unit tests; use pytest fixtures/monkeypatch.

---

## Phase 1 — atlascore primitives

### 1.1 Message and event types
- **Goal:** Implement the agent message protocol and streaming events.
- **Acceptance criteria:**
  - `SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage`, `ToolCallRequest`, `MultiModalMessage` exist and are Pydantic v2 models.
  - `AgentEvent` union covers `TaskStartEvent`, `TaskCompleteEvent`, `ModelCallEvent`, `ModelResponseEvent`, `ToolCallEvent`, `ToolCallResponseEvent`, `ToolApprovalEvent`, `ErrorEvent`, `MemoryUpdateEvent`, `MemoryRetrievalEvent`.
  - Models serialize/deserialize with `model_dump_json()` / `model_validate_json()`.
- **Dependencies:** 0.1.
- **Reference:** `picoagents/src/picoagents/messages.py`, `picoagents/src/picoagents/types.py`.

### 1.2 Context, usage, and cost models
- **Goal:** Build `AgentContext` (messages, metadata, shared_state, approvals) and `Usage`/`AgentResponse` types.
- **Acceptance criteria:**
  - `AgentContext` supports `add_message`, `reset`, `waiting_for_approval`, `add_approval_response`.
  - `Usage` supports aggregation (`__add__`) for parallel execution semantics.
  - `AgentResponse` exposes `messages`, `usage`, `finish_reason`, `needs_approval`.
- **Dependencies:** 1.1.
- **Reference:** `picoagents/src/picoagents/context.py`, `picoagents/src/picoagents/types.py`.

### 1.3 Base LLM client and OpenAI-compatible implementation
- **Goal:** Abstract `BaseChatCompletionClient` with `create()` and `create_stream()`, then implement an `OpenAIChatCompletionClient` that targets any `/v1/chat/completions` endpoint.
- **Acceptance criteria:**
  - Unified interface returns `ChatCompletionResult` with `AssistantMessage`, `usage`, `finish_reason`.
  - Streaming returns `ChatCompletionChunk` and final chunk with usage where available.
  - Tool calls are parsed and returned as `List[ToolCallRequest]`.
- **Dependencies:** 1.1, 1.2.
- **Reference:** `picoagents/src/picoagents/llm/_base.py`, `_openai.py`.

### 1.4 OpenAI-compatible client and provider aliases
- **Goal:** Build one generic `OpenAIChatCompletionClient` that works with any OpenAI-compatible endpoint by setting `base_url` and `api_key` (OpenAI, Groq, OpenRouter, Together, local Ollama/vLLM, Azure OpenAI).
- **Acceptance criteria:**
  - Client can be instantiated from config with `base_url`, `api_key`, `model`, and optional `default_headers`.
  - Provider-specific quirks (streaming, tool calls, token-usage fields) are normalized to the unified `ChatCompletionResult`.
  - `model_alias` mapping lets `cheap`/`strong` point to different providers without code changes.
- **Dependencies:** 1.3.
- **Reference:** `picoagents/src/picoagents/llm/_openai.py` (most providers expose `/v1/chat/completions`).
- **Risks:** Not all OpenAI-compatible endpoints support `response_format` (json_schema) or tool calling; test with the chosen provider and degrade gracefully to text + manual parsing.

---

## Phase 2 — Tool system

### 2.1 BaseTool, FunctionTool, and schema generation
- **Goal:** Tool abstraction with JSON-schema generation from type hints.
- **Acceptance criteria:**
  - `BaseTool` enforces `parameters`, `execute()`, `to_llm_format()`.
  - `FunctionTool` wraps a Python function/coroutine and infers JSON schema including `Literal` enums and required fields.
  - `validate_parameters()` catches missing/typed-mismatched args.
- **Dependencies:** 1.1.
- **Reference:** `picoagents/src/picoagents/tools/_base.py`.

### 2.2 Core deterministic tools
- **Goal:** Think, calculator, datetime, JSON parser, regex, and task-status tools.
- **Acceptance criteria:**
  - Each core tool executes without LLM calls and returns `ToolResult`.
  - Calculator is sandboxed (`eval` with limited namespace).
  - Regex tool supports `search`, `match`, `findall`, `replace` with flags.
- **Dependencies:** 2.1.
- **Reference:** `picoagents/src/picoagents/tools/_core_tools.py`.

### 2.3 Web search and web fetch tools
- **Goal:** Provide `WebSearchTool` and `WebFetchTool` with domain filtering. Primary search is the Exa MCP `web_search_exa` / `web_fetch_exa` endpoint because it works without an API key; Tavily remains a first-class fallback.
- **Acceptance criteria:**
  - Exa MCP search works out of the box with `https://mcp.exa.ai/mcp` (no API key) with rate-limit handling.
  - Tavily keyless and Tavily API-key modes are configurable.
  - `WebFetchTool` fetches via httpx, supports `html`/`text`/`markdown`, respects `allowed_domains`/`blocked_domains`, and truncates to `max_content_length`.
  - Failed fetches return `ToolResult` with `success=False` and error text; agent can recover.
- **Dependencies:** 2.1.
- **Reference:** `picoagents/src/picoagents/tools/_research_tools.py`, `exa.ai/docs/reference/exa-mcp`.
- **Risks:** Exa MCP free tier is ~50 calls/day and ~2 QPS (IP-based) without an API key; Tavily keyless is also rate-limited. Plan a fallback chain from day one.

### 2.4 Tool execution (sequential and parallel)
- **Goal:** Agent executes single and multiple tool calls with `asyncio.gather`.
- **Acceptance criteria:**
  - Multiple independent tool calls run concurrently.
  - Errors in one parallel call do not crash the whole set.
  - Tool results are converted to `ToolMessage` and appended to context.
- **Dependencies:** 2.1, 2.2, 2.3.
- **Reference:** `picoagents/src/picoagents/agents/_agent.py` lines 575-640.

---

## Phase 3 — Agent core

### 3.1 Reasoning/action loop and `run()`
- **Goal:** Implement `Agent` class with `run()` and `run_stream()`.
- **Acceptance criteria:**
  - Agent takes `task`, `context`, `cancellation_token`.
  - Loop processes LLM response, executes tool calls, re-prompts, stops on `finish_reason="stop"` or `max_iterations`.
  - Returns `AgentResponse` with usage, finish reason, and updated context.
- **Dependencies:** 1.1-1.4, 2.1-2.4.
- **Reference:** `picoagents/src/picoagents/agents/_agent.py`.

### 3.2 Streaming and cancellation
- **Goal:** `run_stream()` yields messages, events, and final `AgentResponse`; cancellation token aborts cleanly.
- **Acceptance criteria:**
  - Consumers receive `TaskStartEvent`, `ModelCallEvent`, `ToolCallEvent`, `ToolCallResponseEvent`, `ModelResponseEvent`.
  - `CancellationToken.is_cancelled()` stops the loop and raises `CancelledError` / yields final cancelled response.
- **Dependencies:** 3.1.
- **Reference:** `picoagents/src/picoagents/agents/_agent.py` lines 226-470.

### 3.3 Memory injection
- **Goal:** Agent prepends memory context into the system prompt.
- **Acceptance criteria:**
  - `memory.get_context()` is called before each LLM call.
  - `ListMemory` returns recent messages; `QdrantMemory` returns semantic matches.
  - Agent tolerates memory failures (logs warning, continues).
- **Dependencies:** 3.1, 4.1, 4.2.
- **Reference:** `picoagents/src/picoagents/agents/_base.py` lines 234-300.

### 3.4 Structured output
- **Goal:** `output_format` accepts a Pydantic model and the agent returns `AssistantMessage.structured_content`.
- **Acceptance criteria:**
  - OpenAI `response_format` uses `json_schema` with `strict=True`.
  - Failed parsing falls back to text content with a warning.
  - At least one research schema (`ResearchBrief`, `Citation`) is defined.
- **Dependencies:** 1.3, 3.1.
- **Reference:** `picoagents/src/picoagents/llm/_openai.py` lines 128-155.

### 3.5 Termination conditions
- **Goal:** Pluggable termination: `MaxMessageTermination`, `TokenUsageTermination`, `TimeoutTermination`, `TextMentionTermination`, `ExternalTermination`, plus composite `OR`/`AND`.
- **Acceptance criteria:**
  - Orchestrator and agent loops terminate cleanly when any condition is met.
  - `Termination.check(messages)` returns `Optional[StopMessage]`.
- **Dependencies:** 1.2.
- **Reference:** `picoagents/src/picoagents/termination/`.

### 3.6 Middleware and human-in-the-loop approval
- **Goal:** `MiddlewareChain` wraps `model_call` and `tool_call`; approval middleware pauses execution and surfaces `ToolApprovalRequest`.
- **Acceptance criteria:**
  - Middleware can emit events, transform data, and pause.
  - `ApprovalMiddleware` intercepts tools with `approval_mode=ALWAYS` and yields `ToolApprovalEvent`.
  - Resuming with an `approval_response` continues the pending tool call.
- **Dependencies:** 3.1, 3.2.
- **Reference:** `picoagents/src/picoagents/_middleware.py`, `picoagents/src/picoagents/context.py`.

### 3.7 OpenTelemetry middleware
- **Goal:** Emit spans for agent, model, and tool calls following Gen-AI semantic conventions.
- **Acceptance criteria:**
  - Env flag `ATLAS_ENABLE_OTEL=true` auto-instruments `Agent.__init__` with `OTelMiddleware`.
  - Spans set `gen_ai.system`, `gen_ai.agent.name`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`.
  - Content capture is opt-in (`ATLAS_OTEL_CAPTURE_CONTENT=true`) due to PII.
- **Dependencies:** 3.6.
- **Reference:** `picoagents/src/picoagents/_otel.py`.

---

## Phase 4 — Memory

### 4.1 List and file memory
- **Goal:** `ListMemory` (in-memory) and `FileMemory` (JSON file) implement `BaseMemory`.
- **Acceptance criteria:**
  - `add`, `query`, `get_context`, `clear` work for both.
  - File memory persists across restarts.
- **Dependencies:** 1.2.
- **Reference:** `picoagents/src/picoagents/memory/_base.py`.

### 4.2 Qdrant semantic memory
- **Goal:** Vector memory with sentence-transformer embeddings for RAG.
- **Acceptance criteria:**
  - `QdrantMemory.add()` stores `MemoryContent` as a Qdrant point with payload.
  - `query()` returns relevant memories with distance threshold and optional payload filtering.
  - Supports `:memory:`, local path, and Qdrant Cloud via `url` + `api_key`.
- **Dependencies:** 4.1.
- **Reference:** `qdrant-client` (`QdrantClient(path=...)` and `QdrantClient(url=..., api_key=...)`).
- **Risks:** Qdrant Cloud free clusters are suspended after 1 week of inactivity and deleted after 4 weeks; keep source documents under version control and rebuild the index on startup when needed.

---

## Phase 5 — Workflow engine

### 5.1 DAG builder and typed steps
- **Goal:** `Workflow` class with `add_step`, `add_edge`, `chain`, typed `input_type`/`output_type`.
- **Acceptance criteria:**
  - Steps can be chained into a DAG.
  - `FunctionStep` wraps a typed async function.
  - `AgentStep` wraps an `Agent` and produces structured output.
- **Dependencies:** 1.1, 2.1, 3.1.
- **Reference:** `picoagents/src/picoagents/workflow/core/_workflow.py`.

### 5.2 Workflow runner
- **Goal:** Execute DAG with parallel ready steps and streaming events.
- **Acceptance criteria:**
  - `WorkflowRunner.run_stream()` yields `StepStartedEvent`, `StepCompletedEvent`, `StepFailedEvent`.
  - Fan-in waits for all dependencies; conditional edges decide next step.
- **Dependencies:** 5.1.
- **Reference:** `picoagents/src/picoagents/workflow/core/_runner.py`.

### 5.3 Checkpointing and resume
- **Goal:** Save and load `WorkflowCheckpoint`; resume from latest checkpoint.
- **Acceptance criteria:**
  - `FileCheckpointStore` writes JSON per workflow; `InMemoryCheckpointStore` for tests.
  - `CheckpointConfig` supports auto-save after each step and cleanup.
  - Restarting a workflow with a compatible checkpoint skips completed steps.
- **Dependencies:** 5.2.
- **Reference:** `picoagents/src/picoagents/workflow/core/_checkpoint.py`.

### 5.4 Workflow validation
- **Goal:** Validate DAG before run (start step, end steps, cycles, unreachable, type compatibility).
- **Acceptance criteria:**
  - Cycles are detected and rejected.
  - Type mismatches between connected steps raise a clear error.
- **Dependencies:** 5.1.
- **Reference:** `picoagents/src/picoagents/workflow/core/_workflow.py` lines 335-450.

---

## Phase 6 — Orchestration

### 6.1 Base orchestrator
- **Goal:** `BaseOrchestrator` with universal loop, shared messages, usage aggregation, and termination checks.
- **Acceptance criteria:**
  - `run_stream(task)` iterates, selects agent, prepares context, updates state, checks termination.
  - Returns `OrchestrationResponse` with `messages`, `final_result`, `usage`, `stop_message`.
- **Dependencies:** 3.1, 3.5.
- **Reference:** `picoagents/src/picoagents/orchestration/_base.py`.

### 6.2 Round-robin orchestrator
- **Goal:** Fixed turn order for critic panel.
- **Acceptance criteria:**
  - Agents cycle in order; each gets full shared history.
  - Terminates on `TextMentionTermination` or `MaxMessageTermination`.
- **Dependencies:** 6.1.
- **Reference:** `picoagents/src/picoagents/orchestration/_round_robin.py`.

### 6.3 Plan-based orchestrator (Magentic One style)
- **Goal:** LLM-generated `ExecutionPlan` with agent assignment and step retry.
- **Acceptance criteria:**
  - `create_plan(task)` returns `ExecutionPlan` via structured output.
  - `evaluate_step_progress()` decides retry or advance.
  - Max retries per step is configurable.
- **Dependencies:** 6.1, 3.4.
- **Reference:** `picoagents/src/picoagents/orchestration/_plan.py`.

### 6.4 AI-driven speaker selection
- **Goal:** LLM chooses next agent based on shared history and capabilities.
- **Acceptance criteria:**
  - Selection prompt includes agent names, descriptions, and tool lists.
  - Unknown agent names fall back to the first available agent.
- **Dependencies:** 6.1.
- **Reference:** `picoagents/src/picoagents/orchestration/_ai.py`.

---

## Phase 7 — Research product: agents and workflow

### 7.1 Research schemas
- **Goal:** Define `ResearchBrief`, `Citation`, `Evidence`, `VerificationResult` with Pydantic.
- **Acceptance criteria:**
  - Schemas include `source_url`, `quote`, `assessment`, `confidence`.
  - `ResearchBrief` serializes to markdown with a citation section.
- **Dependencies:** 3.4.

### 7.2 Research agents
- **Goal:** Implement Planner, Researcher, Verifier, Synthesizer.
- **Acceptance criteria:**
  - `Planner` outputs a list of sub-questions and a retrieval plan.
  - `Researcher` calls web search/fetch, returns raw evidence.
  - `Verifier` checks claims against evidence and flags hallucinations/weak citations.
  - `Synthesizer` produces a `ResearchBrief` with citations.
- **Dependencies:** 2.3, 3.1, 3.4, 7.1.

### 7.3 Critic panel
- **Goal:** Round-robin critic agents review the brief and request revisions.
- **Acceptance criteria:**
  - Critic agents inherit shared history and can call out missing evidence.
  - Termination when no critic requests changes or max turns reached.
- **Dependencies:** 6.2, 7.2.

### 7.4 Full research pipeline workflow
- **Goal:** Compose the research flow as a typed DAG: `Plan -> Retrieve(RAG) -> Search -> Verify -> Synthesize -> Critic -> Human Approval -> Grade -> Persist`.
- **Acceptance criteria:**
  - Each stage yields a typed output and emits SSE events.
  - Workflow can be checkpointed and resumed.
  - Final output is a `ResearchBrief` with citations and a usage/cost summary.
- **Dependencies:** 5.1-5.3, 6.1-6.3, 7.2, 7.3.

---

## Phase 8 — Backend API

### 8.1 FastAPI skeleton and health
- **Goal:** FastAPI app with CORS, lifespan, and `/health`.
- **Acceptance criteria:**
  - `uvicorn backend.main:app` starts on port 8000.
  - `/health` reports status and configured model clients.
- **Dependencies:** 0.1.

### 8.2 Session store and run state
- **Goal:** In-memory `SessionManager` mapping `session_id` to `AgentContext` / workflow execution.
- **Acceptance criteria:**
  - New research run generates a UUID session.
  - Context survives across SSE reconnects within the same process (later persisted to SQLite/Chroma).
- **Dependencies:** 8.1.
- **Reference:** `picoagents/src/picoagents/webui/_sessions.py`.

### 8.3 `/run` and `/stream` (SSE)
- **Goal:** Endpoint that starts the research workflow and streams events as Server-Sent Events.
- **Acceptance criteria:**
  - `POST /sessions/{id}/run` accepts a `query` and returns `session_id`.
  - `GET /sessions/{id}/stream` yields `data: <json>` lines until `workflow_completed`.
  - Client disconnect triggers cancellation token.
- **Dependencies:** 7.4, 8.2.
- **Reference:** `examples/app/backend/app.py`.

### 8.4 `/approve` endpoint
- **Goal:** Accept human approval responses and resume a paused workflow.
- **Acceptance criteria:**
  - `POST /sessions/{id}/approve` accepts `ToolApprovalResponse` list.
  - Resumed run continues from the pending tool call.
- **Dependencies:** 3.6, 8.3.

### 8.5 `/eval` endpoint
- **Goal:** Trigger eval harness on a dataset and return results.
- **Acceptance criteria:**
  - `POST /eval` accepts `dataset_path` and `model_client` override.
  - Returns aggregated scores and per-task breakdown.
- **Dependencies:** 14.1-14.3, 8.1.

---

## Phase 9 — Frontend dashboard

### 9.1 Frontend project setup
- **Goal:** React 19 + TypeScript + Vite + Tailwind CSS v4.
- **Acceptance criteria:**
  - `npm install && npm run dev` works.
  - `npm run build` outputs a `dist` that the FastAPI static-files mount can serve.
- **Dependencies:** 0.1.

### 9.2 Live activity feed (SSE)
- **Goal:** Connect to `/stream` and render agent/tool/orchestration events.
- **Acceptance criteria:**
  - Events appear in a scrollable, timestamped feed.
  - Disconnection/reconnection is handled gracefully.
- **Dependencies:** 8.3, 9.1.

### 9.3 Query input and brief display
- **Goal:** Submit a research question and display the final markdown brief.
- **Acceptance criteria:**
  - Markdown renders with clickable citations.
  - Loading and error states are shown.
- **Dependencies:** 9.2.

### 9.4 Citation panel
- **Goal:** Sidebar listing all evidence sources with URLs and snippets.
- **Acceptance criteria:**
  - Clicking a citation in the brief scrolls to the source.
  - Sources can be filtered by status (verified, weak, missing).
- **Dependencies:** 9.3, 7.1.

### 9.5 Approval gate UI
- **Goal:** When `ToolApprovalEvent` occurs, show tool, parameters, and approve/reject buttons.
- **Acceptance criteria:**
  - User can edit parameters before approving.
  - Rejected tool call is recorded and execution continues.
- **Dependencies:** 3.6, 8.4, 9.2.

### 9.6 Session history and eval viewer
- **Goal:** List past runs, view brief/cost/usage, and view eval reports.
- **Acceptance criteria:**
  - `GET /sessions` returns sessions.
  - Session detail shows trace/cost inspector.
- **Dependencies:** 8.2, 14.1-14.3, 9.1.

### 9.7 Cost and trace inspector
- **Goal:** Display token/cost breakdown per step and link to Jaeger trace.
- **Acceptance criteria:**
  - Per-agent and total `Usage` displayed.
  - Trace IDs link to `http://localhost:16686` when OTel is enabled.
- **Dependencies:** 3.7, 9.3.

---

## Phase 10 — RAG knowledge base

### 10.1 Qdrant collection for briefs and sources
- **Goal:** Persistent Qdrant collection with payload metadata for source documents and generated briefs.
- **Acceptance criteria:**
  - Collection stores documents with `kind` (brief/source), `session_id`, `url`, `title`, `timestamp`.
  - `query()` returns top-k relevant items with distance scores and payload filtering.
- **Dependencies:** 4.2.

### 10.2 Ingestion pipeline
- **Goal:** After a run, persist the brief and every fetched source into the vector store; also keep source documents as committed markdown/JSON for cross-session recovery.
- **Acceptance criteria:**
  - Brief text and source markdown are embedded into Qdrant.
  - Duplicate URLs are updated, not duplicated.
  - Raw source documents are written to `data/sources/<url_hash>.md` (git-tracked or persistent volume) and can re-hydrate the index.
- **Dependencies:** 10.1.

### 10.3 Retrieval at planning and research time
- **Goal:** Planner and Researcher query the vector store before web search.
- **Acceptance criteria:**
  - Relevant prior briefs/sources are injected into the prompt.
  - Retrieval event is emitted to SSE stream.
- **Dependencies:** 7.2, 10.2.

### 10.4 Source deduplication and versioning
- **Goal:** Track source URL hashes and brief versions.
- **Acceptance criteria:**
  - Same URL fetched in different sessions updates the stored document.
  - Briefs are versioned by `session_id`/`timestamp`.
- **Dependencies:** 10.2.

---

## Phase 11 — MCP integration

### 11.1 MCP client manager
- **Goal:** Connect to MCP servers via stdio and discover tools.
- **Acceptance criteria:**
  - `MCPTool` wraps an MCP tool and implements `BaseTool`.
  - `MCPClientManager` handles `connect`, `disconnect_all`, and lifecycle.
- **Dependencies:** 2.1.
- **Reference:** `picoagents/src/picoagents/tools/_mcp/`.

### 11.2 Exa MCP, arXiv, GitHub, and news tool configs
- **Goal:** Provide default server configs in `.env.example` / config file. Exa MCP is the first search/default because it works without an API key.
- **Acceptance criteria:**
  - Atlas can start with Exa MCP (remote/SSE) without Node.js, and with arXiv/GitHub MCP if `npx`/Node is available.
  - Missing servers do not block the rest of the app.
- **Dependencies:** 11.1.
- **Risks:** stdio MCP servers require Node.js; use SSE/HTTP MCP servers (Exa, remote) where possible to avoid Node in the runtime.

### 11.3 Approval for destructive MCP tools
- **Goal:** Set `approval_mode=ALWAYS` on write/delete MCP tools.
- **Acceptance criteria:**
  - Write operations surface an approval gate in the UI.
  - Read-only operations proceed automatically.
- **Dependencies:** 3.6, 11.2.

---

## Phase 12 — Computer-use fallback

### 12.1 Playwright interface client
- **Goal:** Initialize browser, navigate, and capture page state.
- **Acceptance criteria:**
  - `PlaywrightInterfaceClient` returns `url`, `title`, `text`, `screenshot`.
  - Supports headless mode and `BrowserType` config.
- **Dependencies:** 0.1 (Playwright install).

### 12.2 Browser action tools
- **Goal:** Tools for `navigate`, `click`, `fill`, `observe_page`, `scroll`.
- **Acceptance criteria:**
  - Each tool maps to a Playwright action.
  - `observe_page` returns accessibility tree + screenshot.
- **Dependencies:** 12.1.

### 12.3 Multimodal browser agent
- **Goal:** `ComputerUseAgent` that uses a vision-capable LLM to operate the browser.
- **Acceptance criteria:**
  - Agent receives screenshot as `MultiModalMessage` and chooses the next action.
  - Stops when answer is found or `max_actions` reached.
- **Dependencies:** 12.2, 1.4 (vision model), 3.1.
- **Reference:** `picoagents/src/picoagents/agents/_computer_use/_computer_use.py`.

### 12.4 Fallback wiring
- **Goal:** When `WebFetchTool` fails or returns insufficient content, route to `ComputerUseAgent`.
- **Acceptance criteria:**
  - Researcher tool loop tries search -> fetch -> browser fallback per source.
  - Browser agent result is returned as additional evidence.
- **Dependencies:** 7.2, 12.3.

---

## Phase 13 — Cost optimization

### 13.1 Two-stage filtering
- **Goal:** Cheap model triages search results; only survivors go to the strong model.
- **Acceptance criteria:**
  - `TriageAgent` rates each search result as `relevant`, `partial`, `irrelevant`.
  - Strong-model `Researcher` receives only `relevant`/`partial` results.
- **Dependencies:** 7.2, 1.4.
- **Reference:** `examples/workflows/yc_analysis/`.

### 13.2 Cost tracking and reporting
- **Goal:** Accumulate token/cost estimates from every LLM call.
- **Acceptance criteria:**
  - `Usage.cost_estimate` is populated for OpenAI-compatible providers where token prices are known; unknown providers skip cost or log a warning.
  - Final `AgentResponse`/`OrchestrationResponse` shows estimated total cost.
  - Dashboard displays cost per run.
- **Dependencies:** 1.2, 7.4.

### 13.3 Target: 90% cost reduction vs naive single-model runs
- **Goal:** Measure and report cost savings.
- **Acceptance criteria:**
  - A benchmark run shows the triage+strong pipeline costs ~10% of a single strong-model run over all results.
  - Result is reproducible on a fixed set of queries.
- **Dependencies:** 13.1, 13.2, 14.2.

---

## Phase 14 — Evaluation harness

### 14.1 LLM-as-judge
- **Goal:** `LLMEvalJudge` scores `ResearchBrief` on accuracy, citation coverage, hallucination, clarity.
- **Acceptance criteria:**
  - Judge uses structured output (`CriterionScore` list).
  - Missing criteria default to 5.0 with error reasoning.
- **Dependencies:** 1.4, 3.4, 7.1.
- **Reference:** `picoagents/src/picoagents/eval/judges/_llm.py`.

### 14.2 Reference-based evaluation
- **Goal:** Golden dataset of questions and expected briefs; compare output with reference.
- **Acceptance criteria:**
  - `ReferenceEvalJudge` computes overlap/citation match.
  - Eval dataset lives in `eval/golden/` and is version-controlled.
- **Dependencies:** 14.1.

### 14.3 Regression runner and dashboard viewer
- **Goal:** `EvalRunner` runs the research pipeline on the golden set and produces `EvalResults`.
- **Acceptance criteria:**
  - `pytest tests/eval/test_regression.py` runs the golden set and compares scores to baseline.
  - Scores are viewable in the frontend eval viewer.
  - CI fails if scores drop below baseline.
- **Dependencies:** 14.2, 7.4, 8.5.
- **Reference:** `picoagents/src/picoagents/eval/_runner.py`.

---

## Phase 15 — Framework comparison benchmark

### 15.1 Re-implement plan-based path in LangGraph or Google ADK
- **Goal:** Build an equivalent plan-based research pipeline in one production framework.
- **Acceptance criteria:**
  - Same inputs produce comparable briefs (not identical, but same shape).
  - Implementation is under `benchmarks/langgraph/` or `benchmarks/adk/`.
- **Dependencies:** 7.4.

### 15.2 Benchmark harness
- **Goal:** Compare from-scratch `atlascore` vs production framework on quality, latency, cost, and dev ergonomics.
- **Acceptance criteria:**
  - Automated benchmark runs on a fixed query set.
  - Report includes per-query cost and latency.
- **Dependencies:** 15.1, 14.3.

### 15.3 Benchmark report
- **Goal:** Markdown report documenting tradeoffs.
- **Acceptance criteria:**
  - Report is checked into `docs/benchmark_report.md`.
  - Includes a table of quality/cost/latency/dev-experience scores.
- **Dependencies:** 15.2.

---

## Phase 16 — Deployment

### 16.1 Docker Compose
- **Goal:** `docker-compose.yml` with backend, frontend (built into backend static files), qdrant, and optional jaeger.
- **Acceptance criteria:**
  - `docker compose up --build` serves the app at `http://localhost:8000`.
  - Qdrant data is persisted in a volume.
- **Dependencies:** 9.1, 10.1.

### 16.2 Production env and health checks
- **Goal:** Production `.env` validation, non-root containers, and FastAPI health checks for DB/MCP.
- **Acceptance criteria:**
  - `docker compose` sets `OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318`.
  - `/health` fails if required external services (e.g., Qdrant) are unreachable.
- **Dependencies:** 16.1.

### 16.3 One-command deploy
- **Goal:** README documents `docker compose up` and common troubleshooting.
- **Acceptance criteria:**
  - A new developer can run Atlas in <15 minutes with only `.env` filled in.
- **Dependencies:** 16.2.

---

## Phase 17 — Stretch goals

### 17.1 Voice I/O
- **Goal:** STT for query input, TTS for brief summary.
- **Acceptance criteria:**
  - UI has a microphone button and a "read summary" button.
- **Dependencies:** 9.3.

### 17.2 SWE-style code verification agent
- **Goal:** Run/verify code snippets found during research.
- **Acceptance criteria:**
  - `CodeVerificationAgent` executes code in a sandboxed subprocess.
  - Results feed into evidence scoring.
- **Dependencies:** 7.2.

### 17.3 Citation graph / knowledge graph
- **Goal:** Visualize relationships between sources, claims, and briefs.
- **Acceptance criteria:**
  - Graph view in the frontend using source/citation edges.
- **Dependencies:** 9.4.

---

## Cross-cutting concerns (the "forgotten" stuff)

1. **Cost caps and rate limiting**
   - Every model client and search call should respect a `max_cost_usd` and `max_calls_per_minute` budget.
   - Add a `BudgetTermination` that stops the run before it overspends.
   - Use the cheapest model defaults for exploration (e.g., Groq `llama-3.3-70b-versatile`, OpenRouter `:free` models, local Ollama).

2. **Secrets and key rotation**
   - No keys in code. Use `.env` and Pydantic Settings.
   - For CI, use GitHub secrets or a test-only keyless search provider.

3. **Data retention and privacy**
   - Decide retention policy for stored briefs/sources (GDPR-style deletion).
   - PII in traces must be opt-in; OTel content capture off by default.

4. **Concurrency and sessions**
   - `SessionManager` must be safe for concurrent requests; use per-session locks.
   - SSE connections should not block the event loop; use cancellation tokens.

5. **Error handling and observability**
   - Every tool call, LLM call, and workflow step emits structured logs and spans.
   - Failed steps produce `ErrorEvent` and continue where possible (or checkpoint).

6. **Testing strategy**
   - Unit tests for each `atlascore` primitive.
   - Integration tests with mocked LLM and search endpoints.
   - End-to-end smoke test for the FastAPI + frontend `docker compose up`.
   - Golden eval set under `eval/`.

7. **Documentation**
   - `README.md` for users, `ARCHITECTURE.md` for contributors, `eval/README.md` for eval methodology.
   - Docstrings follow Google style; generated API docs optional.

8. **CI/CD**
   - GitHub Actions: lint, type-check, unit tests, build frontend, build Docker image, regression eval on PRs.
   - No deployment to prod from CI unless requested.

9. **Model routing**
   - Define a `ModelRouter` that picks `cheap` vs `strong` model aliases from config, not hard-coded strings.
   - Makes the two-stage cost filter and A/B benchmarks trivial.

10. **MCP server availability**
    - Node.js is required for stdio MCP servers. Include it in the Docker image or use SSE/HTTP MCP servers where possible.
    - Provide a fallback: if MCP server is unavailable, research continues with web search.

11. **Browser automation requirements**
    - Playwright browsers must be installed (`playwright install chromium`).
    - Headless mode default; local dev can toggle `PLAYWRIGHT_HEADLESS=false`.
    - Sandbox and resource limits for safety.

12. **Fallback chain for search**
    - Exa MCP (`web_search_exa`) no API key -> Exa MCP with API key -> Tavily keyless -> Tavily API key -> Google CSE / Perplexity -> direct web fetch -> browser fallback.
    - Each fallback emits an event so the UI explains what happened.

13. **Vector persistence across sessions**
    - Local Qdrant (or any local vector DB) survives process restarts but not a fresh Devin VM/session unless its storage files are committed to Git (not recommended for binary vector data).
    - For cross-session development, either: (a) use Qdrant Cloud free cluster and re-create it if it is auto-suspended, or (b) keep the canonical source documents as committed markdown/JSON and rebuild the Qdrant index on startup.
    - The ingestion pipeline should write `data/sources/<url_hash>.md` so the vector index is always reconstructible.

---

## Open questions / decisions for you

1. **Search provider default:** Exa MCP is free without an API key (~50 calls/day) but lower limits; Tavily keyless is also available. Should Exa be the dev default?
2. **LLM provider default:** Groq (free, 30 RPM, 1,000 req/day, OpenAI-compatible) and OpenRouter (`:free` models, 20 RPM, 200 req/day) are the cheapest starts. Do you want one of these as the default instead of paid OpenAI?
3. **Vector store target:** Qdrant Cloud free cluster for cross-session work, or local Qdrant with committed source documents in `data/sources/` and rebuild index on startup?
4. **MCP servers in dev:** Node.js is needed for stdio MCP servers. Is installing Node in the dev environment acceptable, or should we prefer remote/SSE MCP servers like Exa MCP?
5. **Human approval:** Should approval be required for all web fetch/MCP/browser actions initially, or only destructive/write actions?
6. **Deployment target:** Is `docker compose up` on a VM the target, or do you want a cloud platform (Fly, Railway, Vercel/Render)?
7. **Benchmark framework:** LangGraph has the most examples in the reference repo; Google ADK 2.x is newer and rapidly changing. Which one do you want for the comparison?
8. **Golden eval set:** Do you want to curate 5-10 seed questions now, or build the harness first and add questions later?

---

## Suggested first milestone

After approval, implement **Phase 0 + Phase 1.1-1.3 + Phase 2.1-2.2 + Phase 3.1**:
- A single `Agent` that can answer a question using a `calculator` or `datetime` tool, with a typed response and SSE streaming.

This gives a demoable end-to-end in a few files and validates the whole stack before the research domain complexity is added.

---

## Validation/testing approach

- **Unit:** `pytest` for each module (`messages`, `context`, `tools`, `agent`, `workflow`, `orchestration`, `eval`).
- **Integration:** Run the full research pipeline on a small fixed query set (e.g., 3 questions) with mocked search where possible; assert `ResearchBrief` has citations and `usage` is populated.
- **Frontend:** Manual walkthrough of query -> stream -> brief -> citation; add Playwright e2e once the UI stabilizes.
- **Regression:** Maintain `eval/golden/`; CI runs `pytest tests/eval` and fails if `overall` score drops more than 5% from baseline.
- **Cost:** Add a `tests/test_cost_budget.py` that asserts a sample run stays under a hard token/cost limit.
- **Deployment:** `docker compose up --build` smoke test in CI that hits `/health` and runs one `/run`.
