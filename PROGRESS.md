# Atlas — Implementation Progress

This file tracks what has been implemented and what is next for the Atlas project.

## Latest state

- **Current branch target:** `master` (merged via `devin/first-milestone-atlascore`)
- **Last milestone completed:** First milestone from `TASK_BACKLOG.md` — Phase 0 + 1.1-1.3 + 2.1-2.2 + 3.1/3.2
- **Package:** `atlascore` is installable and tested

## Milestones

| Phase | Goal | Status | Notes |
|-------|------|--------|-------|
| 0 | Repo, toolchain, environment | Done | `pyproject.toml`, `README.md`, `.env.example`, `ruff`, `pyright`, `pytest` |
| 1 | atlascore primitives | Done | Messages, events, context, usage, response types, OpenAI-compatible LLM client |
| 2 | Tool system | Done | `BaseTool`, `FunctionTool`, schema generation, core tools (`calculator`, `datetime`, `think`, `json_parser`, `regex`, `task_status`) |
| 3.1 | Agent reasoning loop (`run()`) | Done | Sequential tool-call loop, `AgentResponse`, `max_iterations` |
| 3.2 | Agent streaming (`run_stream()`) | Done | Yields events and final response, cancellation token support |
| 3.3 | Memory injection | Not started | Needs `BaseMemory`, `ListMemory`, `QdrantMemory` |
| 3.4 | Structured output | Partial | LLM client supports `json_schema`; no research schemas yet |
| 3.5 | Termination conditions | Not started | `MaxMessageTermination`, `TokenUsageTermination`, etc. |
| 3.6 | Middleware / approval | Not started | `MiddlewareChain`, `ApprovalMiddleware` |
| 3.7 | OpenTelemetry | Not started | Span emission for agents/tools/models |
| 4 | Memory | Not started | In-memory, file, and Qdrant memory tiers |
| 5 | Workflow engine | Not started | DAG builder, runner, checkpointing |
| 6 | Orchestration | Not started | Round-robin, plan-based, AI speaker selection |
| 7 | Research product | Not started | `ResearchBrief`, agents (Planner, Researcher, Verifier, Synthesizer) |
| 8+ | Backend, frontend, RAG, MCP, computer-use, evals, deployment | Not started | Later phases |

## Verification of completed work

- `pytest tests/` — 16 passed
- `ruff check src tests examples` — clean
- `pyright src` — clean
- Real API smoke test (`examples/calculator_agent.py`) used `calculator` and `datetime` tools correctly.

## Next recommended step

Implement **Phase 3.3-3.5** (memory injection, structured output schemas, termination conditions) before moving to the workflow engine, so the standalone `Agent` is fully production-ready.

## References

- `TASK_BACKLOG.md` — full ordered task list
- `PROJECT_PLAN.md` — architecture and feature overview
- `src/atlascore/` — implemented core framework
