# Atlas — Implementation Progress

This file tracks what has been implemented and what is next for the Atlas project.

## Latest state

- **Current branch target:** `master`
- **Last milestone completed:** Phase 3.3-3.7 (remaining Agent enhancements)
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
| 4 | Memory | Not started | In-memory, file, and Qdrant memory tiers |
| 5 | Workflow engine | Not started | DAG builder, runner, checkpointing |
| 6 | Orchestration | Not started | Round-robin, plan-based, AI speaker selection |
| 7 | Research product | Not started | `ResearchBrief`, agents (Planner, Researcher, Verifier, Synthesizer) |
| 8+ | Backend, frontend, RAG, MCP, computer-use, evals, deployment | Not started | Later phases |

## Verification of completed work

- `pytest tests/` — 34 passed
- `ruff check src tests examples` — clean
- `pyright src` — clean
- Real API smoke test (`examples/calculator_agent.py`) used `calculator` and `datetime` tools correctly.

## Next recommended step

Phase 3 is complete. Next is **Phase 4+** (workflow engine, orchestration, research product, backend/frontend, RAG/MCP, computer-use, evals, deployment).

## References

- `TASK_BACKLOG.md` — full ordered task list
- `PROJECT_PLAN.md` — architecture and feature overview
- `src/atlascore/` — implemented core framework
