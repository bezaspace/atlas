# Atlas — Implementation Progress

This file tracks what has been implemented and what is next for the Atlas project.

## Latest state

- **Current branch target:** `master`
- **Last milestone completed:** Phase 5 (Workflow engine)
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
| 6 | Orchestration | Not started | Round-robin, plan-based, AI speaker selection |
| 7 | Research product | Not started | `ResearchBrief`, agents (Planner, Researcher, Verifier, Synthesizer) |
| 8+ | Backend, frontend, RAG, MCP, computer-use, evals, deployment | Not started | Later phases |

## Verification of completed work

- `pytest tests/` — 59 passed, 1 skipped (cloud integration test skipped by default)
- `tests/test_qdrant_memory.py` covers add/query/get_context/clear for `:memory:` and local path; cloud test verified against Qdrant Cloud
- `tests/test_workflow.py` covers DAG builder/validation, runner, fan-in, conditional edges, checkpoint save/resume, file store, and `AgentStep`
- `ruff check src tests examples` — clean
- `pyright src` — clean
- Real API smoke test (`examples/calculator_agent.py`) used `calculator` and `datetime` tools correctly.

## Next recommended step

Phase 5 is complete. Next is **Phase 6** (orchestration: round-robin, plan-based, and AI speaker selection) or **Phase 7.4** (full research pipeline workflow), depending on product priorities.

## References

- `TASK_BACKLOG.md` — full ordered task list
- `PROJECT_PLAN.md` — architecture and feature overview
- `src/atlascore/` — implemented core framework
