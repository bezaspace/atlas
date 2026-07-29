# Atlas — Autonomous Multi-Agent Deep Research Platform

> A capstone project designed to demonstrate **every** topic taught in
> [victordibia/designing-multiagent-systems](https://github.com/victordibia/designing-multiagent-systems)
> (the code companion to Victor Dibia's *Designing Multi-Agent Systems: Principles,
> Patterns, and Implementation for AI Agents*).
>
> Reference repo to revisit while building:
> https://github.com/victordibia/designing-multiagent-systems

---

## 1. Intention

The goal of Atlas is **not** to clone the reference repository. The reference repo
teaches multi-agent systems by building a from-scratch framework (`picoagents`) and
then layering patterns on top until you can ship a real product. Atlas takes the same
philosophy and applies it to **one coherent, shippable product**: an autonomous
deep-research platform.

A user gives Atlas a research question. A team of agents plans the research, retrieves
from a knowledge base, searches the web, falls back to browser automation when needed,
verifies evidence, synthesizes a cited brief, has a critic panel review it, asks a human
for approval before publishing, grades its own work with an LLM-as-judge, and streams
every step live to a dashboard. The whole thing is containerized and deployable.

**Why this project lands jobs:**

- "Deep research agents" is the most recognizable, in-demand agent category right now
  (OpenAI Deep Research, Perplexity, Gemini Deep Research). Interviewers instantly
  understand the value proposition.
- Building it *well* forces you to touch ~every topic in the book — there is no filler.
- It produces a single, demoable artifact with a live UI, a real eval harness, cost
  numbers, traces, and a Docker one-command deploy. That is a portfolio piece, not a
  notebook.
- The "from scratch" approach means you can explain *why* every design decision was made,
  which is exactly what senior interviews probe.

The build is intentionally **flexible** — there is no rigid file structure prescribed
here. You retain full autonomy over how to organize the code as you build. What follows
is the architecture, the feature set, and the skill each feature demonstrates.

---

## 2. Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React + TS + Vite + Tailwind                 │
│   Dashboard • Live agent activity (SSE) • Citations • Sessions      │
│   Approval gate UI • Eval results viewer • Cost/trace inspector     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ SSE / REST
┌──────────────────────────────▼──────────────────────────────────────┐
│                          FastAPI backend                             │
│   /sessions  /run  /stream  /approve  /eval  /health                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                     atlascore  (your from-scratch framework)         │
│  Agent • Tools • Memory • Middleware • Workflow(DAG) • Orchestration │
│  LLM clients • OTel observability • Eval • Termination               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
        ┌──────────────┬───────┴────────┬───────────────┐
        ▼              ▼                ▼               ▼
   Web search      RAG vector       MCP servers     Computer-use
   + fetch tools   store (prior     (arXiv, GitHub, (Playwright
                   briefs/sources)  news, etc.)     fallback)
```

### The flow of a single research run

1. **Plan** — a Planner agent decomposes the question into sub-questions and a retrieval
   plan (plan-based / Magentic One style).
2. **Retrieve (RAG)** — pull relevant context from a vector store of prior briefs and
   saved sources.
3. **Search (web)** — Researcher agents run web searches and fetch pages; cheap model
   triages which sources are worth deep analysis.
4. **Computer-use fallback** — for sites with no API / blocked scraping, a Playwright
   browser agent with multimodal vision reasons over the rendered page.
5. **MCP sources** — structured data pulled from MCP servers (arXiv, GitHub, news APIs).
6. **Verify** — a Verifier agent checks claims against cited evidence and flags
   hallucinations / weak citations.
7. **Synthesize** — a Synthesizer agent writes the cited brief in structured form.
8. **Critic panel (round-robin)** — a small panel of critic agents reviews the brief
   turn-by-turn and requests revisions.
9. **Human approval gate** — before publishing, a human can approve/edit via the
   dashboard.
10. **Grade** — an LLM-as-judge scores the final brief (accuracy, citation coverage,
    hallucination) and writes the result to the eval set.
11. **Persist** — brief + sources go back into the vector store, improving future runs.

Every step emits OpenTelemetry spans and SSE events to the dashboard. The whole pipeline
is a type-safe DAG with checkpointing, so a failed run can resume.

---

## 3. Features and the skill each demonstrates

Each feature below is mapped to the chapter/topic in the reference repo it exercises, so
you can always go back to
https://github.com/victordibia/designing-multiagent-systems for the canonical teaching
implementation while building your own version.

### 3.1 From-scratch agent core
**What:** Build the `Agent` class yourself — reasoning/action loop, message protocol
(`SystemMessage`, `UserMessage`, `AssistantMessage`, `ToolMessage`,
`MultiModalMessage`), agent context, sync `run()` and streaming `run_stream()`,
lifecycle (init / run / reset).

**Reference:** Ch 4 — `picoagents/src/picoagents/agents/_agent.py`, `_base.py`,
`messages.py`, `context.py`. The `code_along/` dir builds this up in 4 steps:
core loop → tools → memory → streaming.

**Job skill:** Proves you understand *how* an agent framework works internally, not just
how to call one. This is the #1 thing that separates "used LangChain" from "can build
LangChain."

### 3.2 Tool system
**What:** `BaseTool` / `FunctionTool`, tool discovery, sequential and parallel tool
execution, and **agent-as-tool** (one agent callable as a tool by another, enabling
hierarchical composition).

**Reference:** Ch 4 — `picoagents/src/picoagents/tools/_base.py`, `_core_tools.py`,
`examples/agents/agent_as_tool.py`.

**Job skill:** Tool calling is the backbone of every production agent. Demonstrating
parallel tool execution and agent-as-tool shows you can design non-trivial agent
topologies.

### 3.3 Memory (list, conversation, semantic)
**What:** Three memory tiers — simple list memory, conversation memory, and **semantic
memory** backed by a vector store. Memory context is injected into LLM calls; a
memory-tool lets agents store/recall.

**Reference:** Ch 4 — `picoagents/src/picoagents/memory/_list.py`, `_conversation.py`,
`_semantic.py`, `tools/_memory_tool.py`.

**Job skill:** Semantic memory is literally RAG. Showing you can implement it from
scratch (embeddings, retrieval, injection) is a core AI-engineering competency.

### 3.4 Middleware system
**What:** An extensible middleware chain that wraps agent execution — used for control
flow, logging, approval gates, and observability hooks.

**Reference:** Ch 4 — `picoagents/src/picoagents/middleware/_base.py`, `_chain.py`,
`examples/agents/middleware.py`.

**Job skill:** Middleware/interceptor patterns are everywhere in production systems
(web frameworks, RPC, agents). Shows you can design extensible, composable plumbing.

### 3.5 Human-in-the-loop approval
**What:** Approval middleware that pauses a run and surfaces a request to the dashboard;
the user approves/edits/rejects and the run resumes.

**Reference:** Ch 4 — `examples/tools/approval_example.py`.

**Job skill:** Real agents cannot be fully autonomous. Human approval gates are a
compliance and safety requirement in enterprise agent deployments.

### 3.6 Structured output
**What:** Typed response schemas (`ResearchBrief`, `Citation`, `Evidence`,
`VerificationResult`) enforced via structured output from the model.

**Reference:** Ch 4 — `examples/agents/structured-output.py`.

**Job skill:** Typed agent outputs are what let agents compose into pipelines and feed
databases/UIs. Demonstrates schema design and validation discipline.

### 3.7 Multi-provider LLM clients
**What:** A unified model-client interface with implementations for OpenAI, Azure OpenAI,
Anthropic, GitHub Models, and local (Ollama / vLLM / any OpenAI-compatible endpoint).

**Reference:** Ch 4 — `picoagents/src/picoagents/llm/_openai.py`, `_azure_openai.py`,
`_anthropic.py`.

**Job skill:** Vendor lock-in avoidance, cost routing (cheap vs strong models), and
local-model support are all real production concerns. Shows provider-agnostic design.

### 3.8 Two-stage cost optimization
**What:** A cheap model triages retrieved sources / search results; only survivors are
analyzed by the strong model. Target ~90% inference cost reduction vs naive single-model
runs (matching the YC analysis case study's number).

**Reference:** Ch 16 (YC analysis) — `examples/workflows/yc_analysis/`.

**Job skill:** Cost is the #1 blocker for agent products. Being able to quantify
"this design cut cost 90%" is a standout resume line and a common interview question.

### 3.9 Computer-use agent (browser automation)
**What:** A Playwright-based browser agent that uses multimodal vision reasoning to
operate pages that have no API and resist scraping — used as a fallback when web search
+ fetch fail.

**Reference:** Ch 5 — `picoagents/src/picoagents/agents/_computer_use/`,
`examples/agents/computer_use.py`.

**Job skill:** Computer-use / browser agents are a frontier capability (Claude Computer
Use, OpenAI Operator). Demonstrates multimodal reasoning, accessibility-tree reasoning,
and graceful fallback design.

### 3.10 Type-safe DAG workflow engine with checkpointing
**What:** A workflow engine that executes the research pipeline as a typed DAG
(plan → retrieve → search → verify → synthesize → grade), with parallel and conditional
branches, streaming events per step, and **checkpointing** so an interrupted run resumes
from the last completed step.

**Reference:** Ch 6 — `picoagents/src/picoagents/workflow/core/_builder.py`,
`_checkpoint.py`, `examples/workflows/`.

**Job skill:** Workflow engines are the "durable execution" backbone of production
agents (cf. Temporal, Inngest, LangGraph). Checkpointing/resume is what makes agents
survive flaky LLM calls and long runs — a must-have for reliability claims.

### 3.11 Orchestration: round-robin, AI-driven, plan-based (Magentic One)
**What:** Three orchestration strategies, all used in Atlas:
- **Plan-based (Magentic One)** — the Planner drives the research team.
- **AI-driven speaker selection** — the team dynamically picks who speaks next.
- **Round-robin critic panel** — critics review the brief in fixed turn order.

**Reference:** Ch 7 — `picoagents/src/picoagents/orchestration/_round_robin.py`,
`_ai.py`, `_plan.py`; `examples/orchestration/round-robin.py`, `ai-driven.py`,
`plan-based.py`.

**Job skill:** Orchestration is the heart of "multi-agent." Showing you can implement
all three patterns and *choose* the right one per stage demonstrates real architectural
judgment, not just pattern-matching.

### 3.12 Termination conditions
**What:** A set of termination conditions (max turns, token budget, no-progress,
approval-received, judge-score-threshold) that stop runs cleanly.

**Reference:** `picoagents/src/picoagents/termination/` (9 conditions in the repo).

**Job skill:** Unbounded agent loops are a classic production failure. Termination
design is a reliability skill interviewers probe.

### 3.13 Agent UX: FastAPI + SSE backend, React dashboard
**What:** A FastAPI backend exposing session management, run kickoff, SSE streaming of
agent events, an approval endpoint, and an eval endpoint. A React + TypeScript + Vite +
Tailwind frontend with: live agent activity feed, citation panel, session history,
approval gate UI, eval results viewer, and a cost/trace inspector.

**Reference:** Ch 8 — `examples/app/` (minimal FastAPI+SSE), `picoagents/src/picoagents/webui/`
(production React UI with auto-discovery, sessions, real-time streaming).

**Job skill:** Full-stack agent UX is rare. Most candidates can build a notebook; few
can ship a real-time dashboard. This is the difference between "demo" and "product."

### 3.14 Framework comparison (Ch 9)
**What:** Re-implement one orchestration path (e.g. the plan-based team) in a real
framework — Google ADK, Microsoft Agent Framework, or LangGraph — and benchmark it
against your from-scratch version on quality, latency, cost, and dev ergonomics.

**Reference:** Ch 9 — `examples/frameworks/` (Microsoft Agent Framework, Google ADK,
LangGraph comparisons).

**Job skill:** Employers use real frameworks. Showing you can both build from scratch
*and* use production frameworks — and articulate the tradeoffs — is the strongest
possible signal. It also future-proofs you against framework churn.

### 3.15 Evaluation harness: LLM-as-judge + reference-based
**What:** A eval runner that scores briefs on accuracy, citation coverage, and
hallucination rate using (a) an LLM-as-judge and (b) reference-based comparison against a
golden set. Produces a metrics report viewable in the dashboard. Maintains a regression
eval set so changes to prompts/orchestration can be checked for regressions.

**Reference:** Ch 10 — `picoagents/src/picoagents/eval/_runner.py`,
`judges/_llm_judge.py`, `examples/evaluation/agent-evaluation.py`.

**Job skill:** Evaluation is the most under-supplied skill in the AI engineer market.
" I built an eval harness and use it as a regression gate" is a senior-level statement
that very few junior candidates can make credibly.

### 3.16 RAG knowledge base
**What:** A vector store of prior briefs and saved sources; on each run, relevant context
is retrieved and injected into the agents' context. Grows over time, so Atlas improves
with use.

**Reference:** Course samples — RAG/knowledge agent sample.

**Job skill:** RAG is the most common production LLM pattern. Implementing it as the
agent's semantic memory (not a bolt-on) shows deeper understanding.

### 3.17 MCP (Model Context Protocol) integration
**What:** Atlas connects to MCP servers as agent tools — e.g. arXiv search, GitHub repo
queries, news APIs — so agents can pull structured data from external systems through a
standard protocol.

**Reference:** Course samples + `picoagents/src/picoagents/tools/__init__.py` — MCP
integration sample.

**Job skill:** MCP is becoming the standard for agent ↔ tool interoperability (Anthropic,
OpenAI, and others are converging on it). Demonstrating MCP fluency is a 2025/2026
differentiator.

### 3.18 Deep-research multi-agent pattern
**What:** The core research team itself: AssistantAgent (web search + analysis), Verifier
(quality), Summarizer (markdown brief), coordinated by a SelectorGroupChat-style team.

**Reference:** Course samples — deep research / information gathering agent.

**Job skill:** This is the exact pattern behind OpenAI Deep Research / Perplexity.
Building it from scratch is the most direct way to demonstrate you understand those
products end-to-end.

### 3.19 Voice I/O (stretch)
**What:** Speech-to-text for the query input and text-to-speech for the brief summary,
exercising the voice-enabled agent pattern.

**Reference:** Course samples — voice-enabled agent (STT + TTS + groupchat).

**Job skill:** Multimodal I/O breadth; useful for accessibility and mobile use cases.

### 3.20 Observability with OpenTelemetry
**What:** Every agent step, tool call, and LLM call emits OTel spans. Traces are
exported and viewable in the dashboard's trace inspector (and optionally Jaeger/Tempo).

**Reference:** Ch 4 — `picoagents/src/picoagents/_otel.py`, `examples/otel/`.

**Job skill:** Observability is mandatory for production agents. "I can trace a
multi-step agent run and find the failing span" is a real on-call skill.

### 3.21 Deployment with Docker
**What:** Docker Compose that brings up backend, frontend, vector DB, and (optionally)
an OTel collector. One command: `docker compose up`.

**Reference:** `premium-samples/` — Docker / Docker Compose deployment.

**Job skill:** "Works on my machine" is not a portfolio piece. Containerized deploy
proves you can ship.

### 3.22 (Optional) SWE-style code verification agent
**What:** A small agent that can run/verify code snippets found during research (e.g.
checking a claimed algorithm actually compiles/runs), exercising coding tools and
workspace management.

**Reference:** Ch 17 — `examples/agents/swe_agent/`.

**Job skill:** Coding-agent tooling is increasingly relevant; rounds out the "agents that
act on code" competency.

---

## 4. Topic coverage matrix

Confirms Atlas exercises the full surface of the reference repo. Revisit
https://github.com/victordibia/designing-multiagent-systems for the canonical
implementation of each row while building.

| Reference repo topic | Atlas feature | Section |
|---|---|---|
| Foundations: when to use multi-agent (Ch 1) | Justify multi-agent vs single-agent for research | 3.1 |
| Multi-agent patterns taxonomy (Ch 2) | Workflow vs autonomous orchestration, chosen per stage | 3.10, 3.11 |
| UX principles for agent UIs (Ch 3) | Dashboard design: streaming, citations, approvals | 3.13 |
| Core agent + reasoning loop (Ch 4) | From-scratch `Agent` | 3.1 |
| Tools + agent-as-tool (Ch 4) | Tool system | 3.2 |
| Memory (Ch 4) | List / conversation / semantic memory | 3.3 |
| Middleware (Ch 4) | Middleware chain | 3.4 |
| Human-in-the-loop (Ch 4) | Approval gates | 3.5 |
| Structured output (Ch 4) | Typed brief/citation schemas | 3.6 |
| Multi-provider LLM (Ch 4) | OpenAI/Azure/Anthropic/local clients | 3.7 |
| Computer use (Ch 5) | Playwright browser fallback | 3.9 |
| Workflows + checkpointing (Ch 6) | DAG engine with resume | 3.10 |
| Orchestration (Ch 7) | Round-robin + AI-driven + plan-based | 3.11 |
| Termination conditions | Termination design | 3.12 |
| Agent UX app (Ch 8) | FastAPI + SSE + React dashboard | 3.13 |
| Framework comparison (Ch 9) | Re-implement one path in ADK/LangGraph | 3.14 |
| Evaluation (Ch 10) | LLM-as-judge + reference eval harness | 3.15 |
| RAG (course) | Vector knowledge base | 3.16 |
| MCP (course) | MCP server tools | 3.17 |
| Deep research pattern (course) | Researcher/Verifier/Summarizer team | 3.18 |
| Voice (course) | STT/TTS (stretch) | 3.19 |
| Cost optimization (YC case study, Ch 16) | Two-stage cheap→strong filter | 3.8 |
| SWE agent (Ch 17) | Code verification agent (optional) | 3.22 |
| Observability (Ch 4) | OpenTelemetry tracing | 3.20 |
| Deployment | Docker Compose | 3.21 |

---

## 5. The interview story (one paragraph)

> "I built a from-scratch multi-agent framework powering an autonomous deep-research
> platform. It uses plan-based orchestration (Magentic One pattern), a type-safe DAG
> workflow with checkpointing, RAG over a vector store, MCP for external data,
> Playwright computer-use fallback, OpenTelemetry observability, human approval gates,
> an LLM-as-judge eval harness with a regression set, a real-time React/SSE dashboard,
> and Docker deployment — with a two-stage cheap→strong model filter cutting inference
> cost ~90%. I also re-implemented one orchestration path in Google ADK and benchmarked
> it against my from-scratch version."

That single paragraph demonstrates ~22 distinct, hiring-relevant skills, each backed by
code you can show and design decisions you can defend.

---

## 6. Build philosophy and guardrails

- **From scratch first.** Build the framework primitives yourself (`atlascore`) before
  building the product on top. This is the pedagogy of the reference repo and the source
  of interview-defensible depth.
- **Flexible structure.** No file structure is prescribed here. Organize the code as you
  see fit while building. The architecture and feature list above are the contract; the
  layout is up to you.
- **Always be able to reference the canonical implementation.** When in doubt about how
  a primitive should work, revisit the matching file in
  https://github.com/victordibia/designing-multiagent-systems.
- **MVP first, then layers.** A reasonable order:
  1. Core agent + tools + one LLM client + web search → a single agent that answers a
     question with citations. (Covers Ch 4 core.)
  2. Memory + middleware + structured output + termination. (Rest of Ch 4.)
  3. Workflow DAG + checkpointing. (Ch 6.)
  4. Orchestration: plan-based team + round-robin critics. (Ch 7.)
  5. FastAPI + SSE + React dashboard. (Ch 8.)
  6. Eval harness. (Ch 10.)
  7. RAG vector store + MCP servers. (Course.)
  8. Computer-use fallback. (Ch 5.)
  9. Cost optimization two-stage filter. (Ch 16.)
  10. OTel observability. (Ch 4.)
  11. Human approval gates. (Ch 4.)
  12. Docker deployment. (premium-samples.)
  13. Framework comparison benchmark. (Ch 9.)
  14. Voice I/O + code-verification agent (stretches).
- **Every milestone should be demoable on its own.** After each layer, the product
  should still work end-to-end — just better.
- **Measure things.** Track cost, latency, and eval scores from day one. Numbers make
  the resume and interview claims credible.

---

## 7. Reference

- Book landing page: https://buy.multiagentbook.com
- Code repository (revisit while building):
  https://github.com/victordibia/designing-multiagent-systems
- Author's announcement post (what the book covers, in his words):
  https://newsletter.victordibia.com/p/the-designing-multi-agent-systems
- Preview PDF (table of contents):
  https://multiagentbook.com/preview.pdf
