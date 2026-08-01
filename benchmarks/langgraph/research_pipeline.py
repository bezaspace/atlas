"""LangGraph implementation of Atlas's plan-based research pipeline.

This mirrors the ``atlascore`` ``ResearchPipeline`` using LangGraph's
``StateGraph`` so the two frameworks can be compared on the same inputs.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from atlascore import OpenAIChatCompletionClient
from atlascore.base_types import Usage
from atlascore.research._agents import (
    CriticPanel,
    PlannerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    TriageAgent,
    VerifierAgent,
)
from atlascore.research_schemas import (
    Citation,
    CriticReview,
    ResearchBrief,
    ResearchPlan,
    ResearchReport,
    SearchOutput,
    SearchResult,
    VerificationResult,
)
from atlascore.tools import BaseTool


class ResearchState(TypedDict, total=False):
    """LangGraph state for the research pipeline."""

    query: str
    context: str
    plan: Optional[ResearchPlan]
    evidence: List[SearchResult]
    verification: Optional[VerificationResult]
    brief: Optional[ResearchBrief]
    critic_review: Optional[CriticReview]
    report: Optional[ResearchReport]
    error: Optional[str]
    usage: Usage


def _empty_brief(query: str, message: str = "") -> ResearchBrief:
    return ResearchBrief(
        title=query,
        summary=message or "No brief produced.",
        sections=[],
        citations=[Citation(quote="No citations available.", index=0)],
        confidence=0.0,
    )


def _add_usage(total: Optional[Usage], addition: Optional[Usage]) -> Usage:
    if isinstance(total, dict):
        total = Usage(**total)
    if isinstance(addition, dict):
        addition = Usage(**addition)
    if total is None:
        return addition or Usage(duration_ms=0)
    if addition is None:
        return total
    return total + addition


class LangGraphResearchPipeline:
    """Plan-based research pipeline built with LangGraph."""

    def __init__(
        self,
        model_client: OpenAIChatCompletionClient,
        search_tool: BaseTool,
        fetch_tool: BaseTool,
        triage_model_client: Optional[OpenAIChatCompletionClient] = None,
        vision_model_client: Optional[OpenAIChatCompletionClient] = None,
    ) -> None:
        self.model_client = model_client
        self.search_tool = search_tool
        self.fetch_tool = fetch_tool
        self.triage_model_client = triage_model_client or model_client
        self.vision_model_client = vision_model_client

        self.planner = PlannerAgent(model_client)
        self.triage_agent = TriageAgent(self.triage_model_client)
        self.researcher = ResearcherAgent(
            model_client=model_client,
            search_tool=search_tool,
            fetch_tool=fetch_tool,
            triage_agent=self.triage_agent,
        )
        self.verifier = VerifierAgent(model_client)
        self.synthesizer = SynthesizerAgent(model_client)
        self.critic_panel = CriticPanel(model_client)

        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        builder = StateGraph(ResearchState)
        builder.add_node("plan", self._plan_node)
        builder.add_node("search", self._search_node)
        builder.add_node("verify", self._verify_node)
        builder.add_node("synthesize", self._synthesize_node)
        builder.add_node("critique", self._critique_node)
        builder.add_node("finalize", self._finalize_node)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "search")
        builder.add_edge("search", "verify")
        builder.add_edge("verify", "synthesize")
        builder.add_edge("synthesize", "critique")
        builder.add_edge("critique", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile()

    async def run(
        self,
        query: str,
        context: str = "",
        session_id: Optional[str] = None,
    ) -> ResearchReport:
        """Run the LangGraph research pipeline and return a report."""
        start_time = time.time()
        result = await self._graph.ainvoke(
            {"query": query, "context": context, "usage": Usage(duration_ms=0)}
        )
        duration_ms = int((time.time() - start_time) * 1000)

        report = result.get("report")
        if report is None:
            report = ResearchReport(
                query=query,
                brief=_empty_brief(query, result.get("error")),
                brief_markdown="",
                sources=[],
                critic_review=None,
                paths=[],
                approved=True,
                human_feedback="",
                usage={"total": Usage(duration_ms=duration_ms).model_dump()},
            )

        total_usage = _add_usage(report.usage.get("total"), Usage(duration_ms=duration_ms))
        report_usage = dict(report.usage) if isinstance(report.usage, dict) else {}
        report_usage["total"] = total_usage.model_dump()
        return report.model_copy(update={"usage": report_usage})

    async def _plan_node(self, state: ResearchState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            plan = await self.planner.run(
                state["query"], context=state.get("context", "")
            )
            return {
                "plan": plan,
                "usage": _add_usage(state.get("usage"), self.planner.last_usage),
            }
        except Exception as e:
            return {"error": f"Plan failed: {e}", "usage": state.get("usage")}

    async def _search_node(self, state: ResearchState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            search_output: SearchOutput = await self.researcher.run(
                query=state["query"],
                plan=state.get("plan") or ResearchPlan(query=state["query"]),
                rag_context=state.get("context", ""),
            )
            return {
                "evidence": search_output.evidence,
                "usage": _add_usage(state.get("usage"), self.researcher.last_usage),
            }
        except Exception as e:
            return {"error": f"Search failed: {e}", "usage": state.get("usage")}

    async def _verify_node(self, state: ResearchState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            evidence = state.get("evidence") or []
            verification = await self.verifier.run(state["query"], evidence)
            return {
                "verification": verification,
                "usage": _add_usage(state.get("usage"), self.verifier.last_usage),
            }
        except Exception as e:
            return {"error": f"Verify failed: {e}", "usage": state.get("usage")}

    async def _synthesize_node(self, state: ResearchState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            verification = state.get("verification")
            if verification is None:
                raise ValueError("No verification available")
            brief = await self.synthesizer.run(
                state["query"],
                verification,
                rag_context=state.get("context", ""),
            )
            return {
                "brief": brief,
                "usage": _add_usage(state.get("usage"), self.synthesizer.last_usage),
            }
        except Exception as e:
            return {"error": f"Synthesize failed: {e}", "usage": state.get("usage")}

    async def _critique_node(self, state: ResearchState) -> Dict[str, Any]:
        if state.get("error"):
            return {}
        try:
            brief = state.get("brief")
            if brief is None:
                raise ValueError("No brief available")
            review = await self.critic_panel.review(brief, state["query"])
            return {
                "critic_review": review,
                "usage": _add_usage(state.get("usage"), self.critic_panel.last_usage),
            }
        except Exception as e:
            return {"error": f"Critique failed: {e}", "usage": state.get("usage")}

    async def _finalize_node(self, state: ResearchState) -> Dict[str, Any]:
        query = state["query"]
        error = state.get("error")
        brief = state.get("brief") or _empty_brief(query, error or "")
        evidence = state.get("evidence") or []
        critic_review = state.get("critic_review")

        usage = state.get("usage") or Usage(duration_ms=0)

        report = ResearchReport(
            query=query,
            brief=brief,
            brief_markdown=brief.to_markdown(),
            sources=evidence,
            critic_review=critic_review,
            paths=[],
            approved=True,
            human_feedback="",
            usage={"total": usage.model_dump()},
        )
        return {"report": report}
