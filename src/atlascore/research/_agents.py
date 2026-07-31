"""Research agents for the Atlas research pipeline."""

from __future__ import annotations

import json
from typing import Any, List, Optional

from pydantic import BaseModel, ValidationError

from ..agents import Agent
from ..base_types import Usage
from ..llm import BaseChatCompletionClient
from ..messages import AssistantMessage, UserMessage
from ..orchestration import RoundRobinOrchestrator
from ..research_schemas import (
    CriticReview,
    ResearchBrief,
    ResearchPlan,
    SearchOutput,
    SearchResult,
    TriageResult,
    VerificationResult,
)
from ..termination import MaxMessageTermination, TextMentionTermination
from ..tools import BaseTool


def _parse_structured(content: str, model_class: type[BaseModel]) -> BaseModel:
    """Parse JSON assistant output into a Pydantic model."""
    cleaned = content.strip()
    if cleaned.startswith("```"):
        # Strip markdown code fences if present
        lines = cleaned.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    try:
        return model_class.model_validate_json(cleaned)
    except ValidationError as e:
        raise ValueError(f"Failed to parse {model_class.__name__}: {e}") from e


def _find_last_assistant_text(response: Any) -> str:
    """Return the last assistant message content from an AgentResponse."""
    for message in reversed(response.messages):
        if isinstance(message, AssistantMessage):
            return message.content
    return ""


class _BaseResearchAgent:
    """Shared scaffolding for research agents that produce structured output."""

    def __init__(
        self,
        model_client: BaseChatCompletionClient,
        name: str,
        instructions: str,
        output_format: type[BaseModel],
    ):
        self.model_client = model_client
        self.name = name
        self.instructions = instructions
        self.output_format = output_format
        self.agent = Agent(
            name=name,
            instructions=instructions,
            model_client=model_client,
            output_format=output_format,
        )
        self.last_usage: Usage = Usage(duration_ms=0)

    async def _call_agent(self, task: str) -> Any:
        response = await self.agent.run(task)
        self.last_usage = response.usage or Usage(duration_ms=0)
        text = _find_last_assistant_text(response)
        if not text:
            raise ValueError(f"{self.name} produced no assistant output")
        return _parse_structured(text, self.output_format)


class PlannerAgent(_BaseResearchAgent):
    """Breaks a research question into sub-questions and a search plan."""

    def __init__(self, model_client: BaseChatCompletionClient):
        super().__init__(
            model_client,
            "planner",
            (
                "You are a research strategist. Given a research question, produce a concise plan "
                "with sub-questions, concrete search queries, and the minimum number of distinct "
                "sources needed. Explain your reasoning briefly."
            ),
            ResearchPlan,
        )

    async def run(self, query: str, context: str = "") -> ResearchPlan:
        prompt = f"Research question: {query}"
        if context:
            prompt += f"\n\nAdditional context:\n{context}"
        plan = await self._call_agent(prompt)
        if isinstance(plan, ResearchPlan):
            return plan
        return ResearchPlan(query=query, sub_questions=[query], search_queries=[query])


class TriageAgent(_BaseResearchAgent):
    """Cheap model that rates search results as relevant, partial, or irrelevant."""

    def __init__(self, model_client: BaseChatCompletionClient):
        super().__init__(
            model_client,
            "triage_agent",
            (
                "You are a fast relevance classifier. Given a query and a list of search results, "
                "rate each result as 'relevant', 'partial', or 'irrelevant'. Be conservative: only "
                "mark a result 'relevant' if it directly addresses the query."
            ),
            TriageResult,
        )

    async def run(self, query: str, results: List[SearchResult]) -> TriageResult:
        results_json = json.dumps([r.model_dump() for r in results], default=str)
        prompt = f"Query: {query}\n\nSearch results:\n{results_json}\n\nClassify each result."
        return await self._call_agent(prompt)  # type: ignore[return-value]

    def filter_results(
        self, results: List[SearchResult], triage: TriageResult
    ) -> List[SearchResult]:
        """Return only results rated relevant or partial."""
        relevant = set(triage.relevant_urls())
        return [r for r in results if r.url in relevant]


class ResearcherAgent:
    """Gathers raw evidence using web search/fetch and optional triage."""

    def __init__(
        self,
        model_client: BaseChatCompletionClient,
        search_tool: BaseTool,
        fetch_tool: BaseTool,
        triage_agent: Optional[TriageAgent] = None,
        max_sources: int = 5,
    ) -> None:
        self.model_client = model_client
        self.search_tool = search_tool
        self.fetch_tool = fetch_tool
        self.triage_agent = triage_agent
        self.max_sources = max_sources
        self.last_usage: Usage = Usage(duration_ms=0)
        self._extractor = Agent(
            name="researcher_extractor",
            instructions=(
                "You are an evidence extractor. Given a research question and source materials, "
                "extract key facts, quotes, and claims. Return a structured list of sources with "
                "title, URL, snippet, and the most relevant content excerpt. Include a brief summary."
            ),
            model_client=model_client,
            output_format=SearchOutput,
        )

    async def run(
        self, query: str, plan: ResearchPlan, rag_context: str = ""
    ) -> SearchOutput:
        search_queries = plan.search_queries or [query]
        raw_results: List[SearchResult] = []

        for idx, q in enumerate(search_queries):
            if len(raw_results) >= self.max_sources:
                break
            tool_result = await self.search_tool.execute(
                {"query": q, "max_results": self.max_sources}
            )
            if tool_result.success and isinstance(tool_result.result, list):
                for item in tool_result.result:
                    if isinstance(item, dict):
                        raw_results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                snippet=item.get("snippet", ""),
                                source_index=len(raw_results),
                            )
                        )
                    if len(raw_results) >= self.max_sources:
                        break

        if self.triage_agent and raw_results:
            triage = await self.triage_agent.run(query, raw_results)
            raw_results = self.triage_agent.filter_results(raw_results, triage)
            self.last_usage = self.triage_agent.last_usage

        fetched: List[SearchResult] = []
        for result in raw_results[: self.max_sources]:
            fetch_result = await self.fetch_tool.execute(
                {"url": result.url, "output_format": "text"}
            )
            content = ""
            if fetch_result.success:
                content = str(fetch_result.result or "")
            else:
                content = f"Fetch failed: {fetch_result.error}"
            fetched.append(
                result.model_copy(
                    update={"content": content[:4000], "relevance": result.relevance or "partial"}
                )
            )

        source_text = "\n\n---\n\n".join(
            f"[{i}] {r.title}\nURL: {r.url}\nSnippet: {r.snippet}\nContent: {r.content or ''}"
            for i, r in enumerate(fetched)
        )

        prompt = f"Research question: {query}\n\nPlan: {plan.model_dump_json()}\n\n"
        if rag_context:
            prompt += f"Prior knowledge:\n{rag_context}\n\n"
        prompt += f"Sources:\n{source_text}\n\nExtract structured evidence."

        response = await self._extractor.run(prompt)
        self.last_usage = (self.last_usage or Usage(duration_ms=0)) + (response.usage or Usage(duration_ms=0))
        text = _find_last_assistant_text(response)
        if not text:
            return SearchOutput(query=query, evidence=fetched, summary="No extraction output")
        return _parse_structured(text, SearchOutput)  # type: ignore[return-value]


class VerifierAgent(_BaseResearchAgent):
    """Checks claims against evidence and flags weak citations."""

    def __init__(self, model_client: BaseChatCompletionClient):
        super().__init__(
            model_client,
            "verifier",
            (
                "You are a fact-checker. Given a research question and evidence, identify the key "
                "claims, rate each as supported/refuted/unclear, assign confidence, and cite the "
                "specific sources that justify the assessment."
            ),
            VerificationResult,
        )

    async def run(self, query: str, evidence: List[SearchResult]) -> VerificationResult:
        evidence_json = json.dumps([e.model_dump() for e in evidence], default=str)
        prompt = f"Research question: {query}\n\nEvidence:\n{evidence_json}\n\nVerify claims."
        return await self._call_agent(prompt)  # type: ignore[return-value]


class SynthesizerAgent(_BaseResearchAgent):
    """Composes a verified ResearchBrief with citations."""

    def __init__(self, model_client: BaseChatCompletionClient):
        super().__init__(
            model_client,
            "synthesizer",
            (
                "You are a research writer. Given a verified verification result, compose a "
                "well-structured research brief with title, summary, sections, and citations. "
                "Every factual claim must include a citation with a URL quote."
            ),
            ResearchBrief,
        )

    async def run(
        self, query: str, verification: VerificationResult
    ) -> ResearchBrief:
        prompt = (
            f"Research question: {query}\n\n"
            f"Verification result: {verification.model_dump_json()}\n\n"
            "Write the final research brief."
        )
        return await self._call_agent(prompt)  # type: ignore[return-value]


class CriticPanel:
    """Round-robin critic agents that review a brief and request revisions."""

    def __init__(
        self,
        model_client: BaseChatCompletionClient,
        agents: Optional[List[Agent]] = None,
    ) -> None:
        self.model_client = model_client
        if agents is None:
            evidence_critic = Agent(
                name="evidence_critic",
                instructions=(
                    "You review research briefs for evidence quality. If the brief lacks "
                    "specific citations, quotes, or sources, ask for revisions. "
                    "If you are satisfied, end with NO_REVISIONS."
                ),
                model_client=model_client,
            )
            clarity_critic = Agent(
                name="clarity_critic",
                instructions=(
                    "You review research briefs for clarity and structure. If sections are "
                    "confusing or incomplete, ask for revisions. "
                    "If you are satisfied, end with NO_REVISIONS."
                ),
                model_client=model_client,
            )
            agents = [evidence_critic, clarity_critic]

        self.panel = RoundRobinOrchestrator(
            agents=agents,
            termination=MaxMessageTermination(max_messages=6)
            | TextMentionTermination(text="NO_REVISIONS"),
            max_iterations=4,
        )
        self.last_usage: Usage = Usage(duration_ms=0)

    async def review(self, brief: ResearchBrief, query: str) -> CriticReview:
        task = (
            f"Review the following research brief for the query: {query}\n\n"
            f"{brief.to_markdown()}\n\n"
            "Provide feedback. If no changes are needed, respond with NO_REVISIONS."
        )
        panel_result = await self.panel.run(task)
        self.last_usage = panel_result.usage or Usage(duration_ms=0)

        parse_prompt = (
            "Based on the following critic discussion, produce a structured review.\n\n"
            f"Discussion:\n{panel_result.final_result}"
        )
        parse_result = await self.model_client.create(
            messages=[UserMessage(content=parse_prompt, source="user")],
            output_format=CriticReview,
        )
        if parse_result.structured_output:
            self.last_usage = self.last_usage + (parse_result.usage or Usage(duration_ms=0))
            return parse_result.structured_output  # type: ignore[return-value]

        text = parse_result.message.content or panel_result.final_result
        return _parse_structured(text, CriticReview)  # type: ignore[return-value]
