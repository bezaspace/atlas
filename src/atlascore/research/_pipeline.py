"""Full research pipeline composed as a typed Workflow DAG."""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from ..base_types import Usage
from ..llm import BaseChatCompletionClient
from ..memory import BaseMemory, MemoryContent
from ..research_schemas import (
    CriticReview,
    ResearchBrief,
    ResearchPlan,
    ResearchReport,
    SearchOutput,
    SearchResult,
    VerificationResult,
)
from ..tools import BaseTool
from ..workflow import (
    Context,
    FunctionStep,
    StepMetadata,
    Workflow,
    WorkflowMetadata,
    WorkflowRunner,
)
from ._agents import (
    CriticPanel,
    PlannerAgent,
    ResearcherAgent,
    SynthesizerAgent,
    TriageAgent,
    VerifierAgent,
)


class ResearchQuery(BaseModel):
    """Input to the research pipeline."""

    query: str = Field(...)
    context: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)


class PlanOutput(BaseModel):
    """Output of the planning step."""

    query: str = Field(...)
    plan: ResearchPlan = Field(...)


class RAGInput(BaseModel):
    """Input to the retrieval step."""

    query: str = Field(...)
    plan: ResearchPlan = Field(...)


class RAGOutput(BaseModel):
    """Output of the retrieval step."""

    rag_context: str = Field(default="")
    retrieved_sources: List[str] = Field(default_factory=list)


class SearchInput(BaseModel):
    """Input to the search step."""

    query: str = Field(...)
    plan: ResearchPlan = Field(...)
    rag_context: str = Field(default="")
    retrieved_sources: List[str] = Field(default_factory=list)


class VerifyInput(BaseModel):
    """Input to the verification step."""

    query: str = Field(...)
    evidence: List[SearchResult] = Field(...)


class VerifyOutput(BaseModel):
    """Output of the verification step."""

    query: str = Field(...)
    verification: VerificationResult = Field(...)


class SynthesizeInput(BaseModel):
    """Input to the synthesis step."""

    query: str = Field(...)
    verification: VerificationResult = Field(...)


class SynthesizeOutput(BaseModel):
    """Output of the synthesis step."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)


class CriticInput(BaseModel):
    """Input to the critic panel step."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)


class CriticOutput(BaseModel):
    """Output of the critic panel step."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)
    critic_review: CriticReview = Field(...)


class HumanApprovalInput(BaseModel):
    """Input to the human approval gate."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)
    critic_review: CriticReview = Field(...)


class HumanApprovalOutput(BaseModel):
    """Output of the human approval gate."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)
    critic_review: CriticReview = Field(...)
    approved: bool = Field(default=True)
    feedback: str = Field(default="")


class PersistInput(BaseModel):
    """Input to the persistence step."""

    query: str = Field(...)
    brief: ResearchBrief = Field(...)
    evidence: List[SearchResult] = Field(...)
    critic_review: CriticReview = Field(...)
    approved: bool = Field(default=True)
    feedback: str = Field(default="")


class ResearchPipeline:
    """End-to-end research pipeline with human approval gate."""

    def __init__(
        self,
        model_client: BaseChatCompletionClient,
        search_tool: BaseTool,
        fetch_tool: BaseTool,
        triage_model_client: Optional[BaseChatCompletionClient] = None,
        memory: Optional[BaseMemory] = None,
        persist_dir: str = "data/research",
        approval_event_factory: Optional[Callable[[str], asyncio.Event]] = None,
    ) -> None:
        self.model_client = model_client
        self.search_tool = search_tool
        self.fetch_tool = fetch_tool
        self.triage_model_client = triage_model_client or model_client
        self.memory = memory
        self.persist_dir = Path(persist_dir)
        self.approval_event_factory = approval_event_factory

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

    def build_workflow(self) -> Workflow:
        """Build the typed workflow DAG."""
        metadata = WorkflowMetadata(
            name="Research Pipeline",
            description="Plan, retrieve, search, verify, synthesize, critique, and persist.",
        )
        workflow = Workflow(metadata=metadata)

        steps = [
            FunctionStep(
                "plan",
                StepMetadata(name="Plan", description="Create a research plan"),
                ResearchQuery,
                PlanOutput,
                self._plan_step,
            ),
            FunctionStep(
                "retrieve",
                StepMetadata(name="Retrieve", description="Query prior memory/RAG context"),
                RAGInput,
                RAGOutput,
                self._retrieve_step,
            ),
            FunctionStep(
                "search",
                StepMetadata(name="Search", description="Gather raw web evidence"),
                SearchInput,
                SearchOutput,
                self._search_step,
            ),
            FunctionStep(
                "verify",
                StepMetadata(name="Verify", description="Check claims against evidence"),
                VerifyInput,
                VerifyOutput,
                self._verify_step,
            ),
            FunctionStep(
                "synthesize",
                StepMetadata(name="Synthesize", description="Compose the research brief"),
                SynthesizeInput,
                SynthesizeOutput,
                self._synthesize_step,
            ),
            FunctionStep(
                "critic",
                StepMetadata(name="Critic", description="Round-robin critic review"),
                CriticInput,
                CriticOutput,
                self._critic_step,
            ),
            FunctionStep(
                "human_approval",
                StepMetadata(name="Human Approval", description="Human-in-the-loop approval gate"),
                HumanApprovalInput,
                HumanApprovalOutput,
                self._human_approval_step,
            ),
            FunctionStep(
                "persist",
                StepMetadata(name="Persist", description="Save brief and sources"),
                PersistInput,
                ResearchReport,
                self._persist_step,
            ),
        ]

        for step in steps:
            workflow.add_step(step)

        workflow.add_edge("plan", "retrieve")
        workflow.add_edge("plan", "search")
        workflow.add_edge("retrieve", "search")
        workflow.add_edge("search", "verify")
        workflow.add_edge("search", "persist")
        workflow.add_edge("verify", "synthesize")
        workflow.add_edge("synthesize", "critic")
        workflow.add_edge("synthesize", "persist")
        workflow.add_edge("critic", "human_approval")
        workflow.add_edge("human_approval", "persist")
        workflow.set_start_step("plan")
        workflow.add_end_step("persist")

        return workflow

    async def run(
        self,
        query: str,
        context: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> ResearchReport:
        """Run the research pipeline and return the final report."""
        workflow = self.build_workflow()
        runner = WorkflowRunner()
        execution = await runner.run(
            workflow,
            initial_input={"query": query, "context": context, "session_id": session_id},
        )
        report_data = execution.state.get("persist_output")
        if not report_data:
            raise RuntimeError("Pipeline did not produce a report")
        return ResearchReport(**report_data)

    async def _plan_step(self, input: ResearchQuery, context: Context) -> PlanOutput:
        if input.session_id:
            context.set("session_id", input.session_id)
        plan = await self.planner.run(input.query, context=input.context or "")
        output = PlanOutput(query=input.query, plan=plan)
        context.set("plan_output", output.model_dump())
        context.set("plan_usage", self.planner.last_usage.model_dump())
        return output

    async def _retrieve_step(self, input: RAGInput, context: Context) -> RAGOutput:
        rag_context_parts: List[str] = []
        retrieved_sources: List[str] = []

        if self.memory:
            queries = [input.query, *input.plan.sub_questions[:2]]
            seen: set[str] = set()
            for q in queries:
                result = await self.memory.query(q, limit=3)
                for memory in result.results:
                    if memory.content in seen:
                        continue
                    seen.add(memory.content)
                    rag_context_parts.append(memory.content)
                    source = memory.metadata.get("source_url") or memory.metadata.get("title")
                    if source:
                        retrieved_sources.append(source)

        rag_context = "\n\n".join(rag_context_parts)
        output = RAGOutput(rag_context=rag_context, retrieved_sources=retrieved_sources)
        context.set("retrieve_output", output.model_dump())
        return output

    async def _search_step(self, input: SearchInput, context: Context) -> SearchOutput:
        search_output = await self.researcher.run(
            query=input.query,
            plan=input.plan,
            rag_context=input.rag_context,
        )
        context.set("search_output", search_output.model_dump())
        context.set("search_usage", self.researcher.last_usage.model_dump())
        return search_output

    async def _verify_step(self, input: VerifyInput, context: Context) -> VerifyOutput:
        verification = await self.verifier.run(input.query, input.evidence)
        output = VerifyOutput(query=input.query, verification=verification)
        context.set("verify_output", output.model_dump())
        context.set("verify_usage", self.verifier.last_usage.model_dump())
        return output

    async def _synthesize_step(
        self, input: SynthesizeInput, context: Context
    ) -> SynthesizeOutput:
        brief = await self.synthesizer.run(input.query, input.verification)
        output = SynthesizeOutput(query=input.query, brief=brief)
        context.set("synthesize_output", output.model_dump())
        context.set("synthesize_usage", self.synthesizer.last_usage.model_dump())
        return output

    async def _critic_step(self, input: CriticInput, context: Context) -> CriticOutput:
        review = await self.critic_panel.review(input.brief, input.query)
        output = CriticOutput(
            query=input.query,
            brief=input.brief,
            critic_review=review,
        )
        context.set("critic_output", output.model_dump())
        context.set("critic_usage", self.critic_panel.last_usage.model_dump())
        return output

    async def _human_approval_step(
        self, input: HumanApprovalInput, context: Context
    ) -> HumanApprovalOutput:
        session_id = context.get("session_id")
        approved = True
        feedback = ""

        if self.approval_event_factory and session_id:
            event = self.approval_event_factory(session_id)
            try:
                await asyncio.wait_for(event.wait(), timeout=300.0)
                approved = True
            except asyncio.TimeoutError:
                approved = True
                feedback = "Auto-approved after timeout."

        return HumanApprovalOutput(
            query=input.query,
            brief=input.brief,
            critic_review=input.critic_review,
            approved=approved,
            feedback=feedback,
        )

    async def _persist_step(self, input: PersistInput, context: Context) -> ResearchReport:
        slug = re.sub(r"\W+", "-", input.query.lower()).strip("-")[:50]
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = self.persist_dir / slug / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        brief_path = run_dir / "brief.md"
        brief_path.write_text(input.brief.to_markdown(), encoding="utf-8")

        sources_path = run_dir / "sources.json"
        sources_path.write_text(
            json.dumps([s.model_dump() for s in input.evidence], indent=2, default=str),
            encoding="utf-8",
        )

        review_path = run_dir / "critic_review.json"
        review_path.write_text(
            input.critic_review.model_dump_json(indent=2),
            encoding="utf-8",
        )

        paths = [str(brief_path), str(sources_path), str(review_path)]

        usage = self._aggregate_usage(context)

        if self.memory:
            await self.memory.add(
                MemoryContent(
                    content=input.brief.to_markdown(),
                    metadata={
                        "query": input.query,
                        "type": "research_brief",
                        "paths": paths,
                    },
                )
            )
            for source in input.evidence:
                if source.content:
                    await self.memory.add(
                        MemoryContent(
                            content=source.content,
                            metadata={
                                "query": input.query,
                                "type": "source",
                                "source_url": source.url,
                                "title": source.title,
                            },
                        )
                    )

        report = ResearchReport(
            query=input.query,
            brief=input.brief,
            sources=input.evidence,
            critic_review=input.critic_review,
            paths=paths,
            approved=input.approved,
            human_feedback=input.feedback,
            usage=usage,
        )
        context.set("persist_output", report.model_dump())
        return report

    def _aggregate_usage(self, context: Context) -> Dict[str, Any]:
        """Aggregate usage from each step saved in context."""
        total = Usage(duration_ms=0)
        step_usages: Dict[str, Any] = {}
        for key in [
            "plan_usage",
            "search_usage",
            "verify_usage",
            "synthesize_usage",
            "critic_usage",
        ]:
            usage_data = context.get(key)
            if not usage_data:
                continue
            try:
                usage = Usage(**usage_data)
                total = total + usage
                step_usages[key.replace("_usage", "")] = usage.model_dump()
            except Exception:
                pass

        return {
            "total": total.model_dump(),
            "steps": step_usages,
            "workflow_id": context.get("workflow_id"),
        }
