"""AI-driven speaker selection orchestration pattern.

Uses an LLM with structured output to choose the most appropriate next agent.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..agents import Agent
from ..llm import BaseChatCompletionClient
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class AgentSelection(BaseModel):
    """Structured output for agent selection decisions."""

    selected_agent: str = Field(..., description="Name of the selected agent")
    reasoning: str = Field(..., description="Why this agent was chosen")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in the selection (0.0-1.0)"
    )


class AIOrchestrator(BaseOrchestrator):
    """AI-driven orchestration pattern."""

    def __init__(
        self,
        agents: List[Agent],
        termination: BaseTermination,
        model_client: BaseChatCompletionClient,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self.model_client = model_client
        self.selection_history: List[Dict[str, Any]] = []
        self.agent_capabilities_cache: Optional[str] = None

    async def select_next_agent(self) -> Agent:
        """Use an LLM to select the next agent."""
        capabilities = self.get_agent_capabilities_summary()
        conversation_context = self._format_conversation_for_selection()

        selection_prompt = (
            f"You are coordinating a team of AI agents working collaboratively on a task.\n\n"
            f"Available agents and their capabilities:\n{capabilities}\n\n"
            f"Recent conversation history:\n{conversation_context}\n\n"
            "Based on the conversation context and each agent's capabilities, "
            "choose which agent should respond next to move the task forward. Consider:\n"
            "- What type of response is needed right now?\n"
            "- Which agent's skills/tools best match the current need?\n"
            "- Natural flow of the conversation\n"
            "- Avoiding repetitive selections unless justified\n\n"
            "Select the most appropriate agent and explain your reasoning in a single clean line."
        )

        messages: List[Message] = [
            UserMessage(content=selection_prompt, source="orchestrator")
        ]

        try:
            result = await self.model_client.create(
                messages=messages, output_format=AgentSelection
            )

            if result.structured_output and isinstance(
                result.structured_output, AgentSelection
            ):
                selection = result.structured_output
                selected_name = selection.selected_agent
                reasoning = selection.reasoning
                confidence = selection.confidence
            elif result.message and result.message.content:
                selected_name = self._extract_agent_name_from_text(result.message.content)
                reasoning = "Fallback selection due to missing structured output"
                confidence = 0.5
            else:
                selected_name = self._get_fallback_agent_name()
                reasoning = "Fallback selection due to empty model response"
                confidence = 0.1

        except Exception as e:
            selected_name = self._get_fallback_agent_name()
            reasoning = f"Fallback due to LLM error: {e}"
            confidence = 0.1

        selected_agent = self._find_agent_by_name(selected_name)

        self.selection_history.append(
            {
                "selected_agent": selected_agent.name,
                "iteration": self.iteration_count,
                "reasoning": reasoning,
                "confidence": confidence,
                "conversation_length": len(self.shared_messages),
            }
        )

        return selected_agent

    async def prepare_context_for_agent(self, agent: Agent) -> str:
        """Format the full shared history as a single context string."""
        if not self.shared_messages:
            return (
                "You are part of a team taking turns to collaboratively address a task. "
                "It is now your turn. "
            )

        context = (
            "You are part of a team taking turns to collaboratively address a task. "
            "Here's the progress/history so far:\n\n"
        )
        for msg in self.shared_messages:
            context += f"{msg}\n"
        context += "\nIt is now your turn."
        return context

    async def update_shared_state(
        self, result: AgentResponse, new_messages: List[Message]
    ) -> None:
        """Add new messages to the shared conversation."""
        await super().update_shared_state(result, new_messages)

    def get_agent_capabilities_summary(self) -> str:
        """Cached agent capabilities summary."""
        if self.agent_capabilities_cache is None:
            self.agent_capabilities_cache = super().get_agent_capabilities_summary()
        return self.agent_capabilities_cache

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Add AI-driven selection metadata."""
        base = super()._get_pattern_metadata()

        unique_agents = {sel["selected_agent"] for sel in self.selection_history}
        recent_selections = self.selection_history[-5:] if self.selection_history else []
        avg_confidence = (
            sum(sel["confidence"] for sel in self.selection_history)
            / len(self.selection_history)
            if self.selection_history
            else 0.0
        )

        base.update(
            {
                "selection_history": [
                    {
                        "agent": sel["selected_agent"],
                        "iteration": sel["iteration"],
                        "confidence": sel["confidence"],
                    }
                    for sel in self.selection_history
                ],
                "unique_agents_selected": len(unique_agents),
                "agent_diversity": len(unique_agents) / len(self.agents)
                if self.agents
                else 0.0,
                "average_confidence": round(avg_confidence, 3),
                "recent_reasoning": [sel["reasoning"] for sel in recent_selections],
                "model_used": getattr(self.model_client, "model", "unknown"),
            }
        )
        return base

    def _reset_for_run(self) -> None:
        """Reset AI-driven selection state."""
        super()._reset_for_run()
        self.selection_history = []
        self.agent_capabilities_cache = None

    def _format_conversation_for_selection(self) -> str:
        """Format shared messages for the selection prompt."""
        if not self.shared_messages:
            return "No conversation yet."

        context = "History so far:\n\n"
        for msg in self.shared_messages:
            context += f"{msg}\n"
        return context

    def _find_agent_by_name(self, name: str) -> Agent:
        """Find an agent by name with fuzzy matching."""
        name_lower = name.lower().strip()

        for agent in self.agents:
            if agent.name.lower() == name_lower:
                return agent

        for agent in self.agents:
            if name_lower in agent.name.lower() or agent.name.lower() in name_lower:
                return agent

        return self.agents[0]

    def _extract_agent_name_from_text(self, text: str) -> str:
        """Extract an agent name from a plain-text response as a fallback."""
        text_lower = text.lower()
        for agent in self.agents:
            if agent.name.lower() in text_lower:
                return agent.name
        return self._get_fallback_agent_name()

    def _get_fallback_agent_name(self) -> str:
        """Simple round-robin fallback when selection fails."""
        if self.selection_history:
            last_agent_name = self.selection_history[-1]["selected_agent"]
            agent_names = [a.name for a in self.agents]
            try:
                last_index = agent_names.index(last_agent_name)
                return agent_names[(last_index + 1) % len(agent_names)]
            except ValueError:
                pass
        return self.agents[0].name
