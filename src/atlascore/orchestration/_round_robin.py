"""Round-robin orchestration pattern.

Cycles through agents in fixed order, giving each agent access to the complete
shared conversation history.
"""

from typing import Any, Dict, List

from ..agents import Agent
from ..messages import Message
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class RoundRobinOrchestrator(BaseOrchestrator):
    """Round-robin orchestration pattern."""

    def __init__(
        self,
        agents: List[Agent],
        termination: BaseTermination,
        max_iterations: int = 50,
    ):
        super().__init__(agents, termination, max_iterations)
        self.current_agent_index = 0

    async def select_next_agent(self) -> Agent:
        """Select the next agent in round-robin order."""
        agent = self.agents[self.current_agent_index]
        self.current_agent_index = (self.current_agent_index + 1) % len(self.agents)
        return agent

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

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Add round-robin-specific metadata."""
        base = super()._get_pattern_metadata()
        base.update(
            {
                "cycles_completed": self.iteration_count // len(self.agents)
                if self.agents
                else 0,
                "current_agent_index": self.current_agent_index,
                "agents_order": [agent.name for agent in self.agents],
            }
        )
        return base

    def _reset_for_run(self) -> None:
        """Reset round-robin state."""
        super()._reset_for_run()
        self.current_agent_index = 0
