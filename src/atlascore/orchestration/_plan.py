"""Plan-based orchestration pattern (Magentic One style).

Creates an explicit execution plan, assigns agents to each step, and retries
failed steps before advancing.
"""

from typing import Any, Dict, List, Optional, Sequence, Union

from pydantic import BaseModel, Field

from ..agents import Agent
from ..llm import BaseChatCompletionClient
from ..messages import Message, UserMessage
from ..termination import BaseTermination
from ..types import AgentResponse
from ._base import BaseOrchestrator


class StepProgressEvaluation(BaseModel):
    """Structured evaluation of step completion."""

    model_config = {"extra": "forbid"}

    step_completed: bool = Field(description="Whether the step was successfully completed")
    failure_reason: str = Field(description="Brief explanation if step failed; 'None' if successful")
    confidence_score: float = Field(
        description="Confidence in the evaluation (0.0 to 1.0)", ge=0.0, le=1.0
    )
    suggested_improvements: List[str] = Field(
        description="Specific suggestions for retry if step failed"
    )


class PlanStep(BaseModel):
    """A single step in an execution plan."""

    model_config = {"extra": "forbid"}

    task: str = Field(description="Clear, actionable task description")
    agent_name: str = Field(description="Name of the agent that should handle this step")
    reasoning: str = Field(description="Brief explanation for why this agent was chosen")


class ExecutionPlan(BaseModel):
    """A full execution plan composed of ordered steps."""

    model_config = {"extra": "forbid"}

    steps: List[PlanStep] = Field(description="Ordered list of execution steps")


class PlanBasedOrchestrator(BaseOrchestrator):
    """Plan-based orchestrator with LLM-generated plans and step retry logic."""

    def __init__(
        self,
        agents: Sequence[Agent],
        termination: BaseTermination,
        model_client: BaseChatCompletionClient,
        max_iterations: int = 50,
        max_step_retries: int = 3,
    ):
        super().__init__(agents, termination, max_iterations)
        self.model_client = model_client
        self.max_step_retries = max_step_retries

        # Plan execution state
        self.execution_plan: Optional[ExecutionPlan] = None
        self.current_step_index = 0
        self.current_step_retry_count = 0
        self.initial_task: Optional[str] = None

        # Runtime tracking
        self.step_attempts: Dict[int, List[AgentResponse]] = {}
        self.step_results: Dict[int, AgentResponse] = {}
        self.retry_instructions: Dict[int, str] = {}
        self.agent_capabilities_cache: Optional[str] = None

    async def select_next_agent(self) -> Agent:
        """Select the agent for the current plan step."""
        if not self.execution_plan:
            if self.shared_messages:
                self.initial_task = self.shared_messages[0].content
                self.execution_plan = await self.create_plan(self.initial_task)
            else:
                raise ValueError("No initial task found to create plan")

        if self.current_step_index >= len(self.execution_plan.steps):
            # All steps completed; return the first agent as a harmless fallback.
            return self.agents[0]

        current_step = self.execution_plan.steps[self.current_step_index]
        return self._find_agent_by_name(current_step.agent_name)

    async def prepare_context_for_agent(
        self, agent: Agent
    ) -> Union[str, UserMessage, List[Message]]:
        """Prepare step-specific context including retry instructions."""
        if not self.execution_plan or self.current_step_index >= len(
            self.execution_plan.steps
        ):
            return self.shared_messages.copy()

        current_step = self.execution_plan.steps[self.current_step_index]
        context = self.extract_relevant_context(current_step)
        step_message = UserMessage(
            content=self._format_step_task(current_step),
            source="plan_orchestrator",
        )
        context.append(step_message)
        return context

    async def update_shared_state(
        self, result: AgentResponse, new_messages: List[Message]
    ) -> None:
        """Update shared state and evaluate step progress."""
        await super().update_shared_state(result, new_messages)

        if not self.execution_plan or self.current_step_index >= len(
            self.execution_plan.steps
        ):
            return

        if self.current_step_index not in self.step_attempts:
            self.step_attempts[self.current_step_index] = []
        self.step_attempts[self.current_step_index].append(result)

        current_step = self.execution_plan.steps[self.current_step_index]
        progress_eval = await self.evaluate_step_progress(current_step, result)

        if progress_eval.step_completed:
            self.step_results[self.current_step_index] = result
            self.current_step_index += 1
            self.current_step_retry_count = 0
        else:
            self.current_step_retry_count += 1
            if self.current_step_retry_count <= self.max_step_retries:
                self.retry_instructions[
                    self.current_step_index
                ] = self._create_retry_instructions(current_step, progress_eval)
            else:
                self.current_step_index += 1
                self.current_step_retry_count = 0

    async def create_plan(self, task: str) -> ExecutionPlan:
        """Create an execution plan using the LLM with structured output."""
        capabilities = self.get_agent_capabilities_summary()

        planning_prompt = f"""You are a helpful assistant that breaks down tasks into executable steps.

Available agents and their capabilities:
{capabilities}

User task: {task}

Generate a concise set of step-by-step execution plans. For each step:
- Assign it to the agent best suited for that type of work
- Provide a clear, actionable task description
- Explain briefly why that agent was chosen

Keep it simple and focused. If only 2 or 3 steps are needed, that's perfectly fine.
"""

        try:
            result = await self.model_client.create(
                messages=[UserMessage(content=planning_prompt, source="planner")],
                output_format=ExecutionPlan,
            )

            if result.structured_output and isinstance(
                result.structured_output, ExecutionPlan
            ):
                return result.structured_output

            return self._create_fallback_plan(task)
        except Exception:
            return self._create_fallback_plan(task)

    def get_agent_capabilities_summary(self) -> str:
        """Cached agent capabilities summary."""
        if self.agent_capabilities_cache is None:
            self.agent_capabilities_cache = super().get_agent_capabilities_summary()
        return self.agent_capabilities_cache

    def extract_relevant_context(self, _step: PlanStep) -> List[Message]:
        """Return recent messages for focused execution."""
        if len(self.shared_messages) > 5:
            return self.shared_messages[-5:].copy()
        return self.shared_messages.copy()

    async def evaluate_step_progress(
        self, step: PlanStep, result: AgentResponse
    ) -> StepProgressEvaluation:
        """Evaluate whether a step was completed successfully."""
        agent_output = ""
        for msg in result.messages:
            if hasattr(msg, "content") and not isinstance(msg, UserMessage):
                agent_output += f"{msg.content}\n"

        if not agent_output.strip():
            return StepProgressEvaluation(
                step_completed=False,
                failure_reason="No meaningful output detected",
                confidence_score=0.9,
                suggested_improvements=[
                    "Provide more specific instructions",
                    "Add examples of expected output",
                ],
            )

        evaluation_prompt = (
            f"Evaluate whether the following step was completed based on the agent's output.\n\n"
            f"Step Task: {step.task}\n"
            f"Expected Agent: {step.agent_name}\n"
            f"Reasoning: {step.reasoning}\n\n"
            f"Agent's Output:\n{agent_output}\n\n"
            "Evaluate:\n"
            "1. Was the step task completed successfully?\n"
            "2. If not, what was the main failure reason?\n"
            "3. How confident are you in this assessment (0.0 to 1.0)?\n"
            "4. If the step failed, provide 2-3 specific suggestions for improvement.\n\n"
            "Consider the step successful if the agent made meaningful progress."
        )

        try:
            eval_result = await self.model_client.create(
                messages=[
                    UserMessage(content=evaluation_prompt, source="step_evaluator")
                ],
                output_format=StepProgressEvaluation,
            )

            if eval_result.structured_output and isinstance(
                eval_result.structured_output, StepProgressEvaluation
            ):
                return eval_result.structured_output

            return self._fallback_step_evaluation(agent_output)
        except Exception:
            return self._fallback_step_evaluation(agent_output)

    def _fallback_step_evaluation(self, agent_output: str) -> StepProgressEvaluation:
        """Heuristic fallback when the LLM evaluator fails."""
        output_lower = agent_output.lower()
        has_meaningful_content = len(agent_output.strip()) > 20
        has_error_indicators = any(
            word in output_lower for word in ["error", "failed", "cannot", "unable", "sorry"]
        )

        if has_meaningful_content and not has_error_indicators:
            return StepProgressEvaluation(
                step_completed=True,
                failure_reason="None",
                confidence_score=0.7,
                suggested_improvements=[],
            )

        return StepProgressEvaluation(
            step_completed=False,
            failure_reason="Output suggests task was not completed successfully",
            confidence_score=0.6,
            suggested_improvements=[
                "Provide clearer instructions",
                "Break task into smaller parts",
                "Add specific examples",
            ],
        )

    def _create_fallback_plan(self, task: str) -> ExecutionPlan:
        """Create a single-step fallback plan when LLM planning fails."""
        return ExecutionPlan(
            steps=[
                PlanStep(
                    task=f"Complete the task: {task}",
                    agent_name=self.agents[0].name,
                    reasoning="Single step plan fallback",
                )
            ]
        )

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

    def _format_step_task(self, step: PlanStep) -> str:
        """Format the step task, adding retry context when applicable."""
        base_task = f"STEP {self.current_step_index + 1}: {step.task}"

        if (
            self.current_step_retry_count > 0
            and self.current_step_index in self.retry_instructions
        ):
            retry_info = self.retry_instructions[self.current_step_index]
            base_task += f"\n\nRETRY INSTRUCTIONS (Attempt {self.current_step_retry_count + 1}):\n{retry_info}"

        return base_task

    def _create_retry_instructions(
        self, _step: PlanStep, progress_eval: StepProgressEvaluation
    ) -> str:
        """Create enhanced instructions for retry attempts."""
        instructions = f"Previous attempt failed: {progress_eval.failure_reason or 'Unknown reason'}\n"
        attempt_count = len(self.step_attempts.get(self.current_step_index, []))
        if attempt_count > 0:
            instructions += f"This is retry attempt {attempt_count + 1}. Please try a different approach."
        return instructions

    def _get_pattern_metadata(self) -> Dict[str, Any]:
        """Add plan-based orchestrator metadata."""
        base = super()._get_pattern_metadata()

        if self.execution_plan:
            completed_steps = len(self.step_results)
            failed_steps = max(0, self.current_step_index - completed_steps)

            base.update(
                {
                    "plan": self.execution_plan.model_dump(mode="json"),
                    "current_step_index": self.current_step_index,
                    "steps_completed": completed_steps,
                    "steps_failed": failed_steps,
                    "total_retries": sum(
                        len(attempts) - 1 for attempts in self.step_attempts.values()
                    ),
                    "current_step_retry_count": self.current_step_retry_count,
                }
            )

        return base

    def _reset_for_run(self) -> None:
        """Reset plan-based orchestrator state."""
        super()._reset_for_run()
        self.execution_plan = None
        self.current_step_index = 0
        self.current_step_retry_count = 0
        self.initial_task = None
        self.step_attempts = {}
        self.step_results = {}
        self.retry_instructions = {}
        self.agent_capabilities_cache = None
