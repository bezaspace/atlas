"""Agent context management for atlascore."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .messages import AssistantMessage, Message, ToolCallRequest, UserMessage


class ToolApprovalRequest(BaseModel):
    """Request for user approval of a tool call."""

    request_id: str = Field(...)
    tool_call_id: str = Field(...)
    tool_name: str = Field(...)
    parameters: Dict[str, Any] = Field(...)
    original_tool_call: ToolCallRequest = Field(...)

    def create_response(
        self, approved: bool, reason: Optional[str] = None
    ) -> "ToolApprovalResponse":
        return ToolApprovalResponse(
            request_id=self.request_id,
            tool_call_id=self.tool_call_id,
            approved=approved,
            reason=reason,
        )


class ToolApprovalResponse(BaseModel):
    """Response to a tool approval request."""

    request_id: str = Field(...)
    tool_call_id: str = Field(...)
    approved: bool = Field(...)
    reason: Optional[str] = Field(default=None)


class AgentContext(BaseModel):
    """Unified context object for agents."""

    messages: List[Message] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    shared_state: Dict[str, Any] = Field(default_factory=dict)
    environment: Dict[str, Any] = Field(default_factory=dict)
    session_id: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)

    pending_approval_requests: List[ToolApprovalRequest] = Field(default_factory=list)
    approval_responses: Dict[str, ToolApprovalResponse] = Field(default_factory=dict)
    pending_tool_calls: Dict[str, ToolCallRequest] = Field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def get_last_user_message(self) -> Optional[UserMessage]:
        for msg in reversed(self.messages):
            if isinstance(msg, UserMessage):
                return msg
        return None

    def get_last_assistant_message(self) -> Optional[AssistantMessage]:
        for msg in reversed(self.messages):
            if isinstance(msg, AssistantMessage):
                return msg
        return None

    def clear_messages(self) -> None:
        self.messages.clear()

    def reset(self) -> None:
        self.messages.clear()
        self.shared_state.clear()
        self.metadata.clear()

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    @property
    def waiting_for_approval(self) -> bool:
        return len(self.pending_approval_requests) > 0

    def add_approval_request(
        self, tool_call: ToolCallRequest, tool_name: str
    ) -> ToolApprovalRequest:
        request = ToolApprovalRequest(
            request_id=f"approval_{tool_call.call_id}",
            tool_call_id=tool_call.call_id,
            tool_name=tool_name,
            parameters=tool_call.parameters,
            original_tool_call=tool_call,
        )
        self.pending_approval_requests.append(request)
        self.pending_tool_calls[tool_call.call_id] = tool_call
        return request

    def add_approval_response(self, response: ToolApprovalResponse) -> None:
        self.approval_responses[response.tool_call_id] = response
        self.pending_approval_requests = [
            req
            for req in self.pending_approval_requests
            if req.tool_call_id != response.tool_call_id
        ]

    def get_approval_response(self, tool_call_id: str) -> Optional[ToolApprovalResponse]:
        return self.approval_responses.get(tool_call_id)

    def get_approved_tool_calls(self) -> List[ToolCallRequest]:
        approved = []
        processed_ids = []
        for call_id, response in self.approval_responses.items():
            if response.approved and call_id in self.pending_tool_calls:
                approved.append(self.pending_tool_calls[call_id])
                processed_ids.append(call_id)
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]
        return approved

    def get_rejected_tool_calls(self) -> List[tuple[str, ToolCallRequest]]:
        rejected = []
        processed_ids = []
        for call_id, response in self.approval_responses.items():
            if not response.approved and call_id in self.pending_tool_calls:
                rejected.append((call_id, self.pending_tool_calls[call_id]))
                processed_ids.append(call_id)
        for call_id in processed_ids:
            del self.approval_responses[call_id]
            del self.pending_tool_calls[call_id]
        return rejected

    def __str__(self) -> str:
        approval_info = (
            f", {len(self.pending_approval_requests)} pending approvals"
            if self.waiting_for_approval
            else ""
        )
        return (
            f"AgentContext(messages={self.message_count}, session={self.session_id}{approval_info})"
        )
