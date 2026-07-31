export interface HealthResponse {
  status: string;
  model?: string;
  search_provider?: string;
  version?: string;
}

export interface SessionInfo {
  id: string;
  entity_id: string;
  entity_type: string;
  status: string;
  created_at: string;
  last_activity: string;
}

export interface Citation {
  source_url: string;
  source_title: string;
  quote: string;
  index: number;
}

export interface ResearchBrief {
  title: string;
  summary: string;
  sections: { heading: string; content: string }[];
  citations: Citation[];
  confidence: number;
}

export interface SearchResult {
  title: string;
  url: string;
  snippet?: string;
  content?: string;
  relevance?: string;
  source_index: number;
}

export interface ResearchReport {
  query: string;
  brief: ResearchBrief;
  brief_markdown?: string;
  sources: SearchResult[];
  approved: boolean;
  human_feedback?: string;
  usage?: Record<string, any>;
}

export interface WorkflowEvent {
  event_type: string;
  timestamp: string;
  step_id?: string;
  step_name?: string;
  status?: string;
  report?: ResearchReport;
  [key: string]: any;
}

export interface StreamPayload {
  session_id: string;
  event: WorkflowEvent | { type: string; [key: string]: any };
}

export interface ToolApprovalRequest {
  request_id: string;
  tool_call_id: string;
  tool_name: string;
  parameters: Record<string, any>;
}

export interface ToolApprovalResponse {
  request_id: string;
  tool_call_id: string;
  approved: boolean;
  reason?: string;
}
