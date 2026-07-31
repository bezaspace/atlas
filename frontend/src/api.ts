const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (import.meta.env.DEV ? 'http://localhost:8000' : '');

function apiUrl(path: string) {
  return `${API_BASE}${path}`;
}

async function handleResponse(response: Response) {
  if (!response.ok) {
    const text = await response.text().catch(() => 'Unknown error');
    throw new Error(`${response.status}: ${text}`);
  }
  return response.json();
}

export async function getHealth() {
  const response = await fetch(apiUrl('/health'));
  return handleResponse(response);
}

export async function createSession(entity_id = 'research_pipeline') {
  const response = await fetch(apiUrl('/sessions'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ entity_id }),
  });
  return handleResponse(response);
}

export async function listSessions() {
  const response = await fetch(apiUrl('/sessions'));
  return handleResponse(response);
}

export async function deleteSession(sessionId: string) {
  const response = await fetch(apiUrl(`/sessions/${sessionId}`), {
    method: 'DELETE',
  });
  return handleResponse(response);
}

export async function startRun(
  sessionId: string,
  query: string,
  requireHumanApproval = false
) {
  const response = await fetch(apiUrl(`/sessions/${sessionId}/run`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      context: undefined,
      require_human_approval: requireHumanApproval,
    }),
  });
  return handleResponse(response);
}

export async function approveSession(
  sessionId: string,
  approved: boolean,
  approvals: import('./types').ToolApprovalResponse[] = []
) {
  const response = await fetch(apiUrl(`/sessions/${sessionId}/approve`), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved, approvals }),
  });
  return handleResponse(response);
}

export async function startEval(datasetPath: string, maxItems = 5) {
  const response = await fetch(apiUrl('/eval'), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataset_path: datasetPath, max_items: maxItems }),
  });
  return handleResponse(response);
}

export function streamUrl(sessionId: string) {
  return apiUrl(`/sessions/${sessionId}/stream`);
}
