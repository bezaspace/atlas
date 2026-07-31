import { useEffect, useMemo, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  approveSession,
  createSession,
  deleteSession,
  getHealth,
  listSessions,
  startEval,
  startRun,
} from './api';
import { useSSE } from './hooks/useSSE';
import type {
  HealthResponse,
  ResearchReport,
  SearchResult,
  SessionInfo,
  ToolApprovalResponse,
  WorkflowEvent,
} from './types';

function classNames(...classes: (string | false | undefined)[]) {
  return classes.filter(Boolean).join(' ');
}

function formatEvent(event: WorkflowEvent) {
  const type = event.event_type || (event as any).type || 'unknown';
  switch (type) {
    case 'workflow_started':
      return 'Workflow started';
    case 'workflow_completed':
      return 'Workflow completed';
    case 'workflow_failed':
      return `Workflow failed: ${(event as any).error || ''}`;
    case 'workflow_cancelled':
      return 'Workflow cancelled';
    case 'step_started':
      return `Step started: ${(event as any).step_name || (event as any).step_id}`;
    case 'step_completed':
      return `Step completed: ${(event as any).step_name || (event as any).step_id}`;
    case 'step_failed':
      return `Step failed: ${(event as any).step_id || ''} — ${(event as any).error || ''}`;
    case 'tool_call':
      return `Tool call: ${(event as any).tool_name}`;
    case 'tool_approval':
      return `Approval required: ${(event as any).tool_name || ''}`;
    case 'model_call':
      return `Model call: ${(event as any).model || 'unknown'}`;
    case 'model_response':
      return 'Model response received';
    case 'sse_raw':
      return `Raw SSE: ${(event as any).raw}`;
    default:
      return JSON.stringify(event).slice(0, 200);
  }
}

function CitationLink({ href, children }: { href?: string; children?: React.ReactNode }) {
  const handleClick = (e: React.MouseEvent) => {
    if (href?.startsWith('#source-')) {
      e.preventDefault();
      const id = href.slice(1);
      document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  };
  return (
    <a
      href={href}
      onClick={handleClick}
      className="text-indigo-600 hover:underline dark:text-indigo-400"
    >
      {children}
    </a>
  );
}

export default function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [query, setQuery] = useState('What is Atlas?');
  const [requireHumanApproval, setRequireHumanApproval] = useState(false);
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'feed' | 'brief' | 'sources' | 'sessions' | 'eval'>('feed');
  const [report, setReport] = useState<ResearchReport | null>(null);
  const [pendingApproval, setPendingApproval] = useState(false);
  const [approvalRequest, setApprovalRequest] = useState<WorkflowEvent | null>(null);
  const [evalDataset, setEvalDataset] = useState('data/eval.jsonl');
  const [evalSessionId, setEvalSessionId] = useState<string | null>(null);

  const { events, connected, error: sseError, completed, reset } = useSSE(sessionId || evalSessionId);

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'unavailable' }));
    refreshSessions();
  }, []);

  useEffect(() => {
    if (events.length === 0) return;
    const last = events[events.length - 1].event as WorkflowEvent;
    if (last.event_type === 'workflow_completed') {
      setReport((last as any).report || null);
      setActiveTab('brief');
      refreshSessions();
    }
    if (last.event_type === 'tool_approval') {
      setApprovalRequest(last);
    }
    if (last.event_type === 'step_started' && (last as any).step_id === 'human_approval') {
      setPendingApproval(true);
    }
    if (last.event_type === 'step_completed' && (last as any).step_id === 'human_approval') {
      setPendingApproval(false);
    }
  }, [events]);

  useEffect(() => {
    if (completed) {
      setLoading(false);
      refreshSessions();
    }
  }, [completed]);

  const refreshSessions = async () => {
    try {
      const data = await listSessions();
      setSessions(data);
    } catch {
      // ignore
    }
  };

  const handleRun = async () => {
    if (!query.trim()) return;
    setLoading(true);
    setReport(null);
    setPendingApproval(false);
    setApprovalRequest(null);
    reset();
    try {
      const session = await createSession();
      setSessionId(session.id);
      setActiveTab('feed');
      await startRun(session.id, query, requireHumanApproval);
    } catch (e: any) {
      alert(e.message || 'Failed to start run');
      setLoading(false);
    }
  };

  const handleApprove = async (approved: boolean) => {
    if (!sessionId) return;
    const approvals: ToolApprovalResponse[] = approvalRequest
      ? [
          {
            request_id: (approvalRequest as any).request_id || (approvalRequest as any).approval_request?.request_id || '',
            tool_call_id: (approvalRequest as any).tool_call_id || (approvalRequest as any).approval_request?.tool_call_id || '',
            approved,
            reason: '',
          },
        ]
      : [];
    await approveSession(sessionId, approved, approvals);
    setPendingApproval(false);
    setApprovalRequest(null);
  };

  const handleRunEval = async () => {
    setLoading(true);
    reset();
    try {
      const { session_id } = await startEval(evalDataset);
      setEvalSessionId(session_id);
      setSessionId(null);
      setActiveTab('feed');
    } catch (e: any) {
      alert(e.message || 'Failed to start eval');
      setLoading(false);
    }
  };

  const sources = useMemo(() => report?.sources || [], [report]);
  const usage = report?.usage;

  const filteredSources = sources; // could add filter UI later

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      <header className="border-b border-slate-200 bg-white px-6 py-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
        <div className="mx-auto flex max-w-7xl items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-8 w-8 rounded bg-indigo-600" />
            <h1 className="text-xl font-semibold">Atlas</h1>
          </div>
          <div className="flex items-center gap-4 text-sm">
            <span
              className={classNames(
                'rounded-full px-2 py-0.5',
                health?.status === 'healthy'
                  ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                  : 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
              )}
            >
              {health?.status || 'unknown'}
            </span>
            <span className="text-slate-500 dark:text-slate-400">
              {health?.model || 'no model'}
            </span>
            {connected && (
              <span className="text-green-600 dark:text-green-400">SSE connected</span>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl p-6">
        <div className="grid gap-6 lg:grid-cols-12">
          {/* Sidebar */}
          <aside className="lg:col-span-3">
            <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
              <h2 className="mb-4 font-medium">Research</h2>
              <div className="space-y-3">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                  className="w-full rounded-md border border-slate-300 p-2 text-sm dark:border-slate-600 dark:bg-slate-700"
                  placeholder="Enter research question..."
                />
                <label className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={requireHumanApproval}
                    onChange={(e) => setRequireHumanApproval(e.target.checked)}
                    className="rounded border-slate-300"
                  />
                  Require human approval
                </label>
                <button
                  onClick={handleRun}
                  disabled={loading || !query.trim()}
                  className="w-full rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                >
                  {loading ? 'Running...' : 'Run Research'}
                </button>
              </div>
            </div>

            <nav className="mt-6 rounded-lg border border-slate-200 bg-white p-2 shadow-sm dark:border-slate-700 dark:bg-slate-800">
              {(
                [
                  ['feed', 'Activity Feed'],
                  ['brief', 'Brief'],
                  ['sources', 'Sources'],
                  ['sessions', 'Sessions'],
                  ['eval', 'Eval'],
                ] as const
              ).map(([key, label]) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={classNames(
                    'block w-full rounded-md px-3 py-2 text-left text-sm',
                    activeTab === key
                      ? 'bg-indigo-50 text-indigo-700 dark:bg-indigo-900/30 dark:text-indigo-300'
                      : 'hover:bg-slate-50 dark:hover:bg-slate-700'
                  )}
                >
                  {label}
                </button>
              ))}
            </nav>
          </aside>

          {/* Main content */}
          <section className="lg:col-span-9">
            {activeTab === 'feed' && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="font-medium">Activity Feed</h2>
                  {sseError && (
                    <span className="text-sm text-red-600 dark:text-red-400">
                      {sseError}
                    </span>
                  )}
                </div>
                <div className="h-96 overflow-y-auto rounded-md border border-slate-200 bg-slate-50 p-3 text-sm dark:border-slate-700 dark:bg-slate-900/50">
                  {events.length === 0 ? (
                    <p className="text-slate-500 dark:text-slate-400">
                      Submit a query to see the agent activity stream.
                    </p>
                  ) : (
                    <ul className="space-y-2">
                      {events.map((payload, idx) => {
                        const event = payload.event as WorkflowEvent;
                        return (
                          <li
                            key={idx}
                            className="rounded border border-slate-200 bg-white p-2 dark:border-slate-700 dark:bg-slate-800"
                          >
                            <div className="text-xs text-slate-500 dark:text-slate-400">
                              {event.timestamp
                                ? new Date(event.timestamp).toLocaleTimeString()
                                : new Date().toLocaleTimeString()}
                            </div>
                            <div className="font-medium">{formatEvent(event)}</div>
                          </li>
                        );
                      })}
                    </ul>
                  )}
                </div>

                {pendingApproval && !approvalRequest && (
                  <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-900/20">
                    <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                      Waiting for human approval before persisting the brief.
                    </p>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={() => handleApprove(true)}
                        className="rounded bg-green-600 px-3 py-1 text-sm text-white hover:bg-green-700"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleApprove(false)}
                        className="rounded bg-red-600 px-3 py-1 text-sm text-white hover:bg-red-700"
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                )}

                {approvalRequest && (
                  <div className="mt-4 rounded-md border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-900/20">
                    <p className="text-sm font-medium text-amber-800 dark:text-amber-200">
                      Tool approval required:{' '}
                      {(approvalRequest as any).tool_name || (approvalRequest as any).tool_call?.tool_name}
                    </p>
                    <pre className="mt-2 max-h-32 overflow-auto rounded bg-white p-2 text-xs dark:bg-slate-900">
                      {JSON.stringify(
                        (approvalRequest as any).parameters ||
                          (approvalRequest as any).tool_call?.parameters || {},
                        null,
                        2
                      )}
                    </pre>
                    <div className="mt-2 flex gap-2">
                      <button
                        onClick={() => handleApprove(true)}
                        className="rounded bg-green-600 px-3 py-1 text-sm text-white hover:bg-green-700"
                      >
                        Approve
                      </button>
                      <button
                        onClick={() => handleApprove(false)}
                        className="rounded bg-red-600 px-3 py-1 text-sm text-white hover:bg-red-700"
                      >
                        Reject
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'brief' && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <h2 className="mb-4 font-medium">Research Brief</h2>
                {!report ? (
                  <p className="text-slate-500 dark:text-slate-400">
                    Run a query to generate a brief.
                  </p>
                ) : (
                  <div className="prose prose-slate max-w-none dark:prose-invert">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      components={{ a: CitationLink as any }}
                    >
                      {report.brief_markdown || report.brief?.summary || ''}
                    </ReactMarkdown>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'sources' && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <h2 className="mb-4 font-medium">Citations & Sources</h2>
                {filteredSources.length === 0 ? (
                  <p className="text-slate-500 dark:text-slate-400">No sources available.</p>
                ) : (
                  <ul className="space-y-3">
                    {filteredSources.map((source: SearchResult, idx: number) => (
                      <li
                        key={idx}
                        id={`source-${idx}`}
                        className="rounded border border-slate-200 p-3 dark:border-slate-700"
                      >
                        <a
                          href={source.url}
                          target="_blank"
                          rel="noreferrer"
                          className="font-medium text-indigo-600 hover:underline dark:text-indigo-400"
                        >
                          {source.title || source.url}
                        </a>
                        <p className="mt-1 text-sm text-slate-600 dark:text-slate-300">
                          {source.snippet || source.content}
                        </p>
                        <span
                          className={classNames(
                            'mt-2 inline-block rounded px-1.5 py-0.5 text-xs',
                            source.relevance === 'relevant'
                              ? 'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                              : source.relevance === 'irrelevant'
                                ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300'
                                : 'bg-slate-100 text-slate-700 dark:bg-slate-700 dark:text-slate-300'
                          )}
                        >
                          {source.relevance || 'unknown'}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {activeTab === 'sessions' && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <div className="mb-4 flex items-center justify-between">
                  <h2 className="font-medium">Session History</h2>
                  <button
                    onClick={refreshSessions}
                    className="rounded border border-slate-300 px-2 py-1 text-sm hover:bg-slate-50 dark:border-slate-600 dark:hover:bg-slate-700"
                  >
                    Refresh
                  </button>
                </div>
                {sessions.length === 0 ? (
                  <p className="text-slate-500 dark:text-slate-400">No sessions yet.</p>
                ) : (
                  <ul className="space-y-2">
                    {sessions.map((s) => (
                      <li
                        key={s.id}
                        className="flex items-center justify-between rounded border border-slate-200 p-3 dark:border-slate-700"
                      >
                        <div>
                          <div className="text-sm font-medium">{s.id.slice(0, 8)}</div>
                          <div className="text-xs text-slate-500 dark:text-slate-400">
                            {s.entity_id} • {s.status} •{' '}
                            {new Date(s.last_activity).toLocaleString()}
                          </div>
                        </div>
                        <button
                          onClick={() => deleteSession(s.id).then(refreshSessions)}
                          className="rounded border border-red-200 px-2 py-1 text-xs text-red-600 hover:bg-red-50 dark:border-red-900 dark:text-red-400 dark:hover:bg-red-900/20"
                        >
                          Delete
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {activeTab === 'eval' && (
              <div className="rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <h2 className="mb-4 font-medium">Eval</h2>
                <div className="flex gap-2">
                  <input
                    value={evalDataset}
                    onChange={(e) => setEvalDataset(e.target.value)}
                    className="flex-1 rounded-md border border-slate-300 px-3 py-2 text-sm dark:border-slate-600 dark:bg-slate-700"
                    placeholder="Dataset path"
                  />
                  <button
                    onClick={handleRunEval}
                    disabled={loading}
                    className="rounded-md bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-700 disabled:opacity-50"
                  >
                    Run Eval
                  </button>
                </div>
              </div>
            )}

            {usage && activeTab !== 'eval' && (
              <div className="mt-6 rounded-lg border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-800">
                <h2 className="mb-2 font-medium">Cost & Trace Inspector</h2>
                <pre className="max-h-48 overflow-auto rounded-md bg-slate-50 p-2 text-xs dark:bg-slate-900/50">
                  {JSON.stringify(usage, null, 2)}
                </pre>
              </div>
            )}
          </section>
        </div>
      </main>
    </div>
  );
}
