import { useEffect, useRef, useState } from 'react';
import { streamUrl } from '../api';
import type { StreamPayload } from '../types';

export function useSSE(sessionId: string | null) {
  const [events, setEvents] = useState<StreamPayload[]>([]);
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [completed, setCompleted] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    setEvents([]);
    setConnected(false);
    setError(null);
    setCompleted(false);

    const es = new EventSource(streamUrl(sessionId));
    esRef.current = es;

    es.onopen = () => setConnected(true);

    es.onmessage = (event) => {
      if (!event.data) return;
      try {
        const payload: StreamPayload = JSON.parse(event.data);
        setEvents((prev) => [...prev, payload]);
        const et = (payload.event as any)?.event_type;
        const t = (payload.event as any)?.type;
        if (
          et === 'workflow_completed' ||
          et === 'workflow_failed' ||
          et === 'workflow_cancelled' ||
          t === 'eval_complete' ||
          t === 'error'
        ) {
          setCompleted(true);
          es.close();
        }
      } catch (e) {
        setEvents((prev) => [
          ...prev,
          {
            session_id: sessionId,
            event: { type: 'sse_raw', raw: event.data },
          },
        ]);
      }
    };

    es.onerror = () => {
      setConnected(false);
      setError('SSE connection error or closed');
      es.close();
    };

    return () => {
      es.close();
    };
  }, [sessionId]);

  const reset = () => {
    esRef.current?.close();
    setEvents([]);
    setConnected(false);
    setError(null);
    setCompleted(false);
  };

  return { events, connected, error, completed, reset };
}
