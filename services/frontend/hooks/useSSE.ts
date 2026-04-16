'use client';

import { useCallback, useRef, useState } from 'react';

export interface SSEOptions {
  onToken?: (token: string, gpu?: number) => void;
  onDone?: () => void;
  onError?: (err: Error) => void;
}

export interface StreamRequest {
  messages: Array<{ role: string; content: string }>;
  model?: string;
  temperature?: number;
  topK?: number;
}

export function useSSE(endpoint: string, options: SSEOptions = {}) {
  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const start = useCallback(
    async (body: StreamRequest) => {
      stop();
      const ac = new AbortController();
      abortRef.current = ac;
      setIsStreaming(true);

      try {
        const res = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
          signal: ac.signal,
        });

        if (!res.ok || !res.body) {
          throw new Error(`HTTP ${res.status}`);
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, nl).trim();
            buffer = buffer.slice(nl + 1);
            if (!line.startsWith('data:')) continue;
            const payload = line.slice(5).trim();
            if (!payload) continue;
            try {
              const data = JSON.parse(payload);
              if (data.done) {
                options.onDone?.();
                setIsStreaming(false);
                return;
              }
              if (typeof data.token === 'string') {
                options.onToken?.(data.token, data.gpu);
              }
            } catch {
              /* swallow malformed chunks */
            }
          }
        }

        options.onDone?.();
      } catch (err) {
        if ((err as Error).name !== 'AbortError') {
          options.onError?.(err as Error);
        }
      } finally {
        setIsStreaming(false);
        abortRef.current = null;
      }
    },
    [endpoint, options, stop],
  );

  return { start, stop, isStreaming };
}
