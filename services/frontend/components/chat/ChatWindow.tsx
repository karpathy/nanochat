'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { PanelLeftOpen } from 'lucide-react';
import MessageBubble from './MessageBubble';
import EmptyState from './EmptyState';
import ChatInput from './ChatInput';
import { useChatStore } from '@/store/chatStore';
import { useAuth } from '@/hooks/useAuth';
import { authHeaders } from '@/lib/auth-client';
import { parseSlashCommand } from '@/lib/slashCommands';
import type { Message } from '@/types/chat';

export default function ChatWindow() {
  const { user } = useAuth();
  const {
    conversations,
    currentConversationId,
    model,
    temperature,
    topK,
    sidebarOpen,
    toggleSidebar,
    createConversation,
    newConversation,
    appendMessage,
    updateMessage,
    setTemperature,
    setTopK,
  } = useChatStore();

  const [draft, setDraft] = useState('');
  const [streamingMsgId, setStreamingMsgId] = useState<string | null>(null);
  const streamingBufferRef = useRef('');
  const scrollRef = useRef<HTMLDivElement>(null);

  const active = useMemo(
    () => conversations.find((c) => c.id === currentConversationId) ?? null,
    [conversations, currentConversationId],
  );

  const messages: Message[] = active?.messages ?? [];
  const isEmpty = messages.length === 0;

  const scrollToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages.length, streamingMsgId, scrollToBottom]);

  const [isStreaming, setIsStreaming] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const stop = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setIsStreaming(false);
  }, []);

  const streamFromApi = useCallback(async (convId: string, assistantMsgId: string, content: string, temp?: number, topk?: number) => {
    stop();
    const ac = new AbortController();
    abortRef.current = ac;
    setIsStreaming(true);

    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
        ...authHeaders(),
      };

      // Call chat-api directly via nginx — no Next.js proxy
      const res = await fetch(`/api/conversations/${convId}/messages`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ content, temperature: temp, max_tokens: 512, top_k: topk }),
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
              setStreamingMsgId(null);
              streamingBufferRef.current = '';
              setIsStreaming(false);
              return;
            }
            if (typeof data.token === 'string') {
              streamingBufferRef.current += data.token;
              updateMessage(convId, assistantMsgId, streamingBufferRef.current);
            }
          } catch { /* skip malformed */ }
        }
      }

      setStreamingMsgId(null);
      streamingBufferRef.current = '';
    } catch (err) {
      if ((err as Error).name !== 'AbortError') {
        console.error('[chat] stream error:', err);
        if (assistantMsgId && convId) {
          updateMessage(convId, assistantMsgId, `Error: ${(err as Error).message}`);
        }
      }
      setStreamingMsgId(null);
      streamingBufferRef.current = '';
    } finally {
      setIsStreaming(false);
      abortRef.current = null;
    }
  }, [stop, updateMessage]);

  const ensureConversation = useCallback(async () => {
    if (currentConversationId) return currentConversationId;
    // Try creating via the API first
    const apiId = await createConversation();
    if (apiId) return apiId;
    // Fallback to local-only conversation (mock mode)
    return newConversation();
  }, [currentConversationId, createConversation, newConversation]);

  const handleSend = useCallback(
    async (rawInput?: string) => {
      const text = (rawInput ?? draft).trim();
      if (!text || isStreaming) return;

      const convId = await ensureConversation();

      const slash = parseSlashCommand(text, { temperature, topK });
      if (slash.handled) {
        setDraft('');
        if (slash.setTemperature !== undefined) setTemperature(slash.setTemperature);
        if (slash.setTopK !== undefined) setTopK(slash.setTopK);
        if (slash.clear) {
          const apiId = await createConversation();
          if (!apiId) newConversation();
          return;
        }
        if (slash.consoleMessage) {
          appendMessage(convId, { role: 'console', content: slash.consoleMessage });
        }
        return;
      }

      setDraft('');
      appendMessage(convId, { role: 'user', content: text });

      const assistantId = appendMessage(convId, { role: 'assistant', content: '' });
      setStreamingMsgId(assistantId);
      streamingBufferRef.current = '';

      await streamFromApi(convId, assistantId, text, temperature, topK);
    },
    [
      draft,
      isStreaming,
      ensureConversation,
      temperature,
      topK,
      appendMessage,
      streamFromApi,
      setTemperature,
      setTopK,
      createConversation,
      newConversation,
      // streamFromApi in deps via earlier line
    ],
  );

  return (
    <section className="flex-1 flex flex-col min-w-0 bg-white">
      <header className="flex items-center justify-between px-4 md:px-6 py-3 border-b border-cream-border">
        <div className="flex items-center gap-3">
          {!sidebarOpen && (
            <button
              type="button"
              onClick={toggleSidebar}
              aria-label="Open sidebar"
              className="p-1.5 rounded hover:bg-cream text-brown-light"
            >
              <PanelLeftOpen size={18} />
            </button>
          )}
          <span className="text-xs px-2 py-0.5 rounded-full border border-warm-grey bg-cream-light text-brown">
            {model}
          </span>
        </div>
        <div className="text-sm text-gray-600 font-medium">
          {user?.name ? `Hi, ${user.name.split(' ')[0]}` : ''}
        </div>
      </header>

      <div ref={scrollRef} className="flex-1 overflow-y-auto nice-scrollbar">
        <div className="max-w-3xl mx-auto px-4 md:px-6 py-6 flex flex-col min-h-full">
          {isEmpty ? (
            <EmptyState onPick={(p) => handleSend(p)} />
          ) : (
            <div className="flex flex-col">
              {messages.map((m) => (
                <MessageBubble
                  key={m.id}
                  message={m}
                  isStreaming={streamingMsgId === m.id && isStreaming}
                />
              ))}
            </div>
          )}
        </div>
      </div>

      <ChatInput
        value={draft}
        onChange={setDraft}
        onSubmit={() => handleSend()}
        onStop={stop}
        isStreaming={isStreaming}
      />
    </section>
  );
}
