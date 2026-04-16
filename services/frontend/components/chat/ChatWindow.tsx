'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSession } from 'next-auth/react';
import { PanelLeftOpen } from 'lucide-react';
import MessageBubble from './MessageBubble';
import EmptyState from './EmptyState';
import ChatInput from './ChatInput';
import { useChatStore } from '@/store/chatStore';
import { useSSE } from '@/hooks/useSSE';
import { parseSlashCommand } from '@/lib/slashCommands';
import type { Message } from '@/types/chat';

export default function ChatWindow() {
  const { data: session } = useSession();
  const {
    conversations,
    currentConversationId,
    model,
    temperature,
    topK,
    sidebarOpen,
    toggleSidebar,
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

  const { start, stop, isStreaming } = useSSE('/api/chat/stream', {
    onToken: (token) => {
      streamingBufferRef.current += token;
      if (streamingMsgId && currentConversationId) {
        updateMessage(currentConversationId, streamingMsgId, streamingBufferRef.current);
      }
    },
    onDone: () => {
      setStreamingMsgId(null);
      streamingBufferRef.current = '';
    },
    onError: (err) => {
      console.error('[chat] stream error:', err);
      if (streamingMsgId && currentConversationId) {
        updateMessage(
          currentConversationId,
          streamingMsgId,
          `⚠️ Error: ${err.message}. Using mock responses requires only the frontend; cloud streaming requires CHAT_API_URL.`,
        );
      }
      setStreamingMsgId(null);
      streamingBufferRef.current = '';
    },
  });

  const ensureConversation = useCallback(() => {
    if (currentConversationId) return currentConversationId;
    return newConversation();
  }, [currentConversationId, newConversation]);

  const handleSend = useCallback(
    async (rawInput?: string) => {
      const text = (rawInput ?? draft).trim();
      if (!text || isStreaming) return;

      const convId = ensureConversation();

      const slash = parseSlashCommand(text, { temperature, topK });
      if (slash.handled) {
        setDraft('');
        if (slash.setTemperature !== undefined) setTemperature(slash.setTemperature);
        if (slash.setTopK !== undefined) setTopK(slash.setTopK);
        if (slash.clear) {
          newConversation();
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

      const history = [
        ...(useChatStore.getState().conversations.find((c) => c.id === convId)?.messages ?? []),
      ]
        .filter((m) => m.role === 'user' || m.role === 'assistant')
        .slice(0, -1)
        .map((m) => ({ role: m.role, content: m.content }));

      await start({ messages: history, model, temperature, topK });
    },
    [
      draft,
      isStreaming,
      ensureConversation,
      temperature,
      topK,
      appendMessage,
      model,
      setTemperature,
      setTopK,
      newConversation,
      start,
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
          <h1 className="font-baloo font-semibold text-lg text-gray-900">Chat Completions</h1>
          <span className="hidden sm:inline text-xs px-2 py-0.5 rounded-full border border-warm-grey bg-cream-light text-brown">
            {model}
          </span>
        </div>
        <div className="text-xs text-gray-500">
          {session?.user?.name ? `Hi, ${session.user.name.split(' ')[0]}` : ''}
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
