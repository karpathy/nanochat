'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { authHeaders } from '@/lib/auth-client';
import type { Conversation, Message, ModelOption } from '@/types/chat';

const uid = () => Math.random().toString(36).slice(2, 10);
const now = () => Date.now();

export const MODEL_OPTIONS: ModelOption[] = [
  { id: 'nanochat-base', label: 'nanochat · base', description: 'Default cloud model' },
  { id: 'nanochat-chat', label: 'nanochat · chat', description: 'Instruction-tuned' },
  { id: 'nanochat-local', label: 'nanochat · local (WebGPU)', description: 'Browser GPU (experimental)' },
];

/* ------------------------------------------------------------------ */
/*  State shape                                                       */
/* ------------------------------------------------------------------ */

interface ChatState {
  /* data */
  conversations: Conversation[];
  currentConversationId: string | null;

  /* settings (persisted in localStorage) */
  model: string;
  temperature: number;
  topK: number;
  sidebarOpen: boolean;

  /* setting mutators */
  setModel: (m: string) => void;
  setTemperature: (t: number) => void;
  setTopK: (k: number) => void;
  toggleSidebar: () => void;

  /* API-backed actions */
  fetchConversations: () => Promise<void>;
  fetchMessages: (conversationId: string) => Promise<void>;
  createConversation: (title?: string) => Promise<string | null>;
  deleteConversation: (id: string) => Promise<void>;
  selectConversation: (id: string) => void;

  /* local-only helpers (optimistic UI) */
  newConversation: () => string;
  appendMessage: (conversationId: string, message: Omit<Message, 'id' | 'createdAt'>) => string;
  updateMessage: (conversationId: string, messageId: string, content: string) => void;
  setConversationTitle: (id: string, title: string) => void;
}

/* ------------------------------------------------------------------ */
/*  Store                                                             */
/* ------------------------------------------------------------------ */

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      currentConversationId: null,
      model: 'nanochat-base',
      temperature: 0.8,
      topK: 50,
      sidebarOpen: true,

      setModel: (m) => set({ model: m }),
      setTemperature: (t) => set({ temperature: t }),
      setTopK: (k) => set({ topK: k }),
      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),

      /* ---- API actions ---- */

      fetchConversations: async () => {
        try {
          const res = await fetch('/api/conversations', {
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
          });
          if (!res.ok) return;
          const data = await res.json();
          const list: Conversation[] = (data.items ?? []).map(
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            (c: any) => ({
              id: c.id,
              title: c.title ?? 'New chat',
              messages: c.messages ?? [],
              createdAt: c.created_at ? new Date(c.created_at).getTime() : now(),
              updatedAt: c.updated_at ? new Date(c.updated_at).getTime() : now(),
            }),
          );
          set({ conversations: list });
        } catch (err) {
          console.error('[chatStore] fetchConversations error:', err);
        }
      },

      fetchMessages: async (conversationId) => {
        try {
          const res = await fetch(`/api/conversations/${conversationId}`, {
            headers: { ...authHeaders() },
          });
          if (!res.ok) return;
          const data = await res.json();
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const messages: Message[] = (data.messages ?? []).map((m: any) => ({
            id: m.id ?? uid(),
            role: m.role,
            content: m.content,
            createdAt: m.created_at ? new Date(m.created_at).getTime() : now(),
          }));
          set((s) => ({
            conversations: s.conversations.map((c) =>
              c.id === conversationId ? { ...c, messages } : c,
            ),
          }));
        } catch (err) {
          console.error('[chatStore] fetchMessages error:', err);
        }
      },

      createConversation: async (title) => {
        try {
          const res = await fetch('/api/conversations', {
            method: 'POST',
            headers: { ...authHeaders(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: title ?? 'New chat' }),
          });
          if (!res.ok) return null;
          const data = await res.json();
          const conv: Conversation = {
            id: data.id,
            title: data.title ?? title ?? 'New chat',
            messages: [],
            createdAt: data.created_at ? new Date(data.created_at).getTime() : now(),
            updatedAt: data.updated_at ? new Date(data.updated_at).getTime() : now(),
          };
          set((s) => ({
            conversations: [conv, ...s.conversations],
            currentConversationId: conv.id,
          }));
          return conv.id;
        } catch (err) {
          console.error('[chatStore] createConversation error:', err);
          return null;
        }
      },

      deleteConversation: async (id) => {
        try {
          await fetch(`/api/conversations/${id}`, {
            method: 'DELETE',
            headers: { ...authHeaders() },
          });
        } catch (err) {
          console.error('[chatStore] deleteConversation error:', err);
        }
        set((s) => {
          const rest = s.conversations.filter((c) => c.id !== id);
          return {
            conversations: rest,
            currentConversationId:
              s.currentConversationId === id ? rest[0]?.id ?? null : s.currentConversationId,
          };
        });
      },

      selectConversation: (id) => {
        set({ currentConversationId: id });
        // Fetch latest messages for the selected conversation
        get().fetchMessages(id);
      },

      /* ---- local-only helpers (for optimistic UI & mock mode) ---- */

      newConversation: () => {
        const id = uid();
        const conv: Conversation = {
          id,
          title: 'New chat',
          messages: [],
          createdAt: now(),
          updatedAt: now(),
        };
        set((s) => ({ conversations: [conv, ...s.conversations], currentConversationId: id }));
        return id;
      },

      appendMessage: (conversationId, message) => {
        const id = uid();
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: [...c.messages, { ...message, id, createdAt: now() }],
                  updatedAt: now(),
                  title:
                    c.title === 'New chat' && message.role === 'user'
                      ? message.content.slice(0, 48)
                      : c.title,
                }
              : c,
          ),
        }));
        return id;
      },

      updateMessage: (conversationId, messageId, content) =>
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: c.messages.map((m) =>
                    m.id === messageId ? { ...m, content } : m,
                  ),
                  updatedAt: now(),
                }
              : c,
          ),
        })),

      setConversationTitle: (id, title) =>
        set((s) => ({
          conversations: s.conversations.map((c) => (c.id === id ? { ...c, title } : c)),
        })),
    }),
    {
      name: 'samosachaat-settings',
      // Only persist user preferences — conversations live in the DB
      partialize: (s) => ({
        model: s.model,
        temperature: s.temperature,
        topK: s.topK,
        sidebarOpen: s.sidebarOpen,
      }),
    },
  ),
);

/* ------------------------------------------------------------------ */
/*  Grouping helper (unchanged)                                       */
/* ------------------------------------------------------------------ */

export function groupConversations(conversations: Conversation[]) {
  const day = 1000 * 60 * 60 * 24;
  const t = now();
  const startOfToday = new Date(t).setHours(0, 0, 0, 0);
  const startOfYesterday = startOfToday - day;
  const startOfWeek = startOfToday - day * 6;

  const buckets: Record<string, Conversation[]> = {
    Today: [],
    Yesterday: [],
    'Last 7 days': [],
    Older: [],
  };

  for (const c of [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)) {
    if (c.updatedAt >= startOfToday) buckets.Today.push(c);
    else if (c.updatedAt >= startOfYesterday) buckets.Yesterday.push(c);
    else if (c.updatedAt >= startOfWeek) buckets['Last 7 days'].push(c);
    else buckets.Older.push(c);
  }

  return buckets;
}
