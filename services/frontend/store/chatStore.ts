'use client';

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type { Conversation, Message, ModelOption } from '@/types/chat';

const uid = () => Math.random().toString(36).slice(2, 10);

export const MODEL_OPTIONS: ModelOption[] = [
  { id: 'nanochat-base', label: 'nanochat · base', description: 'Default cloud model' },
  { id: 'nanochat-chat', label: 'nanochat · chat', description: 'Instruction-tuned' },
  { id: 'nanochat-local', label: 'nanochat · local (WebGPU)', description: 'Browser GPU (experimental)' },
];

interface ChatState {
  conversations: Conversation[];
  currentConversationId: string | null;
  model: string;
  temperature: number;
  topK: number;
  sidebarOpen: boolean;

  setModel: (m: string) => void;
  setTemperature: (t: number) => void;
  setTopK: (k: number) => void;
  toggleSidebar: () => void;

  newConversation: () => string;
  selectConversation: (id: string) => void;
  deleteConversation: (id: string) => void;
  appendMessage: (conversationId: string, message: Omit<Message, 'id' | 'createdAt'>) => string;
  updateMessage: (conversationId: string, messageId: string, content: string) => void;
  setConversationTitle: (id: string, title: string) => void;
  hydrateMockConversations: () => void;
}

const now = () => Date.now();

const MOCK_CONVERSATIONS: Conversation[] = [
  {
    id: 'mock-1',
    title: 'Why is samosa triangular?',
    messages: [
      { id: 'm1', role: 'user', content: 'Why is samosa triangular?', createdAt: now() - 1000 * 60 * 30 },
      { id: 'm2', role: 'assistant', content: 'The triangular shape likely evolved for easy frying and portability — three edges crisp evenly and the pocket holds filling tightly.', createdAt: now() - 1000 * 60 * 29 },
    ],
    createdAt: now() - 1000 * 60 * 30,
    updatedAt: now() - 1000 * 60 * 29,
  },
  {
    id: 'mock-2',
    title: 'Chai masala recipe',
    messages: [
      { id: 'm1', role: 'user', content: 'Classic masala chai recipe please', createdAt: now() - 1000 * 60 * 60 * 26 },
    ],
    createdAt: now() - 1000 * 60 * 60 * 26,
    updatedAt: now() - 1000 * 60 * 60 * 26,
  },
  {
    id: 'mock-3',
    title: 'Explain transformers simply',
    messages: [],
    createdAt: now() - 1000 * 60 * 60 * 24 * 4,
    updatedAt: now() - 1000 * 60 * 60 * 24 * 4,
  },
  {
    id: 'mock-4',
    title: 'Monsoon pakora tips',
    messages: [],
    createdAt: now() - 1000 * 60 * 60 * 24 * 15,
    updatedAt: now() - 1000 * 60 * 60 * 24 * 15,
  },
];

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

      selectConversation: (id) => set({ currentConversationId: id }),

      deleteConversation: (id) =>
        set((s) => {
          const rest = s.conversations.filter((c) => c.id !== id);
          return {
            conversations: rest,
            currentConversationId: s.currentConversationId === id ? rest[0]?.id ?? null : s.currentConversationId,
          };
        }),

      appendMessage: (conversationId, message) => {
        const id = uid();
        set((s) => ({
          conversations: s.conversations.map((c) =>
            c.id === conversationId
              ? {
                  ...c,
                  messages: [...c.messages, { ...message, id, createdAt: now() }],
                  updatedAt: now(),
                  title: c.title === 'New chat' && message.role === 'user' ? message.content.slice(0, 48) : c.title,
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
                  messages: c.messages.map((m) => (m.id === messageId ? { ...m, content } : m)),
                  updatedAt: now(),
                }
              : c,
          ),
        })),

      setConversationTitle: (id, title) =>
        set((s) => ({
          conversations: s.conversations.map((c) => (c.id === id ? { ...c, title } : c)),
        })),

      hydrateMockConversations: () => {
        const { conversations } = get();
        if (conversations.length === 0) {
          set({ conversations: MOCK_CONVERSATIONS, currentConversationId: MOCK_CONVERSATIONS[0].id });
        }
      },
    }),
    {
      name: 'samosachaat-chat',
      partialize: (s) => ({
        conversations: s.conversations,
        currentConversationId: s.currentConversationId,
        model: s.model,
        temperature: s.temperature,
        topK: s.topK,
      }),
    },
  ),
);

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
