export type Role = 'user' | 'assistant' | 'system' | 'console';

export interface Message {
  id: string;
  role: Role;
  content: string;
  createdAt: number;
}

export interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: number;
  updatedAt: number;
}

export interface ModelOption {
  id: string;
  label: string;
  description: string;
}

export type ConversationGroup = 'Today' | 'Yesterday' | 'Last 7 days' | 'Older';
