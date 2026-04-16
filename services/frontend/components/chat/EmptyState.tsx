'use client';

import { useAuth } from '@/hooks/useAuth';
import { BookOpen, Sparkles, Code2, Smile } from 'lucide-react';

const SUGGESTIONS = [
  { icon: BookOpen, label: 'Summarize a topic', prompt: 'Summarize the history of samosas in 3 paragraphs.' },
  { icon: Sparkles, label: 'Explain a concept', prompt: 'Explain transformers to a curious beginner.' },
  { icon: Code2,    label: 'Write some code',   prompt: 'Write a Python function that reverses a linked list.' },
  { icon: Smile,    label: 'Tell me a joke',    prompt: 'Tell me a joke about chai.' },
];

export default function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  const { user } = useAuth();
  const firstName = (user?.name ?? 'friend').split(' ')[0];

  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4 py-10 text-center">
      <h2 className="font-display font-medium text-[clamp(2.25rem,5vw,3.75rem)] leading-tight tracking-tight text-gray-900 dark:text-ink-text">
        Hello, <span className="italic text-saffron">{firstName}</span>.
      </h2>
      <p className="mt-3 max-w-xl text-base md:text-lg text-gray-600 dark:text-ink-text-soft">
        What shall we cook today — a doubt, a recipe, a code snippet, or a fresh idea?
      </p>

      {/* Pill chips — pick a starter */}
      <div className="mt-10 flex flex-wrap items-center justify-center gap-2.5 max-w-2xl">
        {SUGGESTIONS.map(({ icon: Icon, label, prompt }) => (
          <button
            key={label}
            type="button"
            onClick={() => onPick(prompt)}
            className="inline-flex items-center gap-2 px-4 py-2.5 rounded-full bg-white dark:bg-ink-soft border border-cream-border dark:border-ink-border text-sm font-medium text-gray-700 dark:text-ink-text hover:border-saffron/60 dark:hover:border-saffron/50 hover:text-gray-900 dark:hover:text-white hover:bg-cream/50 dark:hover:bg-ink-elev transition-colors shadow-sm"
          >
            <Icon size={15} className="text-saffron" />
            {label}
          </button>
        ))}
      </div>
    </div>
  );
}
