'use client';

import { Sparkles, BookOpen, Code2, Smile } from 'lucide-react';

const CHIPS = [
  { icon: BookOpen, label: 'Summarize a topic', prompt: 'Summarize the history of samosas in 3 paragraphs.' },
  { icon: Sparkles, label: 'Explain a concept', prompt: 'Explain transformers to a curious beginner.' },
  { icon: Code2, label: 'Write some code', prompt: 'Write a Python function that reverses a linked list.' },
  { icon: Smile, label: 'Tell me a joke', prompt: 'Tell me a joke about chai.' },
];

export default function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex-1 flex flex-col items-center justify-center px-4 text-center">
      <h2 className="font-baloo font-bold text-3xl md:text-4xl text-gray-900 mb-2">
        How can I help you today?
      </h2>
      <p className="font-caveat text-lg text-brown-light mb-8">
        Ask anything — a doubt, a recipe, a code snippet, or a fresh idea.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
        {CHIPS.map(({ icon: Icon, label, prompt }) => (
          <button
            key={label}
            type="button"
            onClick={() => onPick(prompt)}
            className="flex items-center gap-3 px-4 py-3 rounded-xl border border-cream-border bg-cream-light hover:bg-cream text-left transition-colors"
          >
            <Icon size={18} className="text-gold flex-shrink-0" />
            <div>
              <div className="text-sm font-medium text-gray-800">{label}</div>
              <div className="text-xs text-gray-500 truncate">{prompt}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
