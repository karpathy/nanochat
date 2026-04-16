'use client';

import SamosaLogo from '@/components/svg/SamosaLogo';

const SUGGESTIONS = [
  {
    icon: '📚',
    label: 'Summarize a topic',
    description: 'Get a concise overview of any subject',
    prompt: 'Summarize the history of samosas in 3 paragraphs.',
  },
  {
    icon: '✨',
    label: 'Explain a concept',
    description: 'Break down complex ideas simply',
    prompt: 'Explain transformers to a curious beginner.',
  },
  {
    icon: '💻',
    label: 'Write some code',
    description: 'Get help with any programming task',
    prompt: 'Write a Python function that reverses a linked list.',
  },
  {
    icon: '😄',
    label: 'Tell me a joke',
    description: 'Lighten the mood with some humor',
    prompt: 'Tell me a joke about chai.',
  },
];

export default function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 px-4 -mt-20">
      {/* Small logo */}
      <div className="w-16 h-16 mb-6 opacity-20">
        <SamosaLogo size={64} />
      </div>

      <h2 className="font-baloo font-bold text-3xl text-gray-800 mb-2">
        How can I help you today?
      </h2>
      <p className="font-caveat text-lg text-brown/60 mb-10">
        Ask anything — a doubt, a recipe, a code snippet, or a fresh idea.
      </p>

      {/* Bigger suggestion cards - 2x2 grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-xl">
        {SUGGESTIONS.map((s) => (
          <button
            key={s.label}
            type="button"
            onClick={() => onPick(s.prompt)}
            className="flex items-start gap-3 p-4 rounded-xl border border-cream-border bg-white hover:bg-cream/50 hover:border-gold/30 transition-all text-left group"
          >
            <span className="text-xl mt-0.5">{s.icon}</span>
            <div>
              <div className="font-medium text-sm text-gray-800 group-hover:text-brown">
                {s.label}
              </div>
              <div className="text-xs text-gray-500 mt-0.5">{s.description}</div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
