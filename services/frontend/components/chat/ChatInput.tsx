'use client';

import { useEffect, useRef } from 'react';
import { ArrowUp, Brain, Globe, Square } from 'lucide-react';
import clsx from 'clsx';

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
  thinkingMode?: boolean;
  onToggleThinking?: () => void;
  webSearchMode?: boolean;
  onToggleWebSearch?: () => void;
}

export default function ChatInput({ value, onChange, onSubmit, onStop, isStreaming, disabled, thinkingMode, onToggleThinking, webSearchMode, onToggleWebSearch }: Props) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
  }, [value]);

  useEffect(() => {
    ref.current?.focus();
  }, []);

  const canSend = value.trim().length > 0 && !disabled && !isStreaming;

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (canSend) onSubmit();
    }
  };

  return (
    <div className="sticky bottom-0 bg-white/85 dark:bg-ink/85 backdrop-blur pt-3 pb-[calc(1rem+env(safe-area-inset-bottom))] px-4">
      <div className="max-w-3xl mx-auto">
        {/* Input pod — single rounded container with the button inside */}
        <div
          className={clsx(
            'relative flex items-end gap-2 rounded-[26px] border bg-white dark:bg-ink-soft transition-shadow',
            'border-cream-border dark:border-ink-border',
            'focus-within:border-saffron/60 dark:focus-within:border-saffron/50 focus-within:shadow-[0_8px_30px_rgba(255,153,51,0.12)]',
          )}
        >
          <textarea
            ref={ref}
            rows={1}
            placeholder="Ask anything…"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            className="flex-1 resize-none bg-transparent px-5 py-4 pr-2 text-[0.95rem] leading-relaxed text-gray-900 dark:text-ink-text placeholder-gray-400 dark:placeholder-ink-text-soft focus:outline-none min-h-[52px] max-h-[200px]"
          />

          {/* Think toggle */}
          {onToggleThinking && (
            <div className="self-end p-2">
              <button
                type="button"
                onClick={onToggleThinking}
                aria-pressed={!!thinkingMode}
                title={thinkingMode ? 'Reasoning mode ON — model will think step-by-step' : 'Enable reasoning mode'}
                className={clsx(
                  'h-10 px-3 rounded-full flex items-center gap-1.5 text-xs font-medium transition-all border',
                  thinkingMode
                    ? 'bg-saffron/15 dark:bg-saffron/20 border-saffron/40 dark:border-saffron/50 text-saffron dark:text-saffron-soft shadow-[0_4px_14px_rgba(255,153,51,0.15)]'
                    : 'bg-transparent border-cream-border dark:border-ink-border text-gray-500 dark:text-ink-text-soft hover:bg-gray-50 dark:hover:bg-ink-elev',
                )}
              >
                <Brain size={14} />
                <span>Think</span>
              </button>
            </div>
          )}

          {/* Force web-search toggle */}
          {onToggleWebSearch && (
            <div className="self-end p-2">
              <button
                type="button"
                onClick={onToggleWebSearch}
                aria-pressed={!!webSearchMode}
                title={webSearchMode ? 'Web search ON — every message will be searched online' : 'Force a web search for your next message'}
                className={clsx(
                  'h-10 px-3 rounded-full flex items-center gap-1.5 text-xs font-medium transition-all border',
                  webSearchMode
                    ? 'bg-emerald-500/15 dark:bg-emerald-500/20 border-emerald-500/50 text-emerald-600 dark:text-emerald-400 shadow-[0_4px_14px_rgba(16,185,129,0.15)]'
                    : 'bg-transparent border-cream-border dark:border-ink-border text-gray-500 dark:text-ink-text-soft hover:bg-gray-50 dark:hover:bg-ink-elev',
                )}
              >
                <Globe size={14} />
                <span>Search</span>
              </button>
            </div>
          )}

          {/* Send / stop button — vertically centered with the textarea baseline */}
          <div className="self-end p-2">
            {isStreaming && onStop ? (
              <button
                type="button"
                onClick={onStop}
                className="w-10 h-10 rounded-full bg-chutney-red text-white flex items-center justify-center hover:brightness-110 transition shadow-md"
                aria-label="Stop generating"
              >
                <Square size={14} fill="currentColor" />
              </button>
            ) : (
              <button
                type="button"
                onClick={onSubmit}
                disabled={!canSend}
                className={clsx(
                  'w-10 h-10 rounded-full flex items-center justify-center transition-all',
                  canSend
                    ? 'bg-gray-900 dark:bg-ink-text text-white dark:text-ink shadow-[0_6px_18px_rgba(0,0,0,0.2)] hover:-translate-y-px'
                    : 'bg-gray-200 dark:bg-ink-elev text-gray-400 dark:text-ink-text-soft cursor-not-allowed',
                )}
                aria-label="Send message"
              >
                <ArrowUp size={18} strokeWidth={2.4} />
              </button>
            )}
          </div>
        </div>

        <p className="mt-3 text-[11px] text-gray-400 dark:text-ink-text-soft text-center">
          Tip: try <code className="text-brown dark:text-saffron-soft">/temperature 0.7</code>,{' '}
          <code className="text-brown dark:text-saffron-soft">/topk 40</code>, or{' '}
          <code className="text-brown dark:text-saffron-soft">/clear</code>.
        </p>
      </div>
    </div>
  );
}
