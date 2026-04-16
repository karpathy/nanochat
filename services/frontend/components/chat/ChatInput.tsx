'use client';

import { useEffect, useRef } from 'react';
import { Send, Square } from 'lucide-react';
import clsx from 'clsx';

interface Props {
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onStop?: () => void;
  isStreaming?: boolean;
  disabled?: boolean;
}

export default function ChatInput({ value, onChange, onSubmit, onStop, isStreaming, disabled }: Props) {
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
    <div className="sticky bottom-0 bg-white pt-3 pb-[calc(1rem+env(safe-area-inset-bottom))] px-4">
      <div className="max-w-3xl mx-auto flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={ref}
            rows={1}
            placeholder="What's on your mind?"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            className="w-full resize-none px-4 py-3 pr-12 rounded-2xl border border-warm-grey bg-white text-gray-900 placeholder-[#b8a88a] focus:outline-none focus:border-gold focus:ring-2 focus:ring-gold/20 min-h-[54px] max-h-[200px] leading-relaxed text-[0.95rem]"
          />
        </div>
        {isStreaming && onStop ? (
          <button
            type="button"
            onClick={onStop}
            className="w-12 h-12 flex-shrink-0 rounded-full bg-chutney-red text-white flex items-center justify-center hover:brightness-110 transition"
            aria-label="Stop generating"
          >
            <Square size={18} fill="currentColor" />
          </button>
        ) : (
          <button
            type="button"
            onClick={onSubmit}
            disabled={!canSend}
            className={clsx(
              'w-12 h-12 flex-shrink-0 rounded-full flex items-center justify-center transition',
              canSend
                ? 'bg-gold hover:bg-gold-dark text-white'
                : 'bg-gold/30 text-white cursor-not-allowed',
            )}
            aria-label="Send message"
          >
            <Send size={18} />
          </button>
        )}
      </div>
      <p className="max-w-3xl mx-auto mt-2 text-[11px] text-gray-400 text-center">
        Tip: try <code className="text-brown">/temperature 0.7</code>, <code className="text-brown">/topk 40</code>, or <code className="text-brown">/clear</code>.
      </p>
    </div>
  );
}
