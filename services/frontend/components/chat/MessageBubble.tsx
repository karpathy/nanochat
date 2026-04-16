'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import { Check, Copy } from 'lucide-react';
import clsx from 'clsx';
import type { Message } from '@/types/chat';
import SteamTyping from '@/components/svg/SteamTyping';

interface Props {
  message: Message;
  isStreaming?: boolean;
}

function CodeBlock({ inline, className, children, ...props }: {
  inline?: boolean;
  className?: string;
  children?: React.ReactNode;
} & React.HTMLAttributes<HTMLElement>) {
  const [copied, setCopied] = useState(false);
  const content = String(children ?? '').replace(/\n$/, '');

  if (inline) {
    return (
      <code className={className} {...props}>
        {children}
      </code>
    );
  }

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      /* ignore */
    }
  };

  return (
    <div className="relative group">
      <button
        type="button"
        onClick={copy}
        aria-label="Copy code"
        className="absolute top-2 right-2 p-1.5 rounded bg-slate-700/70 text-slate-100 opacity-0 group-hover:opacity-100 hover:bg-slate-600 transition-opacity"
      >
        {copied ? <Check size={14} /> : <Copy size={14} />}
      </button>
      <pre>
        <code className={className} {...props}>
          {children}
        </code>
      </pre>
    </div>
  );
}

export default function MessageBubble({ message, isStreaming }: Props) {
  const isUser = message.role === 'user';
  const isConsole = message.role === 'console';

  if (isConsole) {
    return (
      <div className="flex justify-start mb-2 animate-fade-in">
        <div className="font-mono text-sm bg-cream-light border border-cream-border text-brown-light px-4 py-3 rounded-xl max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('flex mb-3 animate-fade-in', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={clsx(
          'max-w-[85%] md:max-w-[75%]',
          isUser
            ? 'bg-cream border border-cream-border rounded-[1.25rem] px-4 py-3'
            : 'bg-white px-2 py-1',
        )}
      >
        {!isUser && isStreaming && message.content.length === 0 ? (
          <SteamTyping />
        ) : isUser ? (
          <div className="whitespace-pre-wrap leading-relaxed text-[0.95rem] text-gray-900">
            {message.content}
          </div>
        ) : (
          <div className="markdown-body text-[0.95rem] text-gray-900 leading-relaxed">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
              components={{ code: CodeBlock as never }}
            >
              {message.content}
            </ReactMarkdown>
          </div>
        )}
      </div>
    </div>
  );
}
