'use client';

import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github-dark.css';
import { Check, ChevronDown, ChevronRight, Copy, Search, Calculator, Sparkles } from 'lucide-react';
import clsx from 'clsx';
import type { Message } from '@/types/chat';
import SteamTyping from '@/components/svg/SteamTyping';

// ---- Content parser: split into text / think / tool_call / tool_result segments ----
type Segment =
  | { kind: 'text'; content: string }
  | { kind: 'think'; content: string; closed: boolean }
  | { kind: 'tool_call'; content: string; closed: boolean }
  | { kind: 'tool_result'; content: string; closed: boolean };

function parseSegments(raw: string): Segment[] {
  const segs: Segment[] = [];
  let i = 0;
  // marker -> [openTag, closeTag, kind]
  const markers: Array<[string, string, Segment['kind']]> = [
    ['<think>', '</think>', 'think'],
    ['<|python_start|>', '<|python_end|>', 'tool_call'],
    ['<|output_start|>', '<|output_end|>', 'tool_result'],
  ];
  while (i < raw.length) {
    // find the nearest opening marker
    let bestOpen = -1;
    let bestMarker: [string, string, Segment['kind']] | null = null;
    for (const m of markers) {
      const p = raw.indexOf(m[0], i);
      if (p !== -1 && (bestOpen === -1 || p < bestOpen)) { bestOpen = p; bestMarker = m; }
    }
    if (bestOpen === -1) {
      if (i < raw.length) segs.push({ kind: 'text', content: raw.slice(i) });
      break;
    }
    if (bestOpen > i) segs.push({ kind: 'text', content: raw.slice(i, bestOpen) });
    const [openTag, closeTag, kind] = bestMarker!;
    const afterOpen = bestOpen + openTag.length;
    const closeIdx = raw.indexOf(closeTag, afterOpen);
    if (closeIdx === -1) {
      segs.push({ kind, content: raw.slice(afterOpen), closed: false });
      i = raw.length;
    } else {
      segs.push({ kind, content: raw.slice(afterOpen, closeIdx), closed: true });
      i = closeIdx + closeTag.length;
    }
  }
  return segs;
}

function ThinkBlock({ content, closed }: { content: string; closed: boolean }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="my-3 rounded-lg border border-gray-200 dark:border-ink-border bg-gray-50/60 dark:bg-ink-soft/60">
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 px-3 py-2 text-xs uppercase tracking-wider text-gray-500 dark:text-ink-text-soft hover:bg-gray-100 dark:hover:bg-ink-elev/50">
        {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
        <Sparkles size={12} />
        <span>Thinking{closed ? '' : '…'}</span>
      </button>
      {open && (
        <div className="px-4 py-3 text-sm text-gray-600 dark:text-ink-text-soft whitespace-pre-wrap italic leading-relaxed border-t border-gray-200 dark:border-ink-border">
          {content}
        </div>
      )}
    </div>
  );
}

function ToolCallBlock({ content, closed }: { content: string; closed: boolean }) {
  let parsed: { tool?: string; arguments?: Record<string, unknown> } | null = null;
  try { parsed = JSON.parse(content); } catch { /* streaming — partial JSON */ }
  const toolName = parsed?.tool ?? 'tool';
  const icon = toolName === 'web_search' ? <Search size={12} /> : toolName === 'calculator' ? <Calculator size={12} /> : <Sparkles size={12} />;
  const query = parsed?.arguments ? JSON.stringify(parsed.arguments) : content;
  return (
    <div className="my-2 rounded-lg border border-saffron/30 dark:border-saffron/40 bg-saffron/5 dark:bg-saffron/10 px-3 py-2">
      <div className="flex items-center gap-2 text-xs font-medium text-saffron dark:text-saffron-soft uppercase tracking-wider">
        {icon}
        <span>Calling {toolName}{closed ? '' : '…'}</span>
      </div>
      <div className="mt-1 text-xs font-mono text-gray-600 dark:text-ink-text-soft truncate">{query}</div>
    </div>
  );
}

function ToolResultBlock({ content, closed }: { content: string; closed: boolean }) {
  const [open, setOpen] = useState(false);
  let summary = content;
  try {
    const j = JSON.parse(content);
    if (j?.output?.results?.[0]?.snippet) summary = String(j.output.results[0].snippet).slice(0, 160);
    else if (j?.output?.value !== undefined) summary = `= ${j.output.value}`;
    else if (j?.error) summary = `error: ${j.error}`;
  } catch { /* partial */ }
  return (
    <div className="my-2 rounded-lg border border-gray-200 dark:border-ink-border bg-white/60 dark:bg-ink-elev/60">
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-center justify-between gap-2 px-3 py-2 text-xs text-gray-600 dark:text-ink-text-soft hover:bg-gray-50 dark:hover:bg-ink-soft/50">
        <span className="flex items-center gap-2">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="uppercase tracking-wider">Result{closed ? '' : '…'}</span>
          <span className="ml-2 truncate text-gray-500 dark:text-ink-text-soft normal-case">{summary}</span>
        </span>
      </button>
      {open && (
        <pre className="px-3 py-2 text-xs overflow-x-auto border-t border-gray-200 dark:border-ink-border">{content}</pre>
      )}
    </div>
  );
}

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
        <div className="font-mono text-sm bg-cream-light dark:bg-ink-soft border border-cream-border dark:border-ink-border text-brown-light dark:text-ink-text-soft px-4 py-3 rounded-xl max-w-[80%]">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className={clsx('flex mb-5 animate-fade-in', isUser ? 'justify-end' : 'justify-start')}>
      <div
        className={clsx(
          isUser
            ? 'max-w-[85%] md:max-w-[75%] bg-cream dark:bg-ink-elev border border-cream-border dark:border-ink-border rounded-[1.25rem] px-4 py-3'
            : 'w-full bg-transparent px-1 py-1',
        )}
      >
        {!isUser && isStreaming && message.content.length === 0 ? (
          <SteamTyping />
        ) : isUser ? (
          <div className="whitespace-pre-wrap leading-relaxed text-[0.95rem] text-gray-900 dark:text-ink-text">
            {message.content}
          </div>
        ) : (
          <div className="markdown-body text-[0.95rem] text-gray-900 dark:text-ink-text leading-relaxed">
            {parseSegments(message.content).map((seg, idx) => {
              if (seg.kind === 'think') return <ThinkBlock key={idx} content={seg.content} closed={seg.closed} />;
              if (seg.kind === 'tool_call') return <ToolCallBlock key={idx} content={seg.content} closed={seg.closed} />;
              if (seg.kind === 'tool_result') return <ToolResultBlock key={idx} content={seg.content} closed={seg.closed} />;
              return (
                <ReactMarkdown
                  key={idx}
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{ code: CodeBlock as never }}
                >
                  {seg.content}
                </ReactMarkdown>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
