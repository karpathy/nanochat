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

function sanitizeModelOutput(s: string): string {
  // Strip training-artifact leaks the small model sometimes emits:
  //  - HTML bold/italic tags from <b>/<i>/<strong>/<em> that were never meant to render
  //  - Stray leading "<" (a dangling tool marker that was never closed — cosmetic only)
  //  - Leading "Answer:" / "Response:" labels that are training-data prefixes
  //  - Markdown image references whose payload is clearly placeholder text (e.g. [something image])
  //  - Duplicate paragraphs the model sometimes emits verbatim after <|output_end|>
  s = s.replace(/<\/?(?:b|i|strong|em|u|small|big|code)\s*>/gi, '');
  // stray standalone "<" on its own line
  s = s.replace(/^\s*<\s*$/gm, '');
  // leading "Answer:" / "Response:" labels at start of a line
  s = s.replace(/^\s*(?:Answer|Response|Final answer|Reply)\s*:\s*/gim, '');
  // placeholder-image markdown e.g. ![Diary entry for samosa history]
  s = s.replace(/!\[[^\]]*?\](?!\()/g, '');
  // normalize double newlines
  s = s.replace(/\n{3,}/g, '\n\n');
  return s.trim();
}

function parseSegments(raw: string): Segment[] {
  // Sanitize first — remove HTML tag leaks / stray Answer: / orphan "<"
  raw = sanitizeModelOutput(raw);
  // Then strip orphan tool markers (end without open) that the model sometimes
  // emits as loop artifacts.
  raw = stripOrphanMarkers(raw);

  const segs: Segment[] = [];
  let i = 0;
  const markers: Array<[string, string, Segment['kind']]> = [
    ['<think>', '</think>', 'think'],
    ['<|python_start|>', '<|python_end|>', 'tool_call'],
    ['<|output_start|>', '<|output_end|>', 'tool_result'],
  ];
  while (i < raw.length) {
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

  return dedupeAndClean(segs);
}

function stripOrphanMarkers(s: string): string {
  // Walk the string left-to-right. For each opening marker we encounter, keep
  // it only if its matching close exists somewhere after it. For each close
  // marker encountered without a preceding open, drop it.
  const pairs: Array<[string, string]> = [
    ['<think>', '</think>'],
    ['<|python_start|>', '<|python_end|>'],
    ['<|output_start|>', '<|output_end|>'],
  ];
  for (const [open, close] of pairs) {
    // Remove any close-tag that has no preceding open-tag
    const openPositions: number[] = [];
    let idx = 0;
    while (true) {
      const p = s.indexOf(open, idx);
      if (p === -1) break;
      openPositions.push(p);
      idx = p + open.length;
    }
    const closePositions: number[] = [];
    idx = 0;
    while (true) {
      const p = s.indexOf(close, idx);
      if (p === -1) break;
      closePositions.push(p);
      idx = p + close.length;
    }
    // drop close tags that appear before any open tag
    const firstOpen = openPositions[0] ?? Infinity;
    const orphanCloses = closePositions.filter((c) => c < firstOpen);
    if (orphanCloses.length) {
      // remove each orphan close (work in reverse so indices stay valid)
      for (const c of orphanCloses.reverse()) {
        s = s.slice(0, c) + s.slice(c + close.length);
      }
    }
  }
  return s;
}

function dedupeAndClean(segs: Segment[]): Segment[] {
  const out: Segment[] = [];
  let lastResultKey: string | null = null;
  for (const seg of segs) {
    // collapse consecutive duplicate tool_result segments (model re-emits the
    // same block as a training artifact)
    if (seg.kind === 'tool_result') {
      const key = seg.content.replace(/\s+/g, ' ').trim();
      if (key === lastResultKey) continue;
      lastResultKey = key;
    } else {
      lastResultKey = null;
    }
    // drop plain-text segments that are just leftover tool-marker fragments
    if (seg.kind === 'text') {
      const t = seg.content.replace(/<\|?(?:python|output)_(?:start|end)\|?>/g, '').trim();
      if (!t) continue;
      out.push({ kind: 'text', content: seg.content.replace(/<\|?(?:python|output)_(?:start|end)\|?>/g, '') });
      continue;
    }
    out.push(seg);
  }
  return out;
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
    <div className="my-2 w-full max-w-full overflow-hidden rounded-lg border border-saffron/30 dark:border-saffron/40 bg-saffron/5 dark:bg-saffron/10 px-3 py-2">
      <div className="flex items-center gap-2 text-xs font-medium text-saffron dark:text-saffron-soft uppercase tracking-wider">
        {icon}
        <span className="truncate">Calling {toolName}{closed ? '' : '…'}</span>
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
    if (j?.output?.answer) summary = String(j.output.answer).slice(0, 220);
    else if (j?.output?.results?.[0]?.snippet) summary = String(j.output.results[0].snippet).slice(0, 160);
    else if (j?.output?.value !== undefined) summary = `= ${j.output.value}`;
    else if (j?.error) summary = `error: ${j.error}`;
  } catch { /* partial */ }
  return (
    <div className="my-2 w-full max-w-full overflow-hidden rounded-lg border border-gray-200 dark:border-ink-border bg-white/60 dark:bg-ink-elev/60">
      <button type="button" onClick={() => setOpen(!open)} className="w-full flex items-center gap-2 px-3 py-2 text-xs text-gray-600 dark:text-ink-text-soft hover:bg-gray-50 dark:hover:bg-ink-soft/50 min-w-0">
        <span className="flex-shrink-0 inline-flex items-center gap-2">
          {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          <span className="uppercase tracking-wider">Result{closed ? '' : '…'}</span>
        </span>
        <span className="min-w-0 flex-1 truncate text-left text-gray-500 dark:text-ink-text-soft normal-case">{summary}</span>
      </button>
      {open && (
        <pre className="px-3 py-2 text-xs whitespace-pre-wrap break-all border-t border-gray-200 dark:border-ink-border max-w-full">{content}</pre>
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
