import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

interface StreamBody {
  messages: Array<{ role: string; content: string }>;
  model?: string;
  temperature?: number;
  topK?: number;
  conversationId?: string;
}

const encoder = new TextEncoder();

function sseEvent(data: Record<string, unknown>) {
  return encoder.encode(`data: ${JSON.stringify(data)}\n\n`);
}

async function proxyUpstream(body: Record<string, unknown>, upstreamUrl: string, authHeader: string | null) {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (authHeader) headers['Authorization'] = authHeader;

  const upstream = await fetch(upstreamUrl, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
  });

  if (!upstream.ok || !upstream.body) {
    throw new Error(`upstream HTTP ${upstream.status}`);
  }

  return new Response(upstream.body, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}

function mockEcho(body: StreamBody): Response {
  const last = body.messages[body.messages.length - 1]?.content ?? '';
  const greetings = [
    'Namaste! ',
    "Here's what I can offer for that question: ",
    "Let's think about it together. ",
  ];
  const greeting = greetings[Math.floor(Math.random() * greetings.length)];
  const echo = last.trim() ? `You asked: "${last.trim()}".` : 'I am listening.';
  const full = `${greeting}${echo}\n\nThis is a mock response from the samosaChaat frontend — once the chat service is wired, real streaming tokens will land here.`;

  const stream = new ReadableStream({
    async start(controller) {
      const words = full.split(/(\s+)/);
      for (const w of words) {
        controller.enqueue(sseEvent({ token: w, gpu: 0 }));
        await new Promise((r) => setTimeout(r, 25));
      }
      controller.enqueue(sseEvent({ done: true }));
      controller.close();
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  });
}

export async function POST(req: NextRequest) {
  let body: StreamBody;
  try {
    body = (await req.json()) as StreamBody;
  } catch {
    return new Response('Invalid JSON', { status: 400 });
  }

  const upstream = process.env.CHAT_API_URL;
  const authHeader = req.headers.get('authorization');

  if (upstream) {
    try {
      // If we have a conversationId and auth, use the persisted messages endpoint
      const convId = body.conversationId;
      if (convId && authHeader) {
        // Chat-api expects {content, temperature, max_tokens, top_k}
        // Extract the last user message as the content
        const lastUserMsg = [...body.messages].reverse().find(m => m.role === 'user');
        const chatApiBody = {
          content: lastUserMsg?.content ?? '',
          temperature: body.temperature,
          max_tokens: body.maxTokens,
          top_k: body.topK,
        };
        return await proxyUpstream(
          chatApiBody as any,
          `${upstream.replace(/\/$/, '')}/api/conversations/${convId}/messages`,
          authHeader,
        );
      }
      // Fallback to direct chat completions (no persistence)
      return await proxyUpstream(
        body,
        `${upstream.replace(/\/$/, '')}/chat/completions`,
        authHeader,
      );
    } catch (err) {
      console.warn('[chat/stream] upstream failed, falling back to mock:', err);
    }
  }

  return mockEcho(body);
}
