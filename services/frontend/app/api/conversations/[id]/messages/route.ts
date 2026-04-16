import { NextRequest } from 'next/server';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';

const CHAT_API = process.env.CHAT_API_URL || 'http://chat-api:8002';

export async function POST(req: NextRequest, { params }: { params: { id: string } }) {
  const auth = req.headers.get('authorization');
  if (!auth) return new Response('Unauthorized', { status: 401 });

  const body = await req.json();

  try {
    const res = await fetch(`${CHAT_API}/api/conversations/${params.id}/messages`, {
      method: 'POST',
      headers: {
        Authorization: auth,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    });

    if (!res.ok || !res.body) {
      return new Response(`Backend error: ${res.status}`, { status: res.status });
    }

    // Stream SSE through
    return new Response(res.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        Connection: 'keep-alive',
      },
    });
  } catch (err) {
    console.error('[conversations/:id/messages] POST error:', err);
    return new Response('Internal Server Error', { status: 500 });
  }
}
