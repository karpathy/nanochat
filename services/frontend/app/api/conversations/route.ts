import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

const CHAT_API = process.env.CHAT_API_URL || 'http://chat-api:8002';

function getAuthHeader(req: NextRequest): string | null {
  return req.headers.get('authorization');
}

export async function GET(req: NextRequest) {
  const auth = getAuthHeader(req);
  if (!auth) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const res = await fetch(`${CHAT_API}/api/conversations`, {
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    console.error('[conversations] proxy error:', err);
    return NextResponse.json({ conversations: [] });
  }
}

export async function POST(req: NextRequest) {
  const auth = getAuthHeader(req);
  if (!auth) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const body = await req.json();
  try {
    const res = await fetch(`${CHAT_API}/api/conversations`, {
      method: 'POST',
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    return NextResponse.json(data, { status: res.status });
  } catch (err) {
    console.error('[conversations] create error:', err);
    return NextResponse.json({ error: 'Failed to create conversation' }, { status: 500 });
  }
}
