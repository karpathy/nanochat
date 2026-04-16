import { NextRequest, NextResponse } from 'next/server';

export const runtime = 'nodejs';

const CHAT_API = process.env.CHAT_API_URL || 'http://chat-api:8002';

export async function GET(req: NextRequest, { params }: { params: { id: string } }) {
  const auth = req.headers.get('authorization');
  if (!auth) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const res = await fetch(`${CHAT_API}/api/conversations/${params.id}`, {
      headers: { Authorization: auth },
    });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (err) {
    console.error('[conversations/:id] GET error:', err);
    return NextResponse.json({ error: 'Failed to fetch conversation' }, { status: 500 });
  }
}

export async function PUT(req: NextRequest, { params }: { params: { id: string } }) {
  const auth = req.headers.get('authorization');
  if (!auth) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  const body = await req.json();
  try {
    const res = await fetch(`${CHAT_API}/api/conversations/${params.id}`, {
      method: 'PUT',
      headers: { Authorization: auth, 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    return NextResponse.json(await res.json(), { status: res.status });
  } catch (err) {
    console.error('[conversations/:id] PUT error:', err);
    return NextResponse.json({ error: 'Failed to update conversation' }, { status: 500 });
  }
}

export async function DELETE(req: NextRequest, { params }: { params: { id: string } }) {
  const auth = req.headers.get('authorization');
  if (!auth) return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });

  try {
    const res = await fetch(`${CHAT_API}/api/conversations/${params.id}`, {
      method: 'DELETE',
      headers: { Authorization: auth },
    });
    return NextResponse.json({ ok: true }, { status: res.status });
  } catch (err) {
    console.error('[conversations/:id] DELETE error:', err);
    return NextResponse.json({ error: 'Failed to delete conversation' }, { status: 500 });
  }
}
