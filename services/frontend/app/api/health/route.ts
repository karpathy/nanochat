import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'ok',
    service: 'samosachaat-frontend',
    chatApiConfigured: Boolean(process.env.CHAT_API_URL),
    authServiceConfigured: Boolean(process.env.AUTH_SERVICE_URL),
  });
}
