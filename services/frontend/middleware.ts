import { NextResponse } from 'next/server';

export function middleware() {
  // Auth is checked client-side via useAuth / localStorage.
  // Middleware is intentionally a pass-through so the
  // /chat?access_token=... redirect from the auth service works.
  return NextResponse.next();
}

export const config = {
  matcher: [], // No server-side auth blocking
};
