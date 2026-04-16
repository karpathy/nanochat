const TOKEN_KEY = 'samosachaat_access_token';
const USER_KEY = 'samosachaat_user';

export interface TokenUser {
  name: string;
  email: string;
}

export function getToken(): string | null {
  if (typeof window === 'undefined') return null;
  return localStorage.getItem(TOKEN_KEY);
}

export function setToken(token: string): void {
  localStorage.setItem(TOKEN_KEY, token);
  // Decode JWT payload to persist basic user info
  try {
    const payload = JSON.parse(atob(token.split('.')[1]));
    const user: TokenUser = {
      name: payload.name || payload.email || 'User',
      email: payload.email || '',
    };
    localStorage.setItem(USER_KEY, JSON.stringify(user));
  } catch {
    /* malformed JWT — ignore */
  }
}

export function clearToken(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isAuthenticated(): boolean {
  return !!getToken();
}

export function getUser(): TokenUser | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = localStorage.getItem(USER_KEY);
    return raw ? (JSON.parse(raw) as TokenUser) : null;
  } catch {
    return null;
  }
}

export function authHeaders(): Record<string, string> {
  const token = getToken();
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}
