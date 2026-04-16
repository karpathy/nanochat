'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import {
  getToken,
  setToken,
  clearToken,
  isAuthenticated,
  getUser,
  type TokenUser,
} from '@/lib/auth-client';

export function useAuth() {
  const [authenticated, setAuthenticated] = useState(false);
  const [loading, setLoading] = useState(true);
  const [user, setUser] = useState<TokenUser | null>(null);
  const router = useRouter();

  useEffect(() => {
    setAuthenticated(isAuthenticated());
    setUser(getUser());
    setLoading(false);
  }, []);

  const logout = () => {
    clearToken();
    setAuthenticated(false);
    setUser(null);
    router.push('/');
  };

  return { authenticated, loading, user, logout };
}

/** Hook to capture access_token from the OAuth redirect query param */
export function useTokenCapture() {
  const searchParams = useSearchParams();
  const router = useRouter();

  useEffect(() => {
    const token = searchParams.get('access_token');
    if (token) {
      setToken(token);
      // Remove token from URL for cleanliness / security
      router.replace('/chat');
    }
  }, [searchParams, router]);
}
