'use client';

import { Suspense } from 'react';
import { useAuth, useTokenCapture } from '@/hooks/useAuth';
import Sidebar from '@/components/chat/Sidebar';
import ChatWindow from '@/components/chat/ChatWindow';
import { redirect } from 'next/navigation';

function ChatContent() {
  useTokenCapture();
  const { authenticated, loading } = useAuth();

  if (loading) {
    return <div className="flex h-dvh items-center justify-center">Loading...</div>;
  }
  if (!authenticated) {
    redirect('/login');
    return null;
  }

  return (
    <main className="flex h-dvh overflow-hidden">
      <Sidebar />
      <ChatWindow />
    </main>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="flex h-dvh items-center justify-center">Loading...</div>}>
      <ChatContent />
    </Suspense>
  );
}
