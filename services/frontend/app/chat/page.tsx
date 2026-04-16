import Sidebar from '@/components/chat/Sidebar';
import ChatWindow from '@/components/chat/ChatWindow';

export const metadata = { title: 'Chat — samosaChaat' };

export default function ChatPage() {
  return (
    <main className="flex h-dvh overflow-hidden">
      <Sidebar />
      <ChatWindow />
    </main>
  );
}
