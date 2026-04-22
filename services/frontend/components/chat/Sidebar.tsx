'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import {
  Plus,
  PanelLeftClose,
  PanelLeftOpen,
  LogOut,
  Trash2,
  Sun,
  Moon,
} from 'lucide-react';
import SamosaLogo from '@/components/svg/SamosaLogo';
import { useChatStore, groupConversations } from '@/store/chatStore';
import { useAuth } from '@/hooks/useAuth';
import { useTheme } from '@/hooks/useTheme';
import clsx from 'clsx';

export default function Sidebar() {
  const { user, logout } = useAuth();
  const { theme, toggle } = useTheme();
  const {
    conversations,
    currentConversationId,
    sidebarOpen,
    toggleSidebar,
    createConversation,
    selectConversation,
    deleteConversation,
    fetchConversations,
  } = useChatStore();

  useEffect(() => {
    fetchConversations();
  }, [fetchConversations]);

  const grouped = groupConversations(conversations);

  return (
    <aside
      className={clsx(
        'flex flex-col bg-cream-light dark:bg-ink-soft border-r border-cream-border dark:border-ink-border transition-all duration-300 ease-in-out overflow-hidden',
        sidebarOpen ? 'w-[260px]' : 'w-0 md:w-[56px]',
      )}
    >
      <div className="flex items-center justify-between px-3 py-3 border-b border-cream-border dark:border-ink-border">
        <Link href="/" className={clsx('flex items-center gap-2 overflow-hidden', !sidebarOpen && 'md:hidden')}>
          <SamosaLogo size={28} />
          <span className="font-display font-semibold text-base text-gray-900 dark:text-ink-text whitespace-nowrap tracking-tight">
            samosaChaat
          </span>
        </Link>
        <button
          aria-label="Toggle sidebar"
          onClick={toggleSidebar}
          className="p-1.5 rounded-md hover:bg-cream dark:hover:bg-ink-elev text-brown-light dark:text-ink-text-soft"
        >
          {sidebarOpen ? <PanelLeftClose size={18} /> : <PanelLeftOpen size={18} />}
        </button>
      </div>

      {sidebarOpen && (
        <>
          <div className="px-3 py-3">
            <button
              type="button"
              onClick={() => createConversation()}
              className="w-full flex items-center justify-center gap-2 px-3 py-2.5 rounded-full bg-gray-900 dark:bg-ink-text text-white dark:text-ink text-sm font-medium hover:-translate-y-px shadow-[0_6px_20px_rgba(0,0,0,0.18)] transition-all"
            >
              <Plus size={16} />
              New chat
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-2 nice-scrollbar">
            {conversations.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full px-4 text-center py-12">
                <div className="text-3xl mb-3 opacity-30">💬</div>
                <p className="text-sm text-gray-400 dark:text-ink-text-soft font-medium">No conversations yet.</p>
                <p className="text-xs text-gray-400 dark:text-ink-text-soft mt-1">Start your first chat!</p>
              </div>
            ) : (
              Object.entries(grouped).map(([group, items]) => {
                if (items.length === 0) return null;
                return (
                  <div key={group} className="mb-4">
                    <div className="px-2 mb-1 text-[11px] uppercase tracking-wider text-gray-400 dark:text-ink-text-soft font-medium">
                      {group}
                    </div>
                    <ul className="space-y-0.5">
                      {items.map((c) => (
                        <li key={c.id} className="group relative">
                          <button
                            type="button"
                            onClick={() => selectConversation(c.id)}
                            className={clsx(
                              'w-full text-left px-3 py-2 rounded-full text-sm truncate transition-colors pr-9',
                              c.id === currentConversationId
                                ? 'bg-cream dark:bg-ink-elev text-brown dark:text-ink-text font-medium'
                                : 'text-gray-700 dark:text-ink-text-soft hover:bg-cream/70 dark:hover:bg-ink-elev/70',
                            )}
                            title={c.title}
                          >
                            {c.title}
                          </button>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              deleteConversation(c.id);
                            }}
                            className="absolute right-1.5 top-1/2 -translate-y-1/2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-cream dark:hover:bg-ink-elev text-gray-400 hover:text-chutney-red transition-all"
                            aria-label={`Delete ${c.title}`}
                          >
                            <Trash2 size={14} />
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                );
              })
            )}
          </div>

          <div className="px-3 py-3 border-t border-cream-border dark:border-ink-border space-y-3">
            <div className="flex items-center gap-2 pt-1">
              <div className="h-9 w-9 rounded-full bg-gradient-to-br from-saffron to-gold text-white flex items-center justify-center text-sm font-semibold shadow-sm">
                {(user?.name ?? 'G')[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-800 dark:text-ink-text truncate">
                  {user?.name ?? 'Guest'}
                </div>
                <div className="text-xs text-gray-500 dark:text-ink-text-soft truncate">
                  {user?.email ?? 'Not signed in'}
                </div>
              </div>
              <button
                type="button"
                aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                onClick={toggle}
                className="p-1.5 rounded-md hover:bg-cream dark:hover:bg-ink-elev text-gray-500 dark:text-ink-text-soft hover:text-brown dark:hover:text-ink-text transition-colors"
              >
                {theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />}
              </button>
              <button
                type="button"
                aria-label="Sign out"
                onClick={logout}
                className="p-1.5 rounded-md hover:bg-cream dark:hover:bg-ink-elev text-gray-500 dark:text-ink-text-soft hover:text-brown dark:hover:text-ink-text transition-colors"
              >
                <LogOut size={16} />
              </button>
            </div>
          </div>
        </>
      )}
    </aside>
  );
}
