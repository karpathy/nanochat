'use client';

import { useEffect } from 'react';
import Link from 'next/link';
import { Plus, PanelLeftClose, PanelLeftOpen, LogOut, ChevronDown, Trash2 } from 'lucide-react';
import SamosaLogo from '@/components/svg/SamosaLogo';
import { useChatStore, groupConversations, MODEL_OPTIONS } from '@/store/chatStore';
import { useAuth } from '@/hooks/useAuth';
import clsx from 'clsx';

export default function Sidebar() {
  const { user, logout } = useAuth();
  const {
    conversations,
    currentConversationId,
    sidebarOpen,
    model,
    setModel,
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
        'flex flex-col bg-cream-light border-r border-cream-border transition-all duration-300 ease-in-out overflow-hidden',
        sidebarOpen ? 'w-[260px]' : 'w-0 md:w-[56px]',
      )}
    >
      <div className="flex items-center justify-between px-3 py-3 border-b border-cream-border">
        <Link href="/" className={clsx('flex items-center gap-2 overflow-hidden', !sidebarOpen && 'md:hidden')}>
          <SamosaLogo size={28} />
          <span className="font-baloo font-bold text-base text-gray-900 whitespace-nowrap">samosaChaat</span>
        </Link>
        <button
          aria-label="Toggle sidebar"
          onClick={toggleSidebar}
          className="p-1.5 rounded hover:bg-cream text-brown-light"
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
              className="w-full flex items-center gap-2 px-3 py-2 rounded-lg border border-gold/60 bg-white hover:bg-cream text-brown font-medium text-sm transition-colors"
            >
              <Plus size={16} className="text-gold" />
              New chat
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-2 nice-scrollbar">
            {Object.entries(grouped).map(([group, items]) => {
              if (items.length === 0) return null;
              return (
                <div key={group} className="mb-4">
                  <div className="px-2 mb-1 text-[11px] uppercase tracking-wider text-gray-400 font-medium">
                    {group}
                  </div>
                  <ul className="space-y-0.5">
                    {items.map((c) => (
                      <li key={c.id} className="group relative">
                        <button
                          type="button"
                          onClick={() => selectConversation(c.id)}
                          className={clsx(
                            'w-full text-left px-2.5 py-1.5 rounded text-sm truncate transition-colors pr-8',
                            c.id === currentConversationId
                              ? 'bg-cream text-brown font-medium'
                              : 'text-gray-700 hover:bg-cream/70',
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
                          className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-cream text-gray-400 hover:text-chutney-red transition-all"
                          aria-label={`Delete ${c.title}`}
                        >
                          <Trash2 size={14} />
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              );
            })}
          </div>

          <div className="px-3 py-3 border-t border-cream-border space-y-3">
            <div>
              <label htmlFor="model-select" className="block text-[11px] uppercase tracking-wider text-gray-400 mb-1">
                Model
              </label>
              <div className="relative">
                <select
                  id="model-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full appearance-none px-3 py-2 pr-8 rounded-lg border border-cream-border bg-white text-sm text-gray-800 focus:outline-none focus:border-gold"
                >
                  {MODEL_OPTIONS.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.label}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  size={14}
                  className="absolute right-2 top-1/2 -translate-y-1/2 text-gray-400 pointer-events-none"
                />
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1">
              <div className="h-8 w-8 rounded-full bg-gold/20 text-brown flex items-center justify-center text-sm font-semibold">
                {(user?.name ?? 'G')[0].toUpperCase()}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-sm font-medium text-gray-800 truncate">
                  {user?.name ?? 'Guest'}
                </div>
                <div className="text-xs text-gray-500 truncate">
                  {user?.email ?? 'Not signed in'}
                </div>
              </div>
              <button
                type="button"
                aria-label="Sign out"
                onClick={logout}
                className="p-1.5 rounded hover:bg-cream text-gray-500 hover:text-brown"
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
