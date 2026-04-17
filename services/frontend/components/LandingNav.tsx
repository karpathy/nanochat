'use client';

import Link from 'next/link';
import { useAuth } from '@/hooks/useAuth';

export default function LandingNav() {
  const { authenticated } = useAuth();
  const ctaHref = authenticated ? '/chat' : '/login';
  const ctaLabel = authenticated ? 'Open chat' : 'Try samosaChaat';

  return (
    <nav className="relative z-20 px-4 pt-5 flex justify-center">
      <div className="flex items-center gap-1 px-2 py-1.5 rounded-full bg-white/80 dark:bg-ink-soft/80 backdrop-blur-md border border-cream-border/70 dark:border-ink-border shadow-[0_8px_30px_rgba(180,120,40,0.08)] w-full max-w-[1100px]">
        {/* Brand */}
        <Link
          href="/"
          aria-label="samosaChaat home"
          className="flex items-center gap-2 pl-3 pr-4 py-1.5 rounded-full hover:bg-cream/60 dark:hover:bg-ink-elev transition-colors whitespace-nowrap"
        >
          <svg viewBox="0 0 28 28" width={22} height={22} fill="none" stroke="currentColor" strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round" className="text-saffron">
            <path d="M4 22 L14 5 L24 22 Z" />
            <circle cx="14" cy="17" r="1.4" fill="currentColor" />
          </svg>
          <span className="font-display text-[1.05rem] font-semibold text-gray-900 dark:text-ink-text tracking-tight">
            samosaChaat
          </span>
        </Link>

        {/* Center links */}
        <div className="hidden md:flex items-center gap-1 mx-auto text-[0.78rem] font-medium uppercase tracking-[0.08em] text-gray-600 dark:text-ink-text-soft">
          <a href="#features" className="px-3 py-2 rounded-full whitespace-nowrap hover:text-gray-900 dark:hover:text-ink-text transition-colors">Why</a>
          <a href="#how" className="px-3 py-2 rounded-full whitespace-nowrap hover:text-gray-900 dark:hover:text-ink-text transition-colors">How it works</a>
          <a
            href="https://github.com/manmohan659/nanochat"
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-2 rounded-full whitespace-nowrap hover:text-gray-900 dark:hover:text-ink-text transition-colors"
          >
            Github
          </a>
        </div>

        {/* CTAs */}
        <div className="ml-auto flex items-center gap-2">
          <a
            href="https://instagram.com/samosachaat.art"
            target="_blank"
            rel="noopener noreferrer"
            className="hidden lg:inline-flex px-4 py-2 rounded-full text-sm font-medium whitespace-nowrap text-gray-700 dark:text-ink-text-soft hover:text-gray-900 dark:hover:text-ink-text transition-colors"
          >
            @samosachaat
          </a>
          <Link
            href={ctaHref}
            className="px-5 py-2.5 rounded-full bg-gray-900 dark:bg-ink-text text-white dark:text-ink text-sm font-medium whitespace-nowrap shadow-[0_8px_24px_rgba(0,0,0,0.18)] hover:shadow-[0_10px_28px_rgba(0,0,0,0.25)] hover:-translate-y-px transition-all"
          >
            {ctaLabel}
          </Link>
        </div>
      </div>
    </nav>
  );
}
