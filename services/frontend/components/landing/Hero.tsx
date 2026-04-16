'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useAuth } from '@/hooks/useAuth';
import SamosaSvg from '@/components/svg/SamosaSvg';
import KettleSvg from '@/components/svg/KettleSvg';
import KettleSteam from '@/components/svg/KettleSteam';
import ToranSvg from '@/components/svg/ToranSvg';

export default function Hero() {
  const { authenticated } = useAuth();
  const ctaHref = authenticated ? '/chat' : '/login';

  return (
    <section className="relative overflow-hidden">
      {/* Soft saffron glow background */}
      <div aria-hidden className="absolute inset-0 bg-hero-glow pointer-events-none" />
      {/* Toran swing on top center */}
      <div className="absolute left-1/2 top-0 origin-top -translate-x-1/2 animate-pendulum z-[5]">
        <ToranSvg />
      </div>

      <div className="relative z-[2] max-w-6xl mx-auto px-4 pt-24 pb-20 md:pt-28 md:pb-28 flex flex-col items-center text-center">
        {/* Eyebrow pill */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/70 dark:bg-ink-soft/80 border border-cream-border/70 dark:border-ink-border backdrop-blur text-xs md:text-sm font-medium tracking-wide text-saffron"
        >
          <span className="w-1.5 h-1.5 rounded-full bg-saffron" />
          A small, hand-cooked AI · made in India
        </motion.div>

        {/* Hindi script — playful crown */}
        <motion.h2
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.05 }}
          className="mt-7 font-baloo font-extrabold text-[clamp(2.5rem,6vw,4.25rem)] text-gray-900 dark:text-ink-text leading-[1.05] -rotate-1"
        >
          समोसा चाट
        </motion.h2>

        {/* Big serif English headline */}
        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.15 }}
          className="font-display font-medium text-[clamp(2.75rem,7.5vw,5.75rem)] leading-[1.02] tracking-tight text-gray-900 dark:text-ink-text mt-3 max-w-4xl"
          style={{ fontVariationSettings: '"opsz" 144' }}
        >
          A chatbot with a <em className="italic font-display text-saffron">dash</em> of masala.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.3 }}
          className="mt-6 max-w-2xl text-base md:text-lg text-gray-600 dark:text-ink-text-soft leading-relaxed"
        >
          A tiny, full-stack AI cooked from scratch — trained, served and shipped from one tiny kitchen.
          Conversational, a little playful, very desi.
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.45 }}
          className="mt-10 flex flex-wrap items-center justify-center gap-3"
        >
          <Link
            href={ctaHref}
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-gray-900 dark:bg-ink-text text-white dark:text-ink font-medium shadow-[0_14px_40px_rgba(0,0,0,0.25)] hover:shadow-[0_18px_50px_rgba(0,0,0,0.3)] hover:-translate-y-px transition-all"
          >
            Start chatting
            <svg width={16} height={16} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round">
              <path d="M5 12h14" /><path d="M13 5l7 7-7 7" />
            </svg>
          </Link>
          <a
            href="#features"
            className="inline-flex items-center gap-2 px-7 py-3.5 rounded-full bg-white/80 dark:bg-ink-soft/80 border border-cream-border dark:border-ink-border text-gray-800 dark:text-ink-text font-medium hover:bg-white dark:hover:bg-ink-elev transition-colors"
          >
            See the recipe
          </a>
        </motion.div>

        {/* Caveat caption */}
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
          className="mt-6 font-caveat text-lg text-brown-light/80 dark:text-saffron-soft"
        >
          your AI, with a dash of masala
        </motion.p>

        {/* Floating illustrations */}
        <div className="hidden md:block absolute left-6 lg:left-16 top-1/2 -translate-y-1/2 animate-float pointer-events-none">
          <SamosaSvg className="w-36 h-36 lg:w-44 lg:h-44" width={176} height={176} />
          <span className="mt-1 block text-center font-caveat text-[1rem] text-brown-light bg-[#f5edd6] dark:bg-ink-elev dark:text-saffron-soft px-3 py-0.5 border border-[#d4c4a0] dark:border-ink-border rounded-sm -rotate-3 shadow-sm">
            Samosa
          </span>
        </div>
        <div className="hidden md:block absolute right-6 lg:right-16 top-1/2 -translate-y-1/2 animate-wobble pointer-events-none">
          <div className="relative">
            <KettleSteam />
            <KettleSvg className="w-32 h-32 lg:w-40 lg:h-40" width={160} height={160} />
          </div>
          <span className="mt-1 block text-center font-caveat text-[1rem] text-brown-light bg-[#f5edd6] dark:bg-ink-elev dark:text-saffron-soft px-3 py-0.5 border border-[#d4c4a0] dark:border-ink-border rounded-sm rotate-2 shadow-sm">
            Chai
          </span>
        </div>
      </div>
    </section>
  );
}
