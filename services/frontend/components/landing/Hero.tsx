'use client';

import Link from 'next/link';
import { motion } from 'framer-motion';
import { useSession } from 'next-auth/react';

export default function Hero() {
  const { status } = useSession();
  const ctaHref = status === 'authenticated' ? '/chat' : '/login';

  return (
    <section className="relative z-[2] text-center px-4 pt-6">
      <motion.h1
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="font-baloo font-extrabold text-[clamp(3rem,7vw,5.5rem)] text-[#1a1a1a] leading-[1.1] -rotate-1 -mb-[0.35em] relative z-[2]"
      >
        समोसा चाट
      </motion.h1>
      <motion.h2
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.15 }}
        className="font-vibes text-[clamp(2rem,5vw,3.8rem)] text-[rgba(30,30,30,0.55)] leading-none rotate-[0.5deg] relative z-[1] -mt-[0.1em]"
      >
        samosaChaat
      </motion.h2>

      <motion.p
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="mt-6 font-caveat text-lg md:text-xl text-brown-light max-w-xl mx-auto"
      >
        A warm, desi-flavored chat — brewed from the nanochat research model, served with a side of chai.
      </motion.p>

      <motion.div
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.45 }}
        className="mt-6"
      >
        <Link
          href={ctaHref}
          className="inline-flex items-center gap-2 px-6 py-3 rounded-full bg-gold hover:bg-gold-dark text-white font-baloo font-semibold text-base shadow-md transition-colors"
        >
          Start Chatting
          <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
            <path d="M5 12h14" />
            <path d="M13 5l7 7-7 7" />
          </svg>
        </Link>
      </motion.div>
    </section>
  );
}
