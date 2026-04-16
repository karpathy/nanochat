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
    <section className="flex-1 flex flex-col items-center justify-center relative px-4">
      {/* Toran animation at top center */}
      <div className="absolute left-1/2 top-0 origin-top transform -translate-x-1/2 animate-pendulum z-[5]">
        <ToranSvg />
      </div>

      <div className="flex items-center justify-center gap-8 md:gap-16 lg:gap-24 w-full max-w-6xl">
        {/* Left illustration - Samosa */}
        <div className="hidden md:block flex-shrink-0 animate-float">
          <SamosaSvg className="w-44 h-44 lg:w-56 lg:h-56" width={224} height={224} />
          <span className="mt-1.5 block text-center font-caveat text-[1.1rem] text-brown-light bg-[#f5edd6] px-4 py-0.5 border border-[#d4c4a0] rounded-sm -rotate-3 shadow-sm">
            Samosa
          </span>
        </div>

        {/* Center hero text */}
        <div className="text-center relative z-[2]">
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
            Your AI, with a dash of masala
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.45 }}
            className="mt-6"
          >
            <Link
              href={ctaHref}
              className="inline-flex items-center gap-2 px-8 py-3.5 rounded-full bg-gold hover:bg-gold-dark text-white font-baloo font-semibold text-lg shadow-lg shadow-gold/25 hover:shadow-xl hover:shadow-gold/30 transition-all"
            >
              Start Chatting
              <svg width={18} height={18} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
                <path d="M5 12h14" />
                <path d="M13 5l7 7-7 7" />
              </svg>
            </Link>
          </motion.div>
        </div>

        {/* Right illustration - Chai Kettle */}
        <div className="hidden md:block flex-shrink-0 animate-wobble">
          <div className="relative">
            <KettleSteam />
            <KettleSvg className="w-40 h-40 lg:w-48 lg:h-48" width={192} height={192} />
          </div>
          <span className="mt-1.5 block text-center font-caveat text-[1.1rem] text-brown-light bg-[#f5edd6] px-4 py-0.5 border border-[#d4c4a0] rounded-sm rotate-2 shadow-sm">
            Chai
          </span>
        </div>
      </div>
    </section>
  );
}
