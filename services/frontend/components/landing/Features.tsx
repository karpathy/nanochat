'use client';

import { motion } from 'framer-motion';

type Tile = {
  title: string;
  body: string;
  caption: string;
  bg: string;
  glyph: React.ReactNode;
  rotate: string;
};

const TILES: Tile[] = [
  {
    title: 'Conversations that simmer',
    body:
      'Every chat is saved in a warm pot — come back anytime and pick up exactly where the masala was last stirred.',
    caption: 'memory · history',
    bg: 'bg-tile-saffron',
    rotate: '-rotate-1',
    glyph: (
      <svg viewBox="0 0 200 140" className="w-full h-full">
        <defs>
          <linearGradient id="potGrad" x1="0" x2="0" y1="0" y2="1">
            <stop offset="0%" stopColor="#fff" stopOpacity="0.8" />
            <stop offset="100%" stopColor="#fff" stopOpacity="0.1" />
          </linearGradient>
        </defs>
        <g fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {/* steam */}
          <path d="M85 30 Q90 20 85 12 Q80 4 90 0" opacity="0.7" />
          <path d="M105 32 Q110 22 105 14 Q100 6 110 2" opacity="0.55" />
          <path d="M125 30 Q130 20 125 12 Q120 4 130 0" opacity="0.4" />
          {/* pot */}
          <path d="M55 60 L60 122 Q60 130 70 130 L140 130 Q150 130 150 122 L155 60 Z" fill="url(#potGrad)" />
          <ellipse cx="105" cy="60" rx="55" ry="10" fill="#fff" fillOpacity="0.35" />
          <line x1="40" y1="60" x2="170" y2="60" />
          <circle cx="80" cy="55" r="3" fill="#fff" />
          <circle cx="105" cy="50" r="3" fill="#fff" />
          <circle cx="130" cy="55" r="3" fill="#fff" />
        </g>
      </svg>
    ),
  },
  {
    title: 'Swap models like spices',
    body:
      'Pinch of Base, dash of SFT — switch between chefs in a single click and taste the difference for yourself.',
    caption: 'inference · choice',
    bg: 'bg-tile-gold',
    rotate: 'rotate-1',
    glyph: (
      <svg viewBox="0 0 200 140" className="w-full h-full">
        <g fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="60" cy="70" r="28" fill="#fff" fillOpacity="0.18" />
          <circle cx="140" cy="70" r="28" fill="#fff" fillOpacity="0.30" />
          <path d="M75 56 Q100 30 125 56" />
          <path d="M120 50 L128 56 L122 64" />
          <path d="M125 84 Q100 110 75 84" />
          <path d="M80 90 L72 84 L78 76" />
          <text x="60" y="76" textAnchor="middle" fill="#fff" fontFamily="serif" fontSize="14" fontStyle="italic">d20</text>
          <text x="140" y="76" textAnchor="middle" fill="#fff" fontFamily="serif" fontSize="14" fontStyle="italic">d24</text>
        </g>
      </svg>
    ),
  },
  {
    title: 'Desi at heart',
    body:
      'Built with love in saffron, gold and brown. Speaks like a friend, jokes like a cousin, never forgets the chai.',
    caption: 'culture · craft',
    bg: 'bg-tile-chutney',
    rotate: '-rotate-1',
    glyph: (
      <svg viewBox="0 0 200 140" className="w-full h-full">
        <g fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          {/* lotus / ashoka chakra-ish */}
          <circle cx="100" cy="70" r="32" fill="#fff" fillOpacity="0.2" />
          <circle cx="100" cy="70" r="44" />
          {Array.from({ length: 12 }).map((_, i) => {
            const a = (i * Math.PI) / 6;
            const x1 = 100 + Math.cos(a) * 32;
            const y1 = 70 + Math.sin(a) * 32;
            const x2 = 100 + Math.cos(a) * 44;
            const y2 = 70 + Math.sin(a) * 44;
            return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />;
          })}
          <circle cx="100" cy="70" r="4" fill="#fff" />
        </g>
      </svg>
    ),
  },
];

export default function Features() {
  return (
    <section id="features" className="relative px-4 py-20 md:py-28">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-14">
          <p className="text-xs md:text-sm uppercase tracking-[0.18em] text-saffron font-medium">
            For the curious · the hungry · the desi
          </p>
          <h3 className="mt-3 font-display font-medium text-[clamp(2rem,4.5vw,3.25rem)] leading-tight tracking-tight text-gray-900 dark:text-ink-text">
            Why <em className="italic text-saffron">samosaChaat</em>?
          </h3>
          <p className="mt-4 max-w-2xl mx-auto text-gray-600 dark:text-ink-text-soft">
            A small chatbot with a big personality. Every flavor on this plate
            was prepared by hand — model, server, and UI.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 lg:gap-8">
          {TILES.map((t, i) => (
            <motion.article
              key={t.title}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.2 }}
              transition={{ duration: 0.5, delay: i * 0.08 }}
              className="group relative rounded-3xl bg-white dark:bg-ink-soft border border-cream-border/70 dark:border-ink-border p-3 hover:-translate-y-1 transition-transform shadow-[0_10px_40px_rgba(180,120,40,0.08)]"
            >
              {/* Gradient art tile */}
              <div className={`relative rounded-2xl ${t.bg} h-44 overflow-hidden flex items-center justify-center`}>
                <div className={`absolute inset-0 mix-blend-overlay opacity-90 ${t.rotate} transition-transform group-hover:scale-105 duration-500 p-6`}>
                  {t.glyph}
                </div>
                <span className="absolute top-3 left-3 text-[10px] uppercase tracking-[0.16em] text-white/85 bg-black/15 backdrop-blur px-2 py-0.5 rounded-full">
                  {t.caption}
                </span>
              </div>

              {/* Body */}
              <div className="px-3 pt-5 pb-4">
                <h4 className="font-display font-semibold text-[1.35rem] leading-snug text-gray-900 dark:text-ink-text tracking-tight">
                  {t.title}
                </h4>
                <p className="mt-2 text-sm leading-relaxed text-gray-600 dark:text-ink-text-soft">
                  {t.body}
                </p>
              </div>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}
