import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        gold: '#e8a838',
        'gold-dark': '#d4940a',
        brown: '#8b4d0a',
        'brown-light': '#5a3e1b',
        cream: '#fff8e7',
        'cream-light': '#fffdf5',
        'cream-border': '#f0e0b8',
        'chutney-green': '#2d8a4e',
        'chutney-green-light': '#3aa85e',
        'chutney-red': '#c0392b',
        'warm-grey': '#d4c4a0',
        saffron: '#ff9933',
        'saffron-soft': '#ffb063',
        // Dark mode palette (near-black, matches the reference dark UI)
        ink: '#0f0f10',
        'ink-soft': '#161618',
        'ink-elev': '#1e1e21',
        'ink-border': '#2a2a2e',
        'ink-text': '#e7e5e4',
        'ink-text-soft': '#9a9797',
      },
      fontFamily: {
        display: ['var(--font-fraunces)', 'Fraunces', 'ui-serif', 'Georgia', 'serif'],
        baloo: ['var(--font-baloo)', 'Baloo 2', 'cursive'],
        vibes: ['var(--font-vibes)', 'Great Vibes', 'cursive'],
        caveat: ['var(--font-caveat)', 'Caveat', 'cursive'],
        sans: ['var(--font-inter)', 'Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'],
        mono: ['Monaco', 'Menlo', 'Consolas', 'monospace'],
      },
      backgroundImage: {
        'hero-glow':
          'radial-gradient(60% 45% at 50% 0%, rgba(255,153,51,0.28) 0%, rgba(255,255,255,0) 60%), radial-gradient(50% 40% at 20% 20%, rgba(232,168,56,0.18) 0%, rgba(255,255,255,0) 70%), radial-gradient(55% 45% at 85% 30%, rgba(255,176,99,0.22) 0%, rgba(255,255,255,0) 70%)',
        'tile-saffron':
          'linear-gradient(180deg, #ff9933 0%, #ffb063 55%, #ffd2a0 100%)',
        'tile-gold':
          'linear-gradient(180deg, #e8a838 0%, #f3c678 60%, #fde6be 100%)',
        'tile-chutney':
          'linear-gradient(180deg, #2d8a4e 0%, #5cb17a 55%, #cfe7d6 100%)',
      },
      keyframes: {
        pendulum: {
          '0%': { transform: 'translateX(-50%) rotate(-4deg)' },
          '100%': { transform: 'translateX(-50%) rotate(4deg)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        wobble: {
          '0%, 100%': { transform: 'rotate(-1.5deg)' },
          '50%': { transform: 'rotate(1.5deg)' },
        },
        steamFloat: {
          '0%': { opacity: '0', transform: 'translateY(4px) scaleX(0.8)' },
          '35%': { opacity: '0.5' },
          '70%': { opacity: '0.35' },
          '100%': { opacity: '0', transform: 'translateY(-18px) scaleX(1.4)' },
        },
        steamType: {
          '0%, 100%': { opacity: '0.25', transform: 'scaleY(0.6)' },
          '50%': { opacity: '0.8', transform: 'scaleY(1)' },
        },
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
      },
      animation: {
        pendulum: 'pendulum 3s ease-in-out infinite alternate',
        float: 'float 2.5s ease-in-out infinite',
        wobble: 'wobble 3s ease-in-out infinite',
        'steam-1': 'steamFloat 2.8s ease-in-out infinite',
        'steam-2': 'steamFloat 2.8s ease-in-out 0.7s infinite',
        'steam-3': 'steamFloat 2.8s ease-in-out 1.4s infinite',
        'steam-type': 'steamType 1.6s ease-in-out infinite',
        'fade-in': 'fadeIn 0.35s ease-out both',
      },
    },
  },
  plugins: [],
};

export default config;
