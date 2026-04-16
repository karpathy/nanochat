import Link from 'next/link';
import ToranSvg from './svg/ToranSvg';

export default function LandingNav() {
  return (
    <nav className="relative flex justify-between items-start px-4 md:px-9 pt-4 pb-2 z-10 flex-shrink-0">
      <div className="flex items-center gap-2">
        <Link href="/" aria-label="Home" className="transition-transform hover:scale-105">
          <svg viewBox="0 0 30 30" width={30} height={30} fill="none" stroke="#444" strokeWidth={1.3} strokeLinecap="round" strokeLinejoin="round">
            <path d="M4,16 L15.2,5 L26,16" />
            <path d="M7,14.5 L7,26 L23,26 L23,14.5" />
            <rect x="12" y="19" width="6" height="7" rx="0.5" />
            <rect x="8.5" y="16" width="3" height="2.8" rx="0.3" />
            <rect x="18.5" y="16" width="3" height="2.8" rx="0.3" />
          </svg>
        </Link>
        <Link
          href="/"
          className="relative font-caveat text-[1.2rem] md:text-[1.35rem] font-semibold text-gray-800 after:content-[''] after:absolute after:-bottom-0.5 after:left-0 after:w-full after:h-[1.5px] after:bg-gray-500 after:rounded after:-rotate-[0.5deg]"
        >
          samosaChaat
        </Link>
      </div>

      <div className="absolute left-1/2 top-0 origin-top transform -translate-x-1/2 animate-pendulum z-[5]">
        <ToranSvg />
      </div>

      <div className="flex items-center gap-4 font-caveat text-[1.05rem] text-gray-600 pt-1">
        <a
          href="https://instagram.com/samosachaat.art"
          target="_blank"
          rel="noopener noreferrer"
          className="hidden sm:inline hover:text-brown transition-colors"
        >
          @samosachaat
        </a>
        <Link
          href="/login"
          className="px-3 py-1 rounded-full border border-warm-grey text-brown bg-cream-light hover:bg-cream transition-colors"
        >
          Sign in
        </Link>
      </div>
    </nav>
  );
}
