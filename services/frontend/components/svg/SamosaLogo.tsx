export default function SamosaLogo({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 400 400" aria-hidden="true">
      <defs>
        <linearGradient id="logo-fill" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#e8a838" />
          <stop offset="100%" stopColor="#c47f17" />
        </linearGradient>
      </defs>
      <path d="M200 60 L340 320 L60 320 Z" fill="url(#logo-fill)" stroke="#a0620f" strokeWidth={6} strokeLinejoin="round" />
      <path d="M200 100 L160 220" stroke="#c47f17" strokeWidth={3} fill="none" opacity="0.5" />
      <path d="M200 100 L240 220" stroke="#c47f17" strokeWidth={3} fill="none" opacity="0.5" />
      <circle cx="170" cy="230" r="10" fill="#fff" opacity="0.85" />
      <circle cx="200" cy="230" r="10" fill="#fff" opacity="0.85" />
      <circle cx="230" cy="230" r="10" fill="#fff" opacity="0.85" />
    </svg>
  );
}
