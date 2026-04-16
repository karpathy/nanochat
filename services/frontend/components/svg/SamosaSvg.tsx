export default function SamosaSvg({ className = '', width = 200, height = 175 }: { className?: string; width?: number; height?: number }) {
  return (
    <svg className={className} viewBox="0 0 220 195" width={width} height={height} aria-hidden="true">
      <defs>
        <linearGradient id="samosaG" x1="30%" y1="0%" x2="80%" y2="100%">
          <stop offset="0%" stopColor="#edb44c" />
          <stop offset="35%" stopColor="#d4940a" />
          <stop offset="75%" stopColor="#b87a08" />
          <stop offset="100%" stopColor="#8b5e0a" />
        </linearGradient>
        <linearGradient id="samosaHighlight" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#f5d080" stopOpacity="0.5" />
          <stop offset="100%" stopColor="#d4940a" stopOpacity="0" />
        </linearGradient>
        <filter id="samosaSh">
          <feDropShadow dx="1" dy="3" stdDeviation="3" floodColor="#00000015" />
        </filter>
      </defs>
      <ellipse cx="108" cy="178" rx="58" ry="7" fill="#00000008" />
      <path d="M108,18 C88,18 38,72 28,128 C24,150 35,168 55,170 L162,170 C182,168 192,150 188,128 C178,72 128,18 108,18Z" fill="url(#samosaG)" filter="url(#samosaSh)" />
      <path d="M108,18 C88,18 38,72 28,128 C24,150 35,168 55,170 L162,170 C182,168 192,150 188,128 C178,72 128,18 108,18Z" fill="url(#samosaHighlight)" />
      <path d="M108,28 Q102,65 92,105 Q87,128 82,148" stroke="#a06808" strokeWidth="1.2" fill="none" opacity="0.35" />
      <path d="M108,28 Q114,65 124,105 Q129,128 134,148" stroke="#a06808" strokeWidth="1.2" fill="none" opacity="0.35" />
      <path d="M90,50 Q95,47 100,50 Q105,53 110,50 Q115,47 120,50 Q125,53 128,50" stroke="#8b5e0a" strokeWidth="1" fill="none" opacity="0.3" />
      <path d="M55,170 Q80,164 108,166 Q136,168 162,170" stroke="#8b5e0a" strokeWidth="1.5" fill="none" opacity="0.4" />
      <path d="M88,50 Q94,68 96,90" stroke="#f5d080" strokeWidth="2.5" fill="none" opacity="0.35" strokeLinecap="round" />
      <ellipse cx="185" cy="158" rx="14" ry="9" fill="#2d8a4e" opacity="0.65" />
      <ellipse cx="183" cy="156" rx="9" ry="5.5" fill="#3aa85e" opacity="0.45" />
      <circle cx="60" cy="175" r="2" fill="#c4940a" opacity="0.3" />
      <circle cx="165" cy="176" r="1.5" fill="#c4940a" opacity="0.25" />
      <circle cx="145" cy="180" r="1.8" fill="#b87a08" opacity="0.2" />
    </svg>
  );
}
