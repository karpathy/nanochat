export default function KettleSvg({ className = '', width = 180, height = 165 }: { className?: string; width?: number; height?: number }) {
  return (
    <svg className={`kettle-svg ${className}`} viewBox="0 0 200 185" width={width} height={height} aria-hidden="true">
      <defs>
        <linearGradient id="kettleG" x1="20%" y1="0%" x2="80%" y2="100%">
          <stop offset="0%" stopColor="#d4a543" />
          <stop offset="45%" stopColor="#b8862a" />
          <stop offset="100%" stopColor="#8b6914" />
        </linearGradient>
        <radialGradient id="kettleHL" cx="35%" cy="38%">
          <stop offset="0%" stopColor="#e8c860" stopOpacity="0.4" />
          <stop offset="100%" stopColor="#d4a543" stopOpacity="0" />
        </radialGradient>
        <filter id="kettleSh">
          <feDropShadow dx="1" dy="3" stdDeviation="3" floodColor="#00000012" />
        </filter>
      </defs>
      <ellipse cx="95" cy="178" rx="55" ry="6" fill="#00000008" />
      <path d="M28,118 C28,155 55,175 98,175 C141,175 168,155 168,118 C168,92 152,78 142,73 L54,73 C44,78 28,92 28,118Z" fill="url(#kettleG)" filter="url(#kettleSh)" />
      <path d="M28,118 C28,155 55,175 98,175 C141,175 168,155 168,118 C168,92 152,78 142,73 L54,73 C44,78 28,92 28,118Z" fill="url(#kettleHL)" />
      <rect x="58" y="54" width="80" height="19" rx="3" fill="#a07828" />
      <path d="M54,57 Q98,42 142,57" fill="#a07828" stroke="#8b6914" strokeWidth="0.8" />
      <circle cx="98" cy="44" r="5" fill="#8b6914" stroke="#705510" strokeWidth="0.8" />
      <path d="M56,52 Q38,18 98,10 Q158,18 140,52" stroke="#8b6914" strokeWidth="4.5" fill="none" strokeLinecap="round" />
      <path d="M56,52 Q38,18 98,10 Q158,18 140,52" stroke="#c4a040" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.3" />
      <path d="M163,98 Q178,90 182,76 Q184,70 180,66" stroke="#a07828" strokeWidth="6" fill="none" strokeLinecap="round" />
      <path d="M163,98 Q178,90 182,76 Q184,70 180,66" stroke="#c4a040" strokeWidth="2" fill="none" strokeLinecap="round" opacity="0.25" />
      <ellipse cx="98" cy="112" rx="66" ry="2.5" fill="#c4a040" opacity="0.25" />
      <ellipse cx="98" cy="128" rx="64" ry="2" fill="#c4a040" opacity="0.18" />
      <path d="M58,103 Q63,98 68,103 Q73,108 78,103" stroke="#705510" strokeWidth="0.7" fill="none" opacity="0.25" />
      <path d="M118,103 Q123,98 128,103 Q133,108 138,103" stroke="#705510" strokeWidth="0.7" fill="none" opacity="0.25" />
      <path d="M55,95 Q60,110 62,130" stroke="#e8c860" strokeWidth="3" fill="none" opacity="0.2" strokeLinecap="round" />
    </svg>
  );
}
