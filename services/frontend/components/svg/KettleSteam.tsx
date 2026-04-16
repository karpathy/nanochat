export default function KettleSteam({ className = '' }: { className?: string }) {
  return (
    <svg className={`absolute -top-10 -right-2 pointer-events-none ${className}`} viewBox="0 0 60 55" width={60} height={55} aria-hidden="true">
      <g style={{ transformBox: 'fill-box', transformOrigin: 'bottom center' }} className="animate-steam-1">
        <path d="M15,48 Q10,36 18,26 Q26,16 18,4" stroke="#aaa" strokeWidth="2" fill="none" strokeLinecap="round" opacity="0.4" />
      </g>
      <g style={{ transformBox: 'fill-box', transformOrigin: 'bottom center' }} className="animate-steam-2">
        <path d="M30,48 Q36,36 28,26 Q20,16 28,4" stroke="#bbb" strokeWidth="1.8" fill="none" strokeLinecap="round" opacity="0.3" />
      </g>
      <g style={{ transformBox: 'fill-box', transformOrigin: 'bottom center' }} className="animate-steam-3">
        <path d="M45,48 Q40,36 46,28 Q52,18 44,6" stroke="#aaa" strokeWidth="1.5" fill="none" strokeLinecap="round" opacity="0.3" />
      </g>
    </svg>
  );
}
