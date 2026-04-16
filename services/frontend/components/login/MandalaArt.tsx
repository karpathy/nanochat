export default function MandalaArt() {
  return (
    <div className="relative w-full h-full overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          background:
            'linear-gradient(135deg, #ffb347 0%, #e8a838 40%, #fff8e7 100%)',
        }}
      />
      {/* Mandala overlay */}
      <svg
        className="absolute inset-0 w-full h-full opacity-80"
        viewBox="0 0 600 600"
        preserveAspectRatio="xMidYMid slice"
        aria-hidden="true"
      >
        <defs>
          <radialGradient id="mandala-glow" cx="50%" cy="50%">
            <stop offset="0%" stopColor="#fff8e7" stopOpacity="0.55" />
            <stop offset="60%" stopColor="#fff8e7" stopOpacity="0.05" />
            <stop offset="100%" stopColor="#fff8e7" stopOpacity="0" />
          </radialGradient>
        </defs>
        <circle cx="300" cy="300" r="280" fill="url(#mandala-glow)" />
        {Array.from({ length: 24 }).map((_, i) => (
          <g key={i} transform={`rotate(${(360 / 24) * i} 300 300)`}>
            <path d="M300,60 Q312,160 300,260 Q288,160 300,60 Z" fill="#fff8e7" opacity="0.18" />
            <circle cx="300" cy="90" r="6" fill="#fff" opacity="0.45" />
            <circle cx="300" cy="130" r="3" fill="#c47f17" opacity="0.55" />
          </g>
        ))}
        {[260, 210, 160, 110, 60].map((r, i) => (
          <circle
            key={r}
            cx="300"
            cy="300"
            r={r}
            stroke="#fff8e7"
            strokeWidth={i % 2 === 0 ? 1.5 : 0.8}
            strokeDasharray={i % 2 === 0 ? '4 6' : undefined}
            fill="none"
            opacity="0.55"
          />
        ))}
        <g transform="translate(300 300)">
          <path
            d="M0,-80 L32,-32 L80,0 L32,32 L0,80 L-32,32 L-80,0 L-32,-32 Z"
            fill="#8b4d0a"
            opacity="0.25"
          />
          <circle r="28" fill="#fff8e7" opacity="0.9" />
          <circle r="14" fill="#e8a838" />
        </g>
      </svg>

      {/* Large brand type overlay */}
      <div className="absolute inset-0 flex flex-col justify-end p-10 lg:p-14 text-white">
        <h2 className="font-baloo font-extrabold text-5xl lg:text-6xl leading-tight drop-shadow-sm">
          समोसा चाट
        </h2>
        <p className="font-vibes text-3xl lg:text-4xl opacity-90">samosaChaat</p>
        <p className="mt-3 font-caveat text-xl opacity-90 max-w-md">
          Chat in the language of your heart.<br />
          A little desi, a lot thoughtful.
        </p>
      </div>
    </div>
  );
}
