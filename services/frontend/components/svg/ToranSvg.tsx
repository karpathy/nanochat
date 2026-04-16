export default function ToranSvg({ width = 48, height = 100 }: { width?: number; height?: number }) {
  return (
    <svg viewBox="0 0 50 105" width={width} height={height} fill="none" aria-hidden="true">
      <line x1="25" y1="0" x2="25" y2="88" stroke="#555" strokeWidth="1" />
      <path d="M23,16 C18,22 13,34 14,46 C14,49 16,50 18,47 C20,43 22,28 23,20Z" fill="#2E8B57" stroke="#1a6b3a" strokeWidth="0.4" />
      <path d="M23,16 L22,11" stroke="#5a7c4f" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M27,20 C32,26 37,38 36,50 C36,53 34,54 32,51 C30,47 28,32 27,24Z" fill="#34A85A" stroke="#228B44" strokeWidth="0.4" />
      <path d="M27,20 L28,15" stroke="#5a7c4f" strokeWidth="1.4" strokeLinecap="round" />
      <path d="M23,35 C17,42 13,52 15,61 C15,64 17,64 19,62 C21,58 22,45 23,39Z" fill="#2E8B57" stroke="#1a6b3a" strokeWidth="0.4" />
      <path d="M23,35 L22,31" stroke="#5a7c4f" strokeWidth="1.3" strokeLinecap="round" />
      <path d="M27,42 C31,47 34,54 33,61 C33,63 31,63 30,61 C29,58 28,49 27,45Z" fill="#34A85A" stroke="#228B44" strokeWidth="0.4" />
      <path d="M27,42 L28,39" stroke="#5a7c4f" strokeWidth="1.2" strokeLinecap="round" />
      <ellipse cx="25" cy="82" rx="12" ry="10.5" fill="#F4D03F" />
      <ellipse cx="25" cy="82" rx="9" ry="8" fill="#F7DC6F" opacity="0.5" />
      <ellipse cx="23" cy="80" rx="5" ry="4" fill="#FAEA7A" opacity="0.35" />
      <path d="M25,71 Q26,68 25,66" stroke="#C5A828" strokeWidth="1.5" strokeLinecap="round" fill="none" />
      <circle cx="25" cy="65.5" r="1.5" fill="#8B9A46" />
    </svg>
  );
}
