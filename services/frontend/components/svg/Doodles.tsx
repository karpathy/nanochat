const DOODLES = [
  { className: 'top-[18%] right-[6%] rotate-[25deg]', svg: (
      <svg viewBox="0 0 28 14" width={32} height={16}><path d="M2,7 C2,3 8,1 14,2 C20,3 26,7 26,10 C26,12 24,13 22,12 C18,10 10,5 6,6 C4,6.5 2,9 2,7Z" fill="#34A85A" opacity="0.7" /></svg>
    ) },
  { className: 'bottom-[28%] left-[3%] -rotate-[15deg]', svg: (
      <svg viewBox="0 0 18 18" width={20} height={20}><circle cx="9" cy="9" r="7" fill="#F4D03F" opacity="0.6" /><circle cx="9" cy="9" r="5" fill="#F7DC6F" opacity="0.35" /></svg>
    ) },
  { className: 'top-[45%] right-[3%] rotate-[140deg]', svg: (
      <svg viewBox="0 0 24 12" width={28} height={14}><path d="M2,6 C2,3 6,1 12,2 C18,3 22,6 22,9 C22,11 20,11 18,10 C14,8 8,4 4,5 C3,5.5 2,7.5 2,6Z" fill="#2E8B57" opacity="0.6" /></svg>
    ) },
  { className: 'bottom-[18%] left-[42%] rotate-[10deg]', svg: (
      <svg viewBox="0 0 16 16" width={16} height={16}><circle cx="8" cy="8" r="6" fill="#F4D03F" opacity="0.45" /></svg>
    ) },
  { className: 'top-[32%] left-[7%] -rotate-[40deg]', svg: (
      <svg viewBox="0 0 22 11" width={24} height={12}><path d="M2,5 C2,2 6,1 11,2 C16,3 20,5 20,8 C20,10 18,10 16,9 C12,7 8,3 4,4.5 C3,5 2,7 2,5Z" fill="#34A85A" opacity="0.55" /></svg>
    ) },
];

export default function Doodles() {
  return (
    <>
      {DOODLES.map((d, i) => (
        <div
          key={i}
          className={`fixed pointer-events-none opacity-35 z-0 transition-opacity duration-500 hidden md:block ${d.className}`}
          aria-hidden="true"
        >
          {d.svg}
        </div>
      ))}
    </>
  );
}
