export default function SteamTyping() {
  return (
    <div className="inline-flex items-end h-7 px-1" aria-label="assistant thinking">
      <span className="inline-block w-[3px] h-[14px] mx-[3px] rounded-sm bg-warm-grey origin-bottom animate-steam-type" />
      <span className="inline-block w-[3px] h-[18px] mx-[3px] rounded-sm bg-warm-grey origin-bottom animate-steam-type [animation-delay:0.25s]" />
      <span className="inline-block w-[3px] h-[12px] mx-[3px] rounded-sm bg-warm-grey origin-bottom animate-steam-type [animation-delay:0.5s]" />
      <span className="inline-block w-[3px] h-[16px] mx-[3px] rounded-sm bg-warm-grey origin-bottom animate-steam-type [animation-delay:0.75s]" />
    </div>
  );
}
