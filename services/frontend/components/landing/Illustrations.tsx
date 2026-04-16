import SamosaSvg from '@/components/svg/SamosaSvg';
import KettleSvg from '@/components/svg/KettleSvg';
import KettleSteam from '@/components/svg/KettleSteam';

export function SamosaIllustration() {
  return (
    <div className="fixed bottom-[5%] left-[5%] flex flex-col items-center z-[5] pointer-events-none hidden md:flex">
      <div className="animate-float">
        <SamosaSvg />
      </div>
      <span className="mt-1.5 inline-block font-caveat text-[1.1rem] text-brown-light bg-[#f5edd6] px-4 py-0.5 border border-[#d4c4a0] rounded-sm -rotate-3 shadow-sm">
        Samosa
      </span>
    </div>
  );
}

export function KettleIllustration() {
  return (
    <div className="fixed bottom-[5%] right-[5%] flex flex-col items-center z-[5] pointer-events-none hidden md:flex">
      <div className="relative animate-wobble">
        <KettleSteam />
        <KettleSvg />
      </div>
      <span className="mt-1.5 inline-block font-caveat text-[1.1rem] text-brown-light bg-[#f5edd6] px-4 py-0.5 border border-[#d4c4a0] rounded-sm rotate-2 shadow-sm">
        Chai
      </span>
    </div>
  );
}
