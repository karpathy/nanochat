import LandingNav from '@/components/LandingNav';
import LandingFooter from '@/components/LandingFooter';
import Hero from '@/components/landing/Hero';
import Features from '@/components/landing/Features';
import Doodles from '@/components/svg/Doodles';

export default function LandingPage() {
  return (
    <main className="relative flex min-h-dvh flex-col overflow-x-hidden bg-gradient-to-br from-[#fff8e7] via-white to-[#fff8e7]">
      <Doodles />

      {/* Hero section: full viewport height with warm gradient */}
      <div className="relative min-h-dvh flex flex-col">
        <LandingNav />
        <Hero />
      </div>

      {/* Features section */}
      <Features />

      {/* Footer */}
      <LandingFooter />
    </main>
  );
}
