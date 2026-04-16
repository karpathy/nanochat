import LandingNav from '@/components/LandingNav';
import LandingFooter from '@/components/LandingFooter';
import Hero from '@/components/landing/Hero';
import Features from '@/components/landing/Features';
import Doodles from '@/components/svg/Doodles';

export default function LandingPage() {
  return (
    <main className="relative flex min-h-dvh flex-col overflow-x-hidden bg-gradient-to-br from-[#fffaf0] via-white to-[#fff5e1] dark:from-ink dark:via-ink-soft dark:to-ink">
      <Doodles />

      {/* Hero */}
      <div className="relative flex flex-col">
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
