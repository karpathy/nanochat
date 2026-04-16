import LandingNav from '@/components/LandingNav';
import LandingFooter from '@/components/LandingFooter';
import Hero from '@/components/landing/Hero';
import { SamosaIllustration, KettleIllustration } from '@/components/landing/Illustrations';
import Doodles from '@/components/svg/Doodles';

export default function LandingPage() {
  return (
    <main className="relative flex min-h-dvh flex-col bg-white overflow-x-hidden">
      <LandingNav />
      <Doodles />
      <Hero />
      <SamosaIllustration />
      <KettleIllustration />
      <div className="flex-1" />
      <LandingFooter />
    </main>
  );
}
