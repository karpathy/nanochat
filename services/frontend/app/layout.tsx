import type { Metadata, Viewport } from 'next';
import { Baloo_2, Great_Vibes, Caveat, Inter } from 'next/font/google';
import SessionBoundary from '@/components/SessionBoundary';
import './globals.css';

const baloo = Baloo_2({
  subsets: ['latin', 'devanagari'],
  weight: ['400', '600', '700', '800'],
  variable: '--font-baloo',
  display: 'swap',
});

const vibes = Great_Vibes({
  subsets: ['latin'],
  weight: ['400'],
  variable: '--font-vibes',
  display: 'swap',
});

const caveat = Caveat({
  subsets: ['latin'],
  weight: ['400', '600', '700'],
  variable: '--font-caveat',
  display: 'swap',
});

const inter = Inter({
  subsets: ['latin'],
  variable: '--font-inter',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'समोसाचाट — samosaChaat',
  description: 'Crafted with care. For India, from India. A warm, desi-flavored chat experience powered by nanochat.',
  icons: { icon: '/logo.svg' },
};

export const viewport: Viewport = {
  themeColor: '#fff8e7',
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${baloo.variable} ${vibes.variable} ${caveat.variable} ${inter.variable}`}>
      <body className="min-h-dvh bg-white text-gray-900">
        <SessionBoundary>{children}</SessionBoundary>
      </body>
    </html>
  );
}
