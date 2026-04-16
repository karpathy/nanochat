import type { Metadata, Viewport } from 'next';
import { Baloo_2, Great_Vibes, Caveat, Inter, Fraunces } from 'next/font/google';
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

const fraunces = Fraunces({
  subsets: ['latin'],
  weight: ['400', '500', '600', '700'],
  variable: '--font-fraunces',
  display: 'swap',
});

export const metadata: Metadata = {
  title: 'समोसाचाट — samosaChaat',
  description:
    'Crafted with care. For India, from India. A warm, desi-flavored chat experience powered by nanochat.',
  icons: { icon: '/logo.svg' },
};

export const viewport: Viewport = {
  themeColor: '#fff8e7',
  width: 'device-width',
  initialScale: 1,
  viewportFit: 'cover',
};

// Set theme class before paint to avoid flash
const themeInitScript = `
(function(){try{
  var t=localStorage.getItem('theme');
  if(t==='dark'){document.documentElement.classList.add('dark');}
  else if(t==='light'){document.documentElement.classList.remove('dark');}
}catch(e){}})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html
      lang="en"
      className={`${baloo.variable} ${vibes.variable} ${caveat.variable} ${inter.variable} ${fraunces.variable}`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body className="min-h-dvh bg-white text-gray-900 dark:bg-ink dark:text-ink-text">
        {children}
      </body>
    </html>
  );
}
