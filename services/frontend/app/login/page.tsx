import Link from 'next/link';
import { Suspense } from 'react';
import { redirect } from 'next/navigation';
import { auth } from '@/auth';
import MandalaArt from '@/components/login/MandalaArt';
import OAuthButtons from '@/components/login/OAuthButtons';
import SamosaLogo from '@/components/svg/SamosaLogo';

export const metadata = { title: 'Sign in — samosaChaat' };

export default async function LoginPage() {
  const session = await auth();
  if (session?.user) redirect('/chat');
  return (
    <main className="min-h-dvh grid grid-cols-1 lg:grid-cols-2 bg-white">
      <div className="hidden lg:block lg:min-h-dvh">
        <MandalaArt />
      </div>

      <div className="flex items-center justify-center px-5 py-10 md:px-10">
        <div className="w-full max-w-md">
          <Link href="/" className="flex items-center gap-2 mb-8">
            <SamosaLogo size={36} />
            <span className="font-baloo font-bold text-xl text-gray-900">samosaChaat</span>
          </Link>

          <h1 className="font-baloo font-bold text-3xl md:text-4xl text-gray-900">Login to your account</h1>
          <p className="mt-2 text-sm text-gray-500">
            Welcome back. Pick up the conversation where you left it.
          </p>

          <div className="mt-8">
            <Suspense fallback={<div className="h-24" aria-hidden="true" />}>
              <OAuthButtons />
            </Suspense>
          </div>

          <div className="flex items-center gap-3 my-6" aria-hidden="true">
            <span className="flex-1 h-px bg-gray-200" />
            <span className="text-xs uppercase tracking-wider text-gray-400">OR</span>
            <span className="flex-1 h-px bg-gray-200" />
          </div>

          <div className="space-y-3">
            <label htmlFor="email" className="block text-sm font-medium text-gray-700">
              Email (coming soon)
            </label>
            <input
              id="email"
              type="email"
              disabled
              placeholder="you@example.com"
              className="w-full px-4 py-3 rounded-lg border border-gray-200 bg-gray-50 text-gray-400 cursor-not-allowed"
            />
            <button
              disabled
              type="button"
              className="w-full py-3 rounded-lg bg-gold/40 text-white font-medium cursor-not-allowed"
            >
              Continue with email
            </button>
          </div>

          <p className="mt-6 text-sm text-gray-500 text-center">
            Don&apos;t have an account?{' '}
            <Link href="/login" className="text-chutney-green font-medium hover:underline">
              Create one
            </Link>
          </p>

          <p className="mt-10 text-xs text-gray-400 text-center">
            By continuing, you agree to our Terms and acknowledge our Privacy Policy.
          </p>
        </div>
      </div>
    </main>
  );
}
