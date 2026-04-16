export default function LandingFooter() {
  return (
    <footer className="flex flex-col sm:flex-row justify-between items-center gap-1 px-4 md:px-9 py-3 font-caveat text-sm text-gray-400 flex-shrink-0">
      <span>&copy; 2026 samosachaat.art · Crafted with care. For India, from India.</span>
      <span className="text-xs text-gray-400">
        Built on{' '}
        <a
          href="https://github.com/karpathy/nanochat"
          target="_blank"
          rel="noopener noreferrer"
          className="text-warm-grey hover:text-gray-600"
        >
          nanochat
        </a>{' '}
        by Andrej Karpathy
      </span>
      <a href="#" className="hover:text-gray-600">Terms and Policies</a>
    </footer>
  );
}
