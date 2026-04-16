export default function Features() {
  return (
    <section className="bg-[#fff8e7]/60 py-20 px-4">
      <div className="max-w-4xl mx-auto">
        <h3 className="text-center font-baloo text-2xl text-brown mb-12">
          Why samosaChaat?
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-[#f0e0b8]/50 text-center">
            <div className="text-3xl mb-3">💬</div>
            <h4 className="font-baloo font-bold text-lg text-brown mb-2">
              Conversations that stick
            </h4>
            <p className="text-sm text-brown/70">
              Your chats are saved and organized. Pick up right where you left off.
            </p>
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-[#f0e0b8]/50 text-center">
            <div className="text-3xl mb-3">🔄</div>
            <h4 className="font-baloo font-bold text-lg text-brown mb-2">
              Swap models anytime
            </h4>
            <p className="text-sm text-brown/70">
              Switch between different AI models with a click. Your choice, your style.
            </p>
          </div>
          <div className="bg-white rounded-2xl p-6 shadow-sm border border-[#f0e0b8]/50 text-center">
            <div className="text-3xl mb-3">🇮🇳</div>
            <h4 className="font-baloo font-bold text-lg text-brown mb-2">
              Desi at heart
            </h4>
            <p className="text-sm text-brown/70">
              Built with love, inspired by Indian culture. A little desi, a lot thoughtful.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
}
