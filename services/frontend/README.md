# samosaChaat — Frontend Service

Next.js 14 (App Router) application that replaces the legacy `nanochat/ui.html`. It blends Sarvam.ai's clean split-screen layout with samosaChaat's warm desi personality — samosa and chai illustrations, lemon-mirchi toran, gold + cream palette.

## Pages

- `/` Landing — Devanagari + Great Vibes calligraphy, animated samosa (float), chai kettle (wobble + steam), toran (pendulum), ambient doodles, CTA.
- `/login` Sign-in — left: architectural/mandala motif with saffron-cream gradient; right: Google + GitHub OAuth, disabled email input.
- `/chat` Chat — collapsible 260px sidebar (grouped history, model selector, user avatar, logout), main area with empty state + suggestion chips, markdown rendering with code-copy, auto-expanding textarea, slash commands (`/temperature`, `/topk`, `/clear`, `/help`), SSE streaming with steam typing indicator.

## Tech

- Next.js 14 (App Router, standalone output)
- Tailwind CSS (theme tokens carried from `ui.html`)
- Zustand (persisted conversations/settings)
- NextAuth.js v5 (Google + GitHub)
- Framer Motion (hero transitions)
- Lucide React (icons)
- react-markdown + rehype-highlight (assistant rendering)

## Environment

Copy `.env.example` → `.env.local` and fill in:

| Var | Purpose |
| --- | --- |
| `NEXT_PUBLIC_APP_URL` | Public URL (defaults to `http://localhost:3000`) |
| `AUTH_SERVICE_URL` | Auth microservice (BFF only; reserved for future) |
| `CHAT_API_URL` | Upstream chat service. **If unset the frontend serves mock echo responses** — perfect for local dev. |
| `NEXTAUTH_SECRET` | `openssl rand -base64 32` |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Google OAuth (optional in dev) |
| `GITHUB_CLIENT_ID` / `GITHUB_CLIENT_SECRET` | GitHub OAuth (optional in dev) |

Middleware protects `/chat/*` — unauthenticated users are redirected to `/login?callbackUrl=/chat`.

## Run locally

```bash
cd services/frontend
npm install
cp .env.example .env.local    # then edit
npm run dev
# → http://localhost:3000
```

Without any OAuth keys the `/login` buttons will show the NextAuth error page; the rest of the UI works via the mocked API routes.

## API routes (BFF pattern)

All client calls hit Next.js routes — the frontend never talks to the chat backend directly.

- `GET  /api/health` — service info + upstream config flags
- `GET  /api/conversations` — mocked conversation list
- `POST /api/chat/stream` — proxies to `CHAT_API_URL/chat/completions` if set, otherwise streams a mock echo. Emits SSE: `data: {"token": "...", "gpu": 0}\n\n` and terminates with `data: {"done": true}\n\n`.

## Docker

```bash
docker build -t samosachaat-frontend .
docker run --rm -p 3000:3000 \
  -e NEXTAUTH_SECRET=... \
  -e CHAT_API_URL=http://host.docker.internal:8000 \
  samosachaat-frontend
```

The image is based on `node:20-alpine`, uses Next.js `output: standalone`, and runs as a non-root user.

## Porting notes

All SVG assets (samosa, chai kettle, toran, steam, doodles, logo) are componentized under `components/svg/`. CSS keyframes (`pendulum`, `float`, `wobble`, `steamFloat`, `steamType`) moved into `tailwind.config.ts` — reference via the `animate-*` utilities. The original single-file UI lives at `nanochat/ui.html` for reference.
