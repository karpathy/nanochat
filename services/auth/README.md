# samosaChaat Auth Service

FastAPI microservice providing OAuth2 login (Google + GitHub) and JWT session
management for samosaChaat (Issue #5).

## Endpoints

| Method | Path | Purpose |
| ------ | ---- | ------- |
| GET  | `/auth/google` | Redirect to Google consent |
| GET  | `/auth/google/callback` | Complete Google flow, upsert user, issue tokens |
| GET  | `/auth/github` | Redirect to GitHub consent |
| GET  | `/auth/github/callback` | Complete GitHub flow, upsert user, issue tokens |
| POST | `/auth/refresh` | Exchange refresh cookie for new access token |
| GET  | `/auth/me` | Current user profile (Bearer JWT) |
| PUT  | `/auth/me` | Update name / avatar (Bearer JWT) |
| POST | `/auth/validate` | Internal JWT validation (service-to-service) |
| GET  | `/auth/health` | Liveness probe |

## Environment

```
DATABASE_URL=postgresql+asyncpg://user:pass@host/db
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
GITHUB_CLIENT_ID=...
GITHUB_CLIENT_SECRET=...
JWT_PRIVATE_KEY=<RS256 PEM>
JWT_PUBLIC_KEY=<RS256 PEM>
FRONTEND_URL=http://localhost:3000
INTERNAL_API_KEY=<shared secret for /auth/validate>
```

## Local development

```
uv sync
uv run uvicorn src.main:app --reload --port 8001
uv run pytest
```

Database schema is managed by Alembic at `db/migrations`:

```
DATABASE_URL=... uv run alembic -c db/alembic.ini upgrade head
```
