# Chat API Service

Orchestration layer for samosaChaat conversations. Manages conversation state
in PostgreSQL, authenticates every request via the auth service, and proxies
streaming inference requests via Server-Sent Events.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| GET | `/api/health` | Liveness probe (unauthenticated) |
| GET | `/api/conversations` | List the authenticated user's conversations, grouped by date |
| POST | `/api/conversations` | Create a new conversation |
| GET | `/api/conversations/{id}` | Fetch a conversation + full message history |
| PUT | `/api/conversations/{id}` | Update the conversation title |
| DELETE | `/api/conversations/{id}` | Delete a conversation (cascade deletes messages) |
| POST | `/api/conversations/{id}/messages` | Append a user message and stream the assistant response |
| POST | `/api/conversations/{id}/regenerate` | Delete the last assistant message and regenerate it |
| GET | `/api/models` | Proxy to inference `GET /models` |
| POST | `/api/models/swap` | Proxy to inference `POST /models/swap` (admin only) |

All authenticated endpoints expect `Authorization: Bearer <jwt>`. The chat API
validates the token by calling the auth service `POST /auth/validate` with the
shared `X-Internal-API-Key` header and caches the result for 5 minutes.

## Environment

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql+asyncpg://localhost/samosachaat` | PostgreSQL connection string |
| `AUTH_SERVICE_URL` | `http://auth:8001` | Base URL of the auth service |
| `INFERENCE_SERVICE_URL` | `http://inference:8000` | Base URL of the inference service |
| `INTERNAL_API_KEY` | — | Shared key for internal service auth |
| `MAX_CONVERSATION_HISTORY` | `50` | Max messages included in each inference call |
| `MAX_TOKEN_BUDGET` | `6000` | Character budget proxy for the above |
| `FRONTEND_URL` | `http://localhost:3000` | Origin allowed by CORS |
| `LOG_LEVEL` | `INFO` | Python log level |

## Running locally

```
uv pip install -e ".[dev]"
uvicorn src.main:app --reload --port 8002
```

## Running tests

```
cd services/chat-api
pytest
```

Tests use SQLite + aiosqlite for a throwaway database, respx to mock the auth
service, and hand-crafted httpx mocks for the inference SSE stream.
