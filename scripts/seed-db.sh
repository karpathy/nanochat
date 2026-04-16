#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"

if [[ ! -f "${ENV_FILE}" ]]; then
  ENV_FILE="${ROOT_DIR}/.env.example"
fi

set -a
source "${ENV_FILE}"
set +a

docker compose exec -T postgres psql \
  -U "${POSTGRES_USER:-samosachaat_admin}" \
  -d "${POSTGRES_DB:-samosachaat}" <<'SQL'
INSERT INTO users (
    id,
    email,
    name,
    avatar_url,
    provider,
    provider_id,
    last_login_at
) VALUES (
    '11111111-1111-1111-1111-111111111111',
    'demo@samosachaat.local',
    'Demo User',
    'https://example.com/avatar.png',
    'github',
    'demo-user',
    NOW()
) ON CONFLICT (id) DO NOTHING;

INSERT INTO conversations (
    id,
    user_id,
    title,
    model_tag
) VALUES (
    '22222222-2222-2222-2222-222222222222',
    '11111111-1111-1111-1111-111111111111',
    'Welcome to samosaChaat',
    'samosachaat-d12'
) ON CONFLICT (id) DO NOTHING;

INSERT INTO messages (
    id,
    conversation_id,
    role,
    content,
    token_count,
    model_tag,
    inference_time_ms
) VALUES (
    '33333333-3333-3333-3333-333333333333',
    '22222222-2222-2222-2222-222222222222',
    'assistant',
    'The database scaffold is ready for local development.',
    9,
    'samosachaat-d12',
    12
) ON CONFLICT (id) DO NOTHING;
SQL
