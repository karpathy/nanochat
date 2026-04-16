FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.4.30

WORKDIR /app

COPY pyproject.toml README.md /app/

RUN uv pip install --system --no-cache \
      "fastapi>=0.117.1" \
      "uvicorn[standard]>=0.36.0" \
      "pydantic>=2.8.0" \
      "pydantic-settings>=2.4.0" \
      "sqlalchemy[asyncio]>=2.0.36" \
      "asyncpg>=0.29.0" \
      "alembic>=1.13.0" \
      "authlib>=1.3.2" \
      "httpx>=0.27.0" \
      "itsdangerous>=2.2.0" \
      "pyjwt>=2.9.0" \
      "cryptography>=43.0.0" \
      "slowapi>=0.1.9" \
      "python-multipart>=0.0.9" \
      "structlog>=24.4.0" \
      "prometheus-fastapi-instrumentator>=7.0.0"

COPY src /app/src

EXPOSE 8001

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]
