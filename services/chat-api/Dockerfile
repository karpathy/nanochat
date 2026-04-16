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
      "httpx>=0.27.0" \
      "sse-starlette>=2.1.3" \
      "structlog>=24.4.0" \
      "cachetools>=5.5.0" \
      "prometheus-fastapi-instrumentator>=7.0.0"

COPY src /app/src

EXPOSE 8002

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8002"]
