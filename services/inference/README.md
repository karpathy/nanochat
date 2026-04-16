# Inference Service

Standalone FastAPI microservice for nanochat model serving.

## Endpoints

- `POST /generate` streams model output as SSE
- `GET /models` lists registered and loaded weights
- `POST /models/swap` drains workers and hot-swaps the active weights
- `GET /health` reports readiness
- `GET /stats` reports worker pool state

## Environment

- `MODEL_STORAGE_PATH`
- `DEFAULT_MODEL_TAG`
- `HF_TOKEN`
- `INTERNAL_API_KEY`
- `NANOCHAT_DTYPE`
- `NUM_WORKERS`

Run locally with:

```bash
uv run --project services/inference uvicorn main:app --app-dir services/inference/src --reload --port 8003
```
