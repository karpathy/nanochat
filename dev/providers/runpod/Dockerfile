FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404

# ── System packages ──────────────────────────────────────────────────────────
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git nano curl ca-certificates screen && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Python dependencies (into system Python, torch already in base image) ───
RUN pip install --no-cache-dir \
    "datasets>=4.0.0" \
    "fastapi>=0.117.1" \
    "kernels>=0.11.7" \
    "psutil>=7.1.0" \
    "rustbpe>=0.1.0" \
    "tiktoken>=0.11.0" \
    "tokenizers>=0.22.0" \
    "uvicorn>=0.36.0" \
    "wandb>=0.21.3"

WORKDIR /workspace