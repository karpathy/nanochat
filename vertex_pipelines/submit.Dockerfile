FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl build-essential
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

ENV PATH="/root/.local/bin:/root/.cargo/bin:/app/.venv/bin:${PATH}"

COPY . .

RUN uv venv
RUN uv sync --extra gpu
RUN uv pip install maturin
RUN maturin develop --release --manifest-path rustbpe/Cargo.toml

# Install gcloud
RUN apt-get install -y apt-transport-https ca-certificates gnupg
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -
RUN apt-get update && apt-get install -y google-cloud-sdk

ENTRYPOINT ["bash"]
