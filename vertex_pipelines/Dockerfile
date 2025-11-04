# Use the official Python 3.10 image.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Install uv, Rust, and other system dependencies.
RUN apt-get update && apt-get install -y curl build-essential
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the entire project into the Docker image.
COPY . .

# Install Python dependencies using uv.
RUN /root/.local/bin/uv sync --extra gpu

# Build the rustbpe tokenizer.
RUN /root/.local/bin/uv run maturin develop --release --manifest-path rustbpe/Cargo.toml --uv

# Set the entrypoint.
ENTRYPOINT ["python"]
