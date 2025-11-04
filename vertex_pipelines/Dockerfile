# Use the official Python 3.10 image.
FROM python:3.10-slim

# Set the working directory.
WORKDIR /app

# Install uv, Rust, and other system dependencies.
RUN apt-get update && apt-get install -y curl build-essential
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add uv, cargo, and the future venv bin to the PATH
ENV PATH="/root/.local/bin:/root/.cargo/bin:/app/.venv/bin:${PATH}"

# Copy the entire project into the Docker image.
COPY . .

# Create a virtual environment.
RUN uv venv

# Install Python dependencies using uv.
RUN uv sync --extra gpu

# Install maturin, which is a build dependency.
RUN uv pip install maturin

# Build the rustbpe tokenizer.
# The maturin executable from the venv should be on the PATH now.
RUN maturin develop --release --manifest-path rustbpe/Cargo.toml

# Set the entrypoint.
ENTRYPOINT ["python"]
