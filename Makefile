.PHONY: rust-dev rust-dev-gpu

rust-dev:
	uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

rust-dev-gpu:
	uv run --extra gpu maturin develop --release --manifest-path rustbpe/Cargo.toml
