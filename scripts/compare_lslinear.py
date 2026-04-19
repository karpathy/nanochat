"""
Compare nanochat Transformer dense Linear vs LSLinear on random tokens.

Example:
python -m scripts.compare_lslinear --device cuda --depth 4 --model-dim 256 --seq-len 256 --batch-size 4
"""

import argparse
import json
import time

import torch

from nanochat.gpt import BlockDiagonalLinear, GPTConfig, LSLinear, build_model_from_config


def build_config(args, linear_impl):
    head_dim = args.head_dim
    model_dim = ((args.model_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = max(1, model_dim // head_dim)
    return GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",
        architecture="transformer",
        linear_impl=linear_impl,
        ls_num_blocks=args.ls_num_blocks,
        ls_rank=args.ls_rank,
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def bench_model(name, model, idx, targets, warmup, rounds):
    device = idx.device
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        loss.backward()
        optimizer.step()
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    losses = []
    for _ in range(rounds):
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "name": name,
        "loss": losses[-1],
        "ms_per_step": 1000 * elapsed / rounds,
        "tokens_per_sec": idx.numel() * rounds / elapsed,
        "peak_mem_bytes": peak_mem,
        "params": sum(p.numel() for p in model.parameters()),
        "lslinear_layers": sum(isinstance(m, LSLinear) for m in model.modules()),
        "blockdiag_layers": sum(isinstance(m, BlockDiagonalLinear) for m in model.modules()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--model-dim", type=int, default=256)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--ls-num-blocks", type=int, default=16)
    parser.add_argument("--ls-rank", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    idx = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    results = []
    for linear_impl in ("dense", "ls"):
        cfg = build_config(args, linear_impl)
        model = build_model_from_config(cfg).to(device)
        model.init_weights()
        model.train()
        results.append(bench_model(linear_impl, model, idx, targets, args.warmup, args.rounds))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"comparison": "transformer_dense_vs_lslinear", "results": results}, indent=2))


if __name__ == "__main__":
    main()
