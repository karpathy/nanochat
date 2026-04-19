"""
Compare nanochat Transformer vs LSRecurrent on random tokens.

Example:
python -m scripts.compare_lsrecurrent --device cuda --depth 4 --model-dim 256 --seq-len 256 --batch-size 4
"""

import argparse
import json
import time

import torch

from nanochat.gpt import BlockDiagonalLinear, GPTConfig, LSLinear, LSRecurrentScanDrivenGPT, build_model_from_config


def build_config(args, architecture):
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
        architecture=architecture,
        linear_impl="ls" if architecture == "lsrecurrent-scan-driven" else "dense",
        ls_num_blocks=args.ls_num_blocks,
        ls_rank=args.ls_rank,
        lsrec_h_dim=args.lsrec_h_dim,
        lsrec_n_iter=args.lsrec_n_iter,
        lsrec_n_mem=args.lsrec_n_mem,
        lsrec_log_dt_init=args.lsrec_log_dt_init,
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
        "flops_per_token_estimate": model.estimate_flops(),
        "lslinear_layers": sum(isinstance(m, LSLinear) for m in model.modules()),
        "blockdiag_layers": sum(isinstance(m, BlockDiagonalLinear) for m in model.modules()),
        "is_lsrecurrent": isinstance(model, LSRecurrentScanDrivenGPT),
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
    parser.add_argument("--lsrec-h-dim", type=int, default=0)
    parser.add_argument("--lsrec-n-iter", type=int, default=4)
    parser.add_argument("--lsrec-n-mem", type=int, default=0)
    parser.add_argument("--lsrec-log-dt-init", type=float, default=-2.3)
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
    for architecture in ("transformer", "lsrecurrent-scan-driven"):
        cfg = build_config(args, architecture)
        model = build_model_from_config(cfg).to(device)
        model.init_weights()
        model.train()
        results.append(bench_model(architecture, model, idx, targets, args.warmup, args.rounds))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps({"comparison": "transformer_vs_lsrecurrent", "results": results}, indent=2))


if __name__ == "__main__":
    main()
