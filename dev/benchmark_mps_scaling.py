"""Benchmark nanochat training-step capacity on Apple Silicon / MPS.

This is a synthetic benchmark: it does not use the dataset pipeline.
It exists to answer a practical question on a single-machine Apple Silicon box:
which model sizes and token batches are stable, and what throughput do they reach?
"""

import argparse
import gc
import time

import torch

from nanochat.common import get_mps_memory_stats, maybe_torch_compile
from nanochat.gpt import GPT, GPTConfig


def parse_csv_ints(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]

def clear_mps_state() -> None:
    gc.collect()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
        torch.mps.synchronize()


def build_config(depth: int, seq_len: int, aspect_ratio: int, head_dim: int, vocab_size: int, window_pattern: str) -> GPTConfig:
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )


def run_trial(depth: int, batch_size: int, args, device: torch.device) -> dict[str, object]:
    clear_mps_state()
    config = build_config(depth, args.seq_len, args.aspect_ratio, args.head_dim, args.vocab_size, args.window_pattern)
    result = {
        "depth": depth,
        "batch_size": batch_size,
        "seq_len": args.seq_len,
        "status": "ok",
    }
    model = optimizer = inputs = targets = None
    try:
        model = GPT(config)
        model.to(device)
        model.init_weights()
        if args.compile:
            model = maybe_torch_compile(model, "mps", dynamic=False)
        optimizer = model.setup_optimizer(
            unembedding_lr=args.lr,
            embedding_lr=args.lr,
            matrix_lr=args.lr,
            weight_decay=0.0,
        )
        params = model.num_scaling_params()
        flops_per_token = model.estimate_flops()

        inputs = torch.randint(0, args.vocab_size, (batch_size, args.seq_len), dtype=torch.long, device=device)
        targets = inputs.clone()

        warmup_steps = max(args.warmup_steps, 0)
        total_steps = warmup_steps + args.steps
        tokens_measured = batch_size * args.seq_len * args.steps

        for step_idx in range(total_steps):
            loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
            if step_idx == warmup_steps - 1:
                torch.mps.synchronize()

        torch.mps.synchronize()
        start = time.perf_counter()
        for _ in range(args.steps):
            loss = model(inputs, targets)
            loss.backward()
            optimizer.step()
            model.zero_grad(set_to_none=True)
        torch.mps.synchronize()
        elapsed = time.perf_counter() - start

        mem = get_mps_memory_stats(budget_frac=args.recommended_budget_frac)
        result.update({
            "loss": float(loss.item()),
            "params_m": params["total"] / 1e6,
            "model_dim": config.n_embd,
            "heads": config.n_head,
            "tokens_per_s": tokens_measured / elapsed if elapsed > 0 else 0.0,
            "flops_per_token": flops_per_token,
            "allocated_gb": mem["allocated_gb"],
            "driver_gb": mem["driver_gb"],
            "recommended_gb": mem["recommended_gb"],
            "driver_frac": mem["driver_frac"],
            "headroom_gb": mem["headroom_gb"],
            "budget_frac": mem["budget_frac"],
            "budget_limit_gb": mem["budget_limit_gb"],
            "budget_headroom_gb": mem["budget_headroom_gb"],
            "exceeds_budget": mem["exceeds_budget"],
        })
    except Exception as exc:
        result["status"] = "oom" if "out of memory" in str(exc).lower() else "runtime_error"
        result["error"] = str(exc).splitlines()[0][:200]
    finally:
        del model, optimizer, inputs, targets
        clear_mps_state()
    return result


def print_table(results: list[dict[str, object]]) -> None:
    header = (
        f"{'depth':>5}  {'bs':>4}  {'dim':>5}  {'heads':>5}  {'params(M)':>10}  "
        f"{'tok/s':>10}  {'drv GB':>8}  {'rec GB':>8}  {'head GB':>8}  {'drv%':>6}  {'budget':>8}  {'status':>12}"
    )
    print(header)
    print("-" * len(header))
    for row in results:
        if row["status"] == "ok":
            budget_label = "over" if row["exceeds_budget"] else "ok"
            print(
                f"{row['depth']:>5}  {row['batch_size']:>4}  {row['model_dim']:>5}  {row['heads']:>5}  "
                f"{row['params_m']:>10.1f}  {row['tokens_per_s']:>10.0f}  {row['driver_gb']:>8.1f}  "
                f"{row['recommended_gb']:>8.1f}  {row['headroom_gb']:>8.1f}  {100.0 * row['driver_frac']:>5.1f}%  {budget_label:>8}  {row['status']:>12}"
            )
        else:
            print(
                f"{row['depth']:>5}  {row['batch_size']:>4}  {'-':>5}  {'-':>5}  {'-':>10}  {'-':>10}  "
                f"{'-':>8}  {'-':>8}  {'-':>8}  {'-':>6}  {'-':>8}  {row['status']:>12}"
            )


def print_recommendation(results: list[dict[str, object]]) -> None:
    successful = [row for row in results if row["status"] == "ok" and not row.get("exceeds_budget", False)]
    if not successful:
        successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        print("\nNo successful configurations were found.")
        return
    best = max(successful, key=lambda row: (row["params_m"], row["batch_size"], row["tokens_per_s"]))
    print("\nSuggested next base_train starting point:")
    print(
        "python -m scripts.base_train "
        f"--device-type=mps --depth={best['depth']} --head-dim={best['model_dim'] // best['heads']} "
        f"--max-seq-len={best['seq_len']} --device-batch-size={best['batch_size']} "
        f"--window-pattern=L --core-metric-every=-1 --eval-every=100 --sample-every=100"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark MPS scaling for nanochat")
    parser.add_argument("--depths", type=str, default="20,24,28,32,36", help="Comma-separated transformer depths")
    parser.add_argument("--batch-sizes", type=str, default="1,2,4,8", help="Comma-separated device batch sizes")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length")
    parser.add_argument("--aspect-ratio", type=int, default=64, help="Model width scaling factor")
    parser.add_argument("--head-dim", type=int, default=128, help="Attention head dimension")
    parser.add_argument("--vocab-size", type=int, default=32768, help="Synthetic vocab size")
    parser.add_argument("--window-pattern", type=str, default="L", help="Attention window pattern")
    parser.add_argument("--steps", type=int, default=2, help="Measured optimization steps per trial")
    parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup optimization steps per trial")
    parser.add_argument("--lr", type=float, default=1e-3, help="Synthetic learning rate for benchmark")
    parser.add_argument("--compile", action="store_true", help="Try torch.compile during benchmark")
    parser.add_argument("--recommended-budget-frac", type=float, default=0.9, help="Warn when MPS driver memory exceeds this fraction of torch.mps.recommended_max_memory()")
    args = parser.parse_args()

    if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
        raise SystemExit("MPS is not available on this machine.")

    torch.manual_seed(42)
    device = torch.device("mps")
    depths = parse_csv_ints(args.depths)
    batch_sizes = parse_csv_ints(args.batch_sizes)

    print(f"Benchmarking MPS scaling on {device} with seq_len={args.seq_len}, steps={args.steps}")
    print(f"Depths: {depths}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Recommended MPS budget threshold: {100.0 * args.recommended_budget_frac:.1f}% of recommended_max_memory")

    results = []
    for depth in depths:
        for batch_size in batch_sizes:
            print(f"Running depth={depth}, batch_size={batch_size} ...", flush=True)
            results.append(run_trial(depth, batch_size, args, device))

    print()
    print_table(results)
    print_recommendation(results)


if __name__ == "__main__":
    main()