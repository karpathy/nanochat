"""
Compare short training loss curves for nanochat variants on shared synthetic data.

This is a quick trainability check, not a language-quality benchmark. Each
variant in a comparison sees the exact same deterministic next-token batches.
"""

import argparse
import json
import time

import torch

from nanochat.gpt import GPTConfig, build_model_from_config


def build_config(args, architecture, linear_impl):
    model_dim = ((args.model_dim + args.head_dim - 1) // args.head_dim) * args.head_dim
    num_heads = max(1, model_dim // args.head_dim)
    return GPTConfig(
        sequence_len=args.seq_len,
        vocab_size=args.vocab_size,
        n_layer=args.depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern="L",
        architecture=architecture,
        linear_impl=linear_impl,
        ls_num_blocks=args.ls_num_blocks,
        ls_rank=args.ls_rank,
        lsrec_h_dim=args.lsrec_h_dim,
        lsrec_n_iter=args.lsrec_n_iter,
        lsrec_n_mem=args.lsrec_n_mem,
        lsrec_log_dt_init=args.lsrec_log_dt_init,
    )


def make_batches(args, device):
    generator = torch.Generator(device=device)
    generator.manual_seed(args.data_seed)
    batches = []
    positions = torch.arange(args.seq_len, device=device).unsqueeze(0)
    for _ in range(args.steps):
        if args.data_mode == "increment":
            starts = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, 1),
                device=device,
                generator=generator,
            )
            idx = (starts + positions) % args.vocab_size
            targets = (idx + 1) % args.vocab_size
        elif args.data_mode == "random-next":
            idx = torch.randint(
                0,
                args.vocab_size,
                (args.batch_size, args.seq_len),
                device=device,
                generator=generator,
            )
            targets = torch.roll(idx, shifts=-1, dims=1)
            targets[:, -1] = (idx[:, -1] + 1) % args.vocab_size
        else:
            raise ValueError(f"Unknown data_mode: {args.data_mode}")
        batches.append((idx, targets))
    return batches


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def run_variant(args, name, architecture, linear_impl, batches, device):
    torch.manual_seed(args.model_seed)
    cfg = build_config(args, architecture, linear_impl)
    model = build_model_from_config(cfg).to(device)
    model.init_weights()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    losses = []
    start = time.perf_counter()
    for step, (idx, targets) in enumerate(batches):
        optimizer.zero_grad(set_to_none=True)
        loss = model(idx, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        if step == 0 or (step + 1) % args.log_every == 0 or step == len(batches) - 1:
            losses.append({"step": step + 1, "loss": float(loss.detach().cpu())})
    synchronize(device)
    elapsed = time.perf_counter() - start
    return {
        "name": name,
        "architecture": architecture,
        "linear_impl": linear_impl,
        "params": sum(p.numel() for p in model.parameters()),
        "losses": losses,
        "final_loss": losses[-1]["loss"],
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--model-dim", type=int, default=192)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--data-mode", type=str, default="increment", choices=["increment", "random-next"])
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ls-num-blocks", type=int, default=16)
    parser.add_argument("--ls-rank", type=int, default=64)
    parser.add_argument("--lsrec-h-dim", type=int, default=0)
    parser.add_argument("--lsrec-n-iter", type=int, default=4)
    parser.add_argument("--lsrec-n-mem", type=int, default=64)
    parser.add_argument("--lsrec-log-dt-init", type=float, default=-3.5)
    parser.add_argument("--model-seed", type=int, default=1337)
    parser.add_argument("--data-seed", type=int, default=2026)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    batches = make_batches(args, device)
    comparisons = [
        {
            "comparison": "transformer_dense_vs_lslinear",
            "variants": [
                ("dense", "transformer", "dense"),
                ("lslinear", "transformer", "ls"),
            ],
        },
        {
            "comparison": "transformer_vs_lsrecurrent_scan_driven",
            "variants": [
                ("transformer", "transformer", "dense"),
                ("lsrecurrent-scan-driven", "lsrecurrent-scan-driven", "ls"),
            ],
        },
    ]

    output = []
    for comparison in comparisons:
        results = []
        for name, architecture, linear_impl in comparison["variants"]:
            result = run_variant(args, name, architecture, linear_impl, batches, device)
            results.append(result)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        output.append({"comparison": comparison["comparison"], "results": results})

    print(json.dumps({"task": "loss_curve_comparison", "settings": vars(args), "comparisons": output}, indent=2))


if __name__ == "__main__":
    main()
