"""
Benchmark Engine prefill + decode throughput.

Example (AMD MI300X / CUDA / ROCm all use --device-type cuda):

  python -m scripts.bench_infer --source sft --prompt-tokens 512 --max-tokens 128
  python -m scripts.bench_infer --source sft --compile --max-tokens 256
  python -m scripts.bench_infer --source sft --no-compile --num-samples 8
"""

import argparse
import time
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_init
from nanochat.engine import Engine

parser = argparse.ArgumentParser(description="Benchmark nanochat inference throughput")
parser.add_argument("-i", "--source", type=str, default="sft", help="Source of the model: sft|mid|rl|base")
parser.add_argument("-g", "--model-tag", type=str, default=None)
parser.add_argument("-s", "--step", type=int, default=None)
parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
parser.add_argument("-d", "--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None,
                    help="torch.compile (default: on for CUDA/ROCm)")
parser.add_argument("--prompt-tokens", type=int, default=512, help="Synthetic prompt length for prefill")
parser.add_argument("--max-tokens", type=int, default=128, help="Decode tokens to generate")
parser.add_argument("--num-samples", type=int, default=1, help="Parallel decode rows (batch)")
parser.add_argument("--warmup", type=int, default=1, help="Warmup runs (use >=1 after --compile)")
parser.add_argument("--repeats", type=int, default=3)
args = parser.parse_args()

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
_, _, _, _, device = compute_init(device_type)
ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
compile_model = args.compile if args.compile is not None else (device_type == "cuda")
print(f"device={device} dtype={args.dtype} compile={compile_model} config={meta.get('model_config', {})}")
if compile_model:
    print("torch.compile is on: first warmup run can take minutes on MI300X.")
engine = Engine(model, tokenizer, compile_model=compile_model, fuse_qkv=True)

bos = tokenizer.get_bos_token_id()
vocab = tokenizer.get_vocab_size()
# Keep ids in-range and start with BOS so the engine sees a valid sequence
prompt = [bos] + [((i * 17) % max(vocab - 1, 1)) + 1 for i in range(max(args.prompt_tokens - 1, 0))]

kwargs = dict(
    num_samples=args.num_samples,
    max_tokens=args.max_tokens,
    temperature=0.0,  # greedy: isolates kernel time from sampling RNG
)


def run_once():
    ntok = 0
    with autocast_ctx:
        for token_column, _ in engine.generate(prompt, **kwargs):
            ntok += len(token_column)
    return ntok


for i in range(args.warmup):
    print(f"warmup {i + 1}/{args.warmup}...")
    synchronize()
    run_once()
    synchronize()

times = []
toks = []
for i in range(args.repeats):
    synchronize()
    t0 = time.perf_counter()
    ntok = run_once()
    synchronize()
    dt = time.perf_counter() - t0
    times.append(dt)
    toks.append(ntok)
    print(f"repeat {i + 1}/{args.repeats}: {ntok} tokens in {dt:.3f}s ({ntok / dt:.1f} tok/s)")

avg_t = sum(times) / len(times)
avg_n = sum(toks) / len(toks)
print("---")
print(f"prompt_tokens={len(prompt)} decode_tokens={args.max_tokens} batch={args.num_samples}")
print(f"avg wall {avg_t:.3f}s  avg throughput {avg_n / avg_t:.1f} tok/s")
print("Note: tok/s includes prefill + decode. Raise --num-samples to fill MI300X on decode.")
