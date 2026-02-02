#!/usr/bin/env python3
"""Generate model responses for safety eval prompts using local checkpoints."""
from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

try:
    import torch
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: torch. Install with `uv sync` or `pip install torch`.")
try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None

try:
    from nanochat.common import compute_init, autodetect_device_type, get_base_dir
    from nanochat.checkpoint_manager import load_model_from_dir
    from nanochat.engine import Engine
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Failed to import nanochat modules: {exc}")
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    raise SystemExit("Missing dependency: tqdm. Install with `uv sync` or `pip install tqdm`.")


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[3] / "eval_generations"


def _normalize_prompt(text: str) -> str:
    return " ".join(text.strip().split())


def _resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path(get_base_dir()) / p


def _default_dataset_name(input_path: Path) -> str:
    return input_path.stem


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in {"-", "_", "."} else "_" for c in name)


def _load_existing(output_path: Path) -> set[str]:
    existing: set[str] = set()
    if not output_path.exists():
        return existing
    with output_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = obj.get("record_key")
            if isinstance(key, str):
                existing.add(key)
    return existing


def _rank_output_path(output_path: Path, rank: int) -> Path:
    return output_path.parent / f"{output_path.stem}.rank{rank}{output_path.suffix}"


def _merge_rank_outputs(output_path: Path, world_size: int) -> None:
    merged_keys = _load_existing(output_path)
    with output_path.open("a", encoding="utf-8") as merged:
        for rank in range(world_size):
            rank_path = _rank_output_path(output_path, rank)
            if not rank_path.exists():
                continue
            with rank_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = obj.get("record_key")
                    if not isinstance(key, str) or key in merged_keys:
                        continue
                    merged.write(json.dumps(obj, ensure_ascii=True) + "\n")
                    merged_keys.add(key)


def _iter_inputs(input_path: Path, skip: set[str]):
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt = row.get("prompt")
            if not prompt:
                continue
            record_id = row.get("id")
            key = str(record_id) if record_id is not None else _normalize_prompt(prompt)
            if key in skip:
                continue
            yield key, row


def _build_conversation(prompt: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate safety eval responses")
    parser.add_argument("--input", type=str, required=True, help="Input JSONL with prompts")
    parser.add_argument("--dataset-name", type=str, default="", help="Dataset name for output naming")
    parser.add_argument("--output-root", type=str, default=str(_default_output_root()))
    parser.add_argument("--model-name", type=str, default="", help="Name for output subdir")
    parser.add_argument("--checkpoint-dir", type=str, default="", help="Path to model tag dir")
    parser.add_argument("--checkpoints-dir", type=str, default="", help="Path to checkpoints root")
    parser.add_argument("--model-tag", type=str, default="", help="Model tag under checkpoints-dir")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step to load")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16"])
    parser.add_argument("--device-type", type=str, default="", choices=["cuda", "cpu", "mps"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input file not found: {input_path}")

    dataset_name = args.dataset_name or _default_dataset_name(input_path)
    dataset_name = _safe_name(dataset_name)

    if args.checkpoint_dir:
        checkpoint_dir = _resolve_path(args.checkpoint_dir)
        if not checkpoint_dir.exists():
            raise SystemExit(f"Checkpoint dir not found: {checkpoint_dir}")
        checkpoints_dir = checkpoint_dir.parent
        model_tag = args.model_tag or checkpoint_dir.name
    elif args.checkpoints_dir:
        checkpoints_dir = _resolve_path(args.checkpoints_dir)
        if not checkpoints_dir.exists():
            raise SystemExit(f"Checkpoints root not found: {checkpoints_dir}")
        model_tag = args.model_tag or None
    else:
        raise SystemExit("Provide --checkpoint-dir or --checkpoints-dir")

    model_name = args.model_name or model_tag or "model"
    model_name = _safe_name(model_name)

    output_root = Path(args.output_root)
    output_dir = output_root / model_name
    output_dir.mkdir(parents=True, exist_ok=True)
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    output_path = output_dir / f"{dataset_name}__{model_name}.jsonl"
    rank_output_path = _rank_output_path(output_path, ddp_rank) if ddp else output_path

    skip = _load_existing(output_path)
    if ddp:
        skip.update(_load_existing(rank_output_path))
    rows = list(_iter_inputs(input_path, skip))
    if ddp:
        rows = rows[ddp_rank::ddp_world_size]
    if not rows:
        if ddp and dist is not None and dist.is_initialized():
            dist.barrier()
            if ddp_rank == 0:
                _merge_rank_outputs(output_path, ddp_world_size)
        print("No new prompts to process.")
        return

    ptdtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) if device_type == "cuda" else nullcontext()

    model, tokenizer, meta = load_model_from_dir(str(checkpoints_dir), device, phase="eval", model_tag=model_tag, step=args.step)
    engine = Engine(model, tokenizer)

    with rank_output_path.open("a", encoding="utf-8") as f:
        with tqdm(total=len(rows), desc=f"generate rank{ddp_rank}") as progress:
            for idx, (record_key, row) in enumerate(rows):
                prompt = row.get("prompt", "")
                conversation = _build_conversation(prompt)
                prompt_ids = tokenizer.render_for_completion(conversation)
                seed = args.seed + idx
                try:
                    with autocast_ctx:
                        results, _ = engine.generate_batch(
                            prompt_ids,
                            num_samples=1,
                            max_tokens=args.max_new_tokens,
                            temperature=args.temperature,
                            top_k=args.top_k,
                            seed=seed,
                        )
                    full_ids = results[0]
                    response_ids = full_ids[len(prompt_ids):]
                    response = tokenizer.decode(response_ids)
                    record = {
                        "record_key": record_key,
                        "id": row.get("id"),
                        "prompt": prompt,
                        "response": response,
                        "dataset": row.get("dataset"),
                        "source_split": row.get("source_split"),
                        "model_name": model_name,
                        "model_tag": model_tag,
                        "checkpoints_dir": str(checkpoints_dir),
                        "step": args.step,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "seed": seed,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                except Exception as exc:
                    record = {
                        "record_key": record_key,
                        "id": row.get("id"),
                        "prompt": prompt,
                        "response": None,
                        "dataset": row.get("dataset"),
                        "source_split": row.get("source_split"),
                        "model_name": model_name,
                        "model_tag": model_tag,
                        "checkpoints_dir": str(checkpoints_dir),
                        "step": args.step,
                        "temperature": args.temperature,
                        "top_k": args.top_k,
                        "max_new_tokens": args.max_new_tokens,
                        "seed": seed,
                        "error": str(exc),
                        "created_at": datetime.utcnow().isoformat() + "Z",
                    }
                f.write(json.dumps(record, ensure_ascii=True) + "\n")
                f.flush()
                progress.update(1)

    if ddp and dist is not None and dist.is_initialized():
        dist.barrier()
        if ddp_rank == 0:
            _merge_rank_outputs(output_path, ddp_world_size)

    print(f"Wrote generations to {output_path}")


if __name__ == "__main__":
    main()
