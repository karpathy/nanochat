#!/usr/bin/env python
"""Reshard a distributed base-training optimizer checkpoint to a new world size.

This is intended for nanochat's DistMuonAdamW optimizer. It preserves model,
metadata, dataloader state, and optimizer moments while changing the number of
optimizer shard files from old_world_size to new_world_size.
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch

from nanochat.gpt import GPT, GPTConfig
from nanochat.optim import _rank_first_dim_slice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--source-tag", type=str, required=True)
    parser.add_argument("--target-tag", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--old-world-size", type=int, required=True)
    parser.add_argument("--new-world-size", type=int, required=True)
    return parser.parse_args()


def load_optimizer_shards(source_dir: Path, step: int, old_world_size: int):
    shards = []
    for rank in range(old_world_size):
        path = source_dir / f"optim_{step:06d}_rank{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Loading {path}", flush=True)
        shards.append(torch.load(path, map_location="cpu"))
    return shards


def build_param_shapes(meta_data):
    config = GPTConfig(**meta_data["model_config"])
    with torch.device("meta"):
        model = GPT(config)
    optimizer = model.setup_optimizer()
    param_shapes = {}
    group_by_first_param = {}
    for group in optimizer.param_groups:
        param_ids = [id(p) for p in group["params"]]
        # Match PyTorch Optimizer.state_dict's integer parameter IDs.
        # The actual values are assigned in param-group traversal order.
        # Build a temporary state_dict and use its IDs.
    state_dict = optimizer.state_dict()
    object_to_shape = {id(p): tuple(p.shape) for group in optimizer.param_groups for p in group["params"]}
    for real_group, saved_group in zip(optimizer.param_groups, state_dict["param_groups"]):
        for param, param_id in zip(real_group["params"], saved_group["params"]):
            param_shapes[param_id] = tuple(param.shape)
        group_by_first_param[saved_group["params"][0]] = {
            "kind": saved_group["kind"],
            "num_params": len(saved_group["params"]),
            "param_shape": tuple(real_group["params"][0].shape),
        }
    return param_shapes, group_by_first_param


def slice_first_dim(tensor, original_rows, new_world_size, rank):
    rank_size, padded_rows, start, end, valid_rows = _rank_first_dim_slice(original_rows, new_world_size, rank)
    out = torch.zeros((rank_size, *tensor.shape[1:]), dtype=tensor.dtype)
    if valid_rows > 0:
        out[:valid_rows].copy_(tensor[start:end])
    return out


def reshard_first_dim_tensor(shards, state_key, tensor_key, full_shape, old_world_size, new_world_size):
    old_parts = [shard["state"][state_key][tensor_key] for shard in shards]
    full = torch.cat(old_parts, dim=0)[: full_shape[0]].contiguous()
    return [slice_first_dim(full, full_shape[0], new_world_size, rank) for rank in range(new_world_size)]


def reshard_muon_tensor(shards, state_key, tensor_key, num_params, new_world_size):
    old_parts = [shard["state"][state_key][tensor_key] for shard in shards]
    full = torch.cat(old_parts, dim=0)[:num_params].contiguous()
    return [slice_first_dim(full, num_params, new_world_size, rank) for rank in range(new_world_size)]


def main():
    args = parse_args()
    source_dir = args.checkpoint_dir / args.source_tag
    target_dir = args.checkpoint_dir / args.target_tag
    target_dir.mkdir(parents=True, exist_ok=True)

    model_src = source_dir / f"model_{args.step:06d}.pt"
    meta_src = source_dir / f"meta_{args.step:06d}.json"
    model_dst = target_dir / model_src.name
    meta_dst = target_dir / meta_src.name
    if not model_src.exists() or not meta_src.exists():
        raise FileNotFoundError(f"Missing model/meta at step {args.step} in {source_dir}")

    with open(meta_src, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    param_shapes, group_by_first_param = build_param_shapes(meta_data)
    shards = load_optimizer_shards(source_dir, args.step, args.old_world_size)
    template = shards[0]

    print(f"Copying model checkpoint to {model_dst}", flush=True)
    shutil.copy2(model_src, model_dst)

    meta_data = dict(meta_data)
    meta_data.setdefault("user_config", {})["model_tag"] = args.target_tag
    meta_data["resharded_optimizer"] = {
        "source_tag": args.source_tag,
        "old_world_size": args.old_world_size,
        "new_world_size": args.new_world_size,
    }
    with open(meta_dst, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

    new_shards = [
        {"state": {}, "param_groups": template["param_groups"]}
        for _ in range(args.new_world_size)
    ]

    for state_key, state_value in template["state"].items():
        print(f"Resharding state key {state_key}", flush=True)
        if "momentum_buffer" in state_value:
            group_info = group_by_first_param[state_key]
            num_params = group_info["num_params"]
            per_rank = {}
            for tensor_key in ("momentum_buffer", "second_momentum_buffer"):
                per_rank[tensor_key] = reshard_muon_tensor(
                    shards, state_key, tensor_key, num_params, args.new_world_size
                )
            for rank in range(args.new_world_size):
                new_shards[rank]["state"][state_key] = {
                    "momentum_buffer": per_rank["momentum_buffer"][rank],
                    "second_momentum_buffer": per_rank["second_momentum_buffer"][rank],
                }
        elif "exp_avg" in state_value:
            full_shape = param_shapes[state_key]
            replicated = tuple(state_value["exp_avg"].shape) == full_shape
            per_rank = {}
            if replicated:
                for tensor_key in ("exp_avg", "exp_avg_sq"):
                    per_rank[tensor_key] = [state_value[tensor_key].clone() for _ in range(args.new_world_size)]
            else:
                for tensor_key in ("exp_avg", "exp_avg_sq"):
                    per_rank[tensor_key] = reshard_first_dim_tensor(
                        shards, state_key, tensor_key, full_shape, args.old_world_size, args.new_world_size
                    )
            for rank in range(args.new_world_size):
                new_shards[rank]["state"][state_key] = {
                    "step": state_value["step"],
                    "exp_avg": per_rank["exp_avg"][rank],
                    "exp_avg_sq": per_rank["exp_avg_sq"][rank],
                }
        else:
            raise ValueError(f"Unknown optimizer state layout for key {state_key}: {state_value.keys()}")

    for rank, shard in enumerate(new_shards):
        path = target_dir / f"optim_{args.step:06d}_rank{rank}.pt"
        tmp_path = Path(f"{path}.tmp.{os.getpid()}")
        print(f"Saving {path}", flush=True)
        torch.save(shard, tmp_path)
        os.replace(tmp_path, path)

    print("Done", flush=True)


if __name__ == "__main__":
    main()
