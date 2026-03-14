"""
Unified evaluation script for base models.

Supports three evaluation modes:
  --eval core    : CORE benchmark (ICL tasks)
  --eval bpb     : Bits-per-byte evaluation
  --eval sample  : Generate model samples

Example:

torchrun --nproc_per_node=8 -m scripts.base_eval --hf-path openai-community/gpt2
"""

import os
import csv
import time
import json
import yaml
import argparse
import random
import tempfile
import zipfile
import shutil
import torch

from nanochat.common import (
    compute_init,
    compute_cleanup,
    print0,
    get_base_dir,
    autodetect_device_type,
    download_file_with_lock
)

from nanochat.tokenizer import HuggingFaceTokenizer, get_token_bytes
from nanochat.checkpoint_manager import load_model
from nanochat.core_eval import evaluate_task
from nanochat.dataloader import tokenizing_distributed_data_loader_bos_bestfit
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine

# -----------------------------------------------------------------------------
# Model wrapper


class ModelWrapper:
    """
    Adapter to make HuggingFace models compatible with nanochat interface.
    """

    def __init__(self, model, max_seq_len):
        self.model = model
        self.max_seq_len = max_seq_len

    def __call__(self, input_ids, targets=None, loss_reduction="mean"):

        logits = self.model(input_ids).logits

        if targets is None:
            return logits

        # Correct causal language modeling shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_targets = targets[..., 1:].contiguous()

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )

        return loss

    def get_device(self):
        return next(self.model.parameters()).device


# -----------------------------------------------------------------------------
# HuggingFace loading


def load_hf_model(hf_path, device):

    print0(f"Loading HuggingFace model: {hf_path}")

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        hf_path,
        torch_dtype=torch.float16 if device.type == "cuda" else None,
    )

    model.to(device)
    model.eval()

    max_seq_len = getattr(model.config, "max_position_embeddings", 1024)

    tokenizer = HuggingFaceTokenizer.from_pretrained(hf_path)

    return ModelWrapper(model, max_seq_len), tokenizer


# -----------------------------------------------------------------------------
# Token byte computation (cached)


def get_hf_token_bytes(tokenizer, device):

    base_dir = get_base_dir()
    cache_path = os.path.join(base_dir, "token_bytes_cache.pt")

    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)

    vocab_size = tokenizer.tokenizer.get_vocab_size()

    token_bytes = torch.zeros(vocab_size, dtype=torch.int64)

    for token_id in range(vocab_size):
        token_str = tokenizer.tokenizer.decode([token_id])
        token_bytes[token_id] = len(token_str.encode("utf-8"))

    torch.save(token_bytes, cache_path)

    return token_bytes.to(device)


# -----------------------------------------------------------------------------
# CORE evaluation

EVAL_BUNDLE_URL = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip"


def place_eval_bundle(file_path):

    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")

    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(tmpdir)

        shutil.move(os.path.join(tmpdir, "eval_bundle"), eval_bundle_dir)

    print0(f"Eval bundle placed at {eval_bundle_dir}")


def evaluate_core(model, tokenizer, device, max_per_task=-1):

    base_dir = get_base_dir()
    eval_bundle_dir = os.path.join(base_dir, "eval_bundle")

    if not os.path.exists(eval_bundle_dir):
        download_file_with_lock(
            EVAL_BUNDLE_URL,
            "eval_bundle.zip",
            postprocess_fn=place_eval_bundle,
        )

    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    meta_path = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    tasks = config["icl_tasks"]

    random_baselines = {}

    with open(meta_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    results = {}
    centered_results = {}

    with torch.no_grad():

        for task in tasks:

            label = task["label"]

            task_meta = {
                "task_type": task["icl_task_type"],
                "dataset_uri": task["dataset_uri"],
                "num_fewshot": task["num_fewshot"][0],
                "continuation_delimiter": task.get("continuation_delimiter", " "),
            }

            print0(f"Evaluating {label}...")

            data_path = os.path.join(data_base_path, task_meta["dataset_uri"])

            with open(data_path) as f:
                data = [json.loads(x) for x in f]

            random.Random(1337).shuffle(data)

            if max_per_task > 0:
                data = data[:max_per_task]

            acc = evaluate_task(model, tokenizer, data, device, task_meta)

            results[label] = acc

            rb = random_baselines[label]

            centered = (acc - 0.01 * rb) / (1 - 0.01 * rb)

            centered_results[label] = centered

            print0(f"{label}: acc={acc:.4f} centered={centered:.4f}")

    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
    }


# -----------------------------------------------------------------------------
# Sampling


def sample_model(model, tokenizer):

    engine = Engine(model, tokenizer)

    prompts = [
        "The capital of France is",
        "The opposite of hot is",
        "If 5*x + 3 = 13, then x is",
    ]

    outputs = []

    with torch.no_grad():

        for p in prompts:

            tokens = tokenizer(p, prepend="<|bos|>")

            sample, _ = engine.generate_batch(
                tokens,
                num_samples=1,
                max_tokens=16,
                temperature=0,
            )

            text = tokenizer.decode(sample[0])

            outputs.append(text)

            print0(f"{p} -> {text}")

    return outputs


# -----------------------------------------------------------------------------
# Main


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--eval", default="core,bpb,sample")
    parser.add_argument("--hf-path", default=None)
    parser.add_argument("--model-tag", default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--device-batch-size", type=int, default=32)
    parser.add_argument("--split-tokens", type=int, default=40 * 524288)
    parser.add_argument("--max-per-task", type=int, default=-1)

    args = parser.parse_args()

    eval_modes = set(x.strip() for x in args.eval.split(","))

    device_type = autodetect_device_type()

    ddp, rank, local_rank, world_size, device = compute_init(device_type)

    # Load model

    if args.hf_path:

        model, tokenizer = load_hf_model(args.hf_path, device)

        seq_len = model.max_seq_len

        token_bytes = get_hf_token_bytes(tokenizer, device)

        model_name = args.hf_path

    else:

        model, tokenizer, meta = load_model(
            "base",
            device,
            phase="eval",
            model_tag=args.model_tag,
            step=args.step,
        )

        seq_len = meta["model_config"]["sequence_len"]

        token_bytes = get_token_bytes(device)

        model_name = f"base_model_step_{meta['step']}"

    print0(f"Evaluating: {model_name}")

    core_results = None
    bpb_results = {}

    # Sampling

    if "sample" in eval_modes and rank == 0:
        sample_model(model, tokenizer)

    # BPB

    if "bpb" in eval_modes:

        tokens_per_step = args.device_batch_size * seq_len * world_size

        steps = max(1, args.split_tokens // tokens_per_step)

        with torch.no_grad():

            for split in ["train", "val"]:

                loader = tokenizing_distributed_data_loader_bos_bestfit(
                    tokenizer,
                    args.device_batch_size,
                    seq_len,
                    split,
                    device=device,
                )

                bpb = evaluate_bpb(model, loader, steps, token_bytes)

                bpb_results[split] = bpb

                print0(f"{split} BPB: {bpb:.6f}")

    # CORE

    if "core" in eval_modes and rank == 0:

        core_results = evaluate_core(
            model,
            tokenizer,
            device,
            max_per_task=args.max_per_task,
        )

        print0(f"CORE metric: {core_results['core_metric']:.4f}")

    compute_cleanup()


if __name__ == "__main__":
    main()
