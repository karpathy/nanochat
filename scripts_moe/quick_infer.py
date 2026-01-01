"""
Quick local inference for nanochat-MoE checkpoints (pretrain-style, plain text).

Example:
  uv run python scripts_moe/quick_infer.py --model-tag d20 --prompt "what's 1+1 equal to?"
  uv run python scripts_moe/quick_infer.py --hf-path hf-export/moe_gpt2 --prompt "what's 1+1 equal to?"
"""

import argparse
import json
import os
import sys

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import nanochat_moe.manager as _moe_manager
sys.modules.setdefault("manager", _moe_manager)

from nanochat_moe.checkpoint_manager import find_last_step, find_largest_model
from nanochat_moe.common import get_base_dir
from nanochat_moe.standard import GPT, GPTConfig
from nanochat_moe.tokenizer import RustBPETokenizer, get_tokenizer


def pad_vocab_weights(state_dict, target_vocab):
    for key in ("transformer.wte.weight", "lm_head.weight"):
        if key not in state_dict:
            continue
        tensor = state_dict[key]
        if tensor.size(0) >= target_vocab:
            continue
        pad_rows = target_vocab - tensor.size(0)
        pad = tensor.new_zeros((pad_rows, tensor.size(1)))
        state_dict[key] = torch.cat([tensor, pad], dim=0)


def load_tokenizer(mode):
    if mode == "gpt2":
        return RustBPETokenizer.from_pretrained("gpt2")
    if mode == "cache":
        return get_tokenizer()
    raise ValueError(f"Unknown tokenizer mode: {mode}")


def load_hf_model(hf_path, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(hf_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_path, trust_remote_code=True)
    model.to(device)
    model.eval()
    vocab_limit = None
    if tokenizer.vocab_size < model.config.vocab_size:
        vocab_limit = tokenizer.vocab_size
        print(
            f"Warning: tokenizer vocab {tokenizer.vocab_size} < model vocab {model.config.vocab_size}; "
            "clamping logits to tokenizer vocab."
        )
    return model, tokenizer, vocab_limit


def load_standard_model(source, device, model_tag, step, tokenizer):
    base_dir = get_base_dir()
    checkpoint_root = os.path.join(base_dir, f"{source}_checkpoints")
    if model_tag is None:
        model_tag = find_largest_model(checkpoint_root)
    checkpoint_dir = os.path.join(checkpoint_root, model_tag)
    if step is None:
        step = find_last_step(checkpoint_dir)

    model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
    meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
    model_data = torch.load(model_path, map_location=device)
    if device.type in {"cpu", "mps"}:
        model_data = {
            k: (v.float() if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16 else v)
            for k, v in model_data.items()
        }
    model_data = {k.removeprefix("_orig_mod."): v for k, v in model_data.items()}
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    cfg_kwargs = meta["model_config"]
    tok_vocab = tokenizer.get_vocab_size()
    cfg_vocab = cfg_kwargs.get("vocab_size")
    vocab_limit = None
    if tok_vocab > cfg_vocab:
        print(
            f"Warning: tokenizer vocab {tok_vocab} > model vocab {cfg_vocab}; "
            "padding embeddings to match."
        )
        cfg_kwargs = dict(cfg_kwargs)
        cfg_kwargs["vocab_size"] = tok_vocab
        pad_vocab_weights(model_data, tok_vocab)
    elif tok_vocab < cfg_vocab:
        vocab_limit = tok_vocab
        print(
            f"Warning: tokenizer vocab {tok_vocab} < model vocab {cfg_vocab}; "
            "clamping logits to tokenizer vocab."
        )

    model = GPT(GPTConfig(**cfg_kwargs))
    model.load_state_dict(model_data, strict=True)
    model.to(device)
    model.eval()
    return model, tokenizer, meta, model_tag, step, vocab_limit


def build_prompt_tokens(tokenizer, question: str):
    prompt = f"Question: {question}\nAnswer:"
    bos = tokenizer.get_bos_token_id()
    return tokenizer.encode(prompt, prepend=bos), prompt


def build_prompt_tokens_hf(tokenizer, question: str):
    prompt = f"Question: {question}\nAnswer:"
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None:
        ids = [bos_id] + ids
    return ids, prompt


def sample_next_token(logits, temperature, top_k):
    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits = torch.where(
            logits < v[:, [-1]],
            torch.full_like(logits, -float("inf")),
            logits,
        )
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.inference_mode()
def generate_tokens(model, input_ids, max_new_tokens, temperature, top_k, vocab_limit):
    for _ in range(max_new_tokens):
        idx_cond = (
            input_ids
            if input_ids.size(1) <= model.config.block_size
            else input_ids[:, -model.config.block_size :]
        )
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]
        if vocab_limit is not None and vocab_limit < logits.size(-1):
            logits = logits[:, :vocab_limit]
        next_token = sample_next_token(logits, temperature, top_k)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


@torch.inference_mode()
def generate_tokens_hf(model, input_ids, max_new_tokens, temperature, top_k, vocab_limit):
    for _ in range(max_new_tokens):
        logits = model(input_ids).logits
        logits = logits[:, -1, :]
        if vocab_limit is not None and vocab_limit < logits.size(-1):
            logits = logits[:, :vocab_limit]
        next_token = sample_next_token(logits, temperature, top_k)
        input_ids = torch.cat((input_ids, next_token), dim=1)
    return input_ids


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Run a quick nanochat-MoE inference")
    parser.add_argument("--source", type=str, default="base", choices=["base", "mid", "sft", "rl"])
    parser.add_argument("--model-tag", type=str, default="d20")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--hf-path", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="what's 1+1 equal to?")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--tokenizer", type=str, default="gpt2", choices=["gpt2", "cache"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.hf_path:
        model, tokenizer, vocab_limit = load_hf_model(args.hf_path, device)
        prompt_tokens, prompt_text = build_prompt_tokens_hf(tokenizer, args.prompt)
        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output_ids = generate_tokens_hf(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            vocab_limit=vocab_limit,
        )
        generated_tokens = output_ids[0].tolist()[len(prompt_tokens) :]
        output_text = tokenizer.decode(generated_tokens)
        print(f"Loaded HF model: {args.hf_path}")
    else:
        tokenizer = load_tokenizer(args.tokenizer)
        model, tokenizer, _, resolved_tag, resolved_step, vocab_limit = load_standard_model(
            args.source, device, args.model_tag, args.step, tokenizer
        )
        prompt_tokens, prompt_text = build_prompt_tokens(tokenizer, args.prompt)
        if len(prompt_tokens) > model.config.block_size:
            print(
                f"Warning: prompt length {len(prompt_tokens)} exceeds block_size "
                f"{model.config.block_size}; context will be truncated."
            )

        input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        output_ids = generate_tokens(
            model,
            input_ids,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            vocab_limit=vocab_limit,
        )
        generated_tokens = output_ids[0].tolist()[len(prompt_tokens) :]
        output_text = tokenizer.decode(generated_tokens)
        print(f"Loaded: {args.source}/{resolved_tag} step {resolved_step}")
    # print("Prompt:", prompt_text)
    # print("Output:", output_text)
    print("===============Prompt===============")
    print(prompt_text)
    print("===============Output===============")
    print(output_text)


if __name__ == "__main__":
    main()
