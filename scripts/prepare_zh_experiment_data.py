"""
Prepare reproducible Chinese SFT and continued-pretraining datasets.

The outputs are isolated under $NANOCHAT_BASE_DIR/zh_experiment by default:

  sft/train.jsonl
  sft/val.jsonl
  pretrain/shard_00000.parquet ...
  pretrain/shard_99999.parquet  # validation, always sorts last
  manifest.json
"""

import argparse
import hashlib
import json
import os
import random
import shutil

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from nanochat.common import get_base_dir
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer


def has_cjk(text):
    return any(0x3400 <= ord(char) <= 0x9FFF for char in text)


def cjk_language_ratio(text):
    cjk_count = sum(1 for char in text if 0x3400 <= ord(char) <= 0x9FFF)
    latin_count = sum(1 for char in text if "a" <= char.lower() <= "z")
    return cjk_count / (cjk_count + latin_count) if cjk_count + latin_count else 0.0


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_messages(row):
    """Accept common instruction-dataset schemas and return NanoChat messages."""
    if isinstance(row.get("messages"), list):
        raw_messages = row["messages"]
        messages = [
            {"role": message["role"], "content": message["content"]}
            for message in raw_messages
            if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)
        ]
    elif isinstance(row.get("conversations"), list):
        role_map = {
            "human": "user", "user": "user",
            "gpt": "assistant", "assistant": "assistant",
        }
        messages = []
        for message in row["conversations"]:
            role = role_map.get(message.get("from") or message.get("role"))
            content = message.get("value", message.get("content"))
            if role is not None and isinstance(content, str):
                messages.append({"role": role, "content": content})
    elif isinstance(row.get("instruction"), str) and isinstance(row.get("output"), str):
        prompt = row["instruction"].strip()
        extra_input = row.get("input")
        if isinstance(extra_input, str) and extra_input.strip():
            prompt = f"{prompt}\n\n{extra_input.strip()}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": row["output"].strip()},
        ]
    else:
        raise ValueError(f"Unsupported SFT row schema. Available fields: {sorted(row.keys())}")

    while messages and messages[0]["role"] != "user":
        messages.pop(0)
    if len(messages) % 2 == 1:
        messages.pop()
    if len(messages) < 2:
        return None
    for index, message in enumerate(messages):
        expected_role = "user" if index % 2 == 0 else "assistant"
        if message["role"] != expected_role or not message["content"].strip():
            return None
    return messages


def load_hf_dataset(dataset_name, config_name, split, revision, streaming=False):
    kwargs = {
        "path": dataset_name,
        "split": split,
        "revision": revision,
        "streaming": streaming,
    }
    if config_name:
        kwargs["name"] = config_name
    return load_dataset(**kwargs)


def summarize_conversations(conversations, tokenizer, max_tokens):
    rendered_tokens = 0
    assistant_characters = 0
    assistant_cjk_characters = 0
    for messages in conversations:
        ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=max_tokens)
        rendered_tokens += len(ids)
        for message in messages:
            if message["role"] != "assistant":
                continue
            assistant_characters += len(message["content"])
            assistant_cjk_characters += sum(
                1 for char in message["content"] if 0x3400 <= ord(char) <= 0x9FFF
            )
    return {
        "rows": len(conversations),
        "rendered_tokens_at_sequence_limit": rendered_tokens,
        "assistant_characters": assistant_characters,
        "assistant_cjk_characters": assistant_cjk_characters,
    }


def prepare_sft(args, output_root, tokenizer):
    dataset = load_hf_dataset(
        args.sft_dataset, args.sft_config, args.sft_split,
        args.sft_revision, streaming=False,
    )
    conversations = []
    unsupported_schema = None
    for row in dataset:
        try:
            messages = normalize_messages(row)
        except ValueError as exc:
            unsupported_schema = exc
            break
        if messages is None:
            continue
        assistant_text = "\n".join(
            message["content"] for message in messages if message["role"] == "assistant"
        )
        if cjk_language_ratio(assistant_text) < args.sft_min_cjk_ratio:
            continue
        conversations.append(messages)
        if args.sft_max_rows > 0 and len(conversations) >= args.sft_max_rows:
            break
    if unsupported_schema is not None:
        raise unsupported_schema
    if len(conversations) <= args.sft_val_rows:
        raise ValueError(
            f"Only {len(conversations)} Chinese-output conversations remained; "
            f"need more than --sft-val-rows={args.sft_val_rows}"
        )

    rng = random.Random(args.seed)
    rng.shuffle(conversations)
    val_rows = conversations[:args.sft_val_rows]
    train_rows = conversations[args.sft_val_rows:]

    sft_dir = os.path.join(output_root, "sft")
    os.makedirs(sft_dir, exist_ok=True)
    paths = {
        "train": os.path.join(sft_dir, "train.jsonl"),
        "val": os.path.join(sft_dir, "val.jsonl"),
    }
    for split, rows in [("train", train_rows), ("val", val_rows)]:
        with open(paths[split], "w", encoding="utf-8") as handle:
            for messages in rows:
                handle.write(json.dumps(messages, ensure_ascii=False) + "\n")

    return {
        "dataset": args.sft_dataset,
        "config": args.sft_config,
        "revision": args.sft_revision,
        "split": args.sft_split,
        "source_columns": list(dataset.column_names),
        "license_note": "Use for research/non-commercial experiments unless the upstream dataset terms permit more.",
        "minimum_assistant_cjk_ratio": args.sft_min_cjk_ratio,
        "train": summarize_conversations(train_rows, tokenizer, args.sft_sequence_len + 1),
        "validation": summarize_conversations(val_rows, tokenizer, args.sft_sequence_len + 1),
        "train_path": paths["train"],
        "val_path": paths["val"],
        "train_sha256": file_sha256(paths["train"]),
        "val_sha256": file_sha256(paths["val"]),
    }


def iter_local_english(data_dir):
    parquet_paths = list_parquet_files(data_dir=data_dir)
    if len(parquet_paths) < 2:
        raise ValueError(f"Expected train shard(s) plus a validation shard in {data_dir}")
    for filepath in parquet_paths[:-1]:
        parquet_file = pq.ParquetFile(filepath)
        for row_group_index in range(parquet_file.num_row_groups):
            texts = parquet_file.read_row_group(row_group_index, columns=["text"]).column("text")
            yield from texts.to_pylist()


def iter_fineweb_chinese(args):
    dataset = load_hf_dataset(
        args.fineweb_dataset, args.fineweb_config, args.fineweb_split,
        args.fineweb_revision, streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.streaming_shuffle_buffer)
    for row in dataset:
        text = row.get("text")
        if isinstance(text, str) and has_cjk(text):
            yield text


def collect_documents(iterator, tokenizer, target_tokens, doc_cap, language):
    documents = []
    token_count = 0
    for text in iterator:
        text = text.strip()
        if not text:
            continue
        if len(text) > doc_cap:
            text = text[:doc_cap]
        tokens = len(tokenizer.encode(text, prepend="<|bos|>"))
        if tokens < 2:
            continue
        documents.append({"text": text, "tokens": tokens, "language": language})
        token_count += tokens
        if token_count >= target_tokens:
            break
    if token_count < target_tokens:
        raise RuntimeError(
            f"{language} source ended at {token_count:,} tokens, below target {target_tokens:,}"
        )
    return documents


def write_parquet(path, documents):
    table = pa.table({"text": [document["text"] for document in documents]})
    pq.write_table(table, path, row_group_size=1024, compression="zstd")


def summarize_documents(documents):
    summary = {"documents": len(documents), "tokens": 0, "characters": 0, "by_language": {}}
    for document in documents:
        summary["tokens"] += document["tokens"]
        summary["characters"] += len(document["text"])
        lang = document["language"]
        lang_summary = summary["by_language"].setdefault(lang, {"documents": 0, "tokens": 0})
        lang_summary["documents"] += 1
        lang_summary["tokens"] += document["tokens"]
    return summary


def prepare_pretrain(args, output_root, tokenizer):
    chinese_target = round(args.pretrain_tokens * args.zh_token_ratio)
    english_target = args.pretrain_tokens - chinese_target
    chinese_documents = collect_documents(
        iter_fineweb_chinese(args), tokenizer, chinese_target, args.doc_cap, "zh",
    )
    english_documents = collect_documents(
        iter_local_english(args.english_data_dir), tokenizer, english_target, args.doc_cap, "en",
    )
    rng = random.Random(args.seed)
    rng.shuffle(chinese_documents)
    rng.shuffle(english_documents)
    zh_val_count = max(2, round(len(chinese_documents) * args.val_fraction))
    en_val_count = max(1, round(len(english_documents) * args.val_fraction))
    zh_val_documents = chinese_documents[:zh_val_count]
    en_val_documents = english_documents[:en_val_count]
    train_documents = chinese_documents[zh_val_count:] + english_documents[en_val_count:]
    val_documents = zh_val_documents + en_val_documents
    rng.shuffle(train_documents)
    rng.shuffle(val_documents)
    pretrain_dir = os.path.join(output_root, "pretrain")
    os.makedirs(pretrain_dir, exist_ok=True)

    shard_index = 0
    shard_documents = []
    shard_tokens = 0
    train_paths = []
    for document in train_documents:
        shard_documents.append(document)
        shard_tokens += document["tokens"]
        if shard_tokens >= args.tokens_per_shard:
            path = os.path.join(pretrain_dir, f"shard_{shard_index:05d}.parquet")
            write_parquet(path, shard_documents)
            train_paths.append(path)
            shard_index += 1
            shard_documents = []
            shard_tokens = 0
    if shard_documents:
        path = os.path.join(pretrain_dir, f"shard_{shard_index:05d}.parquet")
        write_parquet(path, shard_documents)
        train_paths.append(path)

    val_path = os.path.join(pretrain_dir, "shard_99999.parquet")
    write_parquet(val_path, val_documents)

    zh_eval_dir = os.path.join(output_root, "pretrain_zh_eval")
    os.makedirs(zh_eval_dir, exist_ok=True)
    zh_eval_split = len(zh_val_documents) // 2
    zh_eval_train = zh_val_documents[:zh_eval_split]
    zh_eval_val = zh_val_documents[zh_eval_split:]
    zh_eval_train_path = os.path.join(zh_eval_dir, "shard_00000.parquet")
    zh_eval_val_path = os.path.join(zh_eval_dir, "shard_99999.parquet")
    write_parquet(zh_eval_train_path, zh_eval_train)
    write_parquet(zh_eval_val_path, zh_eval_val)
    return {
        "chinese_source": {
            "dataset": args.fineweb_dataset,
            "config": args.fineweb_config,
            "revision": args.fineweb_revision,
            "split": args.fineweb_split,
        },
        "english_data_dir": args.english_data_dir,
        "requested_tokens": args.pretrain_tokens,
        "requested_zh_token_ratio": args.zh_token_ratio,
        "train": summarize_documents(train_documents),
        "validation": summarize_documents(val_documents),
        "train_shards": train_paths,
        "validation_shard": val_path,
        "validation_sha256": file_sha256(val_path),
        "chinese_eval": {
            "data_dir": zh_eval_dir,
            "train": summarize_documents(zh_eval_train),
            "validation": summarize_documents(zh_eval_val),
            "validation_sha256": file_sha256(zh_eval_val_path),
        },
    }


def main():
    base_dir = get_base_dir()
    parser = argparse.ArgumentParser(description="Prepare data for the NanoChat Chinese experiment")
    parser.add_argument("--prepare", choices=["all", "sft", "pretrain"], default="all")
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "zh_experiment"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--sft-dataset", default="shibing624/alpaca-zh")
    parser.add_argument("--sft-config", default=None)
    parser.add_argument("--sft-revision", default="main")
    parser.add_argument("--sft-split", default="train")
    parser.add_argument("--sft-val-rows", type=int, default=1000)
    parser.add_argument("--sft-max-rows", type=int, default=-1)
    parser.add_argument("--sft-min-cjk-ratio", type=float, default=0.20)
    parser.add_argument("--sft-sequence-len", type=int, default=512)

    parser.add_argument("--fineweb-dataset", default="HuggingFaceFW/fineweb-2")
    parser.add_argument("--fineweb-config", default="cmn_Hani")
    parser.add_argument("--fineweb-revision", default="main")
    parser.add_argument("--fineweb-split", default="train")
    parser.add_argument("--english-data-dir", default=os.path.join(base_dir, "base_data_climbmix"))
    parser.add_argument("--pretrain-tokens", type=int, default=20_000_000)
    parser.add_argument("--zh-token-ratio", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--tokens-per-shard", type=int, default=5_000_000)
    parser.add_argument("--doc-cap", type=int, default=10_000)
    parser.add_argument("--streaming-shuffle-buffer", type=int, default=10_000)
    args = parser.parse_args()

    if not 0 < args.zh_token_ratio < 1:
        parser.error("--zh-token-ratio must be between 0 and 1")
    if not 0 < args.val_fraction < 1:
        parser.error("--val-fraction must be between 0 and 1")
    if not 0 <= args.sft_min_cjk_ratio <= 1:
        parser.error("--sft-min-cjk-ratio must be between 0 and 1")

    output_root = os.path.abspath(os.path.expanduser(args.output_dir))
    manifest_path = os.path.join(output_root, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        raise FileExistsError(f"{manifest_path} already exists; pass --overwrite to regenerate")
    if args.overwrite:
        selected_dirs = {
            "all": ["sft", "pretrain", "pretrain_zh_eval"],
            "sft": ["sft"],
            "pretrain": ["pretrain", "pretrain_zh_eval"],
        }[args.prepare]
        for dirname in selected_dirs:
            shutil.rmtree(os.path.join(output_root, dirname), ignore_errors=True)
    os.makedirs(output_root, exist_ok=True)

    tokenizer = get_tokenizer()
    tokenizer_path = os.path.join(base_dir, "tokenizer", "tokenizer.pkl")
    manifest = {
        "seed": args.seed,
        "tokenizer": {
            "path": tokenizer_path,
            "vocab_size": tokenizer.get_vocab_size(),
            "sha256": file_sha256(tokenizer_path),
        },
    }
    if args.prepare in {"all", "sft"}:
        manifest["sft"] = prepare_sft(args, output_root, tokenizer)
    if args.prepare in {"all", "pretrain"}:
        manifest["pretrain"] = prepare_pretrain(args, output_root, tokenizer)

    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
