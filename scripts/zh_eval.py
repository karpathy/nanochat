"""
Deterministic Chinese/English response-language evaluation.

This intentionally measures language behavior, not factual correctness. It writes
all prompts and responses to JSON so qualitative comparisons remain possible.
"""

import argparse
import json
import os
import re

from nanochat.checkpoint_manager import load_model
from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, get_base_dir, print0
from nanochat.engine import Engine
from nanochat.tokenizer import IncrementalTextDecoder


ZH_TOPICS = [
    "你自己", "机器学习", "春天", "北京", "读书", "健康饮食", "太阳系", "互联网",
    "人工智能", "环境保护", "长城", "音乐", "团队合作", "时间管理", "开源软件",
    "数据库", "操作系统", "数学", "历史", "旅行",
]
ZH_TEMPLATES = [
    "请用中文简单介绍{topic}。",
    "请用两句话解释{topic}。",
    "围绕{topic}写一段简短说明。",
    "如果向初学者介绍{topic}，你会怎么说？",
    "请列出关于{topic}的三个要点。",
]
EN_TOPICS = [
    "yourself", "machine learning", "spring", "Paris", "reading", "healthy food",
    "the solar system", "the internet", "artificial intelligence", "environmental protection",
    "the Great Wall", "music", "teamwork", "time management", "open-source software",
    "databases", "operating systems", "mathematics", "history", "travel",
]
EN_TEMPLATES = [
    "Briefly introduce {topic} in English.",
    "Explain {topic} in two English sentences.",
    "Write a short English description of {topic}.",
    "How would you introduce {topic} to a beginner in English?",
    "List three key points about {topic} in English.",
]


def build_prompts():
    prompts = []
    for topic in ZH_TOPICS:
        prompts.extend({"language": "zh", "prompt": template.format(topic=topic)} for template in ZH_TEMPLATES)
    for topic in EN_TOPICS:
        prompts.extend({"language": "en", "prompt": template.format(topic=topic)} for template in EN_TEMPLATES)
    return prompts


def response_metrics(text):
    cjk_count = sum(1 for char in text if 0x3400 <= ord(char) <= 0x9FFF)
    latin_count = sum(1 for char in text if ("a" <= char.lower() <= "z"))
    language_chars = cjk_count + latin_count
    cjk_ratio = cjk_count / language_chars if language_chars else 0.0
    return {
        "cjk_chars": cjk_count,
        "latin_chars": latin_count,
        "cjk_ratio": cjk_ratio,
        "contains_replacement_character": "\ufffd" in text,
        "non_whitespace_chars": len(re.sub(r"\s", "", text)),
    }


def render_prompt(tokenizer, prompt):
    return [
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<|user_start|>"),
        *tokenizer.encode(prompt),
        tokenizer.encode_special("<|user_end|>"),
        tokenizer.encode_special("<|assistant_start|>"),
    ]


def summarize(records):
    zh_records = [record for record in records if record["language"] == "zh"]
    en_records = [record for record in records if record["language"] == "en"]
    replacement_count = sum(record["metrics"]["contains_replacement_character"] for record in records)
    zh_dominant = sum(
        record["metrics"]["cjk_chars"] > 0 and record["metrics"]["cjk_ratio"] >= 0.5
        for record in zh_records
    )
    en_dominant = sum(
        record["metrics"]["latin_chars"] > 0 and record["metrics"]["cjk_ratio"] < 0.5
        for record in en_records
    )
    return {
        "total_prompts": len(records),
        "zh_prompts": len(zh_records),
        "en_prompts": len(en_records),
        "zh_dominant_responses": zh_dominant,
        "zh_response_rate": zh_dominant / len(zh_records),
        "en_dominant_responses": en_dominant,
        "en_response_rate": en_dominant / len(en_records),
        "replacement_character_responses": replacement_count,
        "mean_zh_cjk_ratio": sum(record["metrics"]["cjk_ratio"] for record in zh_records) / len(zh_records),
        "mean_en_cjk_ratio": sum(record["metrics"]["cjk_ratio"] for record in en_records) / len(en_records),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Chinese/English response language")
    parser.add_argument("-i", "--source", choices=["base", "sft", "rl"], default="sft")
    parser.add_argument("-g", "--model-tag", default=None)
    parser.add_argument("-s", "--step", type=int, default=None)
    parser.add_argument("--device-type", choices=["cuda", "cpu", "mps"], default="")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device_type = autodetect_device_type() if not args.device_type else args.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, meta = load_model(
        args.source, device, phase="eval", model_tag=args.model_tag, step=args.step,
    )
    engine = Engine(model, tokenizer)

    prompts = build_prompts()
    records = []
    for index, item in enumerate(prompts, 1):
        prompt_tokens = render_prompt(tokenizer, item["prompt"])
        results, _ = engine.generate_batch(
            prompt_tokens,
            num_samples=1,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_k=50,
            seed=42,
        )
        response_decoder = IncrementalTextDecoder(tokenizer)
        response = "".join(
            response_decoder.push(token_id)
            for token_id in results[0][len(prompt_tokens):]
        )
        record = {
            **item,
            "response": response,
            "metrics": response_metrics(response),
        }
        records.append(record)
        print0(
            f"[{index:03d}/{len(prompts)}] {item['language']} "
            f"cjk={record['metrics']['cjk_ratio']:.2f} {response[:80]!r}"
        )

    report = {
        "source": args.source,
        "model_tag": args.model_tag,
        "requested_step": args.step,
        "loaded_step": meta["step"],
        "summary": summarize(records),
        "records": records,
    }
    output_path = args.output
    if output_path is None:
        tag = args.model_tag or "auto"
        output_path = os.path.join(
            get_base_dir(), "zh_experiment", "eval",
            f"{args.source}_{tag}_{meta['step']:06d}.json",
        )
    output_path = os.path.abspath(os.path.expanduser(output_path))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print0(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print0(f"Evaluation written to {output_path}")
    compute_cleanup()


if __name__ == "__main__":
    main()
