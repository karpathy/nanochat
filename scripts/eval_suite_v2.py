#!/usr/bin/env python3
"""
Probe-style eval for chat SFT (tools + thinking).

Example:
  TAG=d24-sft-r6 STEP=754 SOURCE=sft WITH_TOOLS=1 \\
    python -m scripts.eval_suite_v2

Env:
  TAG          — model_tag directory under chatsft_checkpoints (or base_checkpoints if SOURCE=base)
  STEP         — checkpoint step (int)
  SOURCE       — base | sft | rl  (default sft)
  WITH_TOOLS   — 1 to run Engine with default tool registry (default 1)
  DEVICE       — optional cuda|cpu|mps override
"""

from __future__ import annotations

import json
import os
import re
import sys

from nanochat.common import autodetect_device_type, compute_cleanup, compute_init, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.tools import TOOL_CALL_END, TOOL_CALL_START, build_default_tool_registry, parse_tool_call_payload

TOOL_BLOCK_RE = re.compile(re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL)
THINK_CLOSE = "</think>"

# Mirrors services/chat-api thinking-mode prefix
SYS_THINK = (
    "You are samosaChaat, a helpful AI assistant created by Manmohan Sharma. "
    "You have access to tools: use web_search for facts that may change over time or "
    "require current information, and use calculator for arithmetic. "
    "Think step by step inside <think>...</think> tags, "
    "then give your final answer after the closing tag."
)


def _tool_calls(assistant_response: str) -> list[str]:
    calls = []
    for payload in TOOL_BLOCK_RE.findall(assistant_response):
        try:
            inv = parse_tool_call_payload(payload)
            calls.append(inv.tool_name)
        except Exception:
            continue
    return calls


def _after_think(text: str) -> str | None:
    if THINK_CLOSE not in text:
        return None
    return text.split(THINK_CLOSE, 1)[1]


def probe_reward(checks: dict, assistant_response: str) -> float:
    total = 0.0
    passed = 0.0
    tool_calls = _tool_calls(assistant_response)

    must_call = checks.get("must_call")
    if must_call:
        total += 1.0
        passed += float(must_call in tool_calls)

    for tool_name in checks.get("must_not_call", []):
        total += 1.0
        passed += float(tool_name not in tool_calls)

    for needle in checks.get("answer_contains", []):
        total += 1.0
        passed += float(needle in assistant_response)

    answer_regex = checks.get("answer_regex")
    if answer_regex:
        total += 1.0
        passed += float(re.search(answer_regex, assistant_response) is not None)

    if checks.get("citation_required", False):
        total += 1.0
        passed += float(("http://" in assistant_response) or ("https://" in assistant_response))

    if checks.get("must_close_think", False):
        total += 1.0
        passed += float(THINK_CLOSE in assistant_response)

    min_after = checks.get("min_chars_after_think")
    if min_after:
        total += 1.0
        tail = _after_think(assistant_response)
        passed += float(tail is not None and len(tail.strip()) >= int(min_after))

    for needle in checks.get("answer_after_think_contains", []):
        total += 1.0
        tail = _after_think(assistant_response)
        passed += float(tail is not None and needle in tail)

    if checks.get("forbid_answer_needles_inside_think_only", False):
        # If needles appear only before </think>, fail (recipe trapped in think)
        total += 1.0
        if THINK_CLOSE in assistant_response:
            head, _, tail = assistant_response.partition(THINK_CLOSE)
            needles = checks.get("_forbid_needles", ("Ingredients", "Step 1", "samosa"))
            bad = any(n in head and n not in tail for n in needles)
            passed += float(not bad)
        else:
            passed += 0.0

    if total == 0.0:
        return 0.0
    return passed / total


def default_probes() -> list[dict]:
    return [
        {
            "name": "president_web_search",
            "category": "think_plus_tool",
            "conversation": {
                "messages": [
                    {
                        "role": "user",
                        "content": SYS_THINK + "\n\nWho is the current president of America?",
                    },
                ]
            },
            "checks": {
                "must_call": "web_search",
                "must_close_think": True,
                "min_chars_after_think": 20,
            },
        },
        {
            "name": "mumbai_weather_web_search",
            "category": "think_plus_tool",
            "conversation": {
                "messages": [
                    {
                        "role": "user",
                        "content": SYS_THINK + "\n\nWhat's the weather in Mumbai today?",
                    },
                ]
            },
            "checks": {
                "must_call": "web_search",
                "must_close_think": True,
            },
        },
        {
            "name": "samosa_chaat_answer_after_think",
            "category": "think_plus_tool",
            "conversation": {
                "messages": [
                    {
                        "role": "user",
                        "content": SYS_THINK + "\n\nHow do I make samosa chaat?",
                    },
                ]
            },
            "checks": {
                "must_close_think": True,
                "min_chars_after_think": 60,
                "answer_after_think_contains": ["samosa"],
                "forbid_answer_needles_inside_think_only": True,
                "_forbid_needles": ("Ingredients", "Step 1", "yogurt"),
            },
        },
        {
            "name": "tip_calculator",
            "category": "think_plus_tool",
            "conversation": {
                "messages": [
                    {
                        "role": "user",
                        "content": SYS_THINK + "\n\nCalculate an 18% tip on a $60 bill.",
                    },
                ]
            },
            "checks": {
                "must_call": "calculator",
                "must_close_think": True,
                "answer_regex": r"10\.8",
            },
        },
    ]


def run_probes(
    *,
    tokenizer,
    engine: Engine,
    probes: list[dict],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[float, dict[str, list[tuple[str, float]]]]:
    by_cat: dict[str, list[tuple[str, float]]] = {}
    scores: list[float] = []

    for probe in probes:
        name = probe.get("name", "unknown")
        category = probe.get("category", "default")
        conv = probe["conversation"]
        checks = probe.get("checks", {})

        encoded = tokenizer.render_for_completion(conv)
        results, _ = engine.generate_batch(
            encoded,
            num_samples=1,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=42,
        )
        completion = tokenizer.decode(results[0][len(encoded) :])
        r = probe_reward(checks, completion)
        scores.append(r)
        by_cat.setdefault(category, []).append((name, r))
        print0(f"[{category}] {name}: reward={r:.3f}\n---\n{completion[:1200]}\n---")

    return sum(scores) / max(len(scores), 1), by_cat


def main():
    device_type = os.environ.get("DEVICE", "") or autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    if ddp_rank != 0:
        return

    tag = os.environ.get("TAG")
    step_s = os.environ.get("STEP")
    source = os.environ.get("SOURCE", "sft")
    if not tag or step_s is None:
        print("Set TAG and STEP in the environment.", file=sys.stderr)
        sys.exit(2)
    step = int(step_s)
    with_tools = os.environ.get("WITH_TOOLS", "1") not in ("0", "false", "False")

    probes = default_probes()
    extra_path = os.environ.get("EVAL_SUITE_EXTRA_JSONL", "")
    if extra_path and os.path.exists(extra_path):
        with open(extra_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    probes.append(json.loads(line))

    model, tokenizer, meta = load_model(source, device, phase="eval", model_tag=tag, step=step)
    tools = build_default_tool_registry() if with_tools else None
    engine = Engine(model, tokenizer, tools=tools)

    mean, by_cat = run_probes(
        tokenizer=tokenizer,
        engine=engine,
        probes=probes,
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "512")),
        temperature=float(os.environ.get("TEMPERATURE", "0.2")),
        top_k=int(os.environ.get("TOP_K", "50")),
    )

    print0("=" * 60)
    print0(f"Overall mean reward: {mean:.4f}")
    for cat, rows in sorted(by_cat.items()):
        m = sum(r for _, r in rows) / len(rows)
        print0(f"  {cat}: {m:.4f} ({len(rows)} probes)")
    compute_cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        compute_cleanup()
