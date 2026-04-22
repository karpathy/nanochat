#!/usr/bin/env python3
"""
Filter reasoning SFT JSONL so thinking blocks stay format-clean:

  - Require a closed </think>
  - Require non-trivial text *after* the closing tag (the answer lives there)
  - Reject if the model likely put the final answer only inside the think block
    (heuristic: strong answer markers appear inside but post-think text is tiny)
"""

from __future__ import annotations

import argparse
import json
import re
import sys

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"

# If these appear inside the thinking span but the tail after </think> is short, drop the row.
INSIDE_THINK_ANSWER_HINTS = re.compile(
    r"(?i)\b(ingredients|instructions|step\s*1|method|preheat|^\s*\d+[\).\s])",
    re.MULTILINE,
)
STRONG_TAIL_NEEDLE = re.compile(r"(?i)\b(is|are|equals|result|answer|you can|first,|mix|serve)\b")


def _assistant_text(messages: list) -> str | None:
    if not messages or messages[-1].get("role") != "assistant":
        return None
    c = messages[-1].get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts) if parts else None
    return None


def _split_think(s: str) -> tuple[str | None, str | None]:
    if THINK_OPEN not in s or THINK_CLOSE not in s:
        return None, None
    try:
        inner_start = s.index(THINK_OPEN) + len(THINK_OPEN)
        close_idx = s.index(THINK_CLOSE, inner_start)
        inner = s[inner_start:close_idx]
        tail = s[close_idx + len(THINK_CLOSE) :]
        return inner, tail
    except ValueError:
        return None, None


def keep_conversation(messages: list, *, min_tail_chars: int) -> tuple[bool, str]:
    text = _assistant_text(messages)
    if not text:
        return False, "no_assistant_string"
    inner, tail = _split_think(text)
    if inner is None:
        return False, "missing_or_unclosed_think"
    tail_stripped = tail.strip() if tail else ""
    if len(tail_stripped) < min_tail_chars:
        return False, "short_tail"
    if INSIDE_THINK_ANSWER_HINTS.search(inner) and len(tail_stripped) < max(min_tail_chars, 80):
        return False, "answer_leaked_into_think"
    if len(tail_stripped) < 40 and not STRONG_TAIL_NEEDLE.search(tail_stripped):
        return False, "weak_tail"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Filter reasoning JSONL for think-block hygiene")
    parser.add_argument("input_jsonl")
    parser.add_argument("--out", required=True, help="Filtered JSONL output path")
    parser.add_argument("--min-tail-chars", type=int, default=25)
    parser.add_argument("--stats-every", type=int, default=10000)
    args = parser.parse_args()

    kept, total = 0, 0
    reasons: dict[str, int] = {}

    with open(args.input_jsonl, encoding="utf-8") as fin, open(args.out, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                messages = json.loads(line)
            except json.JSONDecodeError:
                reasons["bad_json"] = reasons.get("bad_json", 0) + 1
                continue
            if not isinstance(messages, list):
                reasons["not_list"] = reasons.get("not_list", 0) + 1
                continue
            ok, reason = keep_conversation(messages, min_tail_chars=args.min_tail_chars)
            if ok:
                fout.write(json.dumps(messages, ensure_ascii=True) + "\n")
                kept += 1
            else:
                reasons[reason] = reasons.get(reason, 0) + 1

            if args.stats_every and total % args.stats_every == 0:
                print(f"... processed {total} lines, kept {kept}", file=sys.stderr)

    print(f"Done. kept {kept}/{total} -> {args.out}", file=sys.stderr)
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"  dropped {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
