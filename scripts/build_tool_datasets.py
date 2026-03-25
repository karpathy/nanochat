"""
Build small local tool-use datasets for SFT/eval dry runs.

This is not intended to replace larger curated corpora. It creates schema-valid
seed data so tool routing, tokenization, and evaluation can be tested locally.
"""

import argparse
import json
import os


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def calculator_conversation(user_text, expression, value, final_text):
    return [
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me compute that exactly. "},
                {"type": "tool_call", "tool_name": "calculator", "arguments": {"expression": expression}},
                {"type": "tool_result", "tool_name": "calculator", "output": {"expression": expression, "value": value}},
                {"type": "text", "text": final_text},
            ],
        },
    ]


def web_search_conversation(user_text, query, url, final_text):
    return [
        {"role": "user", "content": user_text},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I should verify this with a fresh web lookup. "},
                {"type": "tool_call", "tool_name": "web_search", "arguments": {"query": query, "top_k": 1}},
                {
                    "type": "tool_result",
                    "tool_name": "web_search",
                    "output": {
                        "query": query,
                        "results": [
                            {
                                "url": url,
                                "title": "Source result",
                                "snippet": "Fresh page content retrieved for grounding.",
                            }
                        ],
                    },
                },
                {"type": "text", "text": final_text},
            ],
        },
    ]


def direct_conversation(user_text, answer_text):
    return [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": answer_text},
    ]


def build_train_rows():
    return [
        direct_conversation(
            "Explain what gradient descent is in one short paragraph.",
            "Gradient descent is an optimization method that repeatedly nudges model parameters in the direction that reduces error, using the gradient to decide both direction and magnitude of each update.",
        ),
        calculator_conversation(
            "What is 18% of 250?",
            "percent(250,18)",
            45.0,
            "18% of 250 is 45.",
        ),
        calculator_conversation(
            "If revenue grew from 120 to 150, what is the percent change?",
            "percent_change(120,150)",
            25.0,
            "The percent change from 120 to 150 is 25%.",
        ),
        calculator_conversation(
            "What is the monthly EMI for a 500000 loan at 8% annual interest over 240 months?",
            "emi(500000,8,240)",
            4182.198594391402,
            "The monthly EMI is about 4182.2.",
        ),
        web_search_conversation(
            "Find the official nanochat repository and give me the link.",
            "official nanochat repository",
            "https://github.com/karpathy/nanochat",
            "The official nanochat repository is https://github.com/karpathy/nanochat",
        ),
        web_search_conversation(
            "What do the Cloudflare Browser Rendering docs say about the markdown endpoint?",
            "Cloudflare Browser Rendering markdown endpoint docs",
            "https://developers.cloudflare.com/browser-rendering/rest-api/markdown-endpoint/",
            "Cloudflare documents a markdown endpoint for extracting rendered page content: https://developers.cloudflare.com/browser-rendering/rest-api/markdown-endpoint/",
        ),
    ]


def build_eval_rows():
    return [
        {
            "conversation": {
                "messages": [
                    {"role": "user", "content": "What is 12% of 250?"},
                    {"role": "assistant", "content": "45"},
                ]
            },
            "checks": {
                "must_call": "calculator",
                "must_not_call": ["web_search"],
                "answer_contains": ["45"],
            },
        },
        {
            "conversation": {
                "messages": [
                    {"role": "user", "content": "Find the official nanochat repository and cite the link."},
                    {"role": "assistant", "content": "https://github.com/karpathy/nanochat"},
                ]
            },
            "checks": {
                "must_call": "web_search",
                "citation_required": True,
                "answer_contains": ["github.com/karpathy/nanochat"],
            },
        },
        {
            "conversation": {
                "messages": [
                    {"role": "user", "content": "Explain what overfitting means in one sentence."},
                    {"role": "assistant", "content": "Overfitting means a model memorizes training patterns too closely and generalizes poorly to new data."},
                ]
            },
            "checks": {
                "must_not_call": ["web_search"],
                "answer_contains": ["generalizes poorly"],
            },
        },
    ]


def main():
    parser = argparse.ArgumentParser(description="Build seed tool-use datasets")
    parser.add_argument("--train-out", default="seed_data/tool_sft_seed.jsonl", help="Output JSONL for SFT conversations")
    parser.add_argument("--eval-out", default="seed_data/tool_eval_seed.jsonl", help="Output JSONL for eval/reward objects")
    args = parser.parse_args()

    train_rows = build_train_rows()
    eval_rows = build_eval_rows()
    write_jsonl(args.train_out, train_rows)
    write_jsonl(args.eval_out, eval_rows)
    print(f"Wrote {len(train_rows)} SFT rows to {args.train_out}")
    print(f"Wrote {len(eval_rows)} eval rows to {args.eval_out}")


if __name__ == "__main__":
    main()
