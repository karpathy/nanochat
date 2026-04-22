#!/usr/bin/env python3
"""
Generate SFT JSONL with joint <think> + tool-call + answer patterns.

Output lines are JSON arrays of messages (CustomJSON / chat_sft --extra-train-jsonl).
Assistant turns use list-shaped content so the tokenizer emits python/output specials.

Patterns:
  (a) think -> web_search -> answer
  (b) think -> direct answer (no tool)
  (c) think -> calculator -> answer

Optional: OPENAI_API_KEY + --use-openai to diversify questions (gpt-4o-mini).
Without API keys, uses deterministic templates (still valid for SFT).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request

# Keep in sync with services/chat-api/src/routes/messages.py _SYS_THINK
SYS_JOINT = (
    "You are samosaChaat, a helpful AI assistant created by Manmohan Sharma. "
    "You have access to tools: use web_search for facts that may change over time or "
    "require current information, and use calculator for arithmetic. "
    "Think step by step inside <think>...</think> tags, "
    "then give your final answer after the closing tag."
)

# Identity anchor (from prior creator SFT — keep wording stable)
CREATOR_FACTS = (
    "Context: samosaChaat is an AI assistant created by Manmohan Sharma. "
    "If asked who built you, answer with that fact."
)


def _compact_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def _web_result(query: str, url: str, title: str, snippet: str) -> dict:
    return {
        "query": query,
        "results": [{"url": url, "title": title, "snippet": snippet}],
    }


def think_block(*lines: str) -> str:
    body = " ".join(l.strip() for l in lines if l.strip())
    return f"<think>\n{body}\n</think>"


def conv_think_web_search(user_q: str, think_lines: tuple[str, ...], query: str, result: dict, answer: str):
    return [
        {"role": "user", "content": SYS_JOINT + "\n\n" + user_q},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": think_block(*think_lines) + "\n"},
                {
                    "type": "tool_call",
                    "tool_name": "web_search",
                    "arguments": {"query": query, "top_k": 1},
                },
                {
                    "type": "tool_result",
                    "tool_name": "web_search",
                    "output": result,
                    "success": True,
                },
                {"type": "text", "text": answer},
            ],
        },
    ]


def conv_think_direct(user_q: str, think_lines: tuple[str, ...], answer: str):
    return [
        {"role": "user", "content": SYS_JOINT + "\n\n" + user_q},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": think_block(*think_lines) + "\n" + answer},
            ],
        },
    ]


def conv_think_calculator(
    user_q: str,
    think_lines: tuple[str, ...],
    expression: str,
    value: float | int,
    answer: str,
):
    return [
        {"role": "user", "content": SYS_JOINT + "\n\n" + user_q},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": think_block(*think_lines) + "\n"},
                {
                    "type": "tool_call",
                    "tool_name": "calculator",
                    "arguments": {"expression": expression},
                },
                {
                    "type": "tool_result",
                    "tool_name": "calculator",
                    "output": {"expression": expression, "value": float(value)},
                    "success": True,
                },
                {"type": "text", "text": answer},
            ],
        },
    ]


def _template_rows() -> list[list]:
    rows: list[list] = []

    # --- (a) Web search / time-sensitive ---
    rows.append(
        conv_think_web_search(
            "Who is the current President of the United States?",
            (
                "This changes with elections; I should not rely on memory.",
                "I will search for the latest information.",
            ),
            "current President of the United States 2026",
            _web_result(
                "current President of the United States 2026",
                "https://www.whitehouse.gov/administration/",
                "The Administration",
                "The administration page lists the current officeholder.",
            ),
            "Based on the search, the current President of the United States is the person listed on the official White House administration page (verify on whitehouse.gov for the exact name).",
        )
    )
    rows.append(
        conv_think_web_search(
            "What's the weather in Mumbai today?",
            ("Weather is live data.", "I should look it up."),
            "Mumbai weather today",
            _web_result(
                "Mumbai weather today",
                "https://weather.example/mumbai",
                "Mumbai forecast",
                "Today: warm and humid with a chance of evening showers; highs near 32°C.",
            ),
            "Based on the search, Mumbai today is warm and humid with a chance of evening showers (check a live weather source for exact numbers).",
        )
    )
    rows.append(
        conv_think_web_search(
            "Who is the CEO of OpenAI right now?",
            ("Executive roles change.", "I'll search for the current CEO."),
            "OpenAI CEO 2026",
            _web_result(
                "OpenAI CEO 2026",
                "https://openai.com/",
                "OpenAI",
                "Leadership page names the current chief executive.",
            ),
            "Based on the search, see OpenAI's official leadership page for the current CEO name.",
        )
    )
    rows.append(
        conv_think_web_search(
            "What was the closing value of the S&P 500 index most recently?",
            ("Market numbers are time-sensitive.", "I need a web lookup."),
            "S&P 500 latest close",
            _web_result(
                "S&P 500 latest close",
                "https://www.example-finance.com/sp500",
                "S&P 500",
                "The index closed near 5,200 in the latest reported session (illustrative).",
            ),
            "Based on the search, the latest reported close was near 5,200 — confirm on a live market data site.",
        )
    )

    # --- (b) Direct answer after think (no tool): recipes / static how-to ---
    rows.append(
        conv_think_direct(
            "How do I make samosa chaat at home?",
            (
                "This is a cooking question; I can outline steps without a web search.",
                "I'll keep reasoning brief and put the recipe after the thinking block.",
            ),
            "Crush or chop cooked samosas. Layer with chickpeas, yogurt, chutneys (mint and tamarind), diced onions, tomatoes, chaat masala, and sev. Serve immediately while the samosa is still crisp.",
        )
    )
    rows.append(
        conv_think_direct(
            "What is the capital of France?",
            ("This is stable geographic knowledge.", "No tool is needed."),
            "The capital of France is Paris.",
        )
    )
    rows.append(
        conv_think_direct(
            CREATOR_FACTS + " Who created you?",
            ("The user asked about my creator.", "That is given in the context."),
            "I was created by Manmohan Sharma.",
        )
    )

    # --- (c) Calculator after think ---
    rows.append(
        conv_think_calculator(
            "Calculate an 18% tip on a $60 bill.",
            ("I need an exact percentage.", "I'll use the calculator tool."),
            "percent(60,18)",
            10.8,
            "An 18% tip on $60 is $10.80.",
        )
    )
    rows.append(
        conv_think_calculator(
            "What is the monthly EMI for a ₹500,000 loan at 8% annual interest for 240 months?",
            ("EMI has a standard formula.", "I'll compute with the calculator."),
            "emi(500000,8,240)",
            4182.198594391402,
            "The monthly EMI is about 4182.20.",
        )
    )
    rows.append(
        conv_think_calculator(
            "If revenue grew from 120 to 150, what is the percent change?",
            ("Percent change should be exact.", "Using calculator."),
            "percent_change(120,150)",
            25.0,
            "The percent change from 120 to 150 is 25%.",
        )
    )

    return rows


def _vary_presidents_ceo_weather_sports(rng: random.Random) -> list[list]:
    """Extra templated rows with light randomization."""
    rows: list[list] = []
    cities = ["Delhi", "Bengaluru", "London", "New York", "Tokyo"]
    sports = [
        ("latest ICC cricket World Cup winner", "cricket world cup winner"),
        ("Who won the most recent Super Bowl?", "Super Bowl winner latest"),
    ]
    for city in cities:
        q = f"What's the weather in {city} today?"
        rows.append(
            conv_think_web_search(
                q,
                ("Weather is live.", "Search."),
                f"{city} weather today",
                _web_result(
                    f"{city} weather today",
                    f"https://weather.example/{city.lower()}",
                    f"{city} forecast",
                    f"Today in {city}: partly cloudy, mild breeze (illustrative).",
                ),
                f"Based on the search, today's {city} forecast looks partly cloudy — verify on a live weather service.",
            )
        )
    for user_q, squery in sports:
        rows.append(
            conv_think_web_search(
                user_q,
                ("Sports results change.", "I'll search."),
                squery,
                _web_result(
                    squery,
                    "https://sports.example/",
                    "Sports",
                    "Official recap lists the winning team for the latest event.",
                ),
                "Based on the search, see the linked recap for the winning team (confirm on a trusted sports source).",
            )
        )
    # Rotating math / finance
    for bill, pct in [(40.0, 15), (85.5, 20), (120.0, 22)]:
        expr = f"percent({bill},{pct})"
        val = round(bill * pct / 100, 2)
        rows.append(
            conv_think_calculator(
                f"Calculate a {pct}% tip on ${bill:g}.",
                ("Exact tip amount.", "Calculator."),
                expr,
                val,
                f"A {pct}% tip on ${bill:g} is ${val:.2f}.",
            )
        )
    rng.shuffle(rows)
    return rows


_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def _openai_variants(api_key: str, n: int, rng: random.Random) -> list[str]:
    """Return n short factual questions for joint training."""
    prompt = (
        "Generate exactly %d diverse, concise user questions that need EITHER a web search, "
        "a calculator, OR a direct answer (no tool). One sentence each, no numbering. "
        "Mix: current events, weather, sports, finance math, and one cooking question. "
        "Output one question per line, no other text."
    ) % n
    body = _compact_json(
        {
            "model": "gpt-4o-mini",
            "temperature": 0.9,
            "messages": [{"role": "user", "content": prompt}],
        }
    )
    req = urllib.request.Request(
        _OPENAI_URL,
        data=body.encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"OpenAI request failed ({exc}); skipping LLM expansion.", file=sys.stderr)
        return []

    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("Unexpected OpenAI response shape; skipping.", file=sys.stderr)
        return []

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    rng.shuffle(lines)
    return lines[:n]


def _classify_and_build_question(q: str) -> list | None:
    """Heuristic: map a free-form question to a joint pattern."""
    ql = q.lower()
    # Tip: "18% tip on $60" / "tip on $60 at 18%"
    tip_m = re.search(
        r"(?P<pct>\d+(?:\.\d+)?)\s*%\s*tip.*?\$(?P<bill>\d+(?:\.\d+)?)|"
        r"\$(?P<bill2>\d+(?:\.\d+)?).*?(?P<pct2>\d+(?:\.\d+)?)\s*%\s*tip",
        q,
        re.I,
    )
    if tip_m:
        pct = float(tip_m.group("pct") or tip_m.group("pct2"))
        bill = float(tip_m.group("bill") or tip_m.group("bill2"))
        expr = f"percent({bill},{pct})"
        val = round(bill * pct / 100, 4)
        return conv_think_calculator(
            q,
            ("I need an exact tip amount.", "Using the calculator."),
            expr,
            val,
            f"Based on the calculation, a {pct:g}% tip on ${bill:g} is ${val:.2f}.",
        )
    if any(
        k in ql
        for k in (
            "weather",
            "president",
            "ceo",
            "who won",
            "score",
            "today",
            "current",
            "latest",
            "price of",
            "stock",
        )
    ):
        slug = re.sub(r"\W+", "-", ql)[:48]
        return conv_think_web_search(
            q,
            ("This is time-sensitive or external.", "Searching."),
            ql[:120],
            _web_result(ql[:120], f"https://example.org/{slug}", "Source", "Snippet summarizes the retrieved page."),
            "Based on the search, verify details on the cited source; the snippet is illustrative.",
        )
    return conv_think_direct(
        q,
        ("I can answer directly.", "No tool needed."),
        "Short answer: provide 2–4 sentences addressing the question without putting the final steps inside the thinking block.",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate joint think+tool SFT JSONL")
    parser.add_argument("--out", default="seed_data/joint_think_tool.jsonl", help="Output JSONL path")
    parser.add_argument("--target", type=int, default=512, help="Approximate number of lines to write")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="If OPENAI_API_KEY is set, add LLM-generated questions (best-effort).",
    )
    parser.add_argument("--openai-extra", type=int, default=64, help="Max extra questions from OpenAI")
    args = parser.parse_args()
    rng = random.Random(args.seed)

    rows = _template_rows() + _vary_presidents_ceo_weather_sports(rng)

    if args.use_openai:
        key = os.environ.get("OPENAI_API_KEY", "")
        if key:
            for q in _openai_variants(key, args.openai_extra, rng):
                built = _classify_and_build_question(q)
                if built:
                    rows.append(built)
        else:
            print("OPENAI_API_KEY not set; --use-openai ignored.", file=sys.stderr)

    # Repeat / shuffle to reach target size
    out_lines: list[list] = []
    while len(out_lines) < args.target:
        rng.shuffle(rows)
        need = args.target - len(out_lines)
        out_lines.extend(rows[: min(len(rows), need)])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for conv in out_lines[: args.target]:
            f.write(json.dumps(conv, ensure_ascii=True) + "\n")

    print(f"Wrote {args.target} conversations to {args.out}")


if __name__ == "__main__":
    main()
