"""
Evaluate compression ratio of the tokenizer.
"""

from nanochat.config import Config
from nanochat.dataset import parquets_iter_batched
from nanochat.report import get_report
from nanochat.tokenizer.eval_fixtures import CODE_TEXT, KOREAN_TEXT, MATH_TEXT, NEWS_TEXT, SCIENCE_TEXT
from nanochat.tokenizer.rust_tokenizer import RustBPETokenizer
from nanochat.tokenizer.utils import get_tokenizer

# ANSI color codes
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"


def _build_tokenizer(name: str, base_dir: str) -> RustBPETokenizer:
    if name == "gpt2":
        return RustBPETokenizer.from_pretrained("gpt2")
    if name == "gpt4":
        return RustBPETokenizer.from_pretrained("cl100k_base")
    return get_tokenizer(base_dir=base_dir)


def _encode_text(tokenizer: RustBPETokenizer, label: str, text: str) -> dict[str, object]:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded)
    if decoded != text:
        raise ValueError(f"Tokenizer roundtrip failed for {label!r}")
    n_bytes = len(text.encode("utf-8"))
    return {"bytes": n_bytes, "tokens": len(encoded), "ratio": n_bytes / len(encoded)}


def _print_comparison(
    baseline_name: str,
    baseline_results: dict[str, dict[str, object]],
    ours_results: dict[str, dict[str, object]],
    all_text: list[tuple[str, str]],
):
    """Print comparison table between baseline tokenizer and ours."""
    print(f"\nComparison with {baseline_name}:")
    print("=" * 95)
    print(f"{'Text Type':<10} {'Bytes':<8} {baseline_name:<15} {'Ours':<15} {'Relative':<12} {'Better':<10}")
    print(f"{'':10} {'':8} {'Tokens':<7} {'Ratio':<7} {'Tokens':<7} {'Ratio':<7} {'Diff %':<12}")
    print("-" * 95)

    for name, _ in all_text:
        baseline_data = baseline_results[name]
        ours_data = ours_results[name]

        relative_diff = ((baseline_data["tokens"] - ours_data["tokens"]) / baseline_data["tokens"]) * 100

        if baseline_data["ratio"] > ours_data["ratio"]:
            baseline_color, ours_color = GREEN, RED
            better = baseline_name
            diff_color = RED
        elif ours_data["ratio"] > baseline_data["ratio"]:
            baseline_color, ours_color = RED, GREEN
            better = "Ours"
            diff_color = GREEN
        else:
            baseline_color, ours_color = "", ""
            better = "Tie"
            diff_color = ""

        print(
            f"{name:<10} {baseline_data['bytes']:<8} "
            f"{baseline_color}{baseline_data['tokens']:<7}{RESET} "
            f"{baseline_color}{baseline_data['ratio']:<7.2f}{RESET} "
            f"{ours_color}{ours_data['tokens']:<7}{RESET} "
            f"{ours_color}{ours_data['ratio']:<7.2f}{RESET} "
            f"{diff_color}{relative_diff:+7.1f}%{RESET}     "
            f"{better:<10}"
        )


def tokenizer_eval(config: Config) -> None:
    """Evaluate and compare tokenizer compression ratios against GPT-2 and GPT-4 baselines.

    Encodes a fixed set of text samples (news, Korean, code, math, science, and one
    batch each from the train and val splits) with the GPT-2, GPT-4, and our trained
    tokenizer, then prints a colored comparison table and logs the results to the report.

    Args:
        config: Resolved nanochat config. Uses ``config.common.base_dir`` to locate
            the dataset and the trained tokenizer.
    """
    # The tokenizer was trained on data from earlier shards, so it has seen this data
    train_docs = next(parquets_iter_batched(base_dir=config.common.base_dir, split="train"))
    train_text = "\n".join(train_docs)
    val_docs = next(parquets_iter_batched(base_dir=config.common.base_dir, split="val"))
    val_text = "\n".join(val_docs)

    all_text = [
        ("news", NEWS_TEXT),
        ("korean", KOREAN_TEXT),
        ("code", CODE_TEXT),
        ("math", MATH_TEXT),
        ("science", SCIENCE_TEXT),
        ("fwe-train", train_text),
    ]
    if val_text:
        all_text.append(("fwe-val", val_text))

    # Try out current default compared to GPT-2 and GPT-4 tokenizers
    tokenizer_results = {}
    vocab_sizes = {}

    for tokenizer_name in ["gpt2", "gpt4", "ours"]:
        tokenizer = _build_tokenizer(tokenizer_name, config.common.base_dir)
        vocab_sizes[tokenizer_name] = tokenizer.get_vocab_size()
        tokenizer_results[tokenizer_name] = {}

        for name, text in all_text:
            tokenizer_results[tokenizer_name][name] = _encode_text(tokenizer, name, text)

    # Print vocab sizes
    print("\nVocab sizes:")
    print(f"GPT-2: {vocab_sizes['gpt2']}")
    print(f"GPT-4: {vocab_sizes['gpt4']}")
    print(f"Ours: {vocab_sizes['ours']}")

    # Print comparisons
    _print_comparison("GPT-2", tokenizer_results["gpt2"], tokenizer_results["ours"], all_text)
    _print_comparison("GPT-4", tokenizer_results["gpt4"], tokenizer_results["ours"], all_text)

    # Log to report
    lines = []
    for baseline_name in ["GPT-2", "GPT-4"]:
        baseline_key = baseline_name.lower().replace("-", "")
        baseline_results = tokenizer_results[baseline_key]
        ours_results = tokenizer_results["ours"]
        lines.append(f"### Comparison with {baseline_name}")
        lines.append("")
        lines.append(
            "| Text Type | Bytes | "
            + baseline_name
            + " Tokens | "
            + baseline_name
            + " Ratio | Ours Tokens | Ours Ratio | Relative Diff % |"
        )
        lines.append("|-----------|-------|--------------|--------------|-------------|------------|-----------------|")
        for name, text in all_text:
            baseline_data = baseline_results[name]
            ours_data = ours_results[name]
            relative_diff = ((baseline_data["tokens"] - ours_data["tokens"]) / baseline_data["tokens"]) * 100
            lines.append(
                f"| {name} | {baseline_data['bytes']} | {baseline_data['tokens']} | {baseline_data['ratio']:.2f} | {ours_data['tokens']} | {ours_data['ratio']:.2f} | {relative_diff:+.1f}% |"
            )
        lines.append("")
    report_markdown = "\n".join(lines)
    get_report(base_dir=config.common.base_dir).log(
        section="Tokenizer evaluation",
        data=[
            report_markdown,
        ],
    )
