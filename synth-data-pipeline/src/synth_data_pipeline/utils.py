"""
Common utilities for the synthetic data pipeline.
"""

import asyncio
import json
from pathlib import Path
from typing import List, TypeVar, Callable, Awaitable

import logfire
from tqdm.asyncio import tqdm_asyncio

T = TypeVar('T')
R = TypeVar('R')


async def process_with_concurrency(
    items: List[T],
    process_fn: Callable[[T], Awaitable[R]],
    max_concurrent: int = 10,
    desc: str = "Processing"
) -> List[R]:
    """
    Process items concurrently with a semaphore to limit concurrency.

    Args:
        items: List of items to process
        process_fn: Async function to process each item
        max_concurrent: Maximum number of concurrent operations
        desc: Description for progress bar

    Returns:
        List of results (None entries filtered out)
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_process(item: T) -> R | None:
        async with semaphore:
            try:
                return await process_fn(item)
            except Exception as e:
                logfire.error(f"Error processing item: {e}", item=item)
                return None

    # Process all items with progress bar
    tasks = [bounded_process(item) for item in items]
    results = await tqdm_asyncio.gather(*tasks, desc=desc)

    # Filter out None results (errors)
    return [r for r in results if r is not None]


def save_jsonl(items: List, output_path: str | Path):
    """
    Save items to JSONL file.

    Args:
        items: List of items (must have model_dump_json method or be dicts)
        output_path: Path to output JSONL file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in items:
            if hasattr(item, 'model_dump_json'):
                f.write(item.model_dump_json() + '\n')
            else:
                f.write(json.dumps(item) + '\n')

    logfire.info(f"Saved {len(items)} items to {output_path}")


def load_jsonl(file_path: str | Path, model_class=None) -> List:
    """
    Load items from JSONL file.

    Args:
        file_path: Path to JSONL file
        model_class: Optional Pydantic model class to validate/parse items

    Returns:
        List of items
    """
    items = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if model_class:
                items.append(model_class.model_validate_json(line))
            else:
                items.append(json.loads(line))
    return items


def parse_markdown_chunks(
    file_path: str | Path,
    context_lines: int = 3
) -> List[dict]:
    """
    Parse markdown file and create chunks with context.

    Args:
        file_path: Path to the markdown file
        context_lines: Number of lines before/after to include as context

    Returns:
        List of dicts with 'source_text', 'context_before', 'context_after'
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    chunks = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines and metadata
        if not line or line.startswith('---') or line.startswith('**As of:**'):
            i += 1
            continue

        # Process bullets and significant lines
        if line.startswith('*') or line.startswith('#') or len(line) > 50:
            # Get context before
            context_start = max(0, i - context_lines)
            context_before = ''.join(lines[context_start:i]).strip()

            # Get the main text (current line)
            source_text = line

            # Get context after
            context_end = min(len(lines), i + context_lines + 1)
            context_after = ''.join(lines[i + 1:context_end]).strip()

            chunks.append({
                'source_text': source_text,
                'context_before': context_before,
                'context_after': context_after,
            })

        i += 1

    return chunks


def calculate_overall_score(
    factual_accuracy: float,
    naturalness: float,
    relevance: float,
    diversity: float,
    weights: dict = None
) -> float:
    """
    Calculate overall quality score from individual metrics.

    Args:
        factual_accuracy: Score 0-10
        naturalness: Score 0-10
        relevance: Score 0-10
        diversity: Score 0-10
        weights: Optional custom weights (defaults from config)

    Returns:
        Weighted overall score
    """
    if weights is None:
        from .config import QUALITY_WEIGHTS
        weights = QUALITY_WEIGHTS

    overall = (
        factual_accuracy * weights.get("factual_accuracy", 0.35) +
        naturalness * weights.get("naturalness", 0.25) +
        relevance * weights.get("relevance", 0.25) +
        diversity * weights.get("diversity", 0.15)
    )

    return round(overall, 2)


def print_sample(item, title: str = "SAMPLE"):
    """
    Print a sample item for inspection.

    Args:
        item: Item to print (conversation, Q&A, etc.)
        title: Title for the sample section
    """
    print("\n" + "="*80)
    print(title)
    print("="*80)

    if hasattr(item, 'model_dump'):
        # Pydantic model
        print(json.dumps(item.model_dump(), indent=2))
    elif isinstance(item, dict):
        print(json.dumps(item, indent=2))
    else:
        print(item)

    print("="*80 + "\n")


def print_statistics(scores: List[float], metric_name: str = "Score"):
    """
    Print statistics for a list of scores.

    Args:
        scores: List of numeric scores
        metric_name: Name of the metric being measured
    """
    if not scores:
        print(f"No {metric_name} data available")
        return

    avg = sum(scores) / len(scores)
    min_val = min(scores)
    max_val = max(scores)

    print(f"{metric_name:20s}: avg={avg:5.2f}, min={min_val:5.2f}, max={max_val:5.2f}")
