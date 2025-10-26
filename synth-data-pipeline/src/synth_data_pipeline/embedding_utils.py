"""
Embedding utilities for conversation deduplication.

This module provides functions for:
1. Generating embeddings using OpenAI API
2. Computing cosine similarity between conversations
3. Deduplicating based on similarity threshold
"""

import asyncio
from typing import List
import numpy as np
from openai import AsyncOpenAI
import logfire


async def batch_embed(
    texts: List[str],
    client: AsyncOpenAI,
    model: str = "text-embedding-3-small",
    dimensions: int = 1024,
    batch_size: int = 100,
    max_concurrent: int = 20,
) -> List[np.ndarray]:
    """
    Generate embeddings for a list of texts using OpenAI API.

    Args:
        texts: List of text strings to embed
        client: AsyncOpenAI client instance
        model: OpenAI embedding model name
        dimensions: Number of dimensions for embeddings
        batch_size: Number of texts per API call
        max_concurrent: Maximum concurrent API calls

    Returns:
        List of numpy arrays (embeddings)
    """
    logfire.info(f"Embedding {len(texts)} texts in batches of {batch_size}")

    # Split into batches
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

    # Semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)

    async def embed_batch(batch: List[str]) -> List[List[float]]:
        """Embed a single batch of texts."""
        async with semaphore:
            try:
                response = await client.embeddings.create(
                    model=model, input=batch, dimensions=dimensions
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logfire.error(f"Error embedding batch: {e}")
                raise

    # Process all batches concurrently
    all_embeddings = []
    tasks = [embed_batch(batch) for batch in batches]

    batch_results = await asyncio.gather(*tasks)

    # Flatten results
    for batch_embeds in batch_results:
        all_embeddings.extend(batch_embeds)

    # Convert to numpy arrays
    embeddings_np = [np.array(emb, dtype=np.float32) for emb in all_embeddings]

    logfire.info(f"Generated {len(embeddings_np)} embeddings")
    return embeddings_np


def l2_normalize(embeddings: List[np.ndarray]) -> List[np.ndarray]:
    """
    L2-normalize embeddings for cosine similarity via dot product.

    Args:
        embeddings: List of numpy arrays

    Returns:
        List of L2-normalized numpy arrays
    """
    normalized = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized.append(emb / norm)
        else:
            normalized.append(emb)
    return normalized


def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two L2-normalized embeddings.

    Args:
        emb1: First embedding (L2-normalized)
        emb2: Second embedding (L2-normalized)

    Returns:
        Cosine similarity (0-1, where 1 is identical)
    """
    return float(np.dot(emb1, emb2))


def greedy_deduplicate(
    embeddings: List[np.ndarray],
    scores: List[float],
    similarity_threshold: float = 0.95,
) -> List[int]:
    """
    Greedy deduplication: keep highest-scoring items, remove similar duplicates.

    Args:
        embeddings: List of L2-normalized embeddings
        scores: List of quality scores (higher is better)
        similarity_threshold: Similarity threshold for considering items duplicates

    Returns:
        List of indices to KEEP (deduplicated)
    """
    # Create indices sorted by score (descending)
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    kept_indices = []
    kept_embeddings = []

    for idx in sorted_indices:
        emb = embeddings[idx]

        # Check similarity against all kept items
        is_duplicate = False
        for kept_emb in kept_embeddings:
            similarity = compute_similarity(emb, kept_emb)
            if similarity >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_indices.append(idx)
            kept_embeddings.append(emb)

    # Return indices in original order
    kept_indices.sort()

    logfire.info(
        f"Deduplication: {len(kept_indices)} kept, {len(embeddings) - len(kept_indices)} removed "
        f"({100 * (len(embeddings) - len(kept_indices)) / len(embeddings):.1f}% removed)"
    )

    return kept_indices


def conversation_to_text(messages: List[dict], max_chars: int = 24000) -> str:
    """
    Convert conversation messages to a single text string for embedding.

    Args:
        messages: List of message dicts with 'role' and 'content'
        max_chars: Maximum characters to include

    Returns:
        Concatenated text string
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")

    full_text = "\n\n".join(parts)

    # Truncate if too long
    if len(full_text) > max_chars:
        full_text = full_text[:max_chars]

    return full_text
