"""
L3: Large Lookup Layers
Ref: arXiv:2601.21461v2

L3 generalizes token embeddings by placing per-token lookup tables inside
the decoder stack. Unlike MoE, routing is static (determined by token ID),
eliminating router training and load-balancing losses.
"""

from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lzw_allocation(token_sequences, vocab_size, n_emb, k_max):
    """
    Compute per-token embedding allocation using LZW-style frequency analysis.

    Scans token sequences LZW-style, counting n-gram frequencies and allocating
    embeddings to the last token of frequent n-grams. Every token starts with 1
    embedding, then we greedily add embeddings to tokens involved in the most
    frequent n-grams until we reach n_emb total.

    Args:
        token_sequences: list of token ID lists (training data sample)
        vocab_size: size of vocabulary
        n_emb: target total embeddings
        k_max: max embeddings per token
    Returns:
        alloc: list[int] of length vocab_size (embeddings per token)
    """
    assert n_emb >= vocab_size, f"n_emb ({n_emb}) must be >= vocab_size ({vocab_size})"

    # Count token frequencies across all sequences
    token_freq = Counter()
    for seq in token_sequences:
        for tok in seq:
            token_freq[tok] += 1

    # Also count bigram frequencies (LZW-style: last token of n-gram gets credit)
    bigram_freq = Counter()
    for seq in token_sequences:
        for i in range(len(seq) - 1):
            bigram_freq[seq[i + 1]] += 1  # credit goes to last token

    # Combine unigram and bigram frequencies
    combined_freq = Counter()
    for tok in range(vocab_size):
        combined_freq[tok] = token_freq.get(tok, 0) + bigram_freq.get(tok, 0)

    # Start with 1 embedding per token
    alloc = [1] * vocab_size
    remaining = n_emb - vocab_size

    if remaining <= 0:
        return alloc

    # Sort tokens by frequency (descending) for greedy allocation
    sorted_tokens = sorted(range(vocab_size), key=lambda t: combined_freq[t], reverse=True)

    # Greedily add embeddings to the most frequent tokens
    while remaining > 0:
        added_any = False
        for tok in sorted_tokens:
            if remaining <= 0:
                break
            if alloc[tok] < k_max:
                alloc[tok] += 1
                remaining -= 1
                added_any = True
        if not added_any:
            break  # all tokens at k_max, can't allocate more

    return alloc


def allocation_to_bounds(alloc):
    """
    Convert allocation array to cumulative bounds tensor.

    bounds[0] = 0, bounds[i] = bounds[i-1] + alloc[i-1]
    bounds[-1] = sum(alloc) = n_emb

    Args:
        alloc: list[int] of per-token allocation counts
    Returns:
        bounds: torch.LongTensor of shape [len(alloc) + 1]
    """
    bounds = [0]
    for a in alloc:
        bounds.append(bounds[-1] + a)
    return torch.tensor(bounds, dtype=torch.long)


class L3Layer(nn.Module):
    """
    L3 layer: per-token lookup table with attention-like aggregation.

    Forward pass (vectorized gather+pad approach):
    1. Look up bounds for each token, gather KV embeddings, pad to k_max
    2. Compute scores = K @ x_norm, mask invalid positions, softmax
    3. Aggregate: weighted sum of V embeddings
    4. Up-project, RMSNorm, concat with x, mix-project
    Returns the delta (added residually by caller).
    """

    def __init__(self, n_embd, n_emb, d_up, tie_kv=True):
        super().__init__()
        self.n_embd = n_embd
        self.n_emb = n_emb
        self.d_up = d_up
        self.tie_kv = tie_kv

        if tie_kv:
            # Single shared weight for both keys and values
            self.kv_weight = nn.Parameter(torch.empty(n_emb, n_embd))
        else:
            # Separate key and value weights
            self.k_weight = nn.Parameter(torch.empty(n_emb, n_embd))
            self.v_weight = nn.Parameter(torch.empty(n_emb, n_embd))

        # Up-project from d_emb (= n_embd when tied) to d_up
        self.w_up = nn.Linear(n_embd, d_up, bias=False)
        # Mix-project: concat(up_projected, x) -> n_embd
        self.w_mix = nn.Linear(d_up + n_embd, n_embd, bias=False)

        # Bounds buffer (set after LZW allocation)
        self.register_buffer("bounds", torch.zeros(1, dtype=torch.long), persistent=True)

    def set_bounds(self, bounds):
        """Register the precomputed bounds tensor as a buffer."""
        self.bounds = bounds

    def forward(self, x, token_ids):
        """
        Args:
            x: [B, T, n_embd] hidden states
            token_ids: [B, T] token IDs
        Returns:
            delta: [B, T, n_embd] to be added residually by caller
        """
        B, T, C = x.shape
        device = x.device

        # 1. RMSNorm the input
        x_norm = F.rms_norm(x, (C,))

        # 2. Gather KV embeddings using bounds + token_ids
        # Look up bounds for each token
        flat_ids = token_ids.reshape(-1)  # [B*T]
        starts = self.bounds[flat_ids]     # [B*T]
        ends = self.bounds[flat_ids + 1]   # [B*T]
        lengths = ends - starts            # [B*T]
        k_max = lengths.max().item()

        if k_max == 0:
            # Edge case: no embeddings for any token
            return torch.zeros_like(x)

        # Build index tensor [B*T, k_max] with valid indices and padding
        offsets = torch.arange(k_max, device=device).unsqueeze(0)  # [1, k_max]
        indices = starts.unsqueeze(1) + offsets                     # [B*T, k_max]
        mask = offsets < lengths.unsqueeze(1)                       # [B*T, k_max] True for valid

        # Clamp indices to valid range for gathering (masked ones will be zeroed out)
        indices = indices.clamp(0, self.n_emb - 1)

        # 3. Gather weights
        if self.tie_kv:
            kv = self.kv_weight[indices]  # [B*T, k_max, n_embd]
            k_emb = kv
            v_emb = kv
        else:
            k_emb = self.k_weight[indices]  # [B*T, k_max, n_embd]
            v_emb = self.v_weight[indices]  # [B*T, k_max, n_embd]

        # 4. Compute attention scores: K @ x_norm
        x_flat = x_norm.reshape(B * T, C)           # [B*T, C]
        scores = torch.bmm(k_emb, x_flat.unsqueeze(2)).squeeze(2)  # [B*T, k_max]

        # Mask invalid positions
        scores = scores.masked_fill(~mask, float('-inf'))

        # Softmax over valid positions
        weights = F.softmax(scores, dim=-1)  # [B*T, k_max]
        # Replace NaN from all-inf rows (shouldn't happen since min alloc=1, but safety)
        weights = weights.masked_fill(~mask, 0.0)

        # 5. Aggregate: weighted sum of V embeddings
        agg = torch.bmm(weights.unsqueeze(1), v_emb).squeeze(1)  # [B*T, n_embd]
        agg = agg.view(B, T, C)

        # 6. Up-project, RMSNorm, concat with x, mix-project
        up = self.w_up(agg)                    # [B, T, d_up]
        up = F.rms_norm(up, (self.d_up,))      # normalize
        cat = torch.cat([up, x], dim=-1)       # [B, T, d_up + n_embd]
        delta = self.w_mix(cat)                # [B, T, n_embd]

        return delta
