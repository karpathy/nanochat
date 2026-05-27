"""
Clarinet's BOS-aligned best-fit dataloader.

Differences from nanochat.dataloader.tokenizing_distributed_data_loader_with_state_bos_bestfit:

  1. Documents come from TWO parquet sources — climbmix (general) and the
     reasoning corpus (currently FineMath) — interleaved by a deterministic
     round-robin that honors `reasoning_mix_ratio`. All DDP ranks see the
     same source-ordering so per-rank row-group sharding stays well-defined.

  2. After BOS-prepending, the source-marker token (<|src_reasoning|>,
     <|src_general|>, or <|src_unknown|>) is spliced in at position 1 of
     each tokenized document. With probability `p_uncond`, the true marker
     is replaced with <|src_unknown|> — the CFG-style unconditional dropout
     that makes the unconditional pass at inference well-defined.

  3. Any target whose INPUT position holds <|bos|> is masked to -1.
     That makes the model never train to predict the source-marker (the
     token immediately after BOS in every doc) — we want the marker to
     condition generation, not to be a generation target itself.

Everything else (best-fit packing, DDP row-group sharding, pinned CPU
staging, GPU single-HtoD copy) matches upstream exactly.
"""

import random

import pyarrow.parquet as pq
import torch

from nanochat.common import get_dist_info

from clarinet.dataset import list_parquet_files_with_source


SRC_REASONING = "<|src_reasoning|>"
SRC_GENERAL = "<|src_general|>"
SRC_UNKNOWN = "<|src_unknown|>"


def _interleave_sources(paths_with_source, reasoning_mix_ratio):
    """
    Deterministic round-robin interleave of two source lists into a single
    ordered list, hitting `reasoning_mix_ratio` exactly over each window of
    1 / gcd(ratio_numerator, denominator) files. Returns [(path, is_reasoning)].

    A deterministic schedule (no RNG) keeps all DDP ranks aligned without
    needing a shared seed.
    """
    # Pull just the paths — paths_with_source entries are (path, is_reasoning)
    # tuples, and we re-wrap below. Failing to unwrap here used to produce
    # nested tuples that downstream pyarrow rejected.
    climbmix = [p for p, is_r in paths_with_source if not is_r]
    reasoning = [p for p, is_r in paths_with_source if is_r]
    if not climbmix or not reasoning:
        return list(paths_with_source)

    out = []
    c_idx = r_idx = 0
    # Schedule based on cumulative target ratio: at step k, the desired count
    # of reasoning files so far is round(k * ratio). When that increases, emit
    # a reasoning file; otherwise emit a climbmix file. This produces an
    # evenly-spread interleaving (no clumping).
    emitted_reasoning = 0
    step = 0
    while c_idx < len(climbmix) or r_idx < len(reasoning):
        step += 1
        want_reasoning = round(step * reasoning_mix_ratio)
        if (want_reasoning > emitted_reasoning and r_idx < len(reasoning)) or c_idx >= len(climbmix):
            out.append((reasoning[r_idx], True))
            r_idx += 1
            emitted_reasoning += 1
        else:
            out.append((climbmix[c_idx], False))
            c_idx += 1
    return out


def _document_batches(split, reasoning_mix_ratio, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over (text_batch, is_reasoning, (pq_idx, rg_idx, epoch)).

    pq_idx indexes into the interleaved (climbmix + reasoning) sequence.
    DDP sharding is done at the row-group level inside each parquet file,
    matching upstream.
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    paths_with_source = list_parquet_files_with_source(split)
    interleaved = _interleave_sources(paths_with_source, reasoning_mix_ratio)
    assert interleaved, "No parquet files found across either source"

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    epoch = resume_epoch

    while True:
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(interleaved):
            filepath, is_reasoning = interleaved[pq_idx]
            pf = pq.ParquetFile(filepath)
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column("text").to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i + tokenizer_batch_size], is_reasoning, (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def clarinet_data_loader(
    tokenizer, B, T, split,
    reasoning_mix_ratio=0.3,
    p_uncond=0.1,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    buffer_size=1000,
    seed=0,
):
    """
    BOS-aligned best-fit loader with clarinet source markers + target masking.

    Yields (inputs, targets, state_dict) where:
      - inputs, targets: (B, T) long tensors on `device`
      - state_dict: {"pq_idx", "rg_idx", "epoch"} for resume

    `seed` controls the per-doc p_uncond dropout RNG. We seed per rank so
    different ranks make independent dropout choices, which is fine because
    each rank handles its own row groups.
    """
    assert split in ("train", "val"), "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, reasoning_mix_ratio, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    src_reasoning_id = tokenizer.encode_special(SRC_REASONING)
    src_general_id = tokenizer.encode_special(SRC_GENERAL)
    src_unknown_id = tokenizer.encode_special(SRC_UNKNOWN)

    _, ddp_rank, _, _ = get_dist_info()
    rng = random.Random(seed + ddp_rank)

    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, is_reasoning, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        true_marker = src_reasoning_id if is_reasoning else src_general_id
        for tokens in token_lists:
            marker = src_unknown_id if rng.random() < p_uncond else true_marker
            # tokens already starts with BOS; insert the source marker right after it,
            # giving [BOS, marker, ...doc_tokens]
            tokens.insert(1, marker)
            doc_buffer.append(tokens)

    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        # Mask the marker-prediction targets: any position whose input is BOS
        # has the source marker as its next-token label; we don't want to train
        # the model to predict source.
        cpu_targets[cpu_inputs == bos_token] = -1

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict
