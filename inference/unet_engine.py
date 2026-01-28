"""
Efficient inference engine for UNet LLM model.

Key differences from GPT engine:
- Handles hierarchical encoder-decoder structure with pooling
- Caches encoder outputs at each stage for skip connections
- During T=1 generation, encoder pooling stops early (only stage 0 runs)
- KV cache structure accounts for different sequence lengths at each stage

Everything works with token sequences (no tokenization logic here).
"""

import torch
import torch.nn.functional as F
from collections import deque
from typing import List, Tuple, Optional


# -----------------------------------------------------------------------------
# UNet-specific KV Cache
# -----------------------------------------------------------------------------
class UNetKVCache:
    """
    KV Cache for UNet architecture with hierarchical stages.
    
    The UNet has stages with progressively pooled sequences:
    - Stage 0: seq_len
    - Stage 1: seq_len // 2
    - Stage 2: seq_len // 4
    etc.
    
    Key points:
    - Each layer gets its own K/V cache with stage-appropriate sequence length
    - During T=1 generation, only stage 0 processes (higher stages can't pool)
    - Skip connections use encoder outputs from the current forward pass
    """

    def __init__(self, batch_size, config, max_seq_len, device, dtype):
        self.batch_size = batch_size
        self.config = config
        self.n_stages = len(config.n_layer)
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        
        # Build flat KV cache with varying dimensions per stage
        # Map layer_idx -> (k_cache, v_cache) with stage-appropriate dimensions
        self.layer_caches = []
        self.layer_to_stage = []  # Track which stage each layer belongs to
        
        # Total number of layers across all encoder + decoder blocks
        self.n_layers = sum(config.n_layer)
        
        for stage_idx in range(self.n_stages):
            n_layer = config.n_layer[stage_idx]
            n_kv_head = config.n_kv_head[stage_idx]
            n_embd = config.n_embd[stage_idx]
            head_dim = n_embd // config.n_head[stage_idx]
            
            # Sequence length at this stage (gets pooled by 2^stage_idx)
            stage_seq_len = max_seq_len // (2 ** stage_idx)
            
            # Create caches for encoder blocks at this stage
            for _ in range(n_layer // 2):
                k = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim, 
                              device=device, dtype=dtype)
                v = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                self.layer_caches.append((k, v))
                self.layer_to_stage.append(stage_idx)
            
            # Create caches for decoder blocks at this stage
            for _ in range(n_layer // 2):
                k = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                v = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                self.layer_caches.append((k, v))
                self.layer_to_stage.append(stage_idx)
        
        # Track current position (in terms of original sequence, not pooled)
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) for a specific layer (GPT-compatible interface)."""
        return self.layer_caches[layer_idx]

    def get_pos(self):
        """Get current position in original (unpooled) sequence."""
        return self.cache_seqlens[0].item()

    def advance(self, num_tokens):
        """Advance cache position (called after each forward pass)."""
        self.cache_seqlens += num_tokens

    def reset(self):
        """Reset cache to empty state."""
        self.cache_seqlens.zero_()

    def prefill(self, other):
        """
        Copy cache from another UNetKVCache (for multi-sample generation).
        Used to replicate a single prefill across multiple parallel samples.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty cache"
        assert self.n_layers == other.n_layers
        
        other_pos = other.get_pos()
        
        # Copy KV caches for all layers
        for layer_idx in range(self.n_layers):
            stage_idx = self.layer_to_stage[layer_idx]
            # Calculate how many tokens were cached at this stage
            stage_pos = other_pos // (2 ** stage_idx)
            if stage_pos == 0:
                continue
            
            k_other, v_other = other.layer_caches[layer_idx]
            k_self, v_self = self.layer_caches[layer_idx]
            
            # Copy up to the current position at this stage
            k_self[:, :stage_pos] = k_other[:, :stage_pos]
            v_self[:, :stage_pos] = v_other[:, :stage_pos]
        
        # Copy position
        self.cache_seqlens.fill_(other_pos)


# -----------------------------------------------------------------------------
# Row state for tracking generation per sample
# -----------------------------------------------------------------------------
class RowState:
    """Per-row state tracking during generation."""
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or []
        self.forced_tokens = deque()  # Queue of tokens to force inject (for tool use)
        self.in_python_block = False
        self.python_expr_tokens = []
        self.completed = False


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)


# -----------------------------------------------------------------------------
# UNet Engine
# -----------------------------------------------------------------------------
def fix_unet_layer_indices(model):
    """
    Fix UNet layer indices to be globally unique for KV caching.
    
    The UNet model assigns layer_idx per-stage (0, 1, 2, ...), but KV cache
    needs globally unique indices. This function patches the model in-place
    after loading to make indices globally unique across all blocks.
    
    This allows us to use KV caching without modifying the core unet.py code.
    """
    global_layer_idx = 0
    for stage_idx in range(model.n_stage):
        # Fix encoder blocks
        for block in model.encoder[f"transformer_{stage_idx}"]:
            block.attn.layer_idx = global_layer_idx
            global_layer_idx += 1
        # Fix decoder blocks
        for block in model.decoder[f"transformer_{stage_idx}"]:
            block.attn.layer_idx = global_layer_idx
            global_layer_idx += 1


class UNetEngine:
    """
    Efficient inference engine for UNet LLM with KV caching.
    
    Optimizations:
    - Single prefill pass, then efficient single-token generation
    - KV caching for all transformer blocks (encoder + decoder)
    - Cached encoder outputs for skip connections during generation
    - Batch generation support
    """

    def __init__(self, model, tokenizer):
        # Fix layer indices for KV caching (must be globally unique)
        fix_unet_layer_indices(model)
        self.model = model
        self.tokenizer = tokenizer
    
    def _safe_forward(self, ids, kv_cache):
        """
        Safe wrapper around model.forward() that fixes T=1 edge case.
        
        UNet.forward() has a bug: when T=1, the encoder breaks early but last_stage_idx
        is set OUTSIDE the loop, capturing the wrong value. This causes dimension
        mismatches in the decoder. This method fixes it by tracking last_stage_idx correctly.
        
        For T>1, we use the original model.forward() to avoid any subtle differences.
        """
        B, T = ids.size()
        
        # Only use custom logic for T=1 (the edge case with the bug)
        # For T>1, use original forward (including prefill)
        if T > 1:
            return self.model.forward(ids, kv_cache=kv_cache)
        
        # T=1 case: use fixed forward logic
        model = self.model
        
        #  Get rotary embeddings
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = model.cos[:, T0:T0+T], model.sin[:, T0:T0+T]
        
        def pool_cos_sin(cos, sin, stage_idx):
            return (
                cos[:, (2 ** stage_idx - 1)::(2 ** stage_idx)],
                sin[:, (2 ** stage_idx - 1)::(2 ** stage_idx)]
            )
        
        def norm(x):
            return F.rms_norm(x, (x.size(-1),))
        
        # Input embed
        x = model.wte(ids)
        x = norm(x)
        
        # Encoder - track completed stage INSIDE loop (fix for T=1 bug)
        encoder_outputs = []
        last_stage_idx = 0
        for stage_idx in range(model.n_stage):
            if stage_idx > 0:
                if x.size(1) == 1:
                    break
                x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                x = norm(x)
            
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.encoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            encoder_outputs.append(x)
            last_stage_idx = stage_idx  # Track INSIDE loop (the fix!)
        
        # Decoder
        for stage_idx in reversed(range(last_stage_idx + 1)):
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            if stage_idx > 0:
                x = norm(x)
                x = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                y = encoder_outputs[stage_idx - 1]
                if x.size(1) == y.size(1):
                    x = x[:, :-1]
                shifted_x = torch.zeros_like(y)
                shifted_x[:, 1:] = x
                x = y + shifted_x
        
        # Forward the lm_head
        x = norm(x)
        softcap = 15
        logits = model.lm_head(x)
        logits = logits[..., :model.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Efficient batched generation with KV caching.
        
        Args:
            tokens: List of token IDs (prompt)
            num_samples: Number of parallel samples to generate
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            top_k: Top-k sampling
            seed: Random seed
            
        Yields:
            (token_column, token_masks): List of tokens and masks for each sample
        """
        assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get special tokens (if tokenizer supports them)
        bos = self.tokenizer.get_bos_token_id() if hasattr(self.tokenizer, 'get_bos_token_id') else None
        assistant_end = None
        if hasattr(self.tokenizer, 'encode_special'):
            try:
                assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
            except:
                pass

        # 1) Prefill with batch size 1
        config = self.model.config
        kv_cache_prefill = UNetKVCache(
            batch_size=1,
            config=config,
            max_seq_len=len(tokens) + (max_tokens or config.sequence_len),
            device=device,
            dtype=dtype,
        )
        
        # Run prefill forward pass
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self._safe_forward(ids, kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Clone cache for multi-sample generation
        kv_length_hint = len(tokens) + (max_tokens if max_tokens else config.sequence_len)
        kv_cache_decode = UNetKVCache(
            batch_size=num_samples,
            config=config,
            max_seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        # 3) Initialize row states
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Generation loop
        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            # Sample next tokens
            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row
            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                
                # Handle completion
                if (assistant_end and next_token == assistant_end) or (bos and next_token == bos):
                    state.completed = True

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self._safe_forward(ids, kv_cache_decode)[:, -1, :]

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that returns final token sequences.
        Returns (results, masks) where results is list of token sequences.
        """
        bos = self.tokenizer.get_bos_token_id() if hasattr(self.tokenizer, 'get_bos_token_id') else None
        assistant_end = None
        if hasattr(self.tokenizer, 'encode_special'):
            try:
                assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
            except:
                pass

        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if (assistant_end and token == assistant_end) or (bos and token == bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        
        return results, masks


# -----------------------------------------------------------------------------
# Comparison test: basic generate vs cached generate
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    """
    Test that UNet.generate() (naive) matches UNetEngine.generate() (cached).
    Also compares timing to show speedup from KV caching.
    """
    import time
    from contextlib import nullcontext
    from nanochat.common import compute_init, autodetect_device_type, print0
    from nanochat.checkpoint_manager import load_model

    print0("=" * 80)
    print0("UNet Engine Comparison Test")
    print0("=" * 80)

    # Init compute
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None

    # Load model
    print0("\nLoading UNet model...")
    try:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=None, step=None)
        
        # Check if it's actually a UNet model
        if not hasattr(model, 'encoder') or not hasattr(model, 'decoder'):
            print0("ERROR: Loaded model is not a UNet architecture!")
            print0("This test requires a UNet model checkpoint.")
            print0("Train a UNet model first using: python -m scripts.unet.base_train")
            exit(1)
            
    except Exception as e:
        print0(f"ERROR: Could not load UNet model: {e}")
        print0("Make sure you have a trained UNet checkpoint available.")
        exit(1)

    bos_token_id = tokenizer.get_bos_token_id()
    
    # Test parameters
    prompt = "The chemical formula of water is"
    prompt_tokens = tokenizer.encode(prompt, prepend=bos_token_id)
    kwargs = dict(max_tokens=32, temperature=0.0)  # temperature=0 for determinism

    print0(f"\nPrompt: '{prompt}'")
    print0(f"Prompt tokens: {len(prompt_tokens)}")
    print0(f"Generating {kwargs['max_tokens']} tokens...")
    print0()

    # Method 1: Naive model.generate() (no KV cache)
    print0("-" * 80)
    print0("Method 1: model.generate() [NAIVE - recomputes everything each step]")
    print0("-" * 80)
    generated_tokens = []
    synchronize()
    t0 = time.time()
    with autocast_ctx:
        stream = model.generate(prompt_tokens, **kwargs)
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print0(chunk, end="", flush=True)
    print0()
    synchronize()
    t1 = time.time()
    naive_time = t1 - t0
    print0(f"\nNaive time: {naive_time:.3f}s")
    print0(f"Throughput: {len(generated_tokens) / naive_time:.1f} tokens/sec")
    reference_ids = generated_tokens

    # Method 2: UNetEngine.generate() (with KV cache)
    print0("\n" + "-" * 80)
    print0("Method 2: UNetEngine.generate() [EFFICIENT - with KV caching]")
    print0("-" * 80)
    generated_tokens = []
    engine = UNetEngine(model, tokenizer)
    synchronize()
    t0 = time.time()
    with autocast_ctx:
        stream = engine.generate(prompt_tokens, num_samples=1, **kwargs)
        for token_column, token_masks in stream:
            token = token_column[0]
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print0(chunk, end="", flush=True)
    print0()
    synchronize()
    t1 = time.time()
    engine_time = t1 - t0
    print0(f"\nEngine time: {engine_time:.3f}s")
    print0(f"Throughput: {len(generated_tokens) / engine_time:.1f} tokens/sec")

    # Compare results
    print0("\n" + "=" * 80)
    print0("COMPARISON RESULTS")
    print0("=" * 80)
    
    # Check if outputs match
    match = reference_ids == generated_tokens
    if match:
        print0("✓ SUCCESS: Both methods produced identical output!")
    else:
        print0("✗ FAILURE: Outputs differ!")
        print0(f"  First mismatch at position {next((i for i, (a, b) in enumerate(zip(reference_ids, generated_tokens)) if a != b), len(reference_ids))}")
        print0(f"  Naive:  {reference_ids[:10]}...")
        print0(f"  Engine: {generated_tokens[:10]}...")
    
    # Performance comparison
    speedup = naive_time / engine_time
    print0(f"\nPerformance:")
    print0(f"  Naive method:  {naive_time:.3f}s ({len(reference_ids) / naive_time:.1f} tok/s)")
    print0(f"  Engine method: {engine_time:.3f}s ({len(generated_tokens) / engine_time:.1f} tok/s)")
    print0(f"  Speedup: {speedup:.2f}x")
    
    if speedup > 1.5:
        print0(f"  ✓ Good speedup from KV caching!")
    elif speedup > 1.0:
        print0(f"  ⚠ Modest speedup (KV cache overhead may be significant for short sequences)")
    else:
        print0(f"  ⚠ No speedup - check implementation")
    
    print0("\n" + "=" * 80)
