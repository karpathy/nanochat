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
# UNet-specific KV Cache with Efficient Encoder/Decoder Output Caching
# -----------------------------------------------------------------------------
class UNetKVCache:
    """
    KV Cache for UNet architecture with hierarchical stages.
    
    The UNet has stages with progressively pooled sequences:
    - Stage 0: seq_len
    - Stage 1: seq_len // 2
    - Stage 2: seq_len // 4
    etc.
    
    Key features:
    - Flat layer indexing matching fix_unet_layer_indices (encoder then decoder per stage)
    - Each layer's cache sized for its stage's sequence length
    - Efficient encoder/decoder output caching using ring buffers
    
    Memory optimization for encoder/decoder output caches:
    - Encoder outputs: Ring buffer of size 2^(n_stages - 1 - stage) + 2 per stage
      (only recent tokens needed for pooling and skip connections)
    - Decoder outputs: Small ring buffer per stage (only recent outputs needed)
    
    Efficient decode strategy:
    - Stage 0 runs every token
    - Stage 1 runs every 2 tokens (when we complete a pool pair)
    - Stage 2 runs every 4 tokens
    - Stage S runs every 2^S tokens
    """

    def __init__(self, batch_size, config, max_seq_len, device, dtype):
        self.batch_size = batch_size
        self.config = config
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        self.n_stages = len(config.n_layer)
        
        # Build layer-to-stage mapping (matches fix_unet_layer_indices order)
        # Order: enc_stage0, dec_stage0, enc_stage1, dec_stage1, ...
        self.layer_to_stage = []
        self.layer_is_decoder = []
        for stage_idx in range(self.n_stages):
            n_layers_half = config.n_layer[stage_idx] // 2
            for _ in range(n_layers_half):  # Encoder layers for this stage
                self.layer_to_stage.append(stage_idx)
                self.layer_is_decoder.append(False)
            for _ in range(n_layers_half):  # Decoder layers for this stage
                self.layer_to_stage.append(stage_idx)
                self.layer_is_decoder.append(True)
        
        self.n_layers = len(self.layer_to_stage)
        
        # Create KV caches for each layer (flat indexing)
        # Shape per layer: (batch, stage_seq_len, n_kv_head, head_dim)
        self.k_caches = []
        self.v_caches = []
        for layer_idx in range(self.n_layers):
            stage_idx = self.layer_to_stage[layer_idx]
            n_kv_head = config.n_kv_head[stage_idx]
            head_dim = config.n_embd[stage_idx] // config.n_head[stage_idx]
            stage_seq_len = max_seq_len // (2 ** stage_idx)
            self.k_caches.append(torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim, device=device, dtype=dtype))
            self.v_caches.append(torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim, device=device, dtype=dtype))
        
        # Cache seqlens tensor - updated before each stage's forward pass
        self._cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        
        # Position tracking per stage
        self.stage_pos = [0] * self.n_stages
        
        # Encoder output ring buffers for skip connections
        # Window size: 2^(n_stages - 1 - stage_idx) + 2 (for skip connections + pooling margin)
        # Shape: (batch, window_size, n_embd)
        self.enc_window_sizes = []
        self.enc_ring_buffers = []
        for stage_idx in range(self.n_stages):
            n_embd = config.n_embd[stage_idx]
            # Max window needed: 2^(n_stages-1-stage) for skip, +2 for pooling safety margin
            window_size = (2 ** max(0, self.n_stages - 1 - stage_idx)) + 2
            self.enc_window_sizes.append(window_size)
            self.enc_ring_buffers.append(torch.zeros(batch_size, window_size, n_embd, device=device, dtype=dtype))
        
        # Decoder output ring buffers for skip connections
        # We need a small history because skip connections access the position just BEFORE 
        # the current decoder chunk, which was cached in a previous iteration.
        # Window size: 4 covers all access patterns (current + previous chunks)
        # Shape: (batch, window_size, n_embd)
        self.dec_window_sizes = []
        self.dec_ring_buffers = []
        for stage_idx in range(self.n_stages):
            n_embd = config.n_embd[stage_idx]
            # Window of 4 covers all access patterns (current + previous chunks)
            window_size = 4
            self.dec_window_sizes.append(window_size)
            self.dec_ring_buffers.append(torch.zeros(batch_size, window_size, n_embd, device=device, dtype=dtype))

    def get_layer_cache(self, layer_idx):
        """Get KV cache tensors for a specific layer (called by Block.attn)."""
        return self.k_caches[layer_idx], self.v_caches[layer_idx]
    
    @property
    def cache_seqlens(self):
        """Current cache sequence lengths (set before each stage's forward)."""
        return self._cache_seqlens
    
    def set_cache_seqlens(self, stage_idx):
        """Set cache_seqlens for processing at a given stage."""
        self._cache_seqlens.fill_(self.stage_pos[stage_idx])
    
    def get_pos(self):
        """Get current position at stage 0."""
        return self.stage_pos[0]
    
    def get_stage_pos(self, stage_idx):
        """Get current position at a specific stage."""
        return self.stage_pos[stage_idx]
    
    def advance(self, num_tokens):
        """Called by Block after last layer - we handle advancement ourselves."""
        pass  # No-op, we manage positions in the engine
    
    def advance_stage(self, stage_idx, num_tokens):
        """Advance position at a specific stage."""
        self.stage_pos[stage_idx] += num_tokens
    
    def set_encoder_output(self, stage_idx, output, start_pos=None):
        """Store encoder output in ring buffer for skip connections."""
        if start_pos is None:
            start_pos = self.stage_pos[stage_idx]
        T = output.size(1)
        W = self.enc_window_sizes[stage_idx]
        
        # Store tokens into ring buffer using modular indexing
        if T == 1:
            # Fast path for single token (common during decode)
            self.enc_ring_buffers[stage_idx][:, start_pos % W] = output[:, 0]
        else:
            # General case for prefill
            for i in range(T):
                self.enc_ring_buffers[stage_idx][:, (start_pos + i) % W] = output[:, i]
    
    def get_encoder_output(self, stage_idx, start_pos, length):
        """Retrieve cached encoder output from ring buffer for skip connections."""
        W = self.enc_window_sizes[stage_idx]
        
        if length == 1:
            # Fast path for single token
            idx = start_pos % W
            return self.enc_ring_buffers[stage_idx][:, idx:idx + 1]
        else:
            # General case: gather from ring buffer
            indices = [(start_pos + i) % W for i in range(length)]
            return self.enc_ring_buffers[stage_idx][:, indices]
    
    def set_decoder_output(self, stage_idx, output, start_pos=None):
        """Store decoder output in ring buffer for skip connections."""
        if start_pos is None:
            start_pos = self.stage_pos[stage_idx]
        T = output.size(1)
        W = self.dec_window_sizes[stage_idx]
        
        # Store tokens into ring buffer using modular indexing
        if T == 1:
            # Fast path for single token (common during decode)
            self.dec_ring_buffers[stage_idx][:, start_pos % W] = output[:, 0]
        else:
            # General case for multi-token chunks
            for i in range(T):
                self.dec_ring_buffers[stage_idx][:, (start_pos + i) % W] = output[:, i]
    
    def get_decoder_output(self, stage_idx, start_pos, length):
        """Retrieve cached decoder output from ring buffer for skip connections."""
        W = self.dec_window_sizes[stage_idx]
        
        if length == 1:
            # Fast path for single token (always the case for skip connections)
            idx = start_pos % W
            return self.dec_ring_buffers[stage_idx][:, idx:idx + 1]
        else:
            # General case: gather from ring buffer
            indices = [(start_pos + i) % W for i in range(length)]
            return self.dec_ring_buffers[stage_idx][:, indices]
    
    def prefill(self, other):
        """Copy cache from another UNetKVCache (for multi-sample generation)."""
        # Copy KV caches
        for layer_idx in range(self.n_layers):
            stage_idx = self.layer_to_stage[layer_idx]
            pos = other.stage_pos[stage_idx]
            if pos > 0:
                self.k_caches[layer_idx][:, :pos] = other.k_caches[layer_idx][:, :pos]
                self.v_caches[layer_idx][:, :pos] = other.v_caches[layer_idx][:, :pos]
        
        # Copy encoder ring buffers (full copy since they're small)
        for stage_idx in range(self.n_stages):
            self.enc_ring_buffers[stage_idx][:] = other.enc_ring_buffers[stage_idx]
        
        # Copy decoder ring buffers (full copy since they're small)
        for stage_idx in range(self.n_stages):
            self.dec_ring_buffers[stage_idx][:] = other.dec_ring_buffers[stage_idx]
        
        self.stage_pos = other.stage_pos.copy()

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
    - Hierarchical efficiency: deeper stages only run when needed
      - Stage 0: every token
      - Stage 1: every 2 tokens
      - Stage 2: every 4 tokens
      - etc.
    - Batch generation support
    """

    def __init__(self, model, tokenizer):
        # Fix layer indices for KV caching (must be globally unique)
        fix_unet_layer_indices(model)
        self.model = model
        self.tokenizer = tokenizer
    
    def _prefill_forward(self, ids, kv_cache):
        """
        Prefill forward pass: runs all stages and caches encoder outputs.
        
        This is used for the initial prompt processing. We run the full UNet
        and cache both KV states and encoder outputs for later use.
        """
        B, T = ids.size()
        model = self.model
        
        def norm(x):
            return F.rms_norm(x, (x.size(-1),))
        
        # Get rotary embeddings
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = model.cos[:, T0:T0+T], model.sin[:, T0:T0+T]
        
        def pool_cos_sin(cos, sin, stage_idx):
            return (
                cos[:, (2 ** stage_idx - 1)::(2 ** stage_idx)],
                sin[:, (2 ** stage_idx - 1)::(2 ** stage_idx)]
            )
        
        # Input embed
        x = model.wte(ids)
        x = norm(x)
        
        # Track how many tokens processed at each stage
        stage_token_counts = {}
        
        # Encoder - run all stages and cache outputs
        encoder_outputs = []
        last_stage_idx = 0
        for stage_idx in range(model.n_stage):
            if stage_idx > 0:
                if x.size(1) == 1:
                    break
                x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                x = norm(x)
            
            # Set cache seqlens for this stage before running blocks
            if kv_cache is not None:
                kv_cache.set_cache_seqlens(stage_idx)
            
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.encoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            stage_token_counts[stage_idx] = x.size(1)
            encoder_outputs.append(x)
            last_stage_idx = stage_idx
            
            # Cache encoder output for skip connections
            if kv_cache is not None:
                kv_cache.set_encoder_output(stage_idx, x)
        
        # Advance encoder stage positions before decoder (decoder reads from same positions)
        if kv_cache is not None:
            for stage_idx, count in stage_token_counts.items():
                kv_cache.advance_stage(stage_idx, count)
        
        # Decoder - uses same positions as encoder (reset seqlens for decoder pass)
        for stage_idx in reversed(range(last_stage_idx + 1)):
            if kv_cache is not None:
                # Decoder sees same positions as encoder, so reset to start of this chunk
                old_pos = kv_cache.stage_pos[stage_idx] - stage_token_counts[stage_idx]
                kv_cache._cache_seqlens.fill_(old_pos)
            
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            # Cache decoder output BEFORE unpool (needed for skip connections in efficient decode)
            if kv_cache is not None and stage_idx > 0:
                kv_cache.set_decoder_output(stage_idx, x, start_pos=old_pos)
            
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
    
    def _efficient_decode_forward(self, ids, kv_cache):
        """
        Efficient decode forward: only runs stages that need to process new tokens.
        
        Key insight: due to pooling, deeper stages only get new information periodically:
        - Stage 0: processes every token
        - Stage 1: processes every 2 tokens (when pos % 2 == 1)
        - Stage 2: processes every 4 tokens (when pos % 4 == 3)
        - Stage S: processes every 2^S tokens (when pos % 2^S == 2^S - 1)
        
        When deeper stages run, unpooling produces multiple tokens at lower stages.
        Due to the skip connection shift, previously computed positions get the same
        input (encoder output only), so recomputation gives identical results.
        """
        B, T = ids.size()
        assert T == 1, "Efficient decode expects single token input"
        model = self.model
        
        def norm(x):
            return F.rms_norm(x, (x.size(-1),))
        
        # Current position at stage 0 (before adding the new token)
        pos = kv_cache.get_pos()
        
        # Determine which stages need to run
        # Stage S runs if pos % (2^S) == (2^S - 1), meaning we complete a new pooled token
        stages_to_run = [0]  # Stage 0 always runs
        for s in range(1, model.n_stage):
            if pos % (2 ** s) == (2 ** s) - 1:
                stages_to_run.append(s)
            else:
                break  # Higher stages can't run if lower stages don't
        last_stage = max(stages_to_run)
        
        # Compute how many tokens decoder will produce at stage 0
        # When last_stage runs, decoder produces 2^last_stage tokens at stage 0
        decoder_num_tokens = 2 ** last_stage
        decoder_start_pos = pos - decoder_num_tokens + 1
        
        # Get rotary embeddings
        # Encoder: all stages use rotary for position pos
        # (higher stages process pooled pairs, but the rotary corresponds to the odd position = pos)
        cos_sin_enc = model.cos[:, pos:pos+1], model.sin[:, pos:pos+1]
        # Decoder: needs positions [decoder_start_pos, pos]
        cos_sin_dec = model.cos[:, decoder_start_pos:pos+1], model.sin[:, decoder_start_pos:pos+1]
        
        def pool_cos_sin(cos, sin, stage_idx):
            return (
                cos[:, (2 ** stage_idx - 1)::(2 ** stage_idx)],
                sin[:, (2 ** stage_idx - 1)::(2 ** stage_idx)]
            )
        
        # ============ ENCODER ============
        # Embed the new token
        x = model.wte(ids)
        x = norm(x)
        
        # Process through encoder stages that need to run
        for stage_idx in stages_to_run:
            if stage_idx > 0:
                # Pool: get previous token from encoder output cache and combine with current
                # After processing stage S-1, we advanced stage_pos[S-1], so:
                # - Current token (in x) was cached at stage_pos[S-1] - 1
                # - Previous token (for pooling pair) is at stage_pos[S-1] - 2
                prev_stage_pos = kv_cache.stage_pos[stage_idx - 1]
                prev_x = kv_cache.get_encoder_output(stage_idx - 1, prev_stage_pos - 2, 1)
                x = torch.cat([prev_x, x], dim=1)  # (B, 2, C) = [even_pos, odd_pos]
                x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                x = norm(x)
            
            # Set cache seqlens for this stage
            kv_cache.set_cache_seqlens(stage_idx)
            
            # Run encoder blocks
            # For encoder in efficient decode, all stages use cos_sin_enc directly (no pooling)
            # because we always process 1 token, and the rotary is for position `pos`
            for block in model.encoder[f"transformer_{stage_idx}"]:
                x = block(x, cos_sin_enc, kv_cache)
            
            # Cache encoder output and advance position
            kv_cache.set_encoder_output(stage_idx, x)
            kv_cache.advance_stage(stage_idx, 1)
        
        # ============ DECODER ============
        # Start from deepest stage and work down
        # x currently holds the encoder output from the deepest stage that ran
        #
        # Key insight for skip connections:
        # Position P at stage S gets: encoder_S[P] + shifted(unpool(decoder_(S+1)))[P]
        # Due to shift: this equals encoder_S[P] + unpool(decoder_(S+1)[(P-1)//2])[(P-1)%2] for P>0
        # When stage S+1 didn't run, we use cached decoder_(S+1) output
        
        # If there are higher stages that didn't run, add their skip contribution to x
        # BEFORE running decoder at last_stage
        if last_stage + 1 < model.n_stage:
            # x is encoder output at last_stage. We need to add skip from cached decoder_(last_stage+1).
            # Get the position at last_stage
            stage_start = decoder_start_pos // (2 ** last_stage)
            higher_stage = last_stage + 1
            
            # For a single token at position stage_start:
            if stage_start > 0:
                src_pos_higher = (stage_start - 1) // 2
                src_idx_in_unpool = (stage_start - 1) % 2
                
                if src_pos_higher < kv_cache.stage_pos[higher_stage]:
                    cached_dec = kv_cache.get_decoder_output(higher_stage, src_pos_higher, 1)
                    cached_dec = F.rms_norm(cached_dec, (cached_dec.size(-1),))
                    unpooled = model.decoder[f"unpool_{higher_stage}->{last_stage}"](cached_dec)
                    x = x + unpooled[:, src_idx_in_unpool:src_idx_in_unpool+1]
        
        for stage_idx in reversed(stages_to_run):
            # Set cache seqlens: decoder processes from decoder_start_pos at this stage's resolution
            stage_decoder_start = decoder_start_pos // (2 ** stage_idx)
            kv_cache._cache_seqlens.fill_(stage_decoder_start)
            
            # Get rotary embeddings for this stage
            pooled_cos_sin = pool_cos_sin(*cos_sin_dec, stage_idx)
            
            # Run decoder blocks
            for block in model.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            # Cache decoder output (before unpool) for future skip connections
            if stage_idx > 0:
                kv_cache.set_decoder_output(stage_idx, x, start_pos=stage_decoder_start)
            
            if stage_idx > 0:
                # Unpool to lower stage
                x = norm(x)
                x = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                
                # Skip connection: get encoder outputs from cache for the lower stage
                lower_stage = stage_idx - 1
                lower_start = decoder_start_pos // (2 ** lower_stage)
                lower_len = decoder_num_tokens // (2 ** lower_stage)
                y = kv_cache.get_encoder_output(lower_stage, lower_start, lower_len)
                
                # Apply shift: relative position 0 gets 0, rest get shifted decoder output
                if x.size(1) == y.size(1):
                    x = x[:, :-1]
                shifted_x = torch.zeros_like(y)
                shifted_x[:, 1:] = x
                
                # IMPORTANT: The first position (lower_start) needs skip from the PREVIOUS 
                # decoder position at stage_idx, which was computed in a prior step and is cached.
                # Position P at lower_stage needs: unpool(decoder_stage_idx[(P-1)//2])[(P-1)%2]
                # For lower_start, this is: unpool(decoder_stage_idx[(lower_start-1)//2])[(lower_start-1)%2]
                if lower_start > 0:
                    prev_dec_pos = (lower_start - 1) // 2
                    prev_dec_idx = (lower_start - 1) % 2
                    if prev_dec_pos < kv_cache.stage_pos[stage_idx]:
                        # Get cached decoder output from previous position
                        cached_dec = kv_cache.get_decoder_output(stage_idx, prev_dec_pos, 1)
                        cached_dec = F.rms_norm(cached_dec, (cached_dec.size(-1),))
                        prev_unpooled = model.decoder[f"unpool_{stage_idx}->{lower_stage}"](cached_dec)
                        shifted_x[:, 0:1] = prev_unpooled[:, prev_dec_idx:prev_dec_idx+1]
                
                x = y + shifted_x
        
        # Forward the lm_head
        x = norm(x)
        softcap = 15
        logits = model.lm_head(x)
        logits = logits[..., :model.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        # Return only the last token's logits
        return logits[:, -1:, :]
    
    def _safe_forward(self, ids, kv_cache, use_efficient_decode=True):
        """
        Smart forward that chooses between prefill and efficient decode.
        
        Args:
            ids: Input token IDs (B, T)
            kv_cache: KV cache (UNetKVCache)
            use_efficient_decode: If True, use hierarchical efficiency for T=1
        
        For T > 1 (prefill): runs full forward, caches all encoder outputs
        For T = 1 (decode): 
            - If use_efficient_decode: only runs necessary stages
            - Otherwise: runs full forward (for debugging/comparison)
        """
        B, T = ids.size()
        
        if T > 1:
            # Prefill: run full forward and cache encoder outputs
            return self._prefill_forward(ids, kv_cache)
        
        if use_efficient_decode:
            # Efficient decode: only run necessary stages
            return self._efficient_decode_forward(ids, kv_cache)
        else:
            # Fallback to full forward (for debugging)
            return self._prefill_forward(ids, kv_cache)

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
    Compares full logits at each step, not just argmax tokens.
    """
    import time
    import argparse
    from contextlib import nullcontext
    from nanochat.common import compute_init, autodetect_device_type, print0
    from nanochat.checkpoint_manager import load_model

    parser = argparse.ArgumentParser(description="UNet Engine Comparison Test")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--prompt", type=str, default="The chemical formula of water is", help="Prompt text")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional model tag to load a specific checkpoint")
    parser.add_argument("--compare-logits", action="store_true", help="Compare full logits at each step (slower but more detailed)")
    args = parser.parse_args()

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
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=None)
        
        if not hasattr(model, 'encoder') or not hasattr(model, 'decoder'):
            print0("ERROR: Loaded model is not a UNet architecture!")
            exit(1)
            
    except Exception as e:
        print0(f"ERROR: Could not load UNet model: {e}")
        exit(1)

    bos_token_id = tokenizer.get_bos_token_id()
    
    prompt = args.prompt
    prompt_tokens = tokenizer.encode(prompt, prepend=bos_token_id)
    
    print0(f"\nPrompt: '{prompt}'")
    print0(f"Prompt tokens: {len(prompt_tokens)}")
    print0(f"Max tokens: {args.max_tokens}")
    print0(f"Compare logits: {args.compare_logits}")

    if args.compare_logits:
        # =====================================================================
        # DETAILED LOGITS COMPARISON
        # =====================================================================
        print0("\n" + "=" * 80)
        print0("DETAILED LOGITS COMPARISON (step by step)")
        print0("=" * 80)
        
        # Create engine (this also fixes layer indices)
        engine = UNetEngine(model, tokenizer)
        config = model.config
        dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
        
        # Setup for naive forward (no KV cache)
        ids_naive = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        # Setup for cached forward
        kv_cache = UNetKVCache(
            batch_size=1,
            config=config,
            max_seq_len=len(prompt_tokens) + args.max_tokens,
            device=device,
            dtype=dtype,
        )
        ids_cached = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
        
        all_match = True
        max_diff_overall = 0.0
        first_mismatch_step = -1
        
        with autocast_ctx:
            for step in range(args.max_tokens):
                # Naive forward (full sequence, no cache)
                logits_naive = model.forward(ids_naive)[:, -1, :]  # (1, vocab_size)
                
                # Cached forward
                if step == 0:
                    # Prefill
                    logits_cached = engine._safe_forward(ids_cached, kv_cache)[:, -1, :]
                else:
                    # Decode single token
                    logits_cached = engine._safe_forward(ids_cached[:, -1:], kv_cache)[:, -1, :]
                
                # Compare logits
                diff = (logits_naive - logits_cached).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                max_diff_overall = max(max_diff_overall, max_diff)
                
                # Get tokens
                token_naive = logits_naive.argmax(dim=-1).item()
                token_cached = logits_cached.argmax(dim=-1).item()
                
                # Check which stages ran
                pos = len(prompt_tokens) + step - 1 if step > 0 else len(prompt_tokens) - 1
                stages_that_run = [0]
                for s in range(1, model.n_stage):
                    if pos % (2 ** s) == (2 ** s) - 1:
                        stages_that_run.append(s)
                    else:
                        break
                
                # Report
                status = "✓" if token_naive == token_cached and max_diff < 0.1 else "✗"
                if token_naive != token_cached or max_diff >= 0.1:
                    all_match = False
                    if first_mismatch_step < 0:
                        first_mismatch_step = step
                
                print0(f"Step {step:3d} (pos={pos:3d}, stages={stages_that_run}): "
                       f"max_diff={max_diff:8.4f}, mean_diff={mean_diff:8.6f}, "
                       f"naive={token_naive:5d}, cached={token_cached:5d} {status}")
                
                if token_naive != token_cached:
                    # Show top-5 for both
                    top_naive = logits_naive.topk(5, dim=-1)
                    top_cached = logits_cached.topk(5, dim=-1)
                    print0(f"  Naive  top5: {top_naive.indices[0].tolist()} (logits: {top_naive.values[0].tolist()})")
                    print0(f"  Cached top5: {top_cached.indices[0].tolist()} (logits: {top_cached.values[0].tolist()})")
                
                # Update for next iteration
                next_token = token_naive  # Use naive token to keep sequences aligned
                ids_naive = torch.cat([ids_naive, torch.tensor([[next_token]], device=device)], dim=1)
                ids_cached = torch.cat([ids_cached, torch.tensor([[next_token]], device=device)], dim=1)
        
        print0("\n" + "-" * 80)
        print0("SUMMARY")
        print0("-" * 80)
        print0(f"Max diff overall: {max_diff_overall:.6f}")
        if all_match:
            print0("✓ SUCCESS: All logits match (within tolerance)!")
        else:
            print0(f"✗ FAILURE: First mismatch at step {first_mismatch_step}")
    
    else:
        # =====================================================================
        # SIMPLE TOKEN COMPARISON (original behavior)
        # =====================================================================
        kwargs = dict(max_tokens=args.max_tokens, temperature=0.0)
        
        # Method 1: Naive model.generate() (no KV cache)
        print0("\n" + "-" * 80)
        print0("Method 1: model.generate() [NAIVE]")
        print0("-" * 80)
        naive_tokens = []
        synchronize()
        t0 = time.time()
        with autocast_ctx:
            for token in model.generate(prompt_tokens, **kwargs):
                naive_tokens.append(token)
                print0(tokenizer.decode([token]), end="", flush=True)
        print0()
        synchronize()
        naive_time = time.time() - t0
        print0(f"Time: {naive_time:.3f}s ({len(naive_tokens) / naive_time:.1f} tok/s)")

        # Method 2: UNetEngine.generate() (with KV cache)
        print0("\n" + "-" * 80)
        print0("Method 2: UNetEngine.generate() [CACHED]")
        print0("-" * 80)
        engine_tokens = []
        engine = UNetEngine(model, tokenizer)
        synchronize()
        t0 = time.time()
        with autocast_ctx:
            for token_column, _ in engine.generate(prompt_tokens, num_samples=1, **kwargs):
                engine_tokens.append(token_column[0])
                print0(tokenizer.decode([token_column[0]]), end="", flush=True)
        print0()
        synchronize()
        engine_time = time.time() - t0
        print0(f"Time: {engine_time:.3f}s ({len(engine_tokens) / engine_time:.1f} tok/s)")

        # Compare
        print0("\n" + "-" * 80)
        print0("COMPARISON")
        print0("-" * 80)
        
        if naive_tokens == engine_tokens:
            print0("✓ SUCCESS: Outputs match!")
        else:
            print0("✗ FAILURE: Outputs differ!")
            mismatch = next((i for i, (a, b) in enumerate(zip(naive_tokens, engine_tokens)) if a != b), len(naive_tokens))
            print0(f"  First mismatch at position {mismatch}")
            print0(f"  Naive:  {naive_tokens[:10]}...")
            print0(f"  Engine: {engine_tokens[:10]}...")
        
        speedup = naive_time / engine_time
        print0(f"\nSpeedup: {speedup:.2f}x")
