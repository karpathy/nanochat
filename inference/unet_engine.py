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
# UNet-specific KV Cache with Encoder Output Caching
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
    - Each layer gets its own K/V cache with stage-appropriate sequence length
    - Stores encoder outputs at each stage for skip connections during decode
    - Supports efficient decode where deeper stages only run when needed
    
    Efficient decode strategy:
    - Stage 0 runs every token
    - Stage 1 runs every 2 tokens (when we complete a pool pair)
    - Stage 2 runs every 4 tokens
    - Stage S runs every 2^S tokens
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
        
        # Track the global layer index for each (stage, encoder/decoder, local_layer_idx)
        self.encoder_layer_offset = []  # encoder_layer_offset[stage] = first global idx for encoder at stage
        self.decoder_layer_offset = []  # decoder_layer_offset[stage] = first global idx for decoder at stage
        
        global_idx = 0
        for stage_idx in range(self.n_stages):
            n_layer = config.n_layer[stage_idx]
            n_kv_head = config.n_kv_head[stage_idx]
            n_embd = config.n_embd[stage_idx]
            head_dim = n_embd // config.n_head[stage_idx]
            
            # Sequence length at this stage (gets pooled by 2^stage_idx)
            stage_seq_len = max_seq_len // (2 ** stage_idx)
            
            # Create caches for encoder blocks at this stage
            self.encoder_layer_offset.append(global_idx)
            for _ in range(n_layer // 2):
                k = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim, 
                              device=device, dtype=dtype)
                v = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                self.layer_caches.append((k, v))
                self.layer_to_stage.append(stage_idx)
                global_idx += 1
            
            # Create caches for decoder blocks at this stage
            self.decoder_layer_offset.append(global_idx)
            for _ in range(n_layer // 2):
                k = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                v = torch.zeros(batch_size, stage_seq_len, n_kv_head, head_dim,
                              device=device, dtype=dtype)
                self.layer_caches.append((k, v))
                self.layer_to_stage.append(stage_idx)
                global_idx += 1
        
        # Track current position per-stage (different due to pooling)
        # cache_seqlens_per_stage[stage] = (batch_size,) tensor
        self.cache_seqlens_per_stage = []
        for stage_idx in range(self.n_stages):
            seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
            self.cache_seqlens_per_stage.append(seqlens)
        
        # For compatibility with the attention layer interface, we expose cache_seqlens
        # This will be set dynamically based on which stage is being processed
        self._current_stage = 0
        
        # Encoder output cache for skip connections during efficient decode
        # encoder_outputs[stage] = tensor of shape (batch_size, stage_seq_len, n_embd[stage])
        self.encoder_outputs = []
        for stage_idx in range(self.n_stages):
            stage_seq_len = max_seq_len // (2 ** stage_idx)
            n_embd = config.n_embd[stage_idx]
            enc_out = torch.zeros(batch_size, stage_seq_len, n_embd, device=device, dtype=dtype)
            self.encoder_outputs.append(enc_out)
        
        # Track how many tokens have been written to encoder output cache at each stage
        self.encoder_output_lens = [0] * self.n_stages
        
        # Decoder output cache for skip connections (unpooled outputs at each stage > 0)
        # decoder_outputs[stage] = tensor of shape (batch_size, stage_seq_len_below, n_embd[stage-1])
        # These are the UNPOOLED decoder outputs that get added to encoder outputs in skip connections
        self.decoder_outputs = [None]  # Stage 0 has no skip connection from above
        for stage_idx in range(1, self.n_stages):
            # Decoder at stage S unpools to stage S-1's sequence length
            below_seq_len = max_seq_len // (2 ** (stage_idx - 1))
            n_embd_below = config.n_embd[stage_idx - 1]
            dec_out = torch.zeros(batch_size, below_seq_len, n_embd_below, device=device, dtype=dtype)
            self.decoder_outputs.append(dec_out)
        
        # Track how many tokens have been written to decoder output cache at each stage
        self.decoder_output_lens = [0] * self.n_stages

    @property
    def cache_seqlens(self):
        """Return cache_seqlens for the current stage (for attention layer compatibility)."""
        return self.cache_seqlens_per_stage[self._current_stage]
    
    def set_current_stage(self, stage_idx):
        """Set the current stage for cache_seqlens access."""
        self._current_stage = stage_idx
    
    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) for a specific layer (GPT-compatible interface)."""
        return self.layer_caches[layer_idx]
    
    def get_encoder_layer_offset(self, stage_idx):
        """Get the global layer index for the first encoder layer at a stage."""
        return self.encoder_layer_offset[stage_idx]
    
    def get_decoder_layer_offset(self, stage_idx):
        """Get the global layer index for the first decoder layer at a stage."""
        return self.decoder_layer_offset[stage_idx]

    def get_pos(self):
        """Get current position in original (unpooled) sequence."""
        return self.cache_seqlens_per_stage[0][0].item()
    
    def get_stage_pos(self, stage_idx):
        """Get current position at a specific stage (accounts for pooling)."""
        return self.cache_seqlens_per_stage[stage_idx][0].item()

    def advance_stage(self, stage_idx, num_tokens):
        """Advance cache position for a specific stage."""
        self.cache_seqlens_per_stage[stage_idx] += num_tokens
    
    def advance(self, num_tokens):
        """Advance cache position for stage 0 (called by attention layer)."""
        # This is called by the last layer after processing
        # With efficient decode, we handle advancement manually, so this is a no-op
        # The engine will call advance_stage explicitly
        pass
    
    def advance_all_stages(self, num_tokens_stage0):
        """Advance all stage positions based on stage 0 token count."""
        for stage_idx in range(self.n_stages):
            # Stage S position increases by num_tokens // 2^S
            stage_tokens = num_tokens_stage0 // (2 ** stage_idx)
            if stage_tokens > 0:
                self.cache_seqlens_per_stage[stage_idx] += stage_tokens

    def reset(self):
        """Reset cache to empty state."""
        for stage_idx in range(self.n_stages):
            self.cache_seqlens_per_stage[stage_idx].zero_()
        self.encoder_output_lens = [0] * self.n_stages
        self.decoder_output_lens = [0] * self.n_stages
        self._current_stage = 0
    
    def set_encoder_output(self, stage_idx, output, start_pos=None):
        """
        Store encoder output at a stage for skip connections.
        
        Args:
            stage_idx: Which stage's encoder output
            output: Tensor of shape (batch_size, T, n_embd)
            start_pos: Starting position to write at (defaults to current stage pos)
        """
        if start_pos is None:
            start_pos = self.encoder_output_lens[stage_idx]
        T = output.size(1)
        self.encoder_outputs[stage_idx][:, start_pos:start_pos + T] = output
        self.encoder_output_lens[stage_idx] = start_pos + T
    
    def get_encoder_output(self, stage_idx, start_pos=None, length=None):
        """
        Retrieve encoder output at a stage for skip connections.
        
        Args:
            stage_idx: Which stage's encoder output
            start_pos: Starting position (default 0)
            length: Number of tokens to retrieve (default all cached)
        """
        if start_pos is None:
            start_pos = 0
        if length is None:
            length = self.encoder_output_lens[stage_idx] - start_pos
        return self.encoder_outputs[stage_idx][:, start_pos:start_pos + length]
    
    def set_decoder_output(self, stage_idx, output, start_pos=None):
        """
        Store unpooled decoder output at a stage for skip connections.
        
        Args:
            stage_idx: Which stage's decoder output (must be > 0)
            output: Tensor of shape (batch_size, T, n_embd) - the UNPOOLED output
            start_pos: Starting position to write at (defaults to current len)
        """
        assert stage_idx > 0, "Stage 0 has no skip connection from above"
        if start_pos is None:
            start_pos = self.decoder_output_lens[stage_idx]
        T = output.size(1)
        self.decoder_outputs[stage_idx][:, start_pos:start_pos + T] = output
        self.decoder_output_lens[stage_idx] = start_pos + T
    
    def get_decoder_output(self, stage_idx, start_pos=None, length=None):
        """
        Retrieve unpooled decoder output at a stage for skip connections.
        
        Args:
            stage_idx: Which stage's decoder output (must be > 0)
            start_pos: Starting position (default 0)
            length: Number of tokens to retrieve (default all cached)
        """
        assert stage_idx > 0, "Stage 0 has no skip connection from above"
        if start_pos is None:
            start_pos = 0
        if length is None:
            length = self.decoder_output_lens[stage_idx] - start_pos
        return self.decoder_outputs[stage_idx][:, start_pos:start_pos + length]

    def prefill(self, other):
        """
        Copy cache from another UNetKVCache (for multi-sample generation).
        Used to replicate a single prefill across multiple parallel samples.
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty cache"
        assert self.n_layers == other.n_layers
        
        # Copy KV caches for all layers
        for layer_idx in range(self.n_layers):
            stage_idx = self.layer_to_stage[layer_idx]
            stage_pos = other.get_stage_pos(stage_idx)
            if stage_pos == 0:
                continue
            
            k_other, v_other = other.layer_caches[layer_idx]
            k_self, v_self = self.layer_caches[layer_idx]
            
            # Copy up to the current position at this stage
            k_self[:, :stage_pos] = k_other[:, :stage_pos]
            v_self[:, :stage_pos] = v_other[:, :stage_pos]
        
        # Copy encoder outputs
        for stage_idx in range(self.n_stages):
            enc_len = other.encoder_output_lens[stage_idx]
            if enc_len > 0:
                self.encoder_outputs[stage_idx][:, :enc_len] = other.encoder_outputs[stage_idx][:, :enc_len]
                self.encoder_output_lens[stage_idx] = enc_len
        
        # Copy decoder outputs (for skip connections)
        for stage_idx in range(1, self.n_stages):
            dec_len = other.decoder_output_lens[stage_idx]
            if dec_len > 0:
                self.decoder_outputs[stage_idx][:, :dec_len] = other.decoder_outputs[stage_idx][:, :dec_len]
                self.decoder_output_lens[stage_idx] = dec_len
        
        # Copy positions for all stages
        for stage_idx in range(self.n_stages):
            stage_pos = other.get_stage_pos(stage_idx)
            self.cache_seqlens_per_stage[stage_idx].fill_(stage_pos)
    
    def should_run_stage(self, stage_idx, new_tokens=1):
        """
        Determine if a stage should run given the current position and new tokens.
        
        Stage S should run when we have enough tokens to form a complete pool group:
        - Stage 0: always runs
        - Stage 1: runs when (pos + new_tokens) // 2 > pos // 2
        - Stage 2: runs when (pos + new_tokens) // 4 > pos // 4
        - etc.
        """
        if stage_idx == 0:
            return True
        
        pos = self.get_pos()
        divisor = 2 ** stage_idx
        old_stage_pos = pos // divisor
        new_stage_pos = (pos + new_tokens) // divisor
        return new_stage_pos > old_stage_pos
    
    def get_max_active_stage(self, new_tokens=1):
        """Get the deepest stage that should run for this generation step."""
        for stage_idx in range(self.n_stages - 1, -1, -1):
            if self.should_run_stage(stage_idx, new_tokens):
                return stage_idx
        return 0


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
        
        Encoder and decoder at the same stage have DIFFERENT layer indices,
        so they write to different KV cache tensors. They both write at the
        same positions (0 to T-1 for prefill), and we advance at the end.
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
            
            # Set current stage for KV cache
            if kv_cache is not None:
                kv_cache.set_current_stage(stage_idx)
            
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.encoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            stage_token_counts[stage_idx] = x.size(1)
            encoder_outputs.append(x)
            last_stage_idx = stage_idx
            
            # Cache encoder output for skip connections
            if kv_cache is not None:
                kv_cache.set_encoder_output(stage_idx, x, start_pos=0)
        
        # Decoder - writes to different cache tensors than encoder
        for stage_idx in reversed(range(last_stage_idx + 1)):
            if kv_cache is not None:
                kv_cache.set_current_stage(stage_idx)
            
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in model.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            if stage_idx > 0:
                x = norm(x)
                x = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                
                # Cache the unpooled decoder output for future skip connections
                if kv_cache is not None:
                    kv_cache.set_decoder_output(stage_idx, x, start_pos=0)
                
                y = encoder_outputs[stage_idx - 1]
                if x.size(1) == y.size(1):
                    x = x[:, :-1]
                shifted_x = torch.zeros_like(y)
                shifted_x[:, 1:] = x
                x = y + shifted_x
        
        # Advance all stage positions at the end
        if kv_cache is not None:
            for stage_idx, count in stage_token_counts.items():
                kv_cache.advance_stage(stage_idx, count)
        
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
        - Stage 1: processes every 2 tokens (after pooling pairs)
        - Stage 2: processes every 4 tokens
        - Stage S: processes every 2^S tokens
        
        This method:
        1. Determines which stages need to run based on current position
        2. Runs only those encoder stages, caching outputs
        3. Runs decoder stages in reverse, using cached encoder outputs for skip connections
        
        When deeper stages run, the decoder at lower stages will process multiple tokens
        due to unpooling. We compute all of them but only return the last token's logits.
        """
        B, T = ids.size()
        assert T == 1, "Efficient decode only supports T=1"
        model = self.model
        
        def norm(x):
            return F.rms_norm(x, (x.size(-1),))
        
        # Get current positions at each stage BEFORE processing
        stage_positions = [kv_cache.get_stage_pos(s) for s in range(model.n_stage)]
        pos = stage_positions[0]
        
        # Determine which stages need to run
        max_active_stage = kv_cache.get_max_active_stage(new_tokens=T)
        
        # Track tokens processed at each active stage for final advancement
        stage_token_counts = {}
        
        # =========== ENCODER ===========
        # Stage 0 encoder always runs
        kv_cache.set_current_stage(0)
        
        x = model.wte(ids)
        x = norm(x)
        
        # Get rotary for just the new token at stage 0
        cos_sin_new = model.cos[:, pos:pos+T], model.sin[:, pos:pos+T]
        
        for block in model.encoder[f"transformer_0"]:
            x = block(x, cos_sin_new, kv_cache)
        
        stage_token_counts[0] = T
        
        # Update encoder output cache for stage 0 (must happen before deeper stages pool it)
        kv_cache.set_encoder_output(0, x)
        
        # Process deeper encoder stages if needed
        for stage_idx in range(1, max_active_stage + 1):
            kv_cache.set_current_stage(stage_idx)
            
            # Get the last 2 tokens from previous stage's encoder output to pool
            prev_stage_len = kv_cache.encoder_output_lens[stage_idx - 1]
            pair_start = prev_stage_len - 2
            x_to_pool = kv_cache.get_encoder_output(stage_idx - 1, pair_start, 2)
            
            # Pool
            x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x_to_pool)
            x = norm(x)
            
            # Get rotary for this pooled position
            # IMPORTANT: Rotary uses original sequence indices, not pooled indices
            # For stage S at pooled position P, original index is: 2^S * P + (2^S - 1)
            # This matches pool_cos_sin: cos[:, (2^S - 1)::2^S] takes indices 2^S-1, 2*2^S-1, 3*2^S-1, ...
            stage_pos = stage_positions[stage_idx]
            divisor = 2 ** stage_idx
            original_rotary_idx = divisor * stage_pos + (divisor - 1)
            pooled_cos_sin = (
                model.cos[:, original_rotary_idx:original_rotary_idx+1],
                model.sin[:, original_rotary_idx:original_rotary_idx+1]
            )
            
            for block in model.encoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            stage_token_counts[stage_idx] = 1
            
            # Cache encoder output (must happen before decoder uses skip connections)
            kv_cache.set_encoder_output(stage_idx, x)
        
        # =========== DECODER ===========
        # Run decoder stages in reverse order
        # x currently holds the output from the deepest active encoder stage
        
        # Track the starting position at each stage for decoder rotary embeddings
        # After unpool+skip, we need to use the encoder output positions, not stage_positions
        decoder_start_positions = {}
        for s in range(max_active_stage + 1):
            decoder_start_positions[s] = stage_positions[s]
        
        # Track decoder tokens processed at each stage for final cache position update
        decoder_token_counts = {}
        
        for stage_idx in reversed(range(max_active_stage + 1)):
            kv_cache.set_current_stage(stage_idx)
            
            # FIX: Set cache_seqlens to decoder's starting position
            # This ensures decoder writes to correct cache positions, not where encoder left off
            kv_cache.cache_seqlens_per_stage[stage_idx].fill_(decoder_start_positions[stage_idx])
            
            # Decoder processes at the same positions as encoder
            stage_pos = decoder_start_positions[stage_idx]
            num_tokens = x.size(1)
            
            # Compute rotary indices in original sequence space
            # For stage S at pooled positions P to P+num_tokens-1, 
            # original indices are: 2^S * P + (2^S - 1), 2^S * (P+1) + (2^S - 1), ...
            divisor = 2 ** stage_idx
            original_start = divisor * stage_pos + (divisor - 1)
            # The rotary indices for consecutive pooled positions are spaced by 2^S
            rotary_indices = torch.arange(original_start, original_start + num_tokens * divisor, divisor, 
                                          device=model.cos.device)
            pooled_cos_sin = (
                model.cos[:, rotary_indices],
                model.sin[:, rotary_indices]
            )
            
            for block in model.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            
            # Track tokens processed by decoder at this stage
            decoder_token_counts[stage_idx] = num_tokens
            
            if stage_idx > 0:
                # Unpool and skip connection
                x = norm(x)
                x_unpooled = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                
                # Cache the unpooled decoder output for future skip connections
                kv_cache.set_decoder_output(stage_idx, x_unpooled)
                
                # Get encoder output for skip connection
                # After unpooling, x has 2*num_tokens. We need corresponding encoder outputs.
                prev_stage_len = kv_cache.encoder_output_lens[stage_idx - 1]
                unpool_len = x_unpooled.size(1)
                skip_start_pos = prev_stage_len - unpool_len
                y = kv_cache.get_encoder_output(stage_idx - 1, skip_start_pos, unpool_len)
                
                # The decoder at the next stage will process starting at skip_start_pos
                decoder_start_positions[stage_idx - 1] = skip_start_pos
                
                # For skip connection, we need the SHIFTED unpooled decoder output
                # In naive forward: shifted[P] = unpool[P-1] for P >= 1, 0 otherwise
                # We need to retrieve the cached decoder output for positions before skip_start_pos
                # and combine with the newly computed x_unpooled
                
                # Get the cached decoder output that covers position skip_start_pos - 1 (for the shift)
                # The shift means: output[skip_start_pos] gets decoder_unpool[skip_start_pos - 1]
                #                  output[skip_start_pos + 1] gets decoder_unpool[skip_start_pos]
                #                  etc.
                
                # We need unpool_len positions of shifted decoder output
                # shifted[skip_start_pos : skip_start_pos + unpool_len] comes from 
                # unpool[skip_start_pos - 1 : skip_start_pos + unpool_len - 1]
                
                shifted_x = torch.zeros_like(y)
                
                # The first shifted position (skip_start_pos) needs unpool[skip_start_pos - 1]
                # which is from the CACHED decoder output (computed in previous forward passes)
                if skip_start_pos > 0:
                    # Get cached decoder output at position skip_start_pos - 1
                    cached_dec = kv_cache.get_decoder_output(stage_idx, skip_start_pos - 1, 1)
                    shifted_x[:, 0:1] = cached_dec
                
                # The remaining shifted positions come from x_unpooled (newly computed)
                # shifted[skip_start_pos + 1 : skip_start_pos + unpool_len] = x_unpooled[0 : unpool_len - 1]
                if unpool_len > 1:
                    shifted_x[:, 1:] = x_unpooled[:, :-1]
                
                x = y + shifted_x
        
        # Set final cache positions at the end
        # Both encoder and decoder have processed their tokens; final position is the max
        # For encoder: ends at stage_positions[s] + stage_token_counts[s]
        # For decoder: ends at decoder_start_positions[s] + decoder_token_counts[s]
        for stage_idx in range(max_active_stage + 1):
            encoder_end = stage_positions[stage_idx] + stage_token_counts.get(stage_idx, 0)
            decoder_end = decoder_start_positions[stage_idx] + decoder_token_counts.get(stage_idx, 0)
            final_pos = max(encoder_end, decoder_end)
            kv_cache.cache_seqlens_per_stage[stage_idx].fill_(final_pos)
        
        # x now has shape (B, output_len, dim) where output_len may be > 1 when deeper stages ran
        # We only care about the last token for generation
        x = x[:, -1:, :]
        
        # Forward the lm_head
        x = norm(x)
        softcap = 15
        logits = model.lm_head(x)
        logits = logits[..., :model.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        return logits
    
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
    Also compares timing to show speedup from KV caching.
    """
    import time
    import argparse
    from contextlib import nullcontext
    from nanochat.common import compute_init, autodetect_device_type, print0
    from nanochat.checkpoint_manager import load_model

    parser = argparse.ArgumentParser(description="UNet Engine Comparison Test")
    parser.add_argument("--test", choices=["tokens", "intermediate", "both"], default="both",
                        help="Which test to run: 'tokens' (final output), 'intermediate' (step-by-step logits), or 'both'")
    parser.add_argument("--max-tokens", type=int, default=32, help="Maximum tokens to generate")
    parser.add_argument("--prompt", type=str, default="The chemical formula of water is", help="Prompt text")
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
    prompt = args.prompt
    prompt_tokens = tokenizer.encode(prompt, prepend=bos_token_id)
    kwargs = dict(max_tokens=args.max_tokens, temperature=0.0)  # temperature=0 for determinism

    print0(f"\nPrompt: '{prompt}'")
    print0(f"Prompt tokens: {len(prompt_tokens)}")
    print0(f"Max tokens: {kwargs['max_tokens']}")
    print0()

    # =========================================================================
    # Test 1: Compare final generated tokens (original test)
    # =========================================================================
    if args.test in ["tokens", "both"]:
        print0("=" * 80)
        print0("TEST 1: Final Token Comparison")
        print0("=" * 80)

        # Method 1: Naive model.generate() (no KV cache)
        print0("\n" + "-" * 80)
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
        print0("\n" + "-" * 80)
        print0("COMPARISON RESULTS (Final Tokens)")
        print0("-" * 80)
        
        # Check if outputs match
        match = reference_ids == generated_tokens
        if match:
            print0("✓ SUCCESS: Both methods produced identical output!")
        else:
            print0("✗ FAILURE: Outputs differ!")
            mismatch_pos = next((i for i, (a, b) in enumerate(zip(reference_ids, generated_tokens)) if a != b), len(reference_ids))
            print0(f"  First mismatch at position {mismatch_pos}")
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

    # =========================================================================
    # Test 2: Compare intermediate logits step-by-step with detailed activation diffs
    # =========================================================================
    if args.test in ["intermediate", "both"]:
        print0("\n" + "=" * 80)
        print0("TEST 2: Intermediate Logits Comparison (Step-by-Step)")
        print0("=" * 80)
        print0("\nThis test compares logits at each generation step to find where divergence occurs.")
        print0("Stops at first mismatch and shows detailed per-layer/per-stage activation diffs.")
        
        # Re-fix layer indices (in case they were modified)
        fix_unet_layer_indices(model)
        
        config = model.config
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        max_seq_len = len(prompt_tokens) + kwargs['max_tokens']
        
        # =====================================================================
        # Instrumented forward functions that capture intermediate activations
        # =====================================================================
        
        def naive_forward_instrumented(model, ids):
            """
            Run naive forward pass and capture activations at each stage/layer.
            Returns: (logits, activations_dict)
            """
            B, T = ids.size()
            activations = {}
            
            def norm(x):
                return F.rms_norm(x, (x.size(-1),))
            
            T0 = 0
            cos_sin = model.cos[:, T0:T0+T], model.sin[:, T0:T0+T]
            
            def pool_cos_sin(cos, sin, stage_idx):
                return (
                    cos[:, (2 ** stage_idx - 1)::(2 ** stage_idx)],
                    sin[:, (2 ** stage_idx - 1)::(2 ** stage_idx)]
                )
            
            # Input embed
            x = model.wte(ids)
            x = norm(x)
            activations['embed'] = x[:, -1:, :].clone()  # Last position only
            
            # Encoder
            encoder_outputs = []
            last_stage_idx = 0
            for stage_idx in range(model.n_stage):
                if stage_idx > 0:
                    if x.size(1) == 1:
                        break
                    x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                    x = norm(x)
                    activations[f'enc_pool_{stage_idx}'] = x[:, -1:, :].clone()
                
                pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
                for layer_idx, block in enumerate(model.encoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, None)
                    activations[f'enc_s{stage_idx}_l{layer_idx}'] = x[:, -1:, :].clone()
                
                encoder_outputs.append(x)
                activations[f'enc_s{stage_idx}_out'] = x[:, -1:, :].clone()
                last_stage_idx = stage_idx
            
            # Decoder
            for stage_idx in reversed(range(last_stage_idx + 1)):
                pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
                for layer_idx, block in enumerate(model.decoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, None)
                    activations[f'dec_s{stage_idx}_l{layer_idx}'] = x[:, -1:, :].clone()
                
                activations[f'dec_s{stage_idx}_out'] = x[:, -1:, :].clone()
                
                if stage_idx > 0:
                    x = norm(x)
                    x_unpooled = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                    activations[f'dec_unpool_{stage_idx}'] = x_unpooled[:, -1:, :].clone()
                    # Also capture the position that gets used in shifted_x[-1]
                    # In naive, shifted_x[-1] = x_unpooled[-2] (after potential drop)
                    if x_unpooled.size(1) >= 2:
                        activations[f'dec_unpool_{stage_idx}_first'] = x_unpooled[:, -2:-1, :].clone()
                    
                    y = encoder_outputs[stage_idx - 1]
                    
                    # Capture encoder output used in skip
                    # In naive: y has ALL encoder positions, we need the one that corresponds to 
                    # what cached decode uses. Cached uses get_encoder_output(0, skip_start_pos, unpool_len)
                    # where skip_start_pos = encoder_len - unpool_len
                    # So the last position in y slice is position (encoder_len - 1)
                    # which equals y[:, -1:] - correct!
                    activations[f'skip_enc_{stage_idx}'] = y[:, -1:, :].clone()
                    
                    # Also capture what position 6 looks like (the one from "cache" in efficient decode)
                    skip_start_pos_naive = y.size(1) - x_unpooled.size(1)
                    activations[f'skip_enc_{stage_idx}_pos6'] = y[:, skip_start_pos_naive:skip_start_pos_naive+1, :].clone()
                    
                    x = x_unpooled
                    if x.size(1) == y.size(1):
                        x = x[:, :-1]
                    shifted_x = torch.zeros_like(y)
                    shifted_x[:, 1:] = x
                    
                    # Capture shifted_x last position
                    activations[f'skip_shifted_{stage_idx}'] = shifted_x[:, -1:, :].clone()
                    
                    x = y + shifted_x
                    activations[f'dec_skip_{stage_idx}'] = x[:, -1:, :].clone()
            
            # LM head
            x = norm(x)
            activations['pre_lm_head'] = x[:, -1:, :].clone()
            softcap = 15
            logits = model.lm_head(x)
            logits = logits[..., :model.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            
            return logits, activations
        
        def cached_prefill_instrumented(engine, ids, kv_cache):
            """
            Run cached prefill forward and capture activations.
            Returns: (logits, activations_dict)
            """
            B, T = ids.size()
            model = engine.model
            activations = {}
            
            def norm(x):
                return F.rms_norm(x, (x.size(-1),))
            
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
            activations['embed'] = x[:, -1:, :].clone()
            
            stage_token_counts = {}
            
            # Encoder
            encoder_outputs = []
            last_stage_idx = 0
            for stage_idx in range(model.n_stage):
                if stage_idx > 0:
                    if x.size(1) == 1:
                        break
                    x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                    x = norm(x)
                    activations[f'enc_pool_{stage_idx}'] = x[:, -1:, :].clone()
                
                if kv_cache is not None:
                    kv_cache.set_current_stage(stage_idx)
                
                pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
                for layer_idx, block in enumerate(model.encoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, kv_cache)
                    activations[f'enc_s{stage_idx}_l{layer_idx}'] = x[:, -1:, :].clone()
                
                stage_token_counts[stage_idx] = x.size(1)
                encoder_outputs.append(x)
                activations[f'enc_s{stage_idx}_out'] = x[:, -1:, :].clone()
                last_stage_idx = stage_idx
                
                if kv_cache is not None:
                    kv_cache.set_encoder_output(stage_idx, x, start_pos=0)
            
            # Decoder
            for stage_idx in reversed(range(last_stage_idx + 1)):
                if kv_cache is not None:
                    kv_cache.set_current_stage(stage_idx)
                
                pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
                for layer_idx, block in enumerate(model.decoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, kv_cache)
                    activations[f'dec_s{stage_idx}_l{layer_idx}'] = x[:, -1:, :].clone()
                
                activations[f'dec_s{stage_idx}_out'] = x[:, -1:, :].clone()
                
                if stage_idx > 0:
                    x = norm(x)
                    x = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                    activations[f'dec_unpool_{stage_idx}'] = x[:, -1:, :].clone()
                    
                    if kv_cache is not None:
                        kv_cache.set_decoder_output(stage_idx, x, start_pos=0)
                    
                    y = encoder_outputs[stage_idx - 1]
                    if x.size(1) == y.size(1):
                        x = x[:, :-1]
                    shifted_x = torch.zeros_like(y)
                    shifted_x[:, 1:] = x
                    x = y + shifted_x
                    activations[f'dec_skip_{stage_idx}'] = x[:, -1:, :].clone()
            
            # Advance positions
            if kv_cache is not None:
                for stage_idx, count in stage_token_counts.items():
                    kv_cache.advance_stage(stage_idx, count)
            
            # LM head
            x = norm(x)
            activations['pre_lm_head'] = x[:, -1:, :].clone()
            softcap = 15
            logits = model.lm_head(x)
            logits = logits[..., :model.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            
            return logits, activations
        
        def cached_decode_instrumented(engine, ids, kv_cache):
            """
            Run cached decode forward (T=1) and capture activations.
            Returns: (logits, activations_dict)
            """
            B, T = ids.size()
            assert T == 1, "Decode only supports T=1"
            model = engine.model
            activations = {}
            
            def norm(x):
                return F.rms_norm(x, (x.size(-1),))
            
            stage_positions = [kv_cache.get_stage_pos(s) for s in range(model.n_stage)]
            pos = stage_positions[0]
            
            max_active_stage = kv_cache.get_max_active_stage(new_tokens=T)
            activations['_max_active_stage'] = max_active_stage
            activations['_stage_positions'] = stage_positions.copy()
            
            stage_token_counts = {}
            
            # =========== ENCODER ===========
            kv_cache.set_current_stage(0)
            
            x = model.wte(ids)
            x = norm(x)
            activations['embed'] = x.clone()
            
            cos_sin_new = model.cos[:, pos:pos+T], model.sin[:, pos:pos+T]
            
            for layer_idx, block in enumerate(model.encoder[f"transformer_0"]):
                x = block(x, cos_sin_new, kv_cache)
                activations[f'enc_s0_l{layer_idx}'] = x.clone()
            
            stage_token_counts[0] = T
            activations['enc_s0_out'] = x.clone()
            kv_cache.set_encoder_output(0, x)
            
            # Deeper encoder stages
            for stage_idx in range(1, max_active_stage + 1):
                kv_cache.set_current_stage(stage_idx)
                
                prev_stage_len = kv_cache.encoder_output_lens[stage_idx - 1]
                pair_start = prev_stage_len - 2
                x_to_pool = kv_cache.get_encoder_output(stage_idx - 1, pair_start, 2)
                
                x = model.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x_to_pool)
                x = norm(x)
                activations[f'enc_pool_{stage_idx}'] = x.clone()
                
                stage_pos = stage_positions[stage_idx]
                divisor = 2 ** stage_idx
                original_rotary_idx = divisor * stage_pos + (divisor - 1)
                pooled_cos_sin = (
                    model.cos[:, original_rotary_idx:original_rotary_idx+1],
                    model.sin[:, original_rotary_idx:original_rotary_idx+1]
                )
                
                for layer_idx, block in enumerate(model.encoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, kv_cache)
                    activations[f'enc_s{stage_idx}_l{layer_idx}'] = x.clone()
                
                stage_token_counts[stage_idx] = 1
                activations[f'enc_s{stage_idx}_out'] = x.clone()
                kv_cache.set_encoder_output(stage_idx, x)
            
            # =========== DECODER ===========
            decoder_start_positions = {}
            for s in range(max_active_stage + 1):
                decoder_start_positions[s] = stage_positions[s]
            
            # Track decoder tokens processed at each stage
            decoder_token_counts = {}
            
            for stage_idx in reversed(range(max_active_stage + 1)):
                kv_cache.set_current_stage(stage_idx)
                
                # FIX: Set cache_seqlens to decoder's starting position
                kv_cache.cache_seqlens_per_stage[stage_idx].fill_(decoder_start_positions[stage_idx])
                
                stage_pos = decoder_start_positions[stage_idx]
                num_tokens = x.size(1)
                
                divisor = 2 ** stage_idx
                original_start = divisor * stage_pos + (divisor - 1)
                rotary_indices = torch.arange(original_start, original_start + num_tokens * divisor, divisor,
                                              device=model.cos.device)
                pooled_cos_sin = (
                    model.cos[:, rotary_indices],
                    model.sin[:, rotary_indices]
                )
                
                for layer_idx, block in enumerate(model.decoder[f"transformer_{stage_idx}"]):
                    x = block(x, pooled_cos_sin, kv_cache)
                    activations[f'dec_s{stage_idx}_l{layer_idx}'] = x[:, -1:, :].clone()
                
                # Track tokens processed by decoder at this stage
                decoder_token_counts[stage_idx] = num_tokens
                
                activations[f'dec_s{stage_idx}_out'] = x[:, -1:, :].clone()
                
                if stage_idx > 0:
                    x = norm(x)
                    x_unpooled = model.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                    activations[f'dec_unpool_{stage_idx}'] = x_unpooled[:, -1:, :].clone()
                    # Also capture first position of unpool for debugging
                    activations[f'dec_unpool_{stage_idx}_first'] = x_unpooled[:, 0:1, :].clone()
                    
                    kv_cache.set_decoder_output(stage_idx, x_unpooled)
                    
                    prev_stage_len = kv_cache.encoder_output_lens[stage_idx - 1]
                    unpool_len = x_unpooled.size(1)
                    skip_start_pos = prev_stage_len - unpool_len
                    y = kv_cache.get_encoder_output(stage_idx - 1, skip_start_pos, unpool_len)
                    
                    # Capture encoder output used in skip  
                    activations[f'skip_enc_{stage_idx}'] = y[:, -1:, :].clone()
                    # Also capture position 6 (from cache/prefill)
                    activations[f'skip_enc_{stage_idx}_pos6'] = y[:, 0:1, :].clone()
                    
                    decoder_start_positions[stage_idx - 1] = skip_start_pos
                    
                    shifted_x = torch.zeros_like(y)
                    if skip_start_pos > 0:
                        cached_dec = kv_cache.get_decoder_output(stage_idx, skip_start_pos - 1, 1)
                        shifted_x[:, 0:1] = cached_dec
                        # Capture cached decoder output used in shift
                        activations[f'skip_cached_dec_{stage_idx}'] = cached_dec.clone()
                    if unpool_len > 1:
                        shifted_x[:, 1:] = x_unpooled[:, :-1]
                    
                    # Capture the shifted_x last position (what gets added to encoder for last output)
                    activations[f'skip_shifted_{stage_idx}'] = shifted_x[:, -1:, :].clone()
                    
                    x = y + shifted_x
                    activations[f'dec_skip_{stage_idx}'] = x[:, -1:, :].clone()
            
            # Set final cache positions
            for stage_idx in range(max_active_stage + 1):
                encoder_end = stage_positions[stage_idx] + stage_token_counts.get(stage_idx, 0)
                decoder_end = decoder_start_positions[stage_idx] + decoder_token_counts.get(stage_idx, 0)
                final_pos = max(encoder_end, decoder_end)
                kv_cache.cache_seqlens_per_stage[stage_idx].fill_(final_pos)
            
            x = x[:, -1:, :]
            
            # LM head
            x = norm(x)
            activations['pre_lm_head'] = x.clone()
            softcap = 15
            logits = model.lm_head(x)
            logits = logits[..., :model.config.vocab_size]
            logits = logits.float()
            logits = softcap * torch.tanh(logits / softcap)
            
            return logits, activations
        
        def compare_activations(naive_acts, cached_acts, threshold=1e-3):
            """
            Compare activations and return detailed diff report.
            """
            report = []
            all_keys = sorted(set(naive_acts.keys()) | set(cached_acts.keys()))
            
            for key in all_keys:
                if key.startswith('_'):
                    continue  # Skip metadata
                
                if key not in naive_acts:
                    report.append((key, None, None, None, "MISSING in naive"))
                    continue
                if key not in cached_acts:
                    report.append((key, None, None, None, "MISSING in cached"))
                    continue
                
                naive_val = naive_acts[key]
                cached_val = cached_acts[key]
                
                # Handle shape mismatch
                if naive_val.shape != cached_val.shape:
                    report.append((key, None, None, None, f"SHAPE MISMATCH: naive={naive_val.shape} vs cached={cached_val.shape}"))
                    continue
                
                diff = (naive_val - cached_val).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                match = max_diff < threshold
                
                report.append((key, max_diff, mean_diff, match, None))
            
            return report
        
        def print_activation_report(report, print_fn):
            """Pretty-print the activation comparison report."""
            print_fn(f"\n{'Layer/Stage':<25} {'Max Diff':<15} {'Mean Diff':<15} {'Match':<10} {'Note':<20}")
            print_fn("-" * 85)
            
            for key, max_diff, mean_diff, match, note in report:
                if note:
                    print_fn(f"{key:<25} {'--':<15} {'--':<15} {'--':<10} {note:<20}")
                else:
                    match_str = "✓" if match else "✗"
                    print_fn(f"{key:<25} {max_diff:<15.6f} {mean_diff:<15.6f} {match_str:<10}")
        
        # =====================================================================
        # Run the comparison
        # =====================================================================
        
        # Create KV cache for engine
        kv_cache = UNetKVCache(
            batch_size=1,
            config=config,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        
        # Create engine (but we'll use instrumented methods)
        engine = UNetEngine(model, tokenizer)
        
        # Current sequences
        naive_ids = prompt_tokens.copy()
        cached_ids = prompt_tokens.copy()
        
        # Store all step data for full comparison printout
        all_step_data = []
        
        print0(f"\n{'Step':<6} {'Naive Token':<15} {'Cached Token':<15} {'Logits Match':<15} {'Max Diff':<15}")
        print0("-" * 80)
        
        mismatch_found = False
        mismatch_step = None
        
        with autocast_ctx:
            for step in range(kwargs['max_tokens']):
                # ===== NAIVE: Full forward pass on entire sequence =====
                ids_tensor = torch.tensor([naive_ids], dtype=torch.long, device=device)
                naive_logits, naive_activations = naive_forward_instrumented(model, ids_tensor)
                naive_last_logits = naive_logits[:, -1, :]
                
                # ===== CACHED: Use engine's instrumented forward =====
                if step == 0:
                    ids_tensor = torch.tensor([cached_ids], dtype=torch.long, device=device)
                    cached_logits, cached_activations = cached_prefill_instrumented(engine, ids_tensor, kv_cache)
                    cached_last_logits = cached_logits[:, -1, :]
                else:
                    ids_tensor = torch.tensor([[cached_ids[-1]]], dtype=torch.long, device=device)
                    cached_logits, cached_activations = cached_decode_instrumented(engine, ids_tensor, kv_cache)
                    cached_last_logits = cached_logits[:, -1, :]
                
                # ===== Compare logits =====
                logits_diff = (naive_last_logits - cached_last_logits).abs()
                max_diff = logits_diff.max().item()
                mean_diff = logits_diff.mean().item()
                
                logits_match = max_diff < 1e-3
                
                # ===== Sample next token (greedy for determinism) =====
                naive_next_token = naive_last_logits.argmax(dim=-1).item()
                cached_next_token = cached_last_logits.argmax(dim=-1).item()
                
                # Decode tokens for display
                naive_str = tokenizer.decode([naive_next_token]).replace('\n', '\\n')[:12]
                cached_str = tokenizer.decode([cached_next_token]).replace('\n', '\\n')[:12]
                
                match_str = "✓" if logits_match else "✗"
                
                # Compute activation comparison report
                report = compare_activations(naive_activations, cached_activations)
                
                # Store step data
                step_data = {
                    'step': step,
                    'seq_len': len(naive_ids),
                    'naive_token': naive_next_token,
                    'cached_token': cached_next_token,
                    'naive_str': naive_str,
                    'cached_str': cached_str,
                    'logits_match': logits_match,
                    'max_diff': max_diff,
                    'mean_diff': mean_diff,
                    'report': report,
                    'naive_top5': naive_last_logits.topk(5, dim=-1),
                    'cached_top5': cached_last_logits.topk(5, dim=-1),
                    'max_active_stage': cached_activations.get('_max_active_stage'),
                    'stage_positions': cached_activations.get('_stage_positions'),
                }
                all_step_data.append(step_data)
                
                print0(f"{step:<6} {naive_str:<15} {cached_str:<15} {match_str:<15} {max_diff:<15.6f}")
                
                # Check for mismatch
                if not logits_match or naive_next_token != cached_next_token:
                    mismatch_found = True
                    mismatch_step = step
                    break
                
                # Update sequences
                naive_ids.append(naive_next_token)
                cached_ids.append(cached_next_token)
        
        # =====================================================================
        # Print full comparison for all processed tokens
        # =====================================================================
        print0("\n" + "=" * 100)
        print0("FULL ACTIVATION COMPARISON FOR ALL PROCESSED TOKENS")
        print0("=" * 100)
        
        for step_data in all_step_data:
            step = step_data['step']
            is_mismatch = (step == mismatch_step)
            
            header_char = "!" if is_mismatch else "-"
            print0("\n" + header_char * 100)
            if is_mismatch:
                print0(f"STEP {step} [MISMATCH] - seq_len={step_data['seq_len']}")
            else:
                print0(f"STEP {step} [OK] - seq_len={step_data['seq_len']}")
            print0(header_char * 100)
            
            print0(f"Naive token:  {step_data['naive_token']} ('{step_data['naive_str']}')")
            print0(f"Cached token: {step_data['cached_token']} ('{step_data['cached_str']}')")
            print0(f"Logits max diff: {step_data['max_diff']:.6f}, mean diff: {step_data['mean_diff']:.6f}")
            
            if step_data['max_active_stage'] is not None:
                print0(f"Max active stage: {step_data['max_active_stage']}")
            if step_data['stage_positions'] is not None:
                print0(f"Stage positions: {step_data['stage_positions']}")
            
            # Print activation report
            print_activation_report(step_data['report'], print0)
            
            # Find first divergent layer for this step
            first_divergent = None
            for key, max_d, mean_d, match, note in step_data['report']:
                if note or (match is not None and not match):
                    first_divergent = key
                    break
            
            if first_divergent:
                print0(f"\n>>> First divergence at: {first_divergent}")
            
            if is_mismatch:
                # Show top-5 predictions for mismatch
                print0(f"\nNaive top-5:  {step_data['naive_top5'].indices[0].tolist()}")
                print0(f"             logits: {[f'{v:.3f}' for v in step_data['naive_top5'].values[0].tolist()]}")
                print0(f"Cached top-5: {step_data['cached_top5'].indices[0].tolist()}")
                print0(f"             logits: {[f'{v:.3f}' for v in step_data['cached_top5'].values[0].tolist()]}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        print0("\n" + "=" * 100)
        print0("SUMMARY")
        print0("=" * 100)
        
        if mismatch_found:
            print0(f"\n✗ Mismatch detected at step {mismatch_step}")
            print0(f"  Total steps processed: {len(all_step_data)}")
            print0(f"  Steps OK before mismatch: {mismatch_step}")
        else:
            print0(f"\n✓ All {len(all_step_data)} steps completed with matching outputs!")
        
        # Show activation diff trend across steps
        print0("\n" + "-" * 50)
        print0("Activation diff trend across steps:")
        print0("-" * 50)
        
        # Collect all unique keys
        all_keys = set()
        for sd in all_step_data:
            for key, _, _, _, _ in sd['report']:
                if not key.startswith('_'):
                    all_keys.add(key)
        all_keys = sorted(all_keys)
        
        # Print header
        step_header = "".join([f"{sd['step']:<8}" for sd in all_step_data])
        print0(f"{'Layer':<25} {step_header}")
        print0("-" * (25 + 8 * len(all_step_data)))
        
        # Print max diff for each key across steps
        for key in all_keys:
            row = f"{key:<25} "
            for sd in all_step_data:
                # Find this key in the report
                found = False
                for k, max_d, mean_d, match, note in sd['report']:
                    if k == key:
                        if note:
                            row += f"{'ERR':<8}"
                        elif match:
                            row += f"{max_d:<8.4f}"
                        else:
                            row += f"*{max_d:<7.4f}"  # Asterisk for mismatch
                        found = True
                        break
                if not found:
                    row += f"{'--':<8}"
            print0(row)
        
        print0("\n(* = exceeds threshold)")

    print0("\n" + "=" * 80)
    print0("Test completed.")
    print0("=" * 80)
