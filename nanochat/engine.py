"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from contextlib import nullcontext 

# -----------------------------------------------------------------------------
# Calculator tool helpers
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    # Remove commas from numbers
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    Works hand-in-hand with the GPT model to maintain the KV cache.
    Note that the .pos advances automatically after the last layer of the Transformer inserts.
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        # Each of K/V is of shape (B, H, T, D) and we have one per layer of the Transformer.
        self.kv_shape = (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        self.kv_cache = None
        self.pos = 0 # current position in time in the cache
        self.mla_cache = {} 
    def update_mla(self, layer_idx, kv_latent, k_rope):
        """
        Update MLA cache
        kv_latent: (B, T_new, d_latent)
        k_rope: (B, T_new, n_head, rope_dim)
        Returns: (kv_latent_full, k_rope_full)
        """
        if layer_idx not in self.mla_cache:
            self.mla_cache[layer_idx] = {'latent': kv_latent, 'k_rope': k_rope}
            return kv_latent, k_rope
        else:
            # Concatenate history
            cached = self.mla_cache[layer_idx]
            kv_latent_full = torch.cat([cached['latent'], kv_latent], dim=1)
            k_rope_full = torch.cat([cached['k_rope'], k_rope], dim=1)
            # Update cache
            self.mla_cache[layer_idx] = {'latent': kv_latent_full, 'k_rope': k_rope_full}
            return kv_latent_full, k_rope_full
    def reset(self):
        self.pos = 0

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        """
        Prefill given another KV cache. Optionally expand along batch dim.
        This is used when we do batch 1 prefill and then want to generate
        multiple samples in parallel from there.
        """
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: num_heads, 4: seq_len, 5: head_dim
            if ix in [0, 1, 3, 5]:
                # num_layers, k/v, num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 4:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        self.kv_cache[:, :, :, :, :other.pos, :] = other.kv_cache
        # 4) update the pos
        self.pos = other.pos

    def insert_kv(self, layer_idx, k, v):
        # Lazy initialize the cache here because we need to know the dtype/device
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=k.dtype, device=k.device)
        # Insert new keys/values to the cache and return the full cache so far
        B, H, T_add, D = k.size()
        t0, t1 = self.pos, self.pos + T_add
        # Dynamically grow the cache if needed
        if t1 > self.kv_cache.size(4):
            t_needed = t1 + 1024 # as much as we need plus buffer of 1024
            t_needed = (t_needed + 1023) & ~1023 # then round up to the nearest multiple of 1024
            additional_shape = list(self.kv_cache.shape)
            additional_shape[4] = t_needed - self.kv_cache.size(4)
            additional_cache = torch.empty(additional_shape, dtype=k.dtype, device=k.device)
            self.kv_cache = torch.cat([self.kv_cache, additional_cache], dim=4).contiguous()
            self.kv_shape = self.kv_cache.shape
        # Insert k, v into the cache
        self.kv_cache[layer_idx, 0, :, :, t0:t1] = k
        self.kv_cache[layer_idx, 1, :, :, t0:t1] = v
        # Return the full cached keys/values up to current position (as a view)
        key_view = self.kv_cache[layer_idx, 0, :, :, :t1]
        value_view = self.kv_cache[layer_idx, 1, :, :, :t1]
        # Increment pos after the last layer of the Transformer processes
        if layer_idx == self.kv_cache.size(0) - 1:
            self.pos = t1
        return key_view, value_view


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None:
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
@torch.inference_mode()
def speculative_sample(target_logits, draft_logits, draft_token, rng, temperature=1.0):
    """
    Speculative sampling with rejection correction.
    Ensures output distribution matches original autoregressive decoding
    
    Args:
        target_logits: (vocab_size,) logits from target model
        draft_logits: (vocab_size,) logits from draft model  
        draft_token: int, token proposed by draft model
        rng: torch random generator
        temperature: sampling temperature
        
    Returns:
        accepted: bool, whether draft token is accepted
        final_token: int, the final sampled token
    """
    if temperature == 0.0:
        # Greedy decoding: accept if draft matches target's argmax
        target_token = torch.argmax(target_logits).item()
        return draft_token == target_token, target_token
    
    # Convert to probabilities
    target_probs = F.softmax(target_logits / temperature, dim=-1)
    draft_probs = F.softmax(draft_logits / temperature, dim=-1)
    
    # Acceptance probability: min(1, p_target / p_draft)
    draft_prob = draft_probs[draft_token].item()
    target_prob = target_probs[draft_token].item()
    
    # Avoid division by zero
    if draft_prob < 1e-10:
        acceptance_prob = 0.0
    else:
        acceptance_prob = min(1.0, target_prob / draft_prob)
    
    # Sample acceptance decision
    if torch.rand(1, device=rng.device, generator=rng).item() < acceptance_prob:
        return True, draft_token
    else:
        # Rejection: sample from corrected distribution max(0, p_target - p_draft)
        corrected_probs = torch.clamp(target_probs - draft_probs, min=0.0)
        corrected_probs = corrected_probs / corrected_probs.sum()
        final_token = torch.multinomial(corrected_probs, num_samples=1, generator=rng).item()
        return False, final_token

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
        self.forced_tokens = deque() # Queue of tokens to force inject
        self.in_python_block = False # Whether we are inside a python block
        self.python_expr_tokens = [] # Tokens of the current python expression
        self.completed = False # Whether this row has completed generation

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            if all(state.completed for state in row_states):
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                # TODO: we should sample a token for each row instead of broadcasting
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
                sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            token_column = [] # contains the next token id along each row
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                # Select the next token in this row
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        Returns a list of token sequences (list of lists of ints).
        Terminal tokens (assistant_end, bos) are not included in the results.
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

    @torch.inference_mode()
    def generate_batch_prompts(self, prompts_list, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """
        Batch inference: true dynamic batching supporting prompts of different lengths
        
        Key improvements:
        1. Padding: pad prompts of different lengths to the same length
        2. Attention Mask: use mask to mark padding positions
        3. True batch processing: use batch forward in both Prefill and Decode stages
        4. Shared KV Cache: all sequences share a batched KV cache
        
        Args:
            prompts_list: List[List[int]], list of token sequences for multiple prompts
            max_tokens: int, maximum number of tokens to generate per sequence
            temperature: float, sampling temperature
            top_k: int, top-k sampling
            seed: int, random seed
            
        Yields:
            token_columns: List[int], tokens generated for each sequence at current step
            active_mask: List[bool], whether each sequence is still generating
        """
        batch_size = len(prompts_list)
        assert batch_size > 0, "prompts_list must not be empty"
        device = self.model.get_device()
        
        # Create independent random number generators for each sequence
        rngs = [torch.Generator(device=device) for _ in range(batch_size)]
        for rng in rngs:
            rng.manual_seed(seed)
        
        # Get special tokens
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        pad_token_id = self.tokenizer.get_pad_token_id() if hasattr(self.tokenizer, 'get_pad_token_id') else bos
        
        # ========== Step 1: Padding and creating Attention Mask ==========
        prompt_lengths = [len(p) for p in prompts_list]
        max_prompt_len = max(prompt_lengths)
        
        # Pad all prompts to the same length
        padded_prompts = []
        for prompt in prompts_list:
            pad_len = max_prompt_len - len(prompt)
            # Left padding (keep generation position aligned)
            padded = [pad_token_id] * pad_len + prompt
            padded_prompts.append(padded)
        
        # Create attention mask: 1 for valid positions, 0 for padding
        attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=torch.bool, device=device)
        for i, plen in enumerate(prompt_lengths):
            attention_mask[i, -plen:] = True  # Mark right-side valid portion
        
        # ========== Step 2: Batch Prefill ==========
        m = self.model.config
        kv_length_hint = max_prompt_len + (max_tokens or 100)
        if max_tokens is None:
            kv_length_hint = self.model.config.sequence_len
        
        # Create shared batched KV cache
        kv_cache = KVCache(
            batch_size=batch_size,
            num_heads=m.n_kv_head,
            seq_len=kv_length_hint,
            head_dim=m.n_embd // m.n_head,
            num_layers=m.n_layer
        )
        
        # Batch prefill (one forward pass processes all prompts)
        batch_ids = torch.tensor(padded_prompts, dtype=torch.long, device=device)  # (B, max_prompt_len)
        logits = self.model.forward(batch_ids, kv_cache=kv_cache)  # (B, T, vocab_size)
        logits_last = logits[:, -1, :]  # (B, vocab_size)
        
        # Sample first token (each sequence samples independently)
        if temperature == 0.0:
            next_ids = torch.argmax(logits_last, dim=-1, keepdim=True)  # (B, 1)
        else:
            next_ids_list = []
            for i in range(batch_size):
                logits_i = logits_last[i:i+1, :]  # (1, vocab_size)
                next_id_i = sample_next_token(logits_i, rngs[i], temperature, top_k)
                next_ids_list.append(next_id_i)
            next_ids = torch.cat(next_ids_list, dim=0)  # (B, 1)
        
        # ========== Step 3: Batch Decode ==========
        completed = [False] * batch_size
        num_generated = 0
        
        while True:
            # Stop conditions
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(completed):
                break
            
            # Get current tokens
            current_tokens = next_ids[:, 0].tolist()
            
            # Check termination conditions
            for i in range(batch_size):
                if not completed[i] and (current_tokens[i] == assistant_end or current_tokens[i] == bos):
                    completed[i] = True
            
            # Yield current state
            active_mask = [not c for c in completed]
            yield current_tokens, active_mask
            
            num_generated += 1
            
            if all(completed):
                break
            
            # Batch forward pass (true batch processing: one forward processes all sequences)
            logits = self.model.forward(next_ids, kv_cache=kv_cache)  # (B, 1, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Sample next token
            if temperature == 0.0:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)  # (B, 1)
            else:
                next_ids_list = []
                for i in range(batch_size):
                    if not completed[i]:
                        logits_i = logits[i:i+1, :]  # (1, vocab_size)
                        next_id_i = sample_next_token(logits_i, rngs[i], temperature, top_k)
                        next_ids_list.append(next_id_i)
                    else:
                        # Fill completed sequences with pad token
                        next_ids_list.append(torch.tensor([[pad_token_id]], device=device))
                next_ids = torch.cat(next_ids_list, dim=0)  # (B, 1)

    def generate_batch_prompts_complete(self, prompts_list, **kwargs):
        """
        Non-streaming version of batch inference
        
        Args:
            prompts_list: List[List[int]], list of token sequences for multiple prompts
            **kwargs: other parameters passed to generate_batch_prompts
        
        Returns:
            results: List[List[int]], generation results for each prompt (includes input prompt and generated tokens)
        """
        batch_size = len(prompts_list)
        results = [list(prompt) for prompt in prompts_list]  # Copy original prompts
        
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        for token_column, active_mask in self.generate_batch_prompts(prompts_list, **kwargs):
            for i, (token, is_active) in enumerate(zip(token_column, active_mask)):
                if is_active and token != assistant_end and token != bos:
                    results[i].append(token)
        
        return results
    @torch.inference_mode()
    def generate_speculative(self, tokens, draft_model, num_samples=1, max_tokens=None, 
                            temperature=1.0, top_k=None, seed=42, gamma=4):
        """
        Speculative decoding: use draft model to propose tokens, verify with target model.
        
        Algorithm:
        1. Draft phase: Generate gamma tokens with draft model (autoregressive)
        2. Verify phase: Verify all tokens with target model (parallel forward)
        3. Accept/reject: Use speculative sampling to maintain correct distribution
        
        Args:
            tokens: List[int], input prompt
            draft_model: smaller model for drafting
            num_samples: must be 1
            max_tokens: max tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            seed: random seed
            gamma: speculation length (lookahead)
            
        Yields:
            ([token], [mask], stats_dict)
        """
        assert num_samples == 1, "Speculative decoding only supports num_samples=1"
        
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        
        # Special tokens
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        
        # Setup KV caches for both models
        m = self.model.config
        dm = draft_model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_draft_kwargs = {"num_heads": dm.n_kv_head, "head_dim": dm.n_embd // dm.n_head, "num_layers": dm.n_layer}
        kv_length_hint = len(tokens) + (max_tokens or 100)
        
        kv_cache_target = KVCache(batch_size=1, seq_len=kv_length_hint, **kv_model_kwargs)
        kv_cache_draft = KVCache(batch_size=1, seq_len=kv_length_hint, **kv_draft_kwargs)
        
        # Prefill both models and save logits for first sampling
        prompt_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        target_logits_prefill = self.model.forward(prompt_ids, kv_cache=kv_cache_target)
        draft_logits_prefill = draft_model.forward(prompt_ids, kv_cache=kv_cache_draft)
        
        # Save logits at last position (for sampling first token)
        next_target_logits = target_logits_prefill[0, -1, :].clone()  # (vocab_size,)
        next_draft_logits = draft_logits_prefill[0, -1, :].clone()    # (vocab_size,)
        
        # Statistics
        stats = {"total_drafted": 0, "total_accepted": 0, "iterations": 0}
        num_generated = 0
        
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            
            stats["iterations"] += 1
            
            # ========== DRAFT PHASE ==========
            # Generate gamma candidate tokens with draft model
            draft_tokens = []
            draft_logits_list = []
            
            draft_kv_start = kv_cache_draft.get_pos()
            current_draft_logits = next_draft_logits
            
            for i in range(gamma):
                # Sample from current logits
                if temperature == 0.0:
                    next_token = torch.argmax(current_draft_logits).item()
                else:
                    next_token = sample_next_token(
                        current_draft_logits.unsqueeze(0), rng, temperature, top_k
                    )[0, 0].item()
                
                draft_tokens.append(next_token)
                draft_logits_list.append(current_draft_logits)  # Save logits used for this token
                
                # Check for end tokens
                if next_token == assistant_end or next_token == bos:
                    break
                
                # Forward this token to get logits for next position
                if i < gamma - 1:  # Don't need logits after last token
                    token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)
                    logits_output = draft_model.forward(token_tensor, kv_cache=kv_cache_draft)
                    current_draft_logits = logits_output[0, 0, :]
            
            stats["total_drafted"] += len(draft_tokens)
            
            # ========== VERIFICATION PHASE ==========
            # Forward all draft tokens through target model (parallel)
            target_kv_start = kv_cache_target.get_pos()
            
            # Prepare logits list: start with the "current" logits
            target_logits_list = [next_target_logits]
            
            # Forward all draft tokens at once
            if len(draft_tokens) > 0:
                draft_tensor = torch.tensor([draft_tokens], dtype=torch.long, device=device)
                target_logits_batch = self.model.forward(draft_tensor, kv_cache=kv_cache_target)
                # target_logits_batch[0, i, :] = logits after processing draft_tokens[0:i+1]
                # These are the logits for evaluating draft_tokens[i+1] (if exists)
                for i in range(len(draft_tokens)):
                    target_logits_list.append(target_logits_batch[0, i, :])
            
            # ========== ACCEPT/REJECT PHASE ==========
            num_accepted = 0
            final_token = None
            
            for i in range(len(draft_tokens)):
                # Verify draft_tokens[i] using:
                # - target_logits_list[i]: target's prediction before seeing draft_tokens[i]
                # - draft_logits_list[i]: draft's prediction before seeing draft_tokens[i]
                # - draft_tokens[i]: the proposed token
                
                accepted, sampled_token = speculative_sample(
                    target_logits_list[i], 
                    draft_logits_list[i], 
                    draft_tokens[i],
                    rng, 
                    temperature
                )
                
                final_token = sampled_token
                num_generated += 1
                
                # Yield the token
                yield [final_token], [1], stats
                
                # Check termination
                if final_token == assistant_end or final_token == bos:
                    stats["total_accepted"] += num_accepted
                    return
                
                if max_tokens is not None and num_generated >= max_tokens:
                    stats["total_accepted"] += num_accepted
                    return
                
                if accepted:
                    num_accepted += 1
                    # Continue to next token
                else:
                    # Rejection: need to rollback and restart
                    # Target KV cache is now at position: target_kv_start + i + 1
                    # This includes tokens[0:target_kv_start] + draft_tokens[0:i+1]
                    # But draft_tokens[i] was rejected and replaced with sampled_token
                    # So we need to:
                    # 1. Rollback target KV cache to position target_kv_start + i (before draft_tokens[i])
                    # 2. Forward sampled_token through target model to get correct logits
                    
                    # Rollback target KV cache: set pos back to before the rejected token
                    kv_cache_target.pos = target_kv_start + i
                    
                    # Forward the sampled token (not draft token) through target model
                    token_tensor = torch.tensor([[sampled_token]], dtype=torch.long, device=device)
                    target_logits_out = self.model.forward(token_tensor, kv_cache=kv_cache_target)
                    next_target_logits = target_logits_out[0, 0, :]
                    
                    # Rollback draft KV cache: if some tokens were accepted, keep them
                    # Only rollback to before the rejected token, not before all speculation
                    kv_cache_draft.pos = draft_kv_start + i
                    
                    # Forward the sampled token (not draft token) through draft model
                    draft_logits_out = draft_model.forward(token_tensor, kv_cache=kv_cache_draft)
                    next_draft_logits = draft_logits_out[0, 0, :]
                    
                    break  # Stop verification, start new speculation
            else:
                # All tokens accepted!
                stats["total_accepted"] += num_accepted
                
                # Performance optimization: use the "free" token from target model
                # target_logits_list[-1] is the logits after processing all draft tokens
                # We can sample one more token for free since target model already computed it
                if len(draft_tokens) > 0:
                    # Sample the free token from target model
                    if temperature == 0.0:
                        free_token = torch.argmax(target_logits_list[-1]).item()
                    else:
                        free_token = sample_next_token(
                            target_logits_list[-1].unsqueeze(0), rng, temperature, top_k
                        )[0, 0].item()
                    
                    num_generated += 1
                    yield [free_token], [1], stats
                    
                    # Check termination
                    if free_token == assistant_end or free_token == bos:
                        return
                    
                    if max_tokens is not None and num_generated >= max_tokens:
                        return
                    
                    # Forward the free token through both models to update their states
                    free_token_tensor = torch.tensor([[free_token]], dtype=torch.long, device=device)
                    target_logits_out = self.model.forward(free_token_tensor, kv_cache=kv_cache_target)
                    next_target_logits = target_logits_out[0, 0, :]
                    
                    draft_logits_out = draft_model.forward(free_token_tensor, kv_cache=kv_cache_draft)
                    next_draft_logits = draft_logits_out[0, 0, :]
                else:
                    # No draft tokens were generated, use target logits as fallback
                    next_target_logits = target_logits_list[-1]
                    next_draft_logits = next_target_logits.clone()
            
            # Continue to next speculation round


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    device_type = autodetect_device_type()
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()

    # load the model and tokenizer
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    with autocast_ctx:
        for token in stream:
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
    torch.cuda.synchronize()
    t0 = time.time()
    with autocast_ctx:
        for token_column, token_masks in stream:
            token = token_column[0] # only print out the first row
            generated_tokens.append(token)
            chunk = tokenizer.decode([token])
            print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")
