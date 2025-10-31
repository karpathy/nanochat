"""
Engine for efficient inference of our models.

Everything works around token sequences:
- The user can send token sequences to the engine
- The engine returns the next token

Notes:
- The engine knows nothing about tokenization, it's purely token id sequences.

The whole thing is made as efficient as possible.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT
from nanochat.kv_cache import KVCache
from nanochat.hypercube import HypercubeTopology, HypercubeEmbeddingLayer, LongTermMemory # Import HypercubeTopology
from nanochat.abacus_encoder import AbacusEncoder
from nanochat.self_model import InternalSelfModel
from nanochat.abacus_state_memory import AbacusStateMemory
from nanochat.memetic_learning import MemeticLearningLayer
from nanochat.conscious_integration import ConsciousIntegrationLayer
from nanochat.gpt import GPT, PsycheController

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
                return eval(formula)
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
            if ix in [0, 1, 3, 5]:
                # num_layers, batch_size, num_heads, head_dim must match
                assert dim1 == dim2, f"Batch dim mismatch: {dim1} != {dim2}"
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

    def retrieve_episode(self, start_pos, end_pos):
        """
        Retrieves a slice of the KV cache representing an 'episode' or a specific attention span.
        """
        assert self.kv_cache is not None, "KV cache is empty, cannot retrieve episode."
        assert start_pos >= 0 and end_pos <= self.pos, "Invalid start or end position for episode retrieval."
        assert start_pos < end_pos, "Start position must be less than end position."

        # Return a view of the relevant part of the cache
        # kv_cache shape: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
        # We want to retrieve for all layers, and both k and v
        episode_k = self.kv_cache[:, 0, :, :, start_pos:end_pos, :]
        episode_v = self.kv_cache[:, 1, :, :, start_pos:end_pos, :]
        return episode_k, episode_v


# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_from_logits(logits, rng, temperature, top_k):
    """Sample a single next concept ID from given logits of shape (B, num_concept_ids). Returns (B, 1)."""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < v[:, [-1]]] = -float('Inf')

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    idx_next = torch.multinomial(probs, num_samples=1, generator=rng)
    return idx_next


class Engine:

    def __init__(self, config: GPTConfig):
        self.config = config
        self.model = GPT(config)
        self.model.eval()
        self.model.init_weights()

        # Initialize HypercubeEmbeddingLayer
        self.concept_embedding_layer = HypercubeEmbeddingLayer(
            num_raw_concepts=config.num_concept_ids,
            embedding_dim=config.n_embd,
            hypercube_topology=self.hypercube_topology
        )

        # Initialize AbacusEncoder
        self.abacus_encoder = AbacusEncoder(
            input_dim=config.abacus_input_dim,
            embedding_dim=config.n_embd
        )
        nn.init.normal_(self.concept_embedding_layer.weight, mean=0.0, std=1.0)

        # Initialize HypercubeTopology
        n_hypercube_dim = int(math.log2(config.num_concept_ids))
        if not (2**n_hypercube_dim == config.num_concept_ids):
            raise ValueError("config.num_concept_ids must be a power of 2 for hypercube topology.")
        self.hypercube_topology = HypercubeTopology(n_hypercube_dim)

        # Initialize LongTermMemory
        self.long_term_memory = LongTermMemory(
            embedding_dim=config.n_embd,
            max_memory_size=1000, # Default max memory size
            top_k_retrieval=5     # Default top-k retrieval
        )

        # Initialize InternalSelfModel
        self.internal_self_model = InternalSelfModel(
            embedding_dim=config.n_embd,
            num_concepts=config.num_concept_ids
        )

        # Initialize AbacusStateMemory
        self.abacus_state_memory = AbacusStateMemory(
            max_memory_size=100, # Default max memory size
            abacus_input_dim=config.abacus_input_dim
        )

        # Initialize MemeticLearningLayer
        self.memetic_learning_layer = MemeticLearningLayer(
            config=config,
            abacus_encoder=self.abacus_encoder,
            internal_self_model=self.internal_self_model
        )

        # Initialize PsycheController
        self.psyche_controller = PsycheController(
            config=config
        )

        # Initialize ConsciousIntegrationLayer
        self.conscious_integration_layer = ConsciousIntegrationLayer(
            config=config,
            abacus_encoder=self.abacus_encoder
        )

        self._concept_encoder_placeholder = None # This will be set later
        self._concept_attention_fusion_placeholder = nn.Linear(config.n_embd * 2, config.n_embd)

        # Placeholder for concept encoder
        self.concept_encoder = self._concept_encoder_placeholder

    def _concept_encoder_placeholder(self, concept_list: list[int] | dict) -> torch.Tensor:
        """
        Placeholder for a more sophisticated concept encoder.
        For now, it assumes concept_list directly contains integer concept IDs or a dict with 'abacus_pattern'.
        """
        if isinstance(concept_list, dict) and "abacus_pattern" in concept_list:
            # If an abacus pattern is provided, encode it using the AbacusEncoder
            abacus_pattern = concept_list["abacus_pattern"]
            # Ensure abacus_pattern is a tensor and has the correct shape
            if not isinstance(abacus_pattern, torch.Tensor):
                abacus_pattern = torch.tensor(abacus_pattern, dtype=torch.float32)
            # Add batch dimension if missing
            if abacus_pattern.dim() == 1:
                abacus_pattern = abacus_pattern.unsqueeze(0)
            return self.abacus_encoder(abacus_pattern)
        elif isinstance(concept_list, list):
            # Otherwise, assume it's a list of integer concept IDs
            return torch.tensor(concept_list, dtype=torch.long, device=self.gpt.get_device())
        else:
            raise ValueError("concept_list must be a list of integers or a dict with 'abacus_pattern'.")

    def _concept_memory_retrieve_placeholder(self, current_embedding: torch.Tensor) -> torch.Tensor:
        # Placeholder for concept memory retrieval logic
        # This would typically involve searching a memory bank for relevant concepts
        # based on the current_embedding and returning their embeddings.
        # For now, it returns a zero tensor of appropriate shape.
        return torch.zeros_like(current_embedding)

    def _concept_attention_fusion_placeholder(self, transformer_output: torch.Tensor, retrieved_concepts: torch.Tensor) -> torch.Tensor:
        # Placeholder for concept attention fusion logic
        # This would combine the transformer's output with the retrieved concept embeddings
        # using some attention mechanism.
        # For now, it just returns the transformer_output unchanged.
        return transformer_output

    def sample_from_logits(self, concept_logits: torch.Tensor, temperature: float = 1.0, top_k: int = None) -> torch.Tensor:
        # Apply temperature
        if temperature == 0.0:
            next_concept_id = torch.argmax(concept_logits, dim=-1)
        else:
            concept_logits = concept_logits / temperature
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(concept_logits, min(top_k, concept_logits.size(-1)))
                concept_logits[concept_logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(concept_logits, dim=-1)
            next_concept_id = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return next_concept_id

    @torch.no_grad()
    def generate(self, input_embeddings: torch.Tensor, max_new_concepts: int = 20, temperature: float = 1.0, abacus_embedding: torch.Tensor | None = None, working_memory_window: int = 0) -> list[int]:
        B, T, C = input_embeddings.size()
        generated_embeddings = []

        if abacus_embedding is None:
            abacus_embedding = torch.zeros(B, 1, self.config.abacus_input_dim, device=input_embeddings.device)

        # Long-Term Memory retrieval for prefill
        prefill_long_term_memory_embeddings = self.long_term_memory.retrieve(input_embeddings[:, -1, :].squeeze(0))

        # Get psyche weights from PsycheController for prefill
        prefill_psyche_weights = self.psyche_controller(input_embeddings[:, -1, :])

        # Prefill the model with input_embeddings
        concept_logits, kv_cache, x_id_prefill, x_ego_prefill, x_superego_prefill = self.model.forward_prefill(
            input_embeddings,
            abacus_embedding=abacus_embedding,
            long_term_memory_embeddings=prefill_long_term_memory_embeddings,
            psyche_weights=prefill_psyche_weights
        )

        # Conscious Integration Layer for prefill
        synthesized_state_prefill = self.conscious_integration_layer.forward(
            id_output=x_id_prefill,
            ego_output=x_ego_prefill,
            superego_output=x_superego_prefill,
            long_term_memory_embeddings=prefill_long_term_memory_embeddings,
            memetic_fitness=None, # Memetic fitness is not available during prefill
            abacus_state=abacus_embedding
        )

        # Combine concept_logits from GPT and synthesized_state_prefill
        concept_logits = concept_logits + synthesized_state_prefill

        # Sample the first concept ID from the last token's logits
        next_concept_id = self.sample_from_logits(concept_logits[:, -1, :], temperature)
        next_embedding = self.concept_embedding_layer(next_concept_id)
        generated_embeddings.append(next_embedding)
        self.long_term_memory.store(next_embedding.squeeze(0)) # Store the generated embedding

        # Abacus State Memory and Encoder integration for the first step
        abacus_pattern = self.abacus_encoder(next_embedding)
        self.abacus_state_memory.store(abacus_pattern)
        abacus_embedding = self.abacus_state_memory.retrieve()

        for _ in range(max_new_concepts - 1):
            # Working Memory retrieval
            episodic_kv = None
            if working_memory_window > 0 and kv_cache.get_pos() > working_memory_window:
                start_pos = kv_cache.get_pos() - working_memory_window
                end_pos = kv_cache.get_pos()
                episodic_kv = kv_cache.retrieve_episode(start_pos, end_pos)

            # Long-Term Memory retrieval
            long_term_memory_embeddings = self.long_term_memory.retrieve(next_embedding.squeeze(0))

            # Get psyche weights from PsycheController
            psyche_weights = self.psyche_controller(next_embedding)

            concept_logits, kv_cache, x_id_step, x_ego_step, x_superego_step = self.model.forward_step(
                next_embedding,
                kv_cache,
                abacus_embedding=abacus_embedding,
                episodic_kv=episodic_kv,
                long_term_memory_embeddings=long_term_memory_embeddings,
                psyche_weights=psyche_weights
            )

            # Memetic Learning Layer integration
            memetic_fitness = self.memetic_learning_layer.forward(next_embedding, abacus_pattern)

            # Conscious Integration Layer for step
            synthesized_state_step = self.conscious_integration_layer.forward(
                id_output=x_id_step,
                ego_output=x_ego_step,
                superego_output=x_superego_step,
                long_term_memory_embeddings=long_term_memory_embeddings,
                memetic_fitness=memetic_fitness,
                abacus_state=abacus_embedding
            )

            # Combine concept_logits from GPT and synthesized_state_step
            concept_logits = concept_logits + synthesized_state_step.squeeze(1) # Squeeze to match dimensions

            next_concept_id = self.sample_from_logits(concept_logits, temperature)
            next_embedding = self.concept_embedding_layer(next_concept_id)
            generated_embeddings.append(next_embedding)
            self.long_term_memory.store(next_embedding.squeeze(0)) # Store the generated embedding

            # Abacus State Memory and Encoder integration for subsequent steps
            abacus_pattern = self.abacus_encoder(next_embedding)
            self.abacus_state_memory.store(abacus_pattern)
            abacus_embedding = self.abacus_state_memory.retrieve() # Update abacus_embedding for the next step

            # Memetic Learning Layer integration
            memetic_fitness = self.memetic_learning_layer.forward(next_embedding, abacus_pattern)

        return torch.stack(generated_embeddings, dim=1)

    def generate_from_concepts(self, concept_list: list[int] | dict, max_new_concepts: int = 20, temperature: float = 1.0) -> list[int]:
        # Encode the concept_list into initial input embeddings
        encoded_concepts = self.concept_encoder(concept_list)

        if encoded_concepts.dtype == torch.long:
            # If it's concept IDs, get embeddings from the concept_embedding_layer
            input_embeddings = self.concept_embedding_layer(encoded_concepts)
            abacus_embedding = None # No abacus embedding in this case
        elif encoded_concepts.dtype == torch.float:
            # If it's an abacus embedding, use it directly and set abacus_embedding
            input_embeddings = encoded_concepts
            abacus_embedding = encoded_concepts # The abacus embedding is the input embedding itself
        else:
            raise TypeError("Unexpected return type from concept_encoder.")

        # Call the main generate method
        return self.generate(input_embeddings, max_new_concepts, temperature, abacus_embedding=abacus_embedding)


        # Special tokens are no longer directly used with concept embeddings.
        # The tool use logic will need to be re-evaluated or removed if not applicable.
        # get_special = lambda s: self.tokenizer.encode_special(s)
        # python_start = get_special("<|python_start|>")
        # python_end = get_special("<|python_end|>")
        # output_start = get_special("<|output_start|>")
        # output_end = get_special("<|output_end|>")
        # assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
        # bos = self.tokenizer.get_bos_token_id() # if sampled, ends row

        # 1) Run a batch 1 prefill of the prompt embeddings
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=input_embeddings.size(0),
            seq_len=input_embeddings.size(1),
            **kv_model_kwargs,
        )
        logits = self.model.forward(input_embeddings, kv_cache=kv_cache_prefill)
        # Removed token-based sampling logic (replace with embedding generation logic later)

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (input_embeddings.size(1) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [RowState(input_embeddings[i].tolist()) for i in range(num_samples)] # Assuming input_embeddings is (B, T, C)

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


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    is equivalent to the faster Engine.generate function here.
    """
    import time
    # init compute
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
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
