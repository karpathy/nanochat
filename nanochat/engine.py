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
from nanochat.gpt import GPT, GPTConfig
from nanochat.kv_cache import KVCache
from nanochat.hypercube import HypercubeTopology, HypercubeEmbeddingLayer, LongTermMemory
from nanochat.abacus_encoder import AbacusEncoder
from nanochat.self_model import InternalSelfModel
from nanochat.abacus_state_memory import AbacusStateMemory
from nanochat.conscious_integration import ConsciousIntegrationLayer
from nanochat.memetic_learning import MemeticLearningLayer


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
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    expr = expr.replace(",", "")

    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:
            return None
        return eval_with_timeout(expr)

    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    if '.count(' not in expr:
        return None

    return eval_with_timeout(expr)



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


class Engine(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.model = GPT(config)
        self.model.eval()
        self.model.init_weights()

        # Initialize HypercubeEmbeddingLayer
        self.concept_embedding_layer = HypercubeEmbeddingLayer(
            config=config
        )

        # Initialize AbacusEncoder
        self.abacus_encoder = AbacusEncoder(
            input_dim=config.abacus_input_dim,
            embedding_dim=config.n_embd
        )
        # Placeholder for concept encoder
        self.concept_encoder = self._concept_encoder_placeholder

    def forward(self, concept_ids: torch.Tensor, target_concept_ids: torch.Tensor | None = None, previous_continuous_latent: torch.Tensor | None = None):
        # Get embeddings and losses from HypercubeEmbeddingLayer
        discrete_vertex_embeddings, continuous_latent, predicted_next_continuous_latent, reconstruction_loss, energy_loss = \
            self.concept_embedding_layer(concept_ids, previous_continuous_latent) # target_continuous_latent will be set in the training loop

        # Pass continuous_latent to the GPT model
        concept_logits, kv_cache, id_logits, ego_logits, superego_logits = self.model.forward(
            input_embeddings=continuous_latent,
            kv_cache=None, # kv_cache handling will be more complex for full training
            abacus_embedding=None, # Abacus embedding needs to be handled if used in training
            episodic_kv=None,
            long_term_memory_embeddings=None,
            psyche_weights=None
        )

        total_loss = reconstruction_loss + energy_loss

        # Calculate language modeling loss if target_concept_ids are provided
        lm_loss = torch.tensor(0.0, device=concept_ids.device)
        if target_concept_ids is not None:
            # Slice target_concept_ids to match the sequence length of concept_logits
            target_concept_ids_sliced = target_concept_ids[:, :concept_logits.size(1)]
            # Reshape for cross_entropy: (N, C) for logits, (N) for targets
            lm_loss = F.cross_entropy(concept_logits.reshape(-1, concept_logits.size(-1)), target_concept_ids_sliced.reshape(-1), ignore_index=-1)
            total_loss += lm_loss

            # Calculate auxiliary losses for deep supervision
            id_loss = F.cross_entropy(id_logits.reshape(-1, id_logits.size(-1)), target_concept_ids_sliced.reshape(-1), ignore_index=-1)
            ego_loss = F.cross_entropy(ego_logits.reshape(-1, ego_logits.size(-1)), target_concept_ids_sliced.reshape(-1), ignore_index=-1)
            superego_loss = F.cross_entropy(superego_logits.reshape(-1, superego_logits.size(-1)), target_concept_ids_sliced.reshape(-1), ignore_index=-1)

            # Add weighted auxiliary losses to total_loss
            total_loss += self.config.id_loss_weight * id_loss
            total_loss += self.config.ego_loss_weight * ego_loss
            total_loss += self.config.superego_loss_weight * superego_loss

        return total_loss, lm_loss, reconstruction_loss, energy_loss, id_loss, ego_loss, superego_loss, continuous_latent, predicted_next_continuous_latent

    def _concept_encoder_placeholder(self, concept_list: list[int] | dict) -> torch.Tensor:
        raise NotImplementedError("Concept encoder not yet implemented.")

    def _concept_memory_retrieve_placeholder(self, query_embedding: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Concept memory retrieval not yet implemented.")

    def _concept_attention_fusion_placeholder(self, query_embedding: torch.Tensor, memory_embeddings: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Concept attention fusion not yet implemented.")
