import torch
import torch.nn as nn

from nanochat.abacus_encoder import AbacusEncoder

class ConsciousIntegrationLayer(nn.Module):
    def __init__(self, config, abacus_encoder: AbacusEncoder):
        super().__init__()
        self.config = config
        self.abacus_encoder = abacus_encoder

        # Linear layer to project the integrated state to the vocabulary size
        self.concept_projection = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, id_output: torch.Tensor, ego_output: torch.Tensor, superego_output: torch.Tensor, long_term_memory_embeddings: torch.Tensor, memetic_fitness: torch.Tensor | None, abacus_state: torch.Tensor) -> torch.Tensor:
        # Ensure all inputs are of the same shape for integration
        # For simplicity, let's assume they are all (B, T, C) or (B, C)
        # If they are (B, C), we might need to unsqueeze for sequence dimension if T > 1

        # Example integration: simple summation. More complex mechanisms can be added here.
        # The goal is to synthesize these into a unified conceptual state.
        synthesized_state = id_output + ego_output + superego_output

        if long_term_memory_embeddings is not None:
            # Ensure dimensions match for addition. long_term_memory_embeddings might be (B, C)
            # If synthesized_state is (B, T, C), expand long_term_memory_embeddings
            if synthesized_state.dim() == 3 and long_term_memory_embeddings.dim() == 2:
                long_term_memory_embeddings = long_term_memory_embeddings.unsqueeze(1).expand(-1, synthesized_state.size(1), -1)
            synthesized_state = synthesized_state + long_term_memory_embeddings

        if memetic_fitness is not None:
            # Ensure dimensions match for addition
            if synthesized_state.dim() == 3 and memetic_fitness.dim() == 2:
                memetic_fitness = memetic_fitness.unsqueeze(1).expand(-1, synthesized_state.size(1), -1)
            synthesized_state = synthesized_state + memetic_fitness

        # Abacus Encoder for logical consistency
        # The abacus_state is already an encoded representation, so we might use it to gate or modulate
        # the synthesized_state, or simply include it in the synthesis.
        # For now, let's add it, assuming it's compatible.
        if synthesized_state.dim() == 3 and abacus_state.dim() == 2:
            abacus_state = abacus_state.unsqueeze(1).expand(-1, synthesized_state.size(1), -1)
        synthesized_state = synthesized_state + abacus_state

        # Project the synthesized state to the vocabulary size
        concept_logits_from_synthesis = self.concept_projection(synthesized_state)

        return concept_logits_from_synthesis