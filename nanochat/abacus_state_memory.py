import torch
import torch.nn as nn

class AbacusStateMemory(nn.Module):
    def __init__(self, max_memory_size: int = 100, abacus_input_dim: int = 64):
        super().__init__()
        self.max_memory_size = max_memory_size
        self.abacus_input_dim = abacus_input_dim
        self.memory = [] # Stores abacus patterns (tensors)

    def store(self, abacus_pattern: torch.Tensor):
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0) # Remove the oldest entry
        self.memory.append(abacus_pattern.detach().cpu()) # Store detached CPU tensor

    def retrieve(self, num_to_retrieve: int = 1) -> list[torch.Tensor]:
        # Retrieve the most recent abacus patterns
        return self.memory[-num_to_retrieve:]

    def clear(self):
        self.memory = []

    def forward(self, abacus_pattern: torch.Tensor):
        # For now, forward simply stores the pattern. More complex logic can be added later.
        self.store(abacus_pattern)
        return abacus_pattern