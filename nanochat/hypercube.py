import torch
import torch.nn as nn
import torch.nn.functional as F

class HypercubeTopology:
    def __init__(self, n: int):
        if not (1 <= n <= 16): # Limiting n for practical reasons, can be adjusted
            raise ValueError("Hypercube dimension 'n' must be between 1 and 16.")
        self.n = n
        self.num_vertices = 2**n

    def to_binary(self, vertex_id: int) -> str:
        if not (0 <= vertex_id < self.num_vertices):
            raise ValueError(f"Vertex ID {vertex_id} is out of range for a {self.n}-dimensional hypercube.")
        return format(vertex_id, f'0{self.n}b')

    def from_binary(self, binary_string: str) -> int:
        if len(binary_string) != self.n:
            raise ValueError(f"Binary string length must be {self.n} for a {self.n}-dimensional hypercube.")
        return int(binary_string, 2)

    def get_neighbors(self, vertex_id: int) -> list[int]:
        if not (0 <= vertex_id < self.num_vertices):
            raise ValueError(f"Vertex ID {vertex_id} is out of range for a {self.n}-dimensional hypercube.")

        neighbors = []
        for i in range(self.n):
            # Flip the i-th bit
            neighbor_id = vertex_id ^ (1 << i)
            neighbors.append(neighbor_id)
        return neighbors

    def get_all_edges(self) -> list[tuple[int, int]]:
        edges = []
        for i in range(self.num_vertices):
            for neighbor in self.get_neighbors(i):
                # Ensure each edge is added only once (e.g., (0,1) not (1,0))
                if i < neighbor:
                    edges.append((i, neighbor))
        return edges

    def get_random_vertex(self) -> int:
        return torch.randint(0, self.num_vertices, (1,)).item()

    def get_random_path(self, start_vertex: int, length: int) -> list[int]:
        if not (0 <= start_vertex < self.num_vertices):
            raise ValueError(f"Start vertex ID {start_vertex} is out of range.")
        if length < 1:
            raise ValueError("Path length must be at least 1.")

        path = [start_vertex]
        current_vertex = start_vertex

        for _ in range(length - 1):
            neighbors = self.get_neighbors(current_vertex)
            if not neighbors:
                break # Should not happen in a hypercube unless n=0
            
            # Choose a random neighbor that is not the previous vertex if possible
            next_vertex = neighbors[torch.randint(0, len(neighbors), (1,)).item()]
            path.append(next_vertex)
            current_vertex = next_vertex
        return path

class AutoregressiveLatentGenerator(nn.Module):
    def __init__(self, continuous_latent_dim, conditional_dim, num_layers=2, hidden_dim_multiplier=2):
        super().__init__()
        self.continuous_latent_dim = continuous_latent_dim
        self.conditional_dim = conditional_dim
        self.layers = nn.ModuleList()
        
        # The first layer now takes both the latent vector and the condition
        self.layers.append(nn.Linear(continuous_latent_dim + conditional_dim, continuous_latent_dim * hidden_dim_multiplier))
        self.layers.append(nn.GELU())
        self.layers.append(nn.LayerNorm(continuous_latent_dim * hidden_dim_multiplier))
        self.layers.append(nn.Linear(continuous_latent_dim * hidden_dim_multiplier, continuous_latent_dim))
        self.layers.append(nn.GELU())
        self.layers.append(nn.LayerNorm(continuous_latent_dim))

    def forward(self, latent_vector, condition):
        # Concatenate the latent vector with the condition
        combined_input = torch.cat((latent_vector, condition), dim=-1)
        for layer in self.layers:
            combined_input = layer(combined_input)
        return combined_input

class EnergyFunction(nn.Module):
    def __init__(self, latent_dim, hidden_dim_multiplier=2):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * hidden_dim_multiplier),
            nn.GELU(),
            nn.LayerNorm(latent_dim * hidden_dim_multiplier),
            nn.Linear(latent_dim * hidden_dim_multiplier, 1)  # Outputs a scalar energy value
        )

    def forward(self, z_t_plus_1, z_le_t):
        # z_t_plus_1: (batch_size, latent_dim) - the next latent vector
        # z_le_t: (batch_size, latent_dim) - the previous latent context (e.g., current continuous_latent)
        combined_latent = torch.cat((z_t_plus_1, z_le_t), dim=-1)
        return self.net(combined_latent)


class HypercubeEmbeddingLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_concept_ids = config.num_concept_ids
        self.embedding_dim = config.embedding_dim
        self.hypercube_dim = config.hypercube_dim
        self.hypercube_topology = HypercubeTopology(self.hypercube_dim)

        # Existing discrete embedding setup
        self.raw_embedding_layer = nn.Embedding(self.num_concept_ids, self.embedding_dim)
        self.vertex_embeddings = nn.Parameter(torch.randn(self.hypercube_topology.num_vertices, self.embedding_dim))

        # New CALM-inspired continuous latent space components
        self.continuous_latent_dim = self.embedding_dim  # Scaled for autoencoder capacity
        # Autoencoder encoder (compresses concept embeddings to continuous latent vector)
        self.autoencoder_encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.continuous_latent_dim),
            nn.LayerNorm(self.continuous_latent_dim),
            nn.GELU()
        )
        # Autoencoder decoder (reconstructs concept embeddings from continuous latent vector)
        self.autoencoder_decoder = nn.Sequential(
            nn.Linear(self.continuous_latent_dim, self.embedding_dim),
            nn.LayerNorm(self.embedding_dim),
            nn.GELU()
        )

        # Autoregressive latent generator
        # The conditional_dim is set to continuous_latent_dim for now, meaning it's conditioned on the previous latent state
        self.autoregressive_generator = AutoregressiveLatentGenerator(self.continuous_latent_dim, self.continuous_latent_dim)

        # Energy function for CALM-style training
        self.energy_function = EnergyFunction(self.continuous_latent_dim)


    def forward(self, concept_ids, previous_continuous_latent=None, target_continuous_latent=None):
        # Existing discrete embedding logic
        raw_embeddings = self.raw_embedding_layer(concept_ids)
        distances = torch.cdist(raw_embeddings, self.vertex_embeddings)
        nearest_vertex_indices = torch.argmin(distances, dim=-1)
        discrete_vertex_embeddings = self.vertex_embeddings[nearest_vertex_indices]

        # New continuous latent space processing (CALM integration)
        continuous_latent = self.autoencoder_encoder(discrete_vertex_embeddings)
        reconstructed_embeddings = self.autoencoder_decoder(continuous_latent)
        # Store reconstruction loss for training (to be used in objective later)
        self.reconstruction_loss = F.mse_loss(reconstructed_embeddings, discrete_vertex_embeddings)

        # Autoregressive generation of the next latent vector
        if previous_continuous_latent is not None:
            # If a previous latent is provided, use it and the current continuous_latent as condition
            predicted_next_continuous_latent = self.autoregressive_generator(previous_continuous_latent, continuous_latent)
        else:
            # If no previous latent, the first generated latent is conditioned on itself (or a zero vector if preferred)
            predicted_next_continuous_latent = self.autoregressive_generator(continuous_latent, continuous_latent)

        # Calculate energy loss for CALM-style training
        if target_continuous_latent is not None:
            energy_loss = self.energy_function(continuous_latent, target_continuous_latent)
        else:
            # Initialize energy loss to zero if no target is provided
            energy_loss = torch.tensor(0.0, device=continuous_latent.device)

        return discrete_vertex_embeddings, continuous_latent, predicted_next_continuous_latent, self.reconstruction_loss, energy_loss

        # Calculate energy-based loss if a target is provided
        self.energy_loss = torch.tensor(0.0, device=continuous_latent.device) # Initialize to 0
        if target_continuous_latent is not None:
            # E_theta(z_t+1, z_<=t) - 1)^2
            energy_real = self.energy_function(target_continuous_latent, continuous_latent)
            loss_real = (energy_real - 1)**2

            # E_theta(z_tilde, z_<=t))^2 where z_tilde is a negative sample
            # For now, we'll use the predicted_next_continuous_latent as a negative sample
            # In a full CALM implementation, this would involve sampling from a noise distribution
            energy_fake = self.energy_function(predicted_next_continuous_latent.detach(), continuous_latent)
            loss_fake = energy_fake**2

            self.energy_loss = (loss_real + loss_fake).mean()

        # Return both discrete and continuous representations for transition phase
        # Now also return the generated next_continuous_latent and the losses
        return discrete_vertex_embeddings, continuous_latent, predicted_next_continuous_latent, self.reconstruction_loss, self.energy_loss


class LongTermMemory(nn.Module):
    def __init__(self, embedding_dim: int, max_memory_size: int = 1000, top_k_retrieval: int = 5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_memory_size = max_memory_size
        self.top_k_retrieval = top_k_retrieval

        # Initialize an empty memory bank. We'll use a list of tensors for simplicity initially,
        # but this could be replaced with a more efficient data structure or a learnable embedding layer.
        self.memory_bank = []
        self.memory_bank_tensor = None # Will store concatenated memories for efficient retrieval

    def store(self, embedding: torch.Tensor):
        # Store a new embedding in the memory bank
        if len(self.memory_bank) >= self.max_memory_size:
            # Simple FIFO eviction for now
            self.memory_bank.pop(0)
        self.memory_bank.append(embedding.detach().cpu()) # Store on CPU to save GPU memory
        self.memory_bank_tensor = None # Invalidate cached tensor

    def retrieve(self, query_embedding: torch.Tensor) -> torch.Tensor:
        # Retrieve top-k most similar embeddings from the memory bank
        if not self.memory_bank:
            return torch.zeros(query_embedding.shape[0], self.top_k_retrieval, self.embedding_dim, device=query_embedding.device)

        if self.memory_bank_tensor is None:
            self.memory_bank_tensor = torch.stack(self.memory_bank).to(query_embedding.device)

        # Normalize query and memory bank for cosine similarity
        query_norm = F.normalize(query_embedding, p=2, dim=-1)
        memory_norm = F.normalize(self.memory_bank_tensor, p=2, dim=-1)

        # Calculate cosine similarity
        # query_norm: (batch_size, embedding_dim)
        # memory_norm: (num_memories, embedding_dim)
        # similarities: (batch_size, num_memories)
        similarities = torch.matmul(query_norm, memory_norm.transpose(0, 1))

        # Get top-k similar memories
        # top_k_values: (batch_size, top_k_retrieval)
        # top_k_indices: (batch_size, top_k_retrieval)
        top_k_values, top_k_indices = torch.topk(similarities, min(self.top_k_retrieval, len(self.memory_bank)), dim=-1)

        # Retrieve the actual embeddings
        retrieved_memories = self.memory_bank_tensor[top_k_indices]

        return retrieved_memories

    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        return self.retrieve(query_embedding)