import torch
import torch.nn as nn

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

class HypercubeEmbeddingLayer(nn.Module):
    def __init__(self, num_raw_concepts, embedding_dim, hypercube_topology):
        super().__init__()
        self.raw_embedding_layer = nn.Embedding(num_raw_concepts, embedding_dim)
        self.hypercube_topology = hypercube_topology
        # Embeddings for hypercube vertices. The number of vertices is hypercube_topology.num_vertices
        self.vertex_embeddings = nn.Embedding(hypercube_topology.num_vertices, embedding_dim)

    def forward(self, concept_ids):
        # Get initial embeddings for the input concept_ids
        initial_embeddings = self.raw_embedding_layer(concept_ids) # (batch_size, embedding_dim)

        # Find the nearest hypercube vertex for each initial embedding
        
        # Get all hypercube vertex embeddings
        # Ensure all_vertex_ids are on the same device as concept_ids
        all_vertex_ids = torch.arange(self.hypercube_topology.num_vertices, device=concept_ids.device)
        all_vertex_embeddings = self.vertex_embeddings(all_vertex_ids) # (num_vertices, embedding_dim)

        # Calculate squared Euclidean distance
        # initial_embeddings_expanded: (batch_size, 1, embedding_dim)
        # all_vertex_embeddings_expanded: (1, num_vertices, embedding_dim)
        initial_embeddings_expanded = initial_embeddings.unsqueeze(1)
        all_vertex_embeddings_expanded = all_vertex_embeddings.unsqueeze(0)

        distances = torch.sum((initial_embeddings_expanded - all_vertex_embeddings_expanded)**2, dim=2) # (batch_size, num_vertices)

        # Find the index of the nearest vertex for each initial embedding
        nearest_vertex_indices = torch.argmin(distances, dim=1) # (batch_size,)

        # Retrieve the embeddings of the nearest vertices
        final_embeddings = self.vertex_embeddings(nearest_vertex_indices)

        return final_embeddings


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