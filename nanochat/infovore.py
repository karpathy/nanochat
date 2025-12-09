"""
nanochat/infovore.py

Implements the Novelty-Relation Quotient (NRQ) metric for Online Curriculum Learning.
"""
import torch
import torch.nn.functional as F

class Infovore:
    def __init__(self, d_model, device, beta=0.99):
        """
        Args:
            d_model: Dimension of the model's hidden states (e.g. 768)
            beta: Momentum for the running average of the memory manifold (0.99 = slow moving)
        """
        self.d_model = d_model
        self.device = device
        self.beta = beta

        # The 'Manifold Centroid' represents the center of mass of the model's
        # "known concepts". Initialized as zeros until first update.
        self.manifold_centroid = torch.zeros(d_model, device=device)
        self.is_initialized = False

    @torch.no_grad()
    def update_manifold(self, hidden_states):
        """
        Updates the internal memory manifold based on the current batch's content.
        hidden_states: (B, T, D)
        """
        # Collapse batch and time to get the mean vector of this batch
        # We only want to learn from "valid" data, but for simplicity we take the mean
        batch_centroid = hidden_states.mean(dim=(0, 1))

        if not self.is_initialized:
            self.manifold_centroid = batch_centroid
            self.is_initialized = True
        else:
            # Exponential Moving Average (EMA) to drift slowly
            self.manifold_centroid = (self.beta * self.manifold_centroid) + \
                                     ((1 - self.beta) * batch_centroid)

    def compute_nrq_loss(self, model, idx, targets):
        """
        Computes the NRQ-weighted loss.

        NRQ = Novelty * Relation
        Novelty = Per-token Cross Entropy Loss (Surprisal)
        Relation = Cosine Similarity (Token Embedding, Manifold Centroid)

        Returns:
            weighted_loss: The scalar loss to backpropagate
            metrics: dict containing 'nrq_avg', 'novelty_avg', 'relation_avg'
        """
        # 1. Forward pass with reduction='none' to get raw loss per token
        # We request embeddings to calculate Relation
        # Note: We assume model is wrapped in DDP or compiled, so we call it directly.
        # But wait, if model is compiled or DDP, the signature must support return_embeddings.
        # We modified GPT.forward, but if it's DDP wrapped, forward args pass through.
        raw_loss, hidden_states = model(idx, targets, loss_reduction='none', return_embeddings=True)

        # raw_loss is (B*T,) flattened or (B, T) depending on implementation.
        # GPT.forward returns F.cross_entropy with reduction='none'.
        # F.cross_entropy with reduction='none' returns (N,) if input is (N, C) and target is (N,).
        # In GPT.forward: logits.view(-1, logits.size(-1)), targets.view(-1)
        # So raw_loss is (B*T,).

        B, T = idx.shape
        surprisal = raw_loss.view(B, T) # This is 'Novelty'

        # 2. Calculate Relation: Cosine sim between token embeddings and the manifold centroid

        # We detach hidden_states because we don't want to backpropagate through the relation metric.
        # This saves memory by not extending the graph further.
        hidden_states = hidden_states.detach()

        # manifold: (D) -> reshape to (1, D) for broadcast
        manifold_view = self.manifold_centroid.view(1, -1)

        # Flatten hidden_states to (N, D) for easier chunking
        # This helps avoid OOM on memory constrained devices (like 128GB APUs)
        # because F.cosine_similarity might upcast/allocate large buffers.
        hs_flat = hidden_states.view(-1, self.d_model)

        relations = []
        chunk_size = 4096 # Process 4k tokens at a time to keep peak memory low

        for i in range(0, hs_flat.size(0), chunk_size):
            hs_chunk = hs_flat[i : i + chunk_size]
            # Calculate similarity for this chunk
            # hs_chunk: (chunk_size, D)
            # manifold_view: (1, D)
            # Output: (chunk_size,)
            sim_chunk = F.cosine_similarity(hs_chunk, manifold_view, dim=-1)
            relations.append(sim_chunk)

        # Concatenate and reshape back to (B, T)
        relation = torch.cat(relations).view(B, T)

        # Normalize Relation to [0, 1] roughly (Cosine is [-1, 1])
        # We clamp negative relation (opposites) to 0 for the metric.
        relation = F.relu(relation)

        # 3. Calculate NRQ (Novelty * Relation)
        # We use NRQ as a scalar weight for the gradient.
        # Relation is already detached effectively (computed from detached inputs).
        nrq_score = (surprisal * relation).detach()

        # 4. Filter / Weight the Loss
        # High NRQ samples generate stronger gradients.
        # Low NRQ samples (Boredom or Noise) generate weak gradients.

        # raw_loss is (B*T,), nrq_score is (B, T). Flatten nrq_score.
        weighted_loss = raw_loss * nrq_score.view(-1)

        # Update the manifold for the next step (using the current batch's reality)
        self.update_manifold(hidden_states)

        # Metrics for logging
        metrics = {
            "nrq_avg": nrq_score.mean().item(),
            "novelty_avg": surprisal.mean().item(),
            "relation_avg": relation.mean().item()
        }

        return weighted_loss.mean(), metrics
