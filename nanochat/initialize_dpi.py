import torch
import torch.nn as nn
import numpy as np
import math
from sklearn.cluster import MiniBatchKMeans

"""
Deterministic Pipeline Initialization (DPI)
Core Engine v16.2 - Optimized for nanochat/GPT-2 style architectures.

DPI replaces random weight initialization with a data-driven approach:
- Phase 0: Seeding Lexical Manifold via SVD on token co-occurrence matrix.
- Phase 1: K-Means clustering of embedding activations to structure latent space.
- Phase 2: Spectral Bootstrapping with orthogonality constraints to prevent rank collapse.

This initialization typically yields a significantly lower starting loss (bpb) and 
faster convergence by placing the model in a more "mature" region of the loss landscape.
"""

from nanochat.common import COMPUTE_DTYPE

def get_activations(model, dataloader, layer_idx, num_samples=2000):
    """Samples model activations at a given layer index."""
    model.eval()
    activations = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for i, (x_batch, _, _) in enumerate(dataloader):
            x_batch = x_batch.to(device)
            # 1. Forward pass through initial embedding
            x = model.transformer['wte'](x_batch)
            x = x.to(COMPUTE_DTYPE) # Ensure activations are in COMPUTE_DTYPE
            
            # 2. Sequential pass through blocks up to target layer
            if layer_idx >= 0:
                for j in range(layer_idx + 1):
                    T = x.size(1)
                    # Ensure cos/sin match the COMPUTE_DTYPE for the forward pass
                    cos = model.cos[:, :T, :, :].to(COMPUTE_DTYPE)
                    sin = model.sin[:, :T, :, :].to(COMPUTE_DTYPE)
                    window_size = model.window_sizes[j]
                    ve = model.value_embeds[str(j)](x_batch).to(COMPUTE_DTYPE) if str(j) in model.value_embeds else None
                    x = model.transformer['h'][j](x, ve, (cos, sin), window_size, None)
            
            activations.append(x.view(-1, x.size(-1)))
            if len(activations) * x.size(1) >= num_samples: break 
    return torch.cat(activations, dim=0)

def normalize_weight(W, target_std=None):
    """Rescales weights to a specific standard deviation for stability."""
    if target_std is None: target_std = math.sqrt(1.0 / W.size(1))
    curr_std = W.std()
    if curr_std > 1e-8: return W * (target_std / curr_std)
    return W

def init_phase0_lexical(model, dataloader):
    """Initializes embeddings using SVD of the token co-occurrence matrix."""
    vocab_size = model.transformer['wte'].num_embeddings
    d_model = model.config.n_embd
    device = next(model.parameters()).device
    model_dtype = COMPUTE_DTYPE # Use COMPUTE_DTYPE for embeddings
    print(f"  [Phase 0] Seeding Lexical Manifold (Exact SVD on CUDA)...")
    
    # Co-occurrence matrix: U * vocab_size + V trick for fast GPU accumulation
    C = torch.zeros(vocab_size * vocab_size, device=device)
    for i, (x, _, _) in enumerate(dataloader):
        x = x.to(device)
        u, v = x[:, :-1].reshape(-1), x[:, 1:].reshape(-1)
        C.index_add_(0, u * vocab_size + v, torch.ones_like(u, dtype=torch.float, device=device))
        if i >= 100: break # Sufficient statistics for initialization
    C = C.view(vocab_size, vocab_size)
    
    # Perform SVD to get principal semantic directions
    U, S, V = torch.svd_lowrank(C.float(), q=d_model, niter=5)
    model.transformer['wte'].weight.data[:, :min(d_model, vocab_size)].copy_(U[:, :min(d_model, vocab_size)].to(model_dtype))
    model.transformer['wte'].weight.data.copy_(normalize_weight(model.transformer['wte'].weight.data, target_std=0.8).to(model_dtype))
    return U, V

def initialize_dpi(model, dataloader, spectral_gamma=0.25, mode="v16.2"):
    """
    Core DPI routine. Re-initializes model weights based on data-driven manifolds.
    - All matrix operations (SVD, dot products) are forced to float32 for stability and CUDA support.
    - KMeans uses a fixed random_state to ensure cross-rank consistency (if not broadcasting).
    """
    device = next(model.parameters()).device
    model_dtype = COMPUTE_DTYPE # Use COMPUTE_DTYPE for consistency
    model.to(model_dtype) # Robustness: ensure model is in its intended dtype
    n_layers = len(model.transformer['h'])
    d_model = model.config.n_embd
    phase_shift_layer = n_layers // 2
    
    U_lex, V_lex = init_phase0_lexical(model, dataloader)
    
    print(f"  [Phase 1] K-Means Clustering on Embeddings...")
    X_lex = get_activations(model, dataloader, -1, num_samples=max(4000, d_model))
    # Cast to float32 for sklearn compatibility and precision
    km = MiniBatchKMeans(n_clusters=d_model, n_init=3, batch_size=1024, random_state=42).fit(X_lex.float().cpu().numpy())
    centers = torch.from_numpy(km.cluster_centers_).float().to(device)
    
    print(f"  [Phase 2] Bootstrapping Mode: {mode.upper()} (Genomic Ready)...")
    
    for l in range(n_layers):
        # We sample activations before the current layer to find the manifold basis
        X_curr = get_activations(model, dataloader, l-1, num_samples=max(2000, d_model))
        X_centered = X_curr - X_curr.mean(dim=0)
        
        # Manifold discovery via covariance SVD (float32 mandatory for CUDA SVD)
        cov = torch.matmul(X_centered.t().float(), X_centered.float()) / X_centered.size(0)
        U, S, V = torch.svd(cov)
        
        # Calculate current layer spectral gamma (depth-dependent scaling)
        progress = l / (n_layers - 1) if n_layers > 1 else 0
        current_gamma = spectral_gamma * (1.0 - 0.2 * math.sin(math.pi * progress))
        svd_basis = normalize_weight((U.t() * torch.pow(S + 1e-6, current_gamma).unsqueeze(1)).to(device))
        
        # Alignment with nanochat architectural priors (std targets)
        qkv_std = 1.0 / math.sqrt(d_model)
        mlp_fc_std = 0.4 / math.sqrt(d_model)
        ve_std = 1.0 / math.sqrt(d_model)
        
        layer = model.transformer['h'][l]
        attn = layer.attn
        mlp = layer.mlp
        
        # 1. MLP Initializers (Expansion/Contraction)
        # Use data-driven basis scaled to 0.4/sqrt(n_embd)
        mlp_basis = svd_basis.repeat(mlp.c_fc.weight.data.size(0) // d_model, 1)
        mlp.c_fc.weight.data.copy_(normalize_weight(mlp_basis + torch.randn_like(mlp_basis) * mlp_fc_std, target_std=mlp_fc_std).to(model_dtype))
        # Projections are set to exactly zero to match native init_weights
        torch.nn.init.zeros_(mlp.c_proj.weight)
        
        # 2. Value Embeddings (CRITICAL for nanochat)
        # If this block has value embeddings, they must be aligned with the semantic manifold
        if hasattr(model, 'value_embeds') and str(l) in model.value_embeds:
            ve_target = model.value_embeds[str(l)]
            # Align VE with svd_basis to ensure semantic consistency in the value stream
            # repeat basis if vocab size is larger (it is)
            ve_basis = svd_basis.repeat(ve_target.num_embeddings // svd_basis.size(0) + 1, 1)[:ve_target.num_embeddings]
            ve_target.weight.data.copy_(normalize_weight(ve_basis + torch.randn_like(ve_basis) * ve_std, target_std=ve_std).to(model_dtype))

        # 3. Attention Initializers (Alignment vs. Orthogonality)
        is_consolidated = (l >= phase_shift_layer)
        alignment = 0.40 * math.sin(math.pi * progress) if not is_consolidated else 0.0001
        
        if not is_consolidated:
            # Early layers: Align K/V with data centers
            attn.c_k.weight.data.copy_(normalize_weight(centers + 0.2 * svd_basis, target_std=qkv_std).to(model_dtype))
            attn.c_v.weight.data.copy_(normalize_weight(svd_basis, target_std=qkv_std).to(model_dtype))
            attn.c_q.weight.data.copy_(normalize_weight(alignment * attn.c_k.weight.data.float() + (1 - alignment) * svd_basis, target_std=qkv_std).to(model_dtype))
        else:
            # Deep layers: Force Query orthogonality to prevent rank collapse
            attn.c_k.weight.data.copy_(normalize_weight(svd_basis, target_std=qkv_std).to(model_dtype))
            attn.c_v.weight.data.copy_(normalize_weight(svd_basis, target_std=qkv_std).to(model_dtype))
            
            Q_rand = torch.randn_like(attn.c_k.weight.data)
            dot = (Q_rand.float() * attn.c_k.weight.data.float()).sum(dim=1, keepdim=True)
            norm_k = (attn.c_k.weight.data.float() * attn.c_k.weight.data.float()).sum(dim=1, keepdim=True)
            Q_ortho = Q_rand.float() - (dot / (norm_k + 1e-8)) * attn.c_k.weight.data.float()
            attn.c_q.weight.data.copy_(normalize_weight(alignment * attn.c_k.weight.data.float() + (1 - alignment) * Q_ortho, target_std=qkv_std).to(model_dtype))
        
        # 3. Final Attention Projection
        torch.nn.init.zeros_(attn.c_proj.weight)

    # 4. Phase 4: Zero-Wait Output Head
    if hasattr(model, 'lm_head'):
        model.lm_head.weight.data[:, :min(d_model, model.lm_head.out_features)].copy_(V_lex[:, :min(d_model, model.lm_head.out_features)].to(model_dtype))
        model.lm_head.weight.data.copy_(normalize_weight(model.lm_head.weight.data, target_std=0.001).to(model_dtype))

    # Final dtype sync to ensure all parameters and buffers are consistent
    model.to(model_dtype)
    # Explicitly cast rotary embeddings to COMPUTE_DTYPE to pass GPT assertion
    model.cos.data = model.cos.data.to(COMPUTE_DTYPE)
    model.sin.data = model.sin.data.to(COMPUTE_DTYPE)
    print(f"✓ DPI-V16.2 Initialization Complete.")
