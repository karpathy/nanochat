"""
GPT model (refactored into modular components).

Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from typing import Optional, List, Tuple, Dict, Any, Generator
import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.models.config import GPTConfig
from nanochat.models.attention import CausalSelfAttention, Linear, norm, has_ve
from nanochat.models.mlp import MLP
from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.training.optimizer import MuonAdamW, DistMuonAdamW


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)
    
    def forward(
        self,
        x: torch.Tensor,
        ve: Optional[torch.Tensor],
        cos_sin: Tuple[torch.Tensor, torch.Tensor],
        window_size: Tuple[int, int],
        kv_cache: Optional[Any],
    ) -> torch.Tensor:
        """Forward pass: attention + MLP with residual connections."""
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    """GPT model with modular architecture."""
    
    def __init__(self, config: GPTConfig, pad_vocab_size_to: int = 64):
        """
        NOTE: This __init__ function runs in meta device context (!!).
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config: GPTConfig = config
        
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes: List[Tuple[int, int]] = self._compute_window_sizes(config)
        
        # Pad vocab for efficiency (DDP, tensor cores)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({
            str(i): nn.Embedding(padded_vocab_size, kv_dim)
            for i in range(config.n_layer)
            if has_ve(i, config.n_layer)
        })
        
        # Rotary embeddings (over-compute by 10X for safety)
        self.rotary_seq_len: int = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
    
    @torch.no_grad()
    def init_weights(self) -> None:
        """Initialize the full model in this one function for maximum clarity."""
        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)
        
        # Transformer blocks: uniform init with bound = sqrt(3) * std
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.5, s * 0.5)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
        
        # Per-layer scalars
        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)
        
        # Value embeddings
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)
        
        # Gate weights
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)
        
        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        
        # Cast embeddings to COMPUTE_DTYPE (except fp16 which needs fp32)
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)
    
    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: int = 100000,
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Precompute rotary embeddings (cos, sin) for all positions."""
        if device is None:
            device = self.transformer.wte.weight.device
        
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]  # add batch and head dims
        return cos, sin
    
    def _compute_window_sizes(self, config: GPTConfig) -> List[Tuple[int, int]]:
        """Compute per-layer window sizes for sliding window attention."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        
        long_window = config.sequence_len
        short_window = -(-long_window // 3 // 128) * 128  # ceil to FA3 tile size
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes
    
    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        return self.transformer.wte.weight.device
    
    def estimate_flops(self) -> int:
        """Return the estimated FLOPs per token for the model (forward + backward)."""
        nparams = sum(p.numel() for p in self.parameters())
        
        # Exclude non-matmul params
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (
            self.transformer.wte.weight.numel() +
            value_embeds_numel +
            self.resid_lambdas.numel() +
            self.x0_lambdas.numel()
        )
        
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token
    
    def num_scaling_params(self) -> Dict[str, int]:
        """Return detailed parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }
    
    def setup_optimizer(
        self,
        unembedding_lr: float = 0.004,
        embedding_lr: float = 0.2,
        matrix_lr: float = 0.02,
        weight_decay: float = 0.0,
        scalar_lr: float = 0.5,
    ) -> torch.optim.Optimizer:
        """Setup MuonAdamW optimizer with parameter groups."""
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        
        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        
        assert len(list(self.parameters())) == (
            len(matrix_params) + len(embedding_params) + len(lm_head_params) +
            len(value_embeds_params) + len(resid_params) + len(x0_params)
        )
        
        # Scale the LR for the AdamW parameters by ∝1/√dmodel
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        
        # Build param_groups
        param_groups: List[Dict[str, Any]] = [
            # AdamW groups
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))
        
        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer
    
    def forward(
        self,
        idx: torch.Tensor,  # (B, T) int64
        targets: Optional[torch.Tensor] = None,  # (B, T) int64 or None
        kv_cache: Optional[Any] = None,
        loss_reduction: str = 'mean',
    ) -> torch.Tensor:  # loss (scalar) if targets provided, else logits (B, T, vocab_size)
        """Forward pass through the model."""
        B, T = idx.size()
        
        # Grab the rotary embeddings for the current sequence length
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        
        # Offset rotary embeddings if using KV cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]
        
        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)
        x0 = x  # save initial normalized embedding for x0 residual
        
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
        
        x = norm(x)
        
        # Forward the lm_head (compute logits)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]  # slice to remove padding
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)
        
        if targets is not None:
            # Training: compute and return the loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction
            )
            return loss
        else:
            # Inference: return the logits
            return logits
    
    @torch.inference_mode()
    def generate(
        self,
        tokens: List[int],
        max_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        seed: int = 42,
    ) -> Generator[int, None, None]:
        """Naive autoregressive streaming inference (batch size 1)."""
        assert isinstance(tokens, list)
        device = self.get_device()
        
        rng: Optional[torch.Generator] = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
