"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW
from nanochat.nl_opt import NestedMomentum, DistNestedMomentum
from torch.utils.checkpoint import checkpoint

def _compute_loss_chunk(x_chunk, targets_chunk, lm_head, softcap, ignore_index, reduction):
    logits_chunk = lm_head(x_chunk)
    logits_chunk = logits_chunk.float()
    logits_chunk = softcap * torch.tanh(logits_chunk / softcap)
    return F.cross_entropy(logits_chunk, targets_chunk, ignore_index=ignore_index, reduction=reduction)

def chunked_cross_entropy(x, targets, lm_head, chunk_size=128, softcap=15.0, ignore_index=-1, reduction='mean'):
    # Flatten input and targets
    B, T, C = x.size()
    x_flat = x.view(-1, C)
    targets_flat = targets.view(-1)

    num_elements = x_flat.size(0)
    losses = []
    total_tokens = 0

    # Determine internal reduction for chunks
    # If we need 'none' globally, we must get 'none' from chunks.
    # If we need 'mean' or 'sum' globally, 'sum' from chunks is most efficient.
    chunk_reduction = 'none' if reduction == 'none' else 'sum'

    for i in range(0, num_elements, chunk_size):
        x_chunk = x_flat[i : i + chunk_size]
        target_chunk = targets_flat[i : i + chunk_size]

        # We use checkpointing to save memory for the backward pass
        # Note: We must pass tensors to checkpoint, so softcap is passed as tensor if needed,
        # but here it's a float. checkpoint handles non-tensor args but they aren't differentiable.
        # lm_head is a module, so it's captured.
        loss_chunk = checkpoint(
            _compute_loss_chunk,
            x_chunk,
            target_chunk,
            lm_head,
            torch.tensor(softcap, device=x.device),
            ignore_index,
            chunk_reduction,
            use_reentrant=False
        )

        if reduction == 'none':
            losses.append(loss_chunk)
        else:
            # For sum/mean, we accumulate the sum
            losses.append(loss_chunk.unsqueeze(0)) # keep as tensor list
            # Count valid tokens for mean reduction
            valid_mask = target_chunk != ignore_index
            total_tokens += valid_mask.sum().item()

    if not losses:
        return torch.tensor(0.0, device=x.device, requires_grad=True)

    all_losses = torch.cat(losses)

    if reduction == 'none':
        return all_losses
    elif reduction == 'sum':
        return all_losses.sum()
    elif reduction == 'mean':
        if total_tokens > 0:
            return all_losses.sum() / total_tokens
        else:
             return all_losses.sum() * 0.0 # preserve grad
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

def chunked_cross_entropy(x, targets, lm_head, chunk_size=128, softcap=15.0, ignore_index=-1, reduction='mean'):
    # Flatten input and targets
    B, T, C = x.size()
    x_flat = x.view(-1, C)
    targets_flat = targets.view(-1)

    total_loss = 0.0
    total_tokens = 0

    num_elements = x_flat.size(0)

    for i in range(0, num_elements, chunk_size):
        x_chunk = x_flat[i : i + chunk_size]
        target_chunk = targets_flat[i : i + chunk_size]

        # Valid tokens mask (for accurate averaging)
        valid_mask = target_chunk != ignore_index
        valid_count = valid_mask.sum().item()

        if valid_count > 0:
            # Compute logits for chunk
            logits_chunk = lm_head(x_chunk)
            logits_chunk = logits_chunk.float()
            logits_chunk = softcap * torch.tanh(logits_chunk / softcap)

            # Compute sum of losses for this chunk
            chunk_loss = F.cross_entropy(logits_chunk, target_chunk, ignore_index=ignore_index, reduction='sum')

            total_loss += chunk_loss
            total_tokens += valid_count

    if total_tokens == 0:
        return torch.tensor(0.0, device=x.device, requires_grad=True) # return a zero loss with grad for graph integrity

    final_loss = total_loss / total_tokens if reduction == 'mean' else total_loss
    return final_loss

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        self.init_buffers()

    def init_buffers(self):
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0,
                         matrix_optimizer_backend="muon", general_optimizer_backend="adamw",
                         nested_betas=(0.9, 0.99), nested_level_weights=(0.5, 0.5)):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        use_dist_optim = ddp and world_size > 1

        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)

        # --- General Optimizer (Embeddings & Heads) ---
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the general parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        general_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]

        if general_optimizer_backend == "adamw":
            adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
            AdamWFactory = DistAdamW if use_dist_optim else partial(torch.optim.AdamW, fused=True)
            general_optimizer = AdamWFactory(general_groups, **adamw_kwargs)
        elif general_optimizer_backend == "nested_momentum":
             nm_kwargs = dict(betas=nested_betas, level_weights=nested_level_weights, weight_decay=weight_decay)
             NMFactory = DistNestedMomentum if use_dist_optim else NestedMomentum
             general_optimizer = NMFactory(general_groups, **nm_kwargs)
        else:
            raise ValueError(f"Unknown general_optimizer_backend: {general_optimizer_backend}")

        # --- Matrix Optimizer (Transformer Blocks) ---
        if matrix_optimizer_backend == "muon":
            muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
            MuonFactory = DistMuon if use_dist_optim else Muon
            matrix_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        elif matrix_optimizer_backend == "nested_momentum":
            nm_kwargs = dict(lr=matrix_lr, betas=nested_betas, level_weights=nested_level_weights, weight_decay=weight_decay)
            NMFactory = DistNestedMomentum if use_dist_optim else NestedMomentum
            matrix_optimizer = NMFactory(matrix_params, **nm_kwargs)
        else:
             raise ValueError(f"Unknown matrix_optimizer_backend: {matrix_optimizer_backend}")

        # Combine them the two optimizers into one list
        optimizers = [general_optimizer, matrix_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_embeddings=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15.0
        if targets is not None:
            # training mode: compute and return the loss
            # We use chunked cross entropy to save memory.
            # Instead of materializing (B, T, vocab_size) logits, we compute loss in chunks
            loss = chunked_cross_entropy(x, targets, self.lm_head, softcap=softcap, chunk_size=128, ignore_index=-1, reduction=loss_reduction)
            if return_embeddings:
                return loss, x
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
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
