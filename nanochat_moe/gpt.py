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
from nanochat_moe.muon import Muon, DistMuon
from nanochat_moe.adamw import DistAdamW
# ========== MOE ADDITION START ==========
from nanochat_moe.manager import MANAGER
from contextlib import nullcontext
# ========== MOE ADDITION END ==========

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 6
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 384
    
    # ========== MOE ADDITION START ==========
    # MoE-related configs (added for MoE support)
    n_exp: int = 8 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2 # number of active experts
    use_aux_loss: bool = True # apply auxiliary loss (from Switch Transformer)
    use_router_z_loss: bool = True # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default from Switch Transformer
    router_z_loss_weight: float = 0.001 # default from ST-MoE
    train_capacity: float = 1.25 # default from ST-MoE
    eval_capacity: float = 2.0
    min_capacity: int = 4 # minimum batch size per expert
    stride: int = 2 # one in every stride layers uses MoE
    use_switch_tfm_init: bool = True # use weight init scheme from Switch Transformer
    switch_tfm_init_scale: float = 1.0
    router_use_full_prec: bool = True # use float32 in router
    # ========== MOE ADDITION END ==========





def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)



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
            if prefix_len > 0: # can't be negative but could be zero
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

# ========== MOE ADDITION START ==========
# All MoE-related classes below are NEW additions
class Router(nn.Module):
    """Router for MoE layer - selects top-k experts for each token"""
    def __init__(self, config):
        super().__init__()
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.router_use_full_prec = config.router_use_full_prec
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss
        
        # linear projection for (noisy) softmax gating
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # Save input dtype to ensure output matches
        input_dtype = x.dtype
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if self.router_use_full_prec:
            # When using full precision, convert input to float32 explicitly
            # This ensures compatibility with compiled models
            x_router = x.to(dtype=torch.float32)
            ctx = torch.amp.autocast(device_type=device_type, enabled=False)
        else:
            x_router = x
            ctx = nullcontext()
        
        with ctx:
            B, T, _ = x_router.size()
            num_tokens = B * T
            
            # router logits
            logits = self.w_g(x_router)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                noise = F.softplus(self.w_noise(x_router))
                noise *= torch.randn_like(noise)
                logits += noise
            
            # router z loss
            if self.use_router_z_loss:
                z_loss = torch.logsumexp(logits, dim=-1) ** 2.0
                MANAGER.add_router_z_loss(torch.mean(z_loss))
            
            # find top k experts
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1)  # [B, T, k]
            
            # normalize expert probabilities (only over top-k)
            router_probs = torch.full_like(logits, float('-inf'))
            router_probs.scatter_(-1, top_k_indices, top_k_logits)
            router_probs = F.softmax(router_probs, dim=-1)
            
            # auxiliary load balancing loss
            if self.use_aux_loss:
                with torch.no_grad():
                    one_hot_indices = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
                    one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp]
                    tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))
                prob_per_expert = torch.mean(router_probs.float(), dim=(0, 1))
                aux_loss = self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
                MANAGER.add_aux_loss(aux_loss)
            
            # compute expert capacity
            exp_capacity = self.get_capacity(num_tokens)
            
            # make multi-hot mask of chosen experts
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)
            exp_mask = exp_mask.permute(1, 0, 2)  # [k, B * T, n_exp]
            
            # compute cumulative sum for expert ranking
            exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)
            exp_rank = torch.cumsum(exp_rank, dim=0) - 1
            exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)
            
            # mask out entries beyond expert capacity
            exp_mask *= torch.lt(exp_rank, exp_capacity)
            used_capacity = torch.sum(exp_mask, dim=(0, 1))
            
            # mask rank to only include selected tokens
            exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]
            
            # mask probabilities to only include selected experts
            router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]  # [1, B * T, n_exp]
            exp_weights = exp_mask * router_probs  # [k, B * T, n_exp]
            
            # convert rank to one-hot vectors
            exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity)  # [k, B * T, exp_capacity]
            
            # create weight matrix [B * T, n_exp, exp_capacity]
            cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
            sec_mask = cb_weight.bool()
            
        # Note: cb_weight and sec_mask remain in their computed dtype
        # They will be converted to match input dtype in MOELayer using type_as()
            return used_capacity, cb_weight, sec_mask
    
    def get_capacity(self, tokens_per_batch):
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2  # make even
        capacity = max(capacity, self.min_capacity)
        assert capacity > 0
        return int(capacity)


class MLPExperts(nn.Module):
    """Multiple MLP experts - adapted for nanochat (relu^2, no bias)"""
    def __init__(self, config):
        super().__init__()
        # no bias, using relu^2 activation like nanochat
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, 4 * config.n_embd))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, 4 * config.n_embd, config.n_embd))
    
    def forward(self, x):
        # x: [n_exp, exp_capacity, n_embd]
        x = torch.bmm(x, self.c_fc)  # [n_exp, exp_capacity, 4*n_embd]
        x = F.relu(x).square()  # relu^2 activation (nanochat style)
        x = torch.bmm(x, self.c_proj)  # [n_exp, exp_capacity, n_embd]
        return x


class MOELayer(nn.Module):
    """MoE layer combining Router and MLPExperts"""
    def __init__(self, config):
        super().__init__()
        self.router = Router(config)
        self.experts = MLPExperts(config)
    
    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T
        
        # route tokens to experts
        used_capacity, exp_weight, exp_mask = self.router(x)
        
        # flatten input
        x = x.view(num_tokens, n_embd)
        
        # reshape tokens into batches for each expert
        # [n_exp, exp_capacity, B * T] * [B * T, n_embd] -> [n_exp, exp_capacity, n_embd]
        # Convert exp_mask to match x dtype (matches nanoMoE approach)
        exp_batches = exp_mask.permute(1, 2, 0).type_as(x) @ x
        
        # compute expert output
        exp_out = self.experts(exp_batches)  # [n_exp, exp_capacity, n_embd]
        
        # aggregate expert outputs based on router weights
        exp_weight = exp_weight.view(num_tokens, -1)  # [B * T, n_exp * exp_capacity]
        exp_out = exp_out.view(-1, n_embd)  # [n_exp * exp_capacity, n_embd]
        # Ensure exp_weight matches exp_out dtype (matches nanoMoE approach)
        output = exp_weight.type_as(exp_out) @ exp_out  # [B * T, n_embd]
        
        return output.view(B, T, n_embd)
# ========== MOE ADDITION END ==========


class Block(nn.Module):
    def __init__(self, config, layer_idx, use_moe=False):  # ========== MODIFIED: added use_moe parameter
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        # ========== MOE ADDITION START ==========
        if use_moe:
            self.mlp = MOELayer(config)  # Use MoE layer instead of regular MLP
        else:
            self.mlp = MLP(config)
        self.use_moe = use_moe  # Track if this block uses MoE
        # ========== MOE ADDITION END ==========

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # ========== MOE MODIFICATION START ==========
        # Create blocks, using MoE every stride layers if n_exp > 1
        blocks = []
        for layer_idx in range(config.n_layer):
            use_moe = (config.n_exp > 1) and (layer_idx % config.stride == 0)
            blocks.append(Block(config, layer_idx, use_moe=use_moe))
        # ========== MOE MODIFICATION END ==========
        
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList(blocks),  # ========== MODIFIED: use blocks list instead of list comprehension
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
            # ========== MOE MODIFICATION START ==========
            if block.use_moe:
                # For MoE layers, zero out expert c_proj weights
                torch.nn.init.zeros_(block.mlp.experts.c_proj)
            else:
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            # ========== MOE MODIFICATION END ==========
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        # ========== SWITCH TRANSFORMER INIT START ==========
        # Optionally use Switch Transformer-style initialization
        # See page 10 for switch init explanation: https://arxiv.org/abs/2101.03961
        use_switch_init = self.config.use_switch_tfm_init
        switch_scale = self.config.switch_tfm_init_scale
        # ========== SWITCH TRANSFORMER INIT END ==========
        
        if isinstance(module, nn.Linear):
            if use_switch_init:
                # Switch Transformer initialization: truncated normal with scaled std
                # linear layers have flipped dimensions in torch: [out_dim, in_dim]
                fan_in = module.weight.shape[-1]
                w_std = (switch_scale / fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.weight,
                    mean=0.0,
                    std=w_std,
                    a=-2*w_std,
                    b=2*w_std,
                )
            else:
                # Standard nanochat initialization: https://arxiv.org/pdf/2310.17813
                fan_out = module.weight.size(0)
                fan_in = module.weight.size(1)
                std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                torch.nn.init.normal_(module.weight, mean=0.0, std=std)
                # Always initialize bias to zero
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Always use standard initialization for embeddings
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)
        # ========== MOE ADDITION START ==========
        elif isinstance(module, MLPExperts):
            if use_switch_init:
                # Switch Transformer initialization for experts
                # c_fc: [n_exp, n_embd, 4*n_embd]
                c_fc_fan_in = module.c_fc.shape[-2]
                c_fc_std = (switch_scale / c_fc_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_fc,
                    mean=0.0,
                    std=c_fc_std,
                    a=-2*c_fc_std,
                    b=2*c_fc_std,
                )
                # c_proj: [n_exp, 4*n_embd, n_embd]
                c_proj_fan_in = module.c_proj.shape[-2]
                c_proj_std = (switch_scale / c_proj_fan_in) ** 0.5
                torch.nn.init.trunc_normal_(
                    module.c_proj,
                    mean=0.0,
                    std=c_proj_std,
                    a=-2*c_proj_std,
                    b=2*c_proj_std,
                )
            else:
                # Standard nanochat initialization for experts
                for i in range(module.c_fc.size(0)):
                    # c_fc: [n_exp, n_embd, 4*n_embd]
                    fan_in = module.c_fc.size(1)
                    fan_out = module.c_fc.size(2)
                    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                    torch.nn.init.normal_(module.c_fc[i], mean=0.0, std=std)
                    
                    # c_proj: [n_exp, 4*n_embd, n_embd]
                    fan_in = module.c_proj.size(1)
                    fan_out = module.c_proj.size(2)
                    std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
                    torch.nn.init.normal_(module.c_proj[i], mean=0.0, std=std)
        
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

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        # ========== MOE MODIFICATION START ==========
        # Separate MoE expert parameters (3D) from regular matrix parameters (2D)
        # Muon optimizer only accepts 2D parameters, so we need to filter out 3D (MoE experts) and 1D (bias/norm) params
        matrix_params = []
        moe_params = []  # MoE expert parameters are 3D and need AdamW optimizer
        other_params = []  # 1D parameters (bias, norm weights) also go to AdamW
        for param in self.transformer.h.parameters():
            if param.ndim == 3:  # MoE expert parameters: [n_exp, ...]
                moe_params.append(param)
            elif param.ndim == 2:  # Regular 2D matrix parameters for Muon
                matrix_params.append(param)
            else:  # 1D parameters (bias, norm weights) go to AdamW
                other_params.append(param)
        # ========== MOE MODIFICATION END ==========
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        # Create the AdamW optimizer for the embedding, lm_head, and MoE experts
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
            if moe_params:
                print(f"Found {len(moe_params)} MoE expert parameters (3D) to optimize with AdamW")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        # ========== MOE MODIFICATION START ==========
        # Add MoE expert parameters to AdamW optimizer (use matrix_lr for consistency)
        if moe_params:
            adam_groups.append(dict(params=moe_params, lr=matrix_lr * dmodel_lr_scale))
        # ========== MOE MODIFICATION END ==========
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the 2D linear layers only
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
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
        softcap = 15
        if targets is not None:
            # training mode: compute and return the loss
            # TODO: experiment with Liger Kernels / chunked cross-entropy etc.
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
            logits = logits.float() # use tf32/fp32 for logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            
            # ========== MOE ADDITION START ==========
            # Add MoE auxiliary losses if enabled
            if self.config.n_exp > 1:
                if self.config.use_aux_loss:
                    aux_loss = MANAGER.aggregate_aux_loss()
                    if isinstance(aux_loss, torch.Tensor) and aux_loss.numel() > 0:
                        loss = loss + self.config.aux_loss_weight * aux_loss
                    MANAGER.reset_aux_loss()
                if self.config.use_router_z_loss:
                    router_z_loss = MANAGER.aggregate_router_z_loss()
                    if isinstance(router_z_loss, torch.Tensor) and router_z_loss.numel() > 0:
                        loss = loss + self.config.router_z_loss_weight * router_z_loss
                    MANAGER.reset_router_z_loss()
            # ========== MOE ADDITION END ==========
            
            return loss
        else:
            # inference mode: compute and return the logits
            logits = self.lm_head(x)
            logits = softcap * torch.tanh(logits / softcap) # logits softcap
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
