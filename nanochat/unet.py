from functools import partial
from dataclasses import dataclass
from typing import Sequence, Tuple
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
from nanochat.flash_attention import flash_attn


@dataclass
class UNetConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: Sequence[int] = (6, 12)
    n_head: Sequence[int] = (6, 12) # number of query heads
    n_kv_head: Sequence[int] = (6, 12) # number of key/value heads (GQA)
    n_embd: Sequence[int] = (768, 1536)


def count_params(module):
    return sum(p.numel() for p in module.parameters())


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    def __init__(self, n_head, n_kv_head, n_embd, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
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
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        if kv_cache is None:
            # Training: causal attention
            y = flash_attn.flash_attn_func(q, k, v, causal=True)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, n_head, n_kv_head, n_embd, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(n_head, n_kv_head, n_embd, layer_idx)
        self.mlp = MLP(n_embd)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class Pool(nn.Module):
    def __init__(self, n_embd_1, n_embd_2):
        super().__init__()
        self.c_proj = nn.Linear(2 * n_embd_1, n_embd_2, bias=False)

    def forward(self, x):
        B, T, C = x.size()
        if T % 2 == 1:
            x = x[:, :-1]
            T -= 1
        x = x.view(B, T // 2, 2 * C)
        x = self.c_proj(x)
        return x


class Unpool(nn.Module):
    def __init__(self, n_embd_1, n_embd_2):
        super().__init__()
        self.c_proj = nn.Linear(n_embd_1, 2 * n_embd_2, bias=False)

    def forward(self, x):
        x = self.c_proj(x)
        B, T, C = x.size()
        x = x.view(B, T * 2, C // 2)
        return x


class UNet(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        self.n_stage = len(config.n_layer)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd[0])
        self.encoder = nn.ModuleDict({})
        self.decoder = nn.ModuleDict({})
        for stage_idx in range(self.n_stage):
            n_layer = config.n_layer[stage_idx]
            assert n_layer % 2 == 0
            if stage_idx > 0:
                self.encoder[f"pool_{stage_idx - 1}->{stage_idx}"] = Pool(config.n_embd[stage_idx - 1], config.n_embd[stage_idx])
            self.encoder[f"transformer_{stage_idx}"] = nn.ModuleList([
                Block(config.n_head[stage_idx], config.n_kv_head[stage_idx], config.n_embd[stage_idx], layer_idx)
                for layer_idx in range(n_layer // 2)
            ])
            self.decoder[f"transformer_{stage_idx}"] = nn.ModuleList([
                Block(config.n_head[stage_idx], config.n_kv_head[stage_idx], config.n_embd[stage_idx], layer_idx)
                for layer_idx in range(n_layer // 2)
            ])
            if stage_idx > 0:
                self.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"] = Unpool(config.n_embd[stage_idx], config.n_embd[stage_idx - 1])
        self.lm_head = nn.Linear(config.n_embd[0], padded_vocab_size, bias=False)

        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        for stage_idx in range(self.n_stage):
            assert config.n_embd[stage_idx] // config.n_head[stage_idx] == config.n_embd[0] // config.n_head[0]
        head_dim = config.n_embd[0] // config.n_head[0]
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=1.0)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        for stage_idx in range(self.n_stage):
            s = 3**0.5 * self.config.n_embd[stage_idx]**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
            if stage_idx > 0:
                torch.nn.init.uniform_(self.encoder[f"pool_{stage_idx - 1}->{stage_idx}"].c_proj.weight, -s, s)
            for block in itertools.chain(self.encoder[f"transformer_{stage_idx}"],
                                         self.decoder[f"transformer_{stage_idx}"]):
                torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
                torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
                torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
                torch.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
                torch.nn.init.zeros_(block.mlp.c_proj.weight)
            if stage_idx > 0:
                torch.nn.init.uniform_(self.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"].c_proj.weight, -s, s)

        # Rotary embeddings
        head_dim = self.config.n_embd[0] // self.config.n_head[0]
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to bf16: optimizer can tolerate it and it saves memory
        if self.wte.weight.device.type == "cuda":
            self.wte.to(dtype=torch.bfloat16)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
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
        return self.wte.weight.device

    def estimate_flops(self):
        num_flops_per_token = 0
        for stage_idx in range(self.n_stage):
            nparams = count_params(self.encoder[f"transformer_{stage_idx}"])
            nparams += count_params(self.decoder[f"transformer_{stage_idx}"])
            if stage_idx > 0:
                nparams += count_params(self.encoder[f"pool_{stage_idx - 1}->{stage_idx}"])
                nparams += count_params(self.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"])
            h, q, t = (
                self.config.n_head[stage_idx],
                self.config.n_embd[stage_idx] // self.config.n_head[stage_idx],
                self.config.sequence_len / 2 ** stage_idx
            )
            attn_flops = 12 * h * q * t
            num_flops_per_token += (6 * nparams + attn_flops) / 2 ** stage_idx
        num_flops_per_token += 6 * count_params(self.lm_head)
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return all of the parameters, same as Chinchilla paper.
        Kaplan et al. did not include embedding parameters and said that this led to cleaner scaling laws.
        But Kaplan et al. also had a bug in their results (as pointed out by Chinchilla).
        My own experiments in nanochat confirm the Chinchilla approach gives the much cleaner scaling law.
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper <- good).
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper <- bad)
        """
        return count_params(self)

    def setup_optimizers(self, embedding_lr=0.2, pool_lr=0.004, unpool_lr=0.004, unembedding_lr=0.004, matrix_lr=0.02, weight_decay=0.0, adam_betas=(0.8, 0.95)):
        model_dim = self.config.n_embd[0]
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into groups
        embedding_params = list(self.wte.parameters())
        pool_params = []
        matrix_params = []
        unpool_params = []
        for stage in range(self.n_stage):
            if stage > 0:
                pool_params.extend(list(self.encoder[f"pool_{stage - 1}->{stage}"].parameters()))
            matrix_params.extend(list(self.encoder[f"transformer_{stage}"].parameters()))
            matrix_params.extend(list(self.decoder[f"transformer_{stage}"].parameters()))
            if stage > 0:
                unpool_params.extend(list(self.decoder[f"unpool_{stage}->{stage - 1}"].parameters()))
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == \
            len(embedding_params) + len(pool_params) + len(matrix_params) + len(unpool_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding, pool, unpool, and lm_head layers
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
            dict(params=pool_params, lr=pool_lr * dmodel_lr_scale),
            dict(params=unpool_params, lr=unpool_lr * dmodel_lr_scale),
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=adam_betas, eps=1e-10, weight_decay=0.0) # NOTE: weight decay is hardcoded to 0.0 for AdamW, only used in Muon
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95, weight_decay=weight_decay)
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
        
        def pool_cos_sin(cos, sin, stage_idx):
            return (
                cos[:, (2 ** stage_idx - 1)::(2 ** stage_idx)],
                sin[:, (2 ** stage_idx - 1)::(2 ** stage_idx)]
            )

        # Input embed
        x = self.wte(idx)
        x = norm(x)
        
        # Encoder
        encoder_outputs = []
        for stage_idx in range(self.n_stage):
            if stage_idx > 0:
                if x.size(1) == 1:
                    break
                # Pool
                x = self.encoder[f"pool_{stage_idx - 1}->{stage_idx}"](x)
                x = norm(x)
            # Stage
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in self.encoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            encoder_outputs.append(x)
        last_stage_idx = stage_idx

        # Decoder
        for stage_idx in reversed(range(last_stage_idx + 1)):
            # Stage
            pooled_cos_sin = pool_cos_sin(*cos_sin, stage_idx)
            for block in self.decoder[f"transformer_{stage_idx}"]:
                x = block(x, pooled_cos_sin, kv_cache)
            if stage_idx > 0:
                # Unpool, shift & skip-connection
                x = norm(x)
                x = self.decoder[f"unpool_{stage_idx}->{stage_idx - 1}"](x)
                y = encoder_outputs[stage_idx - 1]
                if x.size(1) == y.size(1):
                    x = x[:, :-1]
                shifted_x = torch.zeros_like(y)
                shifted_x[:, 1:] = x
                x = y + shifted_x

        # Forward the lm_head (compute logits)
        x = norm(x)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
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
