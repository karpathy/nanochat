# pyright: reportMissingImports=false
from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


POLAR_EXPRESS_COEFFS = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

ORTHOGONALIZE_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def has_ve(layer_idx: int, n_layer: int) -> bool:
    return layer_idx % 2 == (n_layer - 1) % 2


def sum_parameter_leaves(tree) -> int:
    if isinstance(tree, dict):
        return sum(sum_parameter_leaves(value) for value in tree.values())
    if isinstance(tree, list):
        return sum(sum_parameter_leaves(value) for value in tree)
    if isinstance(tree, tuple):
        return sum(sum_parameter_leaves(value) for value in tree)
    return int(tree.size) if isinstance(tree, mx.array) else 0


def rms_norm(x: mx.array, eps: float = 1e-5) -> mx.array:
    return x * mx.rsqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)


def build_optimizer_group_trees(tree) -> dict[str, dict]:
    return {
        "lm_head": {"lm_head": tree["lm_head"]},
        "wte": {"wte": tree["wte"]},
        "value_embeds": {"value_embeds": tree["value_embeds"]},
        "resid": {"resid_lambdas": tree["resid_lambdas"]},
        "x0": {"x0_lambdas": tree["x0_lambdas"]},
        "attn_blocks": {"blocks": [{"attn": block["attn"]} for block in tree["blocks"]]},
        "mlp_blocks": {"blocks": [{"mlp": block["mlp"]} for block in tree["blocks"]]},
    }


def _tree_map_leaves(fn, params, grads, path=()):
    if isinstance(grads, dict):
        result = {}
        for key, value in grads.items():
            result[key] = _tree_map_leaves(fn, params[key], value, path + (str(key),))
        return result
    if isinstance(grads, list):
        return [_tree_map_leaves(fn, params[idx], value, path + (str(idx),)) for idx, value in enumerate(grads)]
    if isinstance(grads, tuple):
        return tuple(_tree_map_leaves(fn, params[idx], value, path + (str(idx),)) for idx, value in enumerate(grads))
    return fn(path, params, grads)


@dataclass
class MLXGPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 32768
    n_layer: int = 32
    n_head: int = 16
    n_kv_head: int = 16
    n_embd: int = 2048
    window_pattern: str = "L"


class ReLUSquaredMLP(nn.Module):
    def __init__(self, config: MLXGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight = mx.zeros_like(self.c_proj.weight)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.c_fc(x)
        x = mx.square(nn.relu(x))
        return self.c_proj(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MLXGPTConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight = mx.zeros_like(self.c_proj.weight)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=100000)
        self.ve_gate_channels = 12
        self.ve_gate = nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def __call__(self, x: mx.array, ve: mx.array | None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        q = self.c_q(x).reshape(batch_size, seq_len, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.c_k(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.c_v(x).reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)

        if ve is not None and self.ve_gate is not None:
            gate = 3.0 * mx.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            gate = gate.transpose(0, 2, 1)[..., None]
            ve = ve.reshape(batch_size, seq_len, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
            v = v + gate * ve

        q = rms_norm(self.rope(q)) * 1.15
        k = rms_norm(self.rope(k)) * 1.15
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim ** -0.5, mask="causal")
        y = y.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_embd)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: MLXGPTConfig, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = ReLUSquaredMLP(config)

    def __call__(self, x: mx.array, ve: mx.array | None) -> mx.array:
        x = x + self.attn(rms_norm(x), ve)
        x = x + self.mlp(rms_norm(x))
        return x


class GroupedAdamW:
    def __init__(self, config: MLXGPTConfig, *, unembedding_lr: float, embedding_lr: float, matrix_lr: float, scalar_lr: float, weight_decay: float):
        optim = __import__("mlx.optimizers", fromlist=["AdamW"])

        dmodel_lr_scale = (config.n_embd / 768) ** -0.5
        self.optimizers = {
            "lm_head": optim.AdamW(unembedding_lr * dmodel_lr_scale, betas=[0.8, 0.96], eps=1e-10, weight_decay=0.01),
            "wte": optim.AdamW(embedding_lr * dmodel_lr_scale, betas=[0.8, 0.995], eps=1e-10, weight_decay=0.001),
            "value_embeds": optim.AdamW(embedding_lr * dmodel_lr_scale * 0.5, betas=[0.8, 0.995], eps=1e-10, weight_decay=0.01),
            "resid": optim.AdamW(scalar_lr * 0.01, betas=[0.8, 0.95], eps=1e-10, weight_decay=0.05),
            "x0": optim.AdamW(scalar_lr, betas=[0.96, 0.95], eps=1e-10, weight_decay=0.0),
            "attn_blocks": optim.AdamW(matrix_lr, betas=[0.9, 0.95], eps=1e-10, weight_decay=weight_decay),
            "mlp_blocks": optim.AdamW(matrix_lr, betas=[0.9, 0.95], eps=1e-10, weight_decay=weight_decay),
        }

    def update(self, model: "MLXGPTPrototype", grads: dict) -> None:
        params = build_optimizer_group_trees(model.parameters())
        grad_groups = build_optimizer_group_trees(grads)
        for group_name, optimizer in self.optimizers.items():
            if sum_parameter_leaves(grad_groups[group_name]) == 0:
                continue
            updated = optimizer.apply_gradients(grad_groups[group_name], params[group_name])
            model.update(updated)

    def state_trees(self) -> list[dict]:
        return [optimizer.state for optimizer in self.optimizers.values()]


class MatrixMuon:
    def __init__(self, *, lr: float, momentum: float = 0.95, beta2: float = 0.9, weight_decay: float = 0.28, ns_steps: int = 5, orthogonalize_dtype: str = "bfloat16"):
        self.lr = lr
        self.momentum = momentum
        self.beta2 = beta2
        self.weight_decay = weight_decay
        self.ns_steps = ns_steps
        self.orthogonalize_dtype = orthogonalize_dtype
        self.state: dict[str, dict[str, mx.array]] = {}

    def _state_key(self, path: tuple[str, ...]) -> str:
        return "/".join(path)

    def _orthogonalize(self, grad: mx.array) -> mx.array:
        x = grad.astype(ORTHOGONALIZE_DTYPES[self.orthogonalize_dtype])
        x_norm = mx.sqrt(mx.sum(mx.square(x.astype(mx.float32))))
        x = x / (x_norm * 1.01 + 1e-6)
        for a, b, c in POLAR_EXPRESS_COEFFS[: self.ns_steps]:
            if x.shape[-2] > x.shape[-1]:
                gram = x.swapaxes(-1, -2) @ x
                poly = b * gram + c * (gram @ gram)
                x = a * x + x @ poly
            else:
                gram = x @ x.swapaxes(-1, -2)
                poly = b * gram + c * (gram @ gram)
                x = a * x + poly @ x
        return x.astype(grad.dtype)

    def _apply_one(self, path: tuple[str, ...], param: mx.array, grad: mx.array) -> mx.array:
        if not isinstance(grad, mx.array) or grad.ndim < 2:
            return param

        key = self._state_key(path)
        if key not in self.state:
            second_shape = (param.shape[-2], 1) if param.shape[-2] >= param.shape[-1] else (1, param.shape[-1])
            self.state[key] = {
                "momentum_buffer": mx.zeros_like(param),
                "second_momentum_buffer": mx.zeros(second_shape, dtype=mx.float32),
            }

        momentum_buffer = self.state[key]["momentum_buffer"]
        second_momentum_buffer = self.state[key]["second_momentum_buffer"]

        momentum_buffer = momentum_buffer + (grad - momentum_buffer) * (1.0 - self.momentum)
        update = grad + (momentum_buffer - grad) * self.momentum
        update = self._orthogonalize(update)

        reduction_dim = -1 if update.shape[-2] >= update.shape[-1] else -2
        reduction_size = update.shape[reduction_dim]
        variance_mean = mx.mean(mx.square(update.astype(mx.float32)), axis=reduction_dim, keepdims=True)
        variance_norm = mx.sqrt(mx.sum(variance_mean) * reduction_size)
        second_momentum_buffer = second_momentum_buffer + (variance_mean - second_momentum_buffer) * (1.0 - self.beta2)
        step_scale = mx.rsqrt(mx.maximum(second_momentum_buffer, 1e-10))
        scaled_sq_sum = (variance_mean * reduction_size) * mx.square(step_scale)
        variance_norm_new = mx.sqrt(mx.sum(scaled_sq_sum))
        final_scale = step_scale * (variance_norm / mx.maximum(variance_norm_new, 1e-10))
        update = update * final_scale.astype(update.dtype)

        lr = self.lr * max(1.0, param.shape[-2] / param.shape[-1]) ** 0.5
        mask = (update * param) >= 0
        updated_param = param - lr * update - lr * self.weight_decay * param * mask.astype(param.dtype)

        self.state[key] = {
            "momentum_buffer": momentum_buffer,
            "second_momentum_buffer": second_momentum_buffer,
        }
        return updated_param

    def apply_gradients(self, grads: dict, params: dict) -> dict:
        return _tree_map_leaves(self._apply_one, params, grads)

    def state_tree(self) -> dict[str, dict[str, mx.array]]:
        return self.state

    def metadata(self) -> dict[str, object]:
        return {
            "lr": self.lr,
            "momentum": self.momentum,
            "beta2": self.beta2,
            "weight_decay": self.weight_decay,
            "ns_steps": self.ns_steps,
            "orthogonalize_dtype": self.orthogonalize_dtype,
        }


class GroupedMixedOptimizer:
    def __init__(self, config: MLXGPTConfig, *, unembedding_lr: float, embedding_lr: float, matrix_lr: float, scalar_lr: float, weight_decay: float, matrix_optimizer: str = "adamw", muon_ns_steps: int = 5, muon_orthogonalize_dtype: str = "bfloat16", muon_block_groups: str = "all"):
        self.adamw = GroupedAdamW(
            config,
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            scalar_lr=scalar_lr,
            weight_decay=weight_decay,
        )
        self.matrix_optimizer_name = matrix_optimizer
        self.muon_block_groups = muon_block_groups
        self.matrix_optimizers: dict[str, MatrixMuon] = {}
        self.muon_group_names = self._resolve_muon_group_names(matrix_optimizer, muon_block_groups)
        if matrix_optimizer != "adamw":
            for group_name in self.muon_group_names:
                self.matrix_optimizers[group_name] = MatrixMuon(
                    lr=matrix_lr,
                    momentum=0.95,
                    beta2=0.9,
                    weight_decay=weight_decay,
                    ns_steps=muon_ns_steps,
                    orthogonalize_dtype=muon_orthogonalize_dtype,
                )

    @staticmethod
    def _resolve_muon_group_names(matrix_optimizer: str, muon_block_groups: str) -> set[str]:
        if matrix_optimizer == "adamw":
            return set()
        if muon_block_groups == "all":
            return {"attn_blocks", "mlp_blocks"}
        if muon_block_groups == "mlp_only":
            return {"mlp_blocks"}
        if muon_block_groups == "attn_only":
            return {"attn_blocks"}
        raise ValueError(f"unsupported muon block group selection: {muon_block_groups}")

    def update(self, model: "MLXGPTPrototype", grads: dict) -> None:
        params = build_optimizer_group_trees(model.parameters())
        grad_groups = build_optimizer_group_trees(grads)

        for group_name, optimizer in self.adamw.optimizers.items():
            if group_name in self.muon_group_names:
                continue
            if sum_parameter_leaves(grad_groups[group_name]) == 0:
                continue
            updated = optimizer.apply_gradients(grad_groups[group_name], params[group_name])
            model.update(updated)

        for group_name, optimizer in self.matrix_optimizers.items():
            if sum_parameter_leaves(grad_groups[group_name]) == 0:
                continue
            updated_group = optimizer.apply_gradients(grad_groups[group_name], params[group_name])
            model.update(updated_group)

    def state_trees(self) -> list[dict]:
        trees = [optimizer.state for name, optimizer in self.adamw.optimizers.items() if name not in self.muon_group_names]
        for group_name in sorted(self.muon_group_names):
            trees.append(self.matrix_optimizers[group_name].state_tree())
        return trees

    def metadata(self) -> dict[str, object]:
        return {
            "matrix_optimizer": self.matrix_optimizer_name,
            "muon_block_groups": self.muon_block_groups,
            "muon_groups_active": sorted(self.muon_group_names),
            "muon_group_configs": {group_name: optimizer.metadata() for group_name, optimizer in self.matrix_optimizers.items()},
        }


class MLXGPTPrototype(nn.Module):
    def __init__(self, config: MLXGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = [TransformerBlock(config, idx) for idx in range(config.n_layer)]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = mx.ones((config.n_layer,))
        self.x0_lambdas = mx.full((config.n_layer,), 0.1)
        self.value_embeds = [nn.Embedding(config.vocab_size, config.n_kv_head * (config.n_embd // config.n_head)) if has_ve(idx, config.n_layer) else None for idx in range(config.n_layer)]

    def __call__(self, idx: mx.array) -> mx.array:
        x = self.wte(idx)
        x = rms_norm(x)
        x0 = x
        for idx_layer, block in enumerate(self.blocks):
            x = self.resid_lambdas[idx_layer] * x + self.x0_lambdas[idx_layer] * x0
            ve_module = self.value_embeds[idx_layer]
            ve = ve_module(idx) if ve_module is not None else None
            x = block(x, ve)
        x = rms_norm(x)
        logits = self.lm_head(x).astype(mx.float32)
        softcap = 15.0
        return softcap * mx.tanh(logits / softcap)

    def loss(self, idx: mx.array, targets: mx.array) -> mx.array:
        logits = self(idx)
        flat_logits = logits.reshape(-1, logits.shape[-1])
        flat_targets = targets.reshape(-1)
        return nn.losses.cross_entropy(flat_logits, flat_targets, reduction="mean")

    def num_params(self) -> int:
        return sum_parameter_leaves(self.parameters())

    def build_optimizer(self, *, unembedding_lr: float = 0.008, embedding_lr: float = 0.3, matrix_lr: float = 0.02, scalar_lr: float = 0.5, weight_decay: float = 0.28, matrix_optimizer: str = "adamw", muon_ns_steps: int = 5, muon_orthogonalize_dtype: str = "bfloat16", muon_block_groups: str = "all") -> GroupedMixedOptimizer:
        return GroupedMixedOptimizer(
            self.config,
            unembedding_lr=unembedding_lr,
            embedding_lr=embedding_lr,
            matrix_lr=matrix_lr,
            scalar_lr=scalar_lr,
            weight_decay=weight_decay,
            matrix_optimizer=matrix_optimizer,
            muon_ns_steps=muon_ns_steps,
            muon_orthogonalize_dtype=muon_orthogonalize_dtype,
            muon_block_groups=muon_block_groups,
        )


def build_reference_config(depth: int = 32, sequence_len: int = 1024, aspect_ratio: int = 64, head_dim: int = 128, vocab_size: int = 32768, window_pattern: str = "L") -> MLXGPTConfig:
    model_dim = ((depth * aspect_ratio + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    return MLXGPTConfig(
        sequence_len=sequence_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )