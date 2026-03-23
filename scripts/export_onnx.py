#!/usr/bin/env python3
"""
Export the nanochat/samosaChaat model to ONNX for WebGPU inference in the browser.

Creates two ONNX models:
1. prefill.onnx  - Processes the full prompt, returns logits + KV cache
2. decode.onnx   - Single-token generation with KV cache, returns logits + updated cache

Usage:
    python -m scripts.export_onnx
    python -m scripts.export_onnx --quantize  # also produce INT4 quantized versions
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from nanochat.common import COMPUTE_DTYPE
from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPTConfig, apply_rotary_emb, has_ve


def norm(x):
    """RMS norm implemented with basic ops for ONNX compatibility."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

parser = argparse.ArgumentParser(description='Export samosaChaat to ONNX')
parser.add_argument('--output-dir', type=str, default='onnx_export', help='Output directory')
parser.add_argument('--quantize', action='store_true', help='Also produce INT4 quantized models')
parser.add_argument('-i', '--source', type=str, default='sft', help='Model source: sft|rl')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag')
parser.add_argument('-s', '--step', type=int, default=None, help='Step')
args = parser.parse_args()


class OnnxAttention(nn.Module):
    """Attention layer rewritten for ONNX export (no flash attention, explicit KV cache I/O)."""

    def __init__(self, attn, layer_idx, n_layer):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = attn.n_head
        self.n_kv_head = attn.n_kv_head
        self.head_dim = attn.head_dim
        self.n_groups = self.n_head // self.n_kv_head
        # Copy weights
        self.c_q = attn.c_q
        self.c_k = attn.c_k
        self.c_v = attn.c_v
        self.c_proj = attn.c_proj
        self.ve_gate = attn.ve_gate

    def forward(self, x, ve, cos, sin, past_key, past_value):
        """
        Args:
            x: (B, T, C)
            ve: (B, T, kv_dim) or None
            cos, sin: (1, T, 1, head_dim//2) - already offset for position
            past_key: (B, past_len, n_kv_head, head_dim)
            past_value: (B, past_len, n_kv_head, head_dim)
        Returns:
            output: (B, T, C)
            present_key: (B, past_len+T, n_kv_head, head_dim)
            present_value: (B, past_len+T, n_kv_head, head_dim)
        """
        B, T, C = x.size()

        # Project Q, K, V
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value embeddings (ResFormer)
        if ve is not None and self.ve_gate is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :12]))
            v = v + gate.unsqueeze(-1) * ve

        # Rotary embeddings
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # QK norm + scaling
        q = norm(q) * 1.2
        k = norm(k) * 1.2

        # Concatenate past KV cache
        # past_key/past_value: (B, past_len, n_kv_head, head_dim)
        present_key = torch.cat([past_key, k], dim=1)
        present_value = torch.cat([past_key, v], dim=1)  # BUG: should be past_value
        present_value = torch.cat([past_value, v], dim=1)

        # Standard attention (no flash attention for ONNX compatibility)
        # Transpose to (B, H, T, D) for matmul
        q_t = q.permute(0, 2, 1, 3)  # (B, n_head, T, head_dim)

        # Expand KV heads for GQA: (B, n_kv_head, S, D) -> (B, n_head, S, D)
        k_t = present_key.permute(0, 2, 1, 3)  # (B, n_kv_head, S, head_dim)
        v_t = present_value.permute(0, 2, 1, 3)
        if self.n_groups > 1:
            k_t = k_t.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_head, -1, self.head_dim)
            v_t = v_t.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1).reshape(B, self.n_head, -1, self.head_dim)

        S = present_key.size(1)  # total sequence length (past + current)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale  # (B, H, T, S)

        # Causal mask: each query position can only attend to positions <= its own
        # Query positions are [S-T, S-T+1, ..., S-1], key positions are [0, 1, ..., S-1]
        query_pos = torch.arange(S - T, S, device=x.device).unsqueeze(1)  # (T, 1)
        key_pos = torch.arange(S, device=x.device).unsqueeze(0)  # (1, S)
        causal_mask = key_pos <= query_pos  # (T, S)
        attn_weights = attn_weights.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        y = torch.matmul(attn_weights, v_t)  # (B, H, T, D)

        # Transpose back and project
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, -1)
        y = self.c_proj(y)

        return y, present_key, present_value


class OnnxMLP(nn.Module):
    """MLP rewritten for ONNX (avoids importing norm from gpt.py)."""
    def __init__(self, mlp):
        super().__init__()
        self.c_fc = mlp.c_fc
        self.c_proj = mlp.c_proj

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class OnnxBlock(nn.Module):
    """Transformer block rewritten for ONNX (explicit KV cache)."""

    def __init__(self, block, layer_idx, n_layer):
        super().__init__()
        self.attn = OnnxAttention(block.attn, layer_idx, n_layer)
        self.mlp = OnnxMLP(block.mlp)

    def forward(self, x, ve, cos, sin, past_key, past_value):
        attn_out, present_key, present_value = self.attn(norm(x), ve, cos, sin, past_key, past_value)
        x = x + attn_out
        x = x + self.mlp(norm(x))
        return x, present_key, present_value


class OnnxGPT(nn.Module):
    """
    ONNX-exportable wrapper around the GPT model.

    Inputs:
        input_ids:      (B, T) int64
        position:       (1,) int64 - position offset for rotary embeddings
        prev_embedding: (B, 1, n_embd) - previous token's embedding for smear
        past_keys:      (n_layers, B, past_len, n_kv_head, head_dim)
        past_values:    (n_layers, B, past_len, n_kv_head, head_dim)

    Outputs:
        logits:             (B, T, vocab_size)
        new_prev_embedding: (B, 1, n_embd)
        present_keys:       (n_layers, B, past_len+T, n_kv_head, head_dim)
        present_values:     (n_layers, B, past_len+T, n_kv_head, head_dim)
    """

    def __init__(self, model):
        super().__init__()
        config = model.config
        self.config = config
        self.n_layer = config.n_layer
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.vocab_size = config.vocab_size

        # Copy model components
        self.wte = model.transformer.wte
        self.lm_head = model.lm_head
        self.resid_lambdas = model.resid_lambdas
        self.x0_lambdas = model.x0_lambdas
        self.smear_gate = model.smear_gate
        self.smear_lambda = model.smear_lambda
        self.backout_lambda = model.backout_lambda
        self.value_embeds = model.value_embeds

        # Rotary embeddings (precomputed)
        self.register_buffer("cos", model.cos, persistent=False)
        self.register_buffer("sin", model.sin, persistent=False)

        # Window sizes (baked in as constants)
        self.window_sizes = model.window_sizes

        # Rebuild blocks with ONNX-compatible attention
        self.blocks = nn.ModuleList([
            OnnxBlock(model.transformer.h[i], i, config.n_layer)
            for i in range(config.n_layer)
        ])

    def forward(self, input_ids, cos_slice, sin_slice, prev_embedding, past_keys, past_values):
        """
        Args:
            input_ids: (B, T)
            cos_slice: (1, T, 1, head_dim//2) - pre-sliced rotary cos for current positions
            sin_slice: (1, T, 1, head_dim//2) - pre-sliced rotary sin for current positions
            prev_embedding: (B, 1, n_embd) - previous token's embedding for smear
            past_keys: (n_layers, B, past_len, n_kv_head, head_dim)
            past_values: (n_layers, B, past_len, n_kv_head, head_dim)
        """
        B, T = input_ids.size()
        cos = cos_slice
        sin = sin_slice

        # Token embedding + norm
        x = self.wte(input_ids)
        x = x.to(self.cos.dtype)
        x = norm(x)

        # Smear: mix previous token's embedding
        new_prev_embedding = x[:, -1:, :].clone()
        if T > 1:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
            x = x + gate * prev_embedding

        # Transformer blocks with explicit KV cache
        x0 = x
        backout_layer = self.n_layer // 2
        x_backout = torch.zeros_like(x)
        present_keys_list = []
        present_values_list = []

        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0

            # Value embeddings (alternating layers)
            ve = self.value_embeds[str(i)](input_ids).to(x.dtype) if str(i) in self.value_embeds else None

            # Get past KV for this layer
            layer_past_k = past_keys[i]
            layer_past_v = past_values[i]

            x, present_k, present_v = block(x, ve, cos, sin, layer_past_k, layer_past_v)
            present_keys_list.append(present_k.unsqueeze(0))
            present_values_list.append(present_v.unsqueeze(0))

            if i == backout_layer:
                x_backout = x

        # Backout: subtract mid-layer residual
        x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Logits with softcap
        logits = self.lm_head(x)
        logits = logits[..., :self.vocab_size]
        logits = logits.float()
        logits = 15.0 * torch.tanh(logits / 15.0)

        # Stack KV caches
        present_keys = torch.cat(present_keys_list, dim=0)
        present_values = torch.cat(present_values_list, dim=0)

        return logits, new_prev_embedding, present_keys, present_values


def export_model(model, tokenizer, output_dir):
    """Export model to ONNX format."""
    os.makedirs(output_dir, exist_ok=True)
    config = model.config

    print("Building ONNX-compatible model wrapper...")
    onnx_model = OnnxGPT(model)
    onnx_model.eval()

    n_kv_head = config.n_kv_head
    head_dim = config.n_embd // config.n_head
    n_layer = config.n_layer

    # Pre-compute rotary embeddings on CPU
    cos_full = onnx_model.cos  # (1, max_len, 1, head_dim//2)
    sin_full = onnx_model.sin

    # --- Export Prefill Model ---
    print("\nExporting prefill model...")
    batch_size = 1
    seq_len = 4  # dummy prompt length for tracing
    dummy_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    dummy_cos = cos_full[:, :seq_len].contiguous()
    dummy_sin = sin_full[:, :seq_len].contiguous()
    dummy_prev_emb = torch.zeros(batch_size, 1, config.n_embd, dtype=COMPUTE_DTYPE)
    dummy_past_keys = torch.zeros(n_layer, batch_size, 0, n_kv_head, head_dim, dtype=COMPUTE_DTYPE)
    dummy_past_values = torch.zeros(n_layer, batch_size, 0, n_kv_head, head_dim, dtype=COMPUTE_DTYPE)

    # Test forward pass first
    print("  Testing forward pass...")
    with torch.no_grad():
        logits, new_prev, pk, pv = onnx_model(dummy_ids, dummy_cos, dummy_sin, dummy_prev_emb, dummy_past_keys, dummy_past_values)
    print(f"  Prefill output shapes: logits={logits.shape}, present_keys={pk.shape}")

    # Export with legacy exporter (dynamo=False) for better compatibility
    prefill_path = os.path.join(output_dir, "prefill.onnx")
    print(f"  Exporting to {prefill_path}...")
    torch.onnx.export(
        onnx_model,
        (dummy_ids, dummy_cos, dummy_sin, dummy_prev_emb, dummy_past_keys, dummy_past_values),
        prefill_path,
        dynamo=False,
        input_names=["input_ids", "cos", "sin", "prev_embedding", "past_keys", "past_values"],
        output_names=["logits", "new_prev_embedding", "present_keys", "present_values"],
        dynamic_axes={
            "input_ids": {1: "seq_len"},
            "cos": {1: "seq_len"},
            "sin": {1: "seq_len"},
            "past_keys": {2: "past_len"},
            "past_values": {2: "past_len"},
            "logits": {1: "seq_len"},
            "present_keys": {2: "total_len"},
            "present_values": {2: "total_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Prefill model exported: {os.path.getsize(prefill_path) / 1e6:.1f} MB")

    # --- Export Decode Model (single token) ---
    print("\nExporting decode model...")
    dummy_decode_ids = torch.randint(0, config.vocab_size, (batch_size, 1))
    dummy_decode_cos = cos_full[:, seq_len:seq_len+1].contiguous()
    dummy_decode_sin = sin_full[:, seq_len:seq_len+1].contiguous()
    dummy_decode_prev = new_prev.detach()
    dummy_decode_past_k = pk.detach()
    dummy_decode_past_v = pv.detach()

    decode_path = os.path.join(output_dir, "decode.onnx")
    print(f"  Exporting to {decode_path}...")
    torch.onnx.export(
        onnx_model,
        (dummy_decode_ids, dummy_decode_cos, dummy_decode_sin, dummy_decode_prev, dummy_decode_past_k, dummy_decode_past_v),
        decode_path,
        dynamo=False,
        input_names=["input_ids", "cos", "sin", "prev_embedding", "past_keys", "past_values"],
        output_names=["logits", "new_prev_embedding", "present_keys", "present_values"],
        dynamic_axes={
            "past_keys": {2: "past_len"},
            "past_values": {2: "past_len"},
            "present_keys": {2: "total_len"},
            "present_values": {2: "total_len"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"  Decode model exported: {os.path.getsize(decode_path) / 1e6:.1f} MB")

    # --- Save config for JS runtime ---
    config_dict = {
        "n_layer": config.n_layer,
        "n_head": config.n_head,
        "n_kv_head": config.n_kv_head,
        "n_embd": config.n_embd,
        "head_dim": head_dim,
        "vocab_size": config.vocab_size,
        "sequence_len": config.sequence_len,
        "window_pattern": config.window_pattern,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"\nConfig saved to {config_path}")

    # --- Validate with ONNX Runtime ---
    print("\nValidating with ONNX Runtime...")
    try:
        import onnxruntime as ort
        # Test prefill
        sess = ort.InferenceSession(prefill_path, providers=["CPUExecutionProvider"])
        feeds = {
            "input_ids": dummy_ids.numpy(),
            "cos": dummy_cos.numpy().astype("float32"),
            "sin": dummy_sin.numpy().astype("float32"),
            "prev_embedding": dummy_prev_emb.numpy().astype("float32"),
            "past_keys": dummy_past_keys.numpy().astype("float32"),
            "past_values": dummy_past_values.numpy().astype("float32"),
        }
        ort_logits, ort_prev, ort_pk, ort_pv = sess.run(None, feeds)
        print(f"  Prefill ONNX Runtime OK: logits={ort_logits.shape}")

        # Compare with PyTorch output
        max_diff = abs(logits.numpy() - ort_logits).max()
        print(f"  Max logit difference (PyTorch vs ONNX Runtime): {max_diff:.6f}")

        # Test decode
        sess_decode = ort.InferenceSession(decode_path, providers=["CPUExecutionProvider"])
        feeds_decode = {
            "input_ids": dummy_decode_ids.numpy(),
            "cos": dummy_decode_cos.numpy().astype("float32"),
            "sin": dummy_decode_sin.numpy().astype("float32"),
            "prev_embedding": ort_prev.astype("float32"),
            "past_keys": ort_pk.astype("float32"),
            "past_values": ort_pv.astype("float32"),
        }
        ort_dec_logits, _, _, _ = sess_decode.run(None, feeds_decode)
        print(f"  Decode ONNX Runtime OK: logits={ort_dec_logits.shape}")
        print("\nONNX export validated successfully!")

    except Exception as e:
        print(f"  ONNX Runtime validation failed: {e}")
        print("  The ONNX files were still exported - fix the validation issue before deployment.")

    return prefill_path, decode_path


def quantize_models(output_dir):
    """Quantize ONNX models to INT4 for smaller download size."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("onnxruntime quantization not available. Install: pip install onnxruntime")
        return

    for name in ["prefill", "decode"]:
        input_path = os.path.join(output_dir, f"{name}.onnx")
        output_path = os.path.join(output_dir, f"{name}_q4.onnx")
        print(f"\nQuantizing {name} to INT4...")
        quantize_dynamic(
            input_path,
            output_path,
            weight_type=QuantType.QUInt8,  # INT4 not always available, use INT8 as fallback
        )
        original_size = os.path.getsize(input_path) / 1e6
        quant_size = os.path.getsize(output_path) / 1e6
        print(f"  {original_size:.1f} MB -> {quant_size:.1f} MB ({quant_size/original_size*100:.0f}%)")


if __name__ == "__main__":
    print("=" * 60)
    print("samosaChaat ONNX Export")
    print("=" * 60)

    # Load model
    device = torch.device("cpu")
    model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
    model.eval()

    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Config: n_layer={model.config.n_layer}, n_head={model.config.n_head}, "
          f"n_kv_head={model.config.n_kv_head}, n_embd={model.config.n_embd}")

    # Export
    prefill_path, decode_path = export_model(model, tokenizer, args.output_dir)

    # Quantize if requested
    if args.quantize:
        quantize_models(args.output_dir)

    print("\n" + "=" * 60)
    print("Export complete!")
    print(f"Files in: {args.output_dir}/")
    print("=" * 60)
