# Model Configuration Examples

This directory contains example configurations for training hybrid models with different block patterns.

## Usage

Pass configuration files to training scripts:

```bash
# Pure transformer (default)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train

# Pure Mamba
torchrun --standalone --nproc_per_node=8 -m scripts.base_train configs/mamba_d20.py

# Hybrid: early transformer, late Mamba
torchrun --standalone --nproc_per_node=8 -m scripts.base_train configs/hybrid_early_t_late_m_d20.py

# Alternating transformer and Mamba
torchrun --standalone --nproc_per_node=8 -m scripts.base_train configs/hybrid_alternating_d20.py
```

## Configuration Options

### `block_pattern`
List of block types, one per layer. Valid types:
- `"T"` or `"transformer"`: Transformer block with attention
- `"M"` or `"mamba"`: Mamba block with SSM

Example: `["T", "T", "M", "M"]` for 4 layers (2 transformer, 2 Mamba)

If `None` or omitted, defaults to all transformer blocks (backward compatible).

### Mamba-specific Parameters

- `mamba_d_state`: State space dimension (default: 16, range: 16-64)
- `mamba_d_conv`: Convolution kernel size (default: 4)
- `mamba_expand`: Inner dimension expansion factor (default: 2)
- `mamba_use_mlp`: Add MLP after Mamba (default: False, usually not needed)

## GPU Memory Considerations

For 12GB GPUs (RTX 3060/3070):
- Use smaller models (d16 or d20 with reduced batch size)
- Reduce `device_batch_size` to 2-4
- Consider `max_seq_len=1024` instead of 2048

For 8GB GPUs (RTX 3070 non-Ti):
- Use d12 or d16 models
- Set `device_batch_size=2`
- Set `max_seq_len=512` or `1024`

## Block Pattern Strategies

### Strategy 1: Early Transformer, Late Mamba
**Rationale**: Transformers excel at token-level patterns, Mamba handles long-range dependencies.
```python
block_pattern = ["T"] * 12 + ["M"] * 8  # For d20
```

### Strategy 2: Alternating
**Rationale**: Mix local attention with long-range SSM processing.
```python
block_pattern = ["T", "M"] * 10  # For d20
```

### Strategy 3: Strategic Transformer Placement
**Rationale**: Use attention at key positions (early, middle, late).
```python
block_pattern = ["T", "T"] + ["M"] * 6 + ["T"] + ["M"] * 6 + ["T", "M", "T", "T"]
```

### Strategy 4: Pure Mamba
**Rationale**: Maximum efficiency for long sequences.
```python
block_pattern = ["M"] * 20
```

## Performance Notes

- **Training Speed**: Mamba is 10-20% faster for sequences >2048 tokens
- **Inference Speed**: Mamba is 30-50% faster due to smaller state cache
- **Memory Usage**: Mamba uses 30-40% less activation memory
- **Quality**: Hybrid models often match or exceed pure transformer quality

