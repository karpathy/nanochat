# nanochat Model Extraction & Inference Scripts

Scripts to extract trained nanochat models from UPPMAX and run inference locally or in Google Colab.

## Quick Start

### 1. Download Weights from UPPMAX

```bash
# Make sure you can SSH to UPPMAX first
ssh birger@pelle.uppmax.uu.se

# Download all checkpoints and logs
./download_weights.sh
```

This downloads:
- All model checkpoints from `~/.cache/nanochat/base_checkpoints/d20/`
- Training logs (`nanochat-*.out` files)
- Creates local `checkpoints/nanochat-d20/` directory

### 2. Convert to Hugging Face Format

```bash
# Convert best checkpoint (usually the highest step number)
python convert_to_hf.py checkpoints/nanochat-d20/step_2500.pt nanochat-d20-hf
```

This creates:
- `nanochat-d20-hf/pytorch_model.bin` - model weights
- `nanochat-d20-hf/config.json` - HF-compatible configuration  
- `nanochat-d20-hf/training_metadata.json` - training info

### 3. Run Inference

#### Option A: Google Colab

1. Upload the `nanochat-d20-hf/` folder to Colab files
2. Upload `colab_inference.py` to Colab
3. Run the script:

```python
# In Colab notebook
!pip install torch transformers tokenizers

# Update MODEL_DIR to match your uploaded folder
exec(open('colab_inference.py').read())
```

#### Option B: Mac Local (with GPU)

```bash
# Run with MPS GPU support on Mac
python mac_inference.py nanochat-d20-hf "The meaning of life is"
```

## Script Details

### `download_weights.sh`
- Downloads model checkpoints from UPPMAX Pelle
- Fetches training logs for debugging
- Creates organized local directory structure

### `convert_to_hf.py`
- Converts nanochat `.pt` checkpoints to Hugging Face format
- Preserves original configuration for reference
- Adds training metadata (step, loss, etc.)

### `colab_inference.py`
- Google Colab-ready inference wrapper
- Handles GPU detection and model loading
- Shows model information and parameter counts
- **Note:** Placeholder generation - needs nanochat model implementation

### `mac_inference.py`  
- Mac-optimized inference with MPS (Metal Performance Shaders) GPU support
- Includes speed benchmarking
- Automatic device selection (MPS → CPU fallback)
- **Note:** Placeholder generation - needs nanochat model implementation

## Your Training Results

Based on your recent training run:

- **Run:** `nanochat-d20-ckpt-1853832`
- **Steps:** 2,600 completed ✅
- **Loss:** 2.655 (final training loss)
- **Validation:** 0.884 bits-per-byte
- **Training Time:** ~23 hours
- **Checkpoints:** Every 500 steps (5+ checkpoints available)

Best checkpoint is likely: `step_2500.pt` or `step_2000.pt`

## Troubleshooting

### SSH Access Issues
If `download_weights.sh` fails:
```bash
# Test SSH access manually
ssh birger@pelle.uppmax.uu.se "ls ~/.cache/nanochat/base_checkpoints/d20/"

# Check if files exist
ssh birger@pelle.uppmax.uu.se "ls -la ~/nanochat-*.out"
```

### Missing Checkpoints
Check which checkpoints are available:
```bash
ssh birger@pelle.uppmax.uu.se "ls -la ~/.cache/nanochat/base_checkpoints/d20/"
```

### Generation Not Working
The current scripts are **inference-ready wrappers**. To enable actual text generation:

1. Import nanochat's model classes (`GPT`, `GPTConfig`)
2. Load the state dict into the model
3. Implement tokenization (nanochat uses GPT tokenizer)
4. Add the generation loop

Example integration:
```python
from nanochat.model import GPT, GPTConfig

# Load config and create model
config = GPTConfig(**checkpoint['config'])  
model = GPT(config)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Generate text
with torch.no_grad():
    # Add tokenization and generation logic here
    pass
```

## Next Steps

1. **Download weights:** `./download_weights.sh`
2. **Convert format:** `python convert_to_hf.py checkpoints/nanochat-d20/step_2500.pt nanochat-d20-hf`
3. **Test locally:** `python mac_inference.py nanochat-d20-hf`
4. **Try Colab:** Upload `nanochat-d20-hf/` and run `colab_inference.py`

Your depth=20 model trained successfully for 2,600 steps - it's ready to extract and run! 🚀