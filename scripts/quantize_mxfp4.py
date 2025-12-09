import os
import torch
import argparse
from torch.utils.data import DataLoader, TensorDataset
from nanochat.checkpoint_manager import load_model
from nanochat.dataloader import tokenizing_distributed_data_loader
from nanochat.dataset import list_parquet_files
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Config, OCP_MXFP4Spec, QuantizationConfig

def get_calibration_data(tokenizer, device, num_samples=128, batch_size=1, sequence_len=1024):
    """
    Creates a simple dataloader for calibration.
    Tries to use real data if available, otherwise falls back to random data.
    """
    # Check if we have data
    parquet_files = list_parquet_files()
    if parquet_files:
        print(f"Found {len(parquet_files)} parquet files. Using real data for calibration.")
        # We use the validation split (last file) or training split if enough files
        # Note: tokenizing_distributed_data_loader expects 'train' (all but last) or 'val' (last only)
        # If we have only 1 file, 'train' gets 0 files, 'val' gets 1 file.
        split = "val" if len(parquet_files) > 0 else "train"

        # Generator that yields batches from the main dataloader
        loader = tokenizing_distributed_data_loader(
            B=batch_size,
            T=sequence_len,
            split=split,
            tokenizer_batch_size=32, # batch size for tokenizer parallelism
            device=device
        )

        samples = []
        count = 0
        print(f"Collecting {num_samples} samples...")
        for inputs, _ in loader: # ignore targets
            samples.append(inputs)
            count += batch_size
            if count >= num_samples:
                break

        # Concatenate all collected batches
        if len(samples) > 0:
            all_data = torch.cat(samples, dim=0) # (num_samples, T)
            # Truncate if we got more than needed (though loop check handles it)
            all_data = all_data[:num_samples]
            dataset = TensorDataset(all_data)
            return DataLoader(dataset, batch_size=batch_size)
        else:
            print("Warning: Could not collect any samples from dataloader. Falling back to dummy data.")

    print("Using random dummy data for calibration.")
    # Generate dummy data
    vocab_size = tokenizer.get_vocab_size()
    dummy_data = torch.randint(0, vocab_size, (num_samples, sequence_len), device=device)
    # TensorDataset wraps tensors, DataLoader yields tuples of tensors
    dataset = TensorDataset(dummy_data)
    return DataLoader(dataset, batch_size=batch_size)

def main():
    parser = argparse.ArgumentParser(description="Quantize nanochat model to MXFP4 using AMD Quark")
    parser.add_argument("--source", type=str, default="base", choices=["base", "mid", "sft", "rl"], help="Model source to load")
    parser.add_argument("--model-tag", type=str, default=None, help="Specific model tag (e.g. d12)")
    parser.add_argument("--step", type=int, default=None, help="Specific step to load")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run quantization on (cpu or cuda)")
    parser.add_argument("--output", type=str, default="mxfp4_model.pt", help="Output path for quantized model")
    parser.add_argument("--num-calib-samples", type=int, default=128, help="Number of samples for calibration")

    args = parser.parse_args()

    # 1. Configuration
    # We target the standard OCP MXFP4 format (E2M1)
    # group_size=32 is standard for OCP Microscaling, but implicitly handled by OCP_MXFP4Spec
    # ch_axis=0 is the default (per output channel)
    print("Configuring OCP MXFP4 Quantization...")

    ocp_spec = OCP_MXFP4Spec(
        is_dynamic=False,       # Static quantization (requires calibration) usually performs better
        ch_axis=0               # Quantize per output channel
    )

    # We convert to the low-level quantization spec as Quark's QuantizationConfig expects a spec
    # object with a 'dtype' attribute, which the high-level OCP_MXFP4Spec wrapper doesn't expose directly.
    quant_spec = ocp_spec.to_quantization_spec()

    # Apply this spec to all weights
    quant_config = QuantizationConfig(weight=quant_spec)
    config = Config(global_quant_config=quant_config)

    # 2. Load the Model
    device = torch.device(args.device)
    print(f"Loading model '{args.source}' on {device}...")
    try:
        model, tokenizer, meta = load_model(args.source, device=device, phase="eval", model_tag=args.model_tag, step=args.step)
    except FileNotFoundError as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have a trained model checkpoint available.")
        return

    # 3. Create Calibration Data
    print("Preparing calibration data...")
    calib_loader = get_calibration_data(
        tokenizer,
        device=device,
        num_samples=args.num_calib_samples,
        batch_size=1, # Quark often prefers batch_size=1 for calibration
        sequence_len=model.config.sequence_len
    )

    # 4. Quantize
    print("Starting quantization (this may take a while)...")
    quantizer = ModelQuantizer(config)
    # quantize_model replaces nn.Linear layers with Quark's quantized equivalents
    quantized_model = quantizer.quantize_model(model, calib_loader)

    # 5. Export/Save
    print(f"Saving quantized model to {args.output}...")
    # Quark models usually need to be saved with a specific utility to preserve the quantization metadata
    # or exported to a format like ONNX or GGUF if the goal is external inference.
    # For now, we use standard torch.save on the state_dict.
    torch.save(quantized_model.state_dict(), args.output)
    print("Model quantized to MXFP4 and saved.")

if __name__ == "__main__":
    main()
