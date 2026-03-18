import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

def main():
    parser = argparse.ArgumentParser(description='Sample from a base model checkpoint')
    parser.add_argument('-m', '--model-tag', type=str, default="d2", help='Model tag to load (e.g. d2)')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load (default: last)')
    parser.add_argument('-p', '--prompt', type=str, default='The capital of France is', help='Prompt to start with')
    parser.add_argument('-n', '--num-tokens', type=int, default=100, help='Number of tokens to generate')
    parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device to use')
    args = parser.parse_args()

    # 1. Initialize Device
    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    _, _, _, _, device = compute_init(device_type)

    # 2. Load Model and Tokenizer
    print(f"Loading base model '{args.model_tag}' at step {args.step or 'latest'}...")
    try:
        model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=args.model_tag, step=args.step)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Prepare Engine
    engine = Engine(model, tokenizer)

    # 4. Encode Prompt
    prompt_tokens = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Generating {args.num_tokens} tokens...\n")

    # 5. Generate
    print(f"--- Output ---")
    print(args.prompt, end="", flush=True)
    
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": args.num_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }

    # engine.generate yields tokens as they are produced
    for token_column, _ in engine.generate(prompt_tokens, **generate_kwargs):
        token = token_column[0]
        text = tokenizer.decode([token])
        print(text, end="", flush=True)
    
    print("\n\n--- End of Sampling ---")
    print("(Note: At 100 steps, a tiny model will likely produce gibberish!)")

if __name__ == "__main__":
    main()
