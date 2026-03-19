"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm:
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model
from nanochat.swift_stub_engine import (
    SWIFT_AUTO_MIN_OUTPUT_TOKENS,
    SwiftStubEngine,
    choose_swift_backend,
    repo_root,
)
from nanochat.tokenizer import get_tokenizer

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Chat with the model')
    parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
    parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
    parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
    parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
    parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
    parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
    parser.add_argument('-m', '--max-tokens', type=int, default=256, help='Maximum number of generated tokens')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('--swift-manifest', type=str, default=None, help='Use the Swift MLX stub with the given export manifest instead of loading a PyTorch checkpoint')
    parser.add_argument('--swift-device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Execution device for the Swift MLX stub')
    parser.add_argument('--swift-rebuild', action='store_true', help='Rebuild the Swift stub before first use')
    parser.add_argument('--no-swift-auto', action='store_true', help='Disable automatic use of a matching MLX export manifest when one is available')
    parser.add_argument('--swift-min-output-tokens', type=int, default=SWIFT_AUTO_MIN_OUTPUT_TOKENS, help='Minimum expected output tokens required for automatic Swift routing')
    return parser


def select_backend(args):
    routing = choose_swift_backend(
        repo_root(),
        source=args.source,
        model_tag=args.model_tag,
        step=args.step,
        explicit_manifest_path=args.swift_manifest,
        allow_auto=not args.no_swift_auto,
        temperature=args.temperature,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        min_output_tokens=args.swift_min_output_tokens,
    )

    if not routing.use_swift:
        device_type = autodetect_device_type() if args.device_type == "" else args.device_type
        ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
        model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)
        engine = Engine(model, tokenizer)
        return engine, tokenizer, None, routing

    tokenizer = get_tokenizer()
    engine = SwiftStubEngine(
        tokenizer,
        routing.manifest_path,
        device=args.swift_device,
        rebuild=args.swift_rebuild,
    )
    return engine, tokenizer, routing.manifest_path, routing


def main():
    args = build_parser().parse_args()
    engine, tokenizer, selected_swift_manifest, routing = select_backend(args)

    print(f"[backend] selected={routing.backend} reason={routing.reason_code} {routing.reason_detail}")

    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]

    while True:

        if args.prompt:
            # Get the prompt from the launch command
            user_input = args.prompt
        else:
            # Get the prompt interactively from the console
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            conversation_tokens = [bos]
            print("Conversation cleared.")
            continue

        if not user_input:
            continue

        # Add User message to the conversation
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tokenizer.encode(user_input))
        conversation_tokens.append(user_end)

        # Kick off the assistant
        conversation_tokens.append(assistant_start)
        generate_kwargs = {
            "num_samples": 1,
            "max_tokens": args.max_tokens,
            "temperature": 0.0 if selected_swift_manifest is not None else args.temperature,
            "top_k": 0 if selected_swift_manifest is not None else args.top_k,
        }
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)
        for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0] # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        print()
        # we have to ensure that the assistant end token is the last token
        # so even if generation ends due to max tokens, we have to append it to the end
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        if selected_swift_manifest is not None and getattr(engine, "last_request_telemetry", None):
            telemetry = engine.last_request_telemetry
            print(
                f"[swift telemetry] device={telemetry.get('device', '?')} "
                f"ttft={telemetry.get('ttft_ms', '?')}ms "
                f"decode={telemetry.get('decode_ms_per_token', '?')}ms/token "
                f"active={telemetry.get('active_memory_gb', '?')}GB "
                f"peak={telemetry.get('peak_memory_gb', '?')}GB "
                f"cache={telemetry.get('cache_memory_gb', '?')}GB "
                f"reuse={telemetry.get('persistent_worker_reuse_count', '?')} "
                f"tokens_decoded={telemetry.get('tokens_decoded', '?')}"
            )

        # In the prompt mode, we only want a single response and exit
        if args.prompt:
            break


if __name__ == "__main__":
    main()
