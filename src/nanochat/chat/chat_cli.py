"""
New and upgraded chat mode because a lot of the code has changed since the last one.

Intended to be run single GPU only atm.
"""

from nanochat.config import Config
from nanochat.common import autodetect_device_type, compute_init
from nanochat.evaluation.engine import Engine
from nanochat.training.checkpoint import load_model


def chat_cli(config: Config, source: str, model_tag: str, step: int, prompt: str, temperature: float, top_k: int) -> None:
    """Run an interactive chat session with a trained model.

    Loads the model from ``config.common.base_dir`` using ``source`` and optional
    ``model_tag``/``step``, then enters a REPL loop. In prompt mode prints a single
    response and exits.

    Args:
        config: Resolved nanochat config. Uses ``config.common.device_type`` and ``config.common.base_dir``.
        source: Checkpoint source to load from: ``sft`` or ``rl``.
        model_tag: Optional model tag to select a specific checkpoint.
        step: Optional step number to load a specific checkpoint.
        prompt: If non-empty, send this prompt, print the response, and exit.
        temperature: Sampling temperature.
        top_k: Top-k sampling parameter.
    """
    # Init the model and tokenizer
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, _, _, _, device = compute_init(device_type)
    model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)

    # Special tokens for the chat state machine
    bos = tokenizer.get_bos_token_id()
    user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
    assistant_start, assistant_end = (
        tokenizer.encode_special("<|assistant_start|>"),
        tokenizer.encode_special("<|assistant_end|>"),
    )

    # Create Engine for efficient generation
    engine = Engine(model, tokenizer)

    print("\nNanoChat Interactive Mode")
    print("-" * 50)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("-" * 50)

    conversation_tokens = [bos]

    while True:
        if prompt:
            # Get the prompt from the launch command
            user_input = prompt
        else:
            # Get the prompt interactively from the console
            try:
                user_input = input("\nUser: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

        # Handle special commands
        if user_input.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break

        if user_input.lower() == "clear":
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
            "max_tokens": 256,
            "temperature": temperature,
            "top_k": top_k,
        }
        response_tokens = []
        print("\nAssistant: ", end="", flush=True)
        for token_column, _ in engine.generate(conversation_tokens, **generate_kwargs):
            token = token_column[0]  # pop the batch dimension (num_samples=1)
            response_tokens.append(token)
            token_text = tokenizer.decode([token])
            print(token_text, end="", flush=True)
        print()
        # we have to ensure that the assistant end token is the last token
        # so even if generation ends due to max tokens, we have to append it to the end
        if response_tokens[-1] != assistant_end:
            response_tokens.append(assistant_end)
        conversation_tokens.extend(response_tokens)

        # In the prompt mode, we only want a single response and exit
        if prompt:
            break
