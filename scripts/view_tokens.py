"""
A simple utility to view the contents of a trained tokenizer.
It loads a tokenizer from the default directory and prints the string
representation for each token ID up to a given limit.
"""
import os
import argparse
from nanochat.common import get_base_dir
from nanochat.tokenizer import RustBPETokenizer

def view_tokens(limit):
    """
    Loads the tokenizer and prints the decoded representation of each token ID.
    """
    # --- Initialization ---
    try:
        base_dir = get_base_dir()
        tokenizer_dir = os.path.join(base_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir) or not os.listdir(tokenizer_dir):
            print(f"Error: Tokenizer directory not found or is empty at {tokenizer_dir}")
            print("Please train a tokenizer first using scripts/tok_train.py")
            return

        print(f"Loading tokenizer from {tokenizer_dir}...")
        tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return

    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary Size: {vocab_size}")
    print("-" * 30)

    # Determine the number of tokens to print
    num_to_print = min(limit, vocab_size) if limit != -1 else vocab_size
    print(f"Displaying the first {num_to_print} tokens...\n")

    # --- Loop and Decode ---
    for i in range(num_to_print):
        try:
            # The decode method expects a list of IDs, even for a single token
            decoded_token = tokenizer.decode([i])
            # We use repr() to make whitespace, newlines, and control characters visible
            print(f"Token {i:<5}: {repr(decoded_token)}")
        except Exception as e:
            print(f"Could not decode token {i}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="View tokens from a trained nanochat tokenizer.")
    parser.add_argument(
        '--limit',
        type=int,
        default=300,
        help="Number of tokens to print. Use -1 to print all tokens."
    )
    args = parser.parse_args()
    view_tokens(args.limit)
