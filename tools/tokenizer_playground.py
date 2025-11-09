#!/usr/bin/env python3
"""
Interactive Tokenizer Playground

Visualize how text is tokenized, explore token boundaries, and understand BPE.
Perfect for understanding how "Hello world" becomes [15496, 995].

Usage:
    python tools/tokenizer_playground.py "Hello world!"
    python tools/tokenizer_playground.py --interactive
    python tools/tokenizer_playground.py --special
"""

import argparse
import sys


class TokenizerPlayground:
    """Interactive tool for exploring tokenization."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

        # Terminal color codes
        self.COLORS = [
            '\033[91m',  # Red
            '\033[92m',  # Green
            '\033[93m',  # Yellow
            '\033[94m',  # Blue
            '\033[95m',  # Magenta
            '\033[96m',  # Cyan
        ]
        self.RESET = '\033[0m'
        self.BOLD = '\033[1m'
        self.DIM = '\033[2m'

    def tokenize_and_visualize(self, text: str) -> None:
        """
        Tokenize text and display results with colors and details.

        Args:
            text: Input text to tokenize
        """
        # Encode text
        token_ids = self.tokenizer.encode(text)

        print(f"\n{self.BOLD}{'='*70}{self.RESET}")
        print(f"{self.BOLD}TOKENIZATION VISUALIZATION{self.RESET}")
        print(f"{self.BOLD}{'='*70}{self.RESET}\n")

        print(f"{self.BOLD}Original Text:{self.RESET}")
        print(f'  "{text}"\n')

        print(f"{self.BOLD}Quick Stats:{self.RESET}")
        print(f"  Total tokens:      {len(token_ids)}")
        print(f"  Total characters:  {len(text)}")
        print(f"  Total bytes:       {len(text.encode('utf-8'))}")

        if len(text.encode('utf-8')) > 0:
            ratio = len(token_ids) / len(text.encode('utf-8'))
            print(f"  Compression ratio: {ratio:.3f} tokens/byte")
            print(f"  Efficiency:        {len(text)/len(token_ids) if len(token_ids) > 0 else 0:.2f} chars/token")

        # Colorized token display
        print(f"\n{self.BOLD}Colored Token Breakdown:{self.RESET}")
        print(f"{self.DIM}(Each color represents a different token){self.RESET}\n")

        reconstructed = ""
        for i, token_id in enumerate(token_ids):
            token_str = self.tokenizer.decode([token_id])
            color = self.COLORS[i % len(self.COLORS)]

            # Show with color
            print(f"{color}{token_str}{self.RESET}", end="")
            reconstructed += token_str

        print("\n")

        # Verify reconstruction
        if reconstructed != text:
            print(f"{self.BOLD}⚠️  Warning:{self.RESET} Reconstruction doesn't match original!")
            print(f"  Original:      {repr(text)}")
            print(f"  Reconstructed: {repr(reconstructed)}\n")

        # Detailed table
        print(f"\n{self.BOLD}Detailed Token Information:{self.RESET}\n")
        print(f"{'Index':<6} {'Token ID':<10} {'Text':<35} {'Bytes':<8} {'Type':<15}")
        print("-" * 80)

        for i, token_id in enumerate(token_ids):
            token_str = self.tokenizer.decode([token_id])
            token_bytes = token_str.encode('utf-8')
            num_bytes = len(token_bytes)

            # Determine token type (simple heuristics)
            if token_str in self.tokenizer.get_special_tokens():
                token_type = "Special"
            elif token_id < 256:
                token_type = "Single byte"
            elif token_str.strip() == "":
                token_type = "Whitespace"
            elif token_str.isalpha():
                token_type = "Alphabetic"
            elif token_str.isdigit():
                token_type = "Numeric"
            elif token_str.isalnum():
                token_type = "Alphanumeric"
            else:
                token_type = "Mixed/Other"

            # Format for display (handle special chars)
            display_str = repr(token_str)[1:-1]  # Remove outer quotes
            if len(display_str) > 33:
                display_str = display_str[:30] + "..."

            print(f"{i:<6} {token_id:<10} {display_str:<35} {num_bytes:<8} {token_type:<15}")

        print(f"\n{self.BOLD}{'='*70}{self.RESET}\n")

    def analyze_special_tokens(self) -> None:
        """Display all special tokens and their IDs."""
        special_tokens = sorted(list(self.tokenizer.get_special_tokens()))

        print(f"\n{self.BOLD}{'='*70}{self.RESET}")
        print(f"{self.BOLD}SPECIAL TOKENS{self.RESET}")
        print(f"{self.BOLD}{'='*70}{self.RESET}\n")

        print(f"Total special tokens: {len(special_tokens)}\n")

        print(f"{'Token':<25} {'ID':<10} {'Purpose':<30}")
        print("-" * 70)

        # Add helpful descriptions
        descriptions = {
            "<|bos|>": "Beginning of sequence",
            "<|user_start|>": "Start of user message",
            "<|user_end|>": "End of user message",
            "<|assistant_start|>": "Start of assistant message",
            "<|assistant_end|>": "End of assistant message",
            "<|python_start|>": "Start of Python code",
            "<|python_end|>": "End of Python code",
            "<|output_start|>": "Start of tool output",
            "<|output_end|>": "End of tool output",
        }

        for token in special_tokens:
            token_id = self.tokenizer.encode_special(token)
            desc = descriptions.get(token, "")
            print(f"{token:<25} {token_id:<10} {desc:<30}")

        print(f"\n{self.BOLD}Learning Note:{self.RESET}")
        print("  Special tokens are used to structure conversations and tool use.")
        print("  They don't appear in regular text but are added during fine-tuning.")
        print(f"\n{self.BOLD}{'='*70}{self.RESET}\n")

    def show_vocab_info(self) -> None:
        """Show information about the vocabulary."""
        vocab_size = self.tokenizer.get_vocab_size()
        special_tokens = self.tokenizer.get_special_tokens()

        print(f"\n{self.BOLD}{'='*70}{self.RESET}")
        print(f"{self.BOLD}VOCABULARY INFORMATION{self.RESET}")
        print(f"{self.BOLD}{'='*70}{self.RESET}\n")

        print(f"Total vocabulary size:  {vocab_size:,}")
        print(f"Special tokens:         {len(special_tokens)}")
        print(f"Regular tokens:         {vocab_size - len(special_tokens):,}")
        print(f"Single-byte tokens:     256 (first 256 IDs)")
        print(f"Merged BPE tokens:      {vocab_size - 256 - len(special_tokens):,}")

        print(f"\n{self.BOLD}Vocabulary Breakdown:{self.RESET}")
        print(f"  [0-255]           Single bytes (UTF-8)")
        print(f"  [256-{vocab_size-len(special_tokens)-1}]      BPE merged tokens")
        print(f"  [{vocab_size-len(special_tokens)}-{vocab_size-1}]       Special tokens")

        print(f"\n{self.BOLD}{'='*70}{self.RESET}\n")

    def compare_texts(self, texts: list) -> None:
        """Compare tokenization of multiple texts."""
        print(f"\n{self.BOLD}{'='*70}{self.RESET}")
        print(f"{self.BOLD}TOKENIZATION COMPARISON{self.RESET}")
        print(f"{self.BOLD}{'='*70}{self.RESET}\n")

        results = []
        for text in texts:
            tokens = self.tokenizer.encode(text)
            results.append({
                'text': text,
                'tokens': tokens,
                'num_tokens': len(tokens),
                'num_chars': len(text),
                'num_bytes': len(text.encode('utf-8')),
            })

        print(f"{'Text':<40} {'Tokens':<8} {'Chars':<8} {'Bytes':<8} {'Tok/Byte':<10}")
        print("-" * 80)

        for r in results:
            text_display = r['text'][:37] + "..." if len(r['text']) > 40 else r['text']
            ratio = r['num_tokens'] / r['num_bytes'] if r['num_bytes'] > 0 else 0
            print(f"{text_display:<40} {r['num_tokens']:<8} {r['num_chars']:<8} {r['num_bytes']:<8} {ratio:<10.3f}")

        print(f"\n{self.BOLD}Learning Insight:{self.RESET}")
        print("  Notice how different texts have different compression ratios.")
        print("  Common words/patterns usually tokenize more efficiently!")

        print(f"\n{self.BOLD}{'='*70}{self.RESET}\n")

    def interactive_mode(self) -> None:
        """Run interactive mode where user can input text repeatedly."""
        print(f"\n{self.BOLD}{'='*70}{self.RESET}")
        print(f"{self.BOLD}TOKENIZER PLAYGROUND - Interactive Mode{self.RESET}")
        print(f"{self.BOLD}{'='*70}{self.RESET}\n")

        print("Commands:")
        print("  • Type text to tokenize it")
        print("  • 'special' - Show all special tokens")
        print("  • 'vocab' - Show vocabulary information")
        print("  • 'compare' - Compare multiple texts (enter one per line, empty line to finish)")
        print("  • 'help' - Show this help message")
        print("  • 'quit' or 'exit' - Exit the playground\n")

        while True:
            try:
                user_input = input(f"{self.BOLD}>{self.RESET} ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Happy tokenizing!\n")
                    break

                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  • Type text to tokenize it")
                    print("  • 'special' - Show all special tokens")
                    print("  • 'vocab' - Show vocabulary information")
                    print("  • 'compare' - Compare multiple texts")
                    print("  • 'quit' or 'exit' - Exit the playground\n")
                    continue

                if user_input.lower() == 'special':
                    self.analyze_special_tokens()
                    continue

                if user_input.lower() == 'vocab':
                    self.show_vocab_info()
                    continue

                if user_input.lower() == 'compare':
                    print("\nEnter texts to compare (one per line, empty line to finish):")
                    texts = []
                    while True:
                        text = input(f"  {len(texts)+1}. ")
                        if not text:
                            break
                        texts.append(text)
                    if texts:
                        self.compare_texts(texts)
                    else:
                        print("No texts entered.\n")
                    continue

                # Default: tokenize the input
                self.tokenize_and_visualize(user_input)

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Happy tokenizing!\n")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Interactive Tokenizer Playground - Visualize and understand tokenization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize a single text
  python tools/tokenizer_playground.py "Hello world!"

  # Interactive mode
  python tools/tokenizer_playground.py --interactive
  python tools/tokenizer_playground.py -i

  # Show special tokens
  python tools/tokenizer_playground.py --special

  # Show vocabulary info
  python tools/tokenizer_playground.py --vocab

  # Compare multiple texts
  python tools/tokenizer_playground.py --compare "Hello" "Hi" "Hey there"
        """
    )

    parser.add_argument('text', nargs='?',
                       help='Text to tokenize (if not provided, enters interactive mode)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Run in interactive mode')
    parser.add_argument('--special', action='store_true',
                       help='Show all special tokens')
    parser.add_argument('--vocab', action='store_true',
                       help='Show vocabulary information')
    parser.add_argument('--compare', nargs='+',
                       help='Compare tokenization of multiple texts')

    args = parser.parse_args()

    # Load tokenizer
    print("Loading tokenizer...", file=sys.stderr)
    try:
        from nanochat.tokenizer import get_tokenizer
        tokenizer = get_tokenizer()
    except Exception as e:
        print(f"\n❌ Error loading tokenizer: {e}", file=sys.stderr)
        print("\nMake sure you have a trained tokenizer in the 'tokenizer/' directory.", file=sys.stderr)
        print("Run 'python scripts/tok_train.py' to train a tokenizer first.\n", file=sys.stderr)
        sys.exit(1)

    print("Tokenizer loaded!\n", file=sys.stderr)

    playground = TokenizerPlayground(tokenizer)

    # Run appropriate mode
    if args.special:
        playground.analyze_special_tokens()
    elif args.vocab:
        playground.show_vocab_info()
    elif args.compare:
        playground.compare_texts(args.compare)
    elif args.interactive or args.text is None:
        playground.interactive_mode()
    else:
        playground.tokenize_and_visualize(args.text)


if __name__ == '__main__':
    main()
