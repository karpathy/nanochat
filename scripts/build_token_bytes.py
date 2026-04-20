import argparse
import os
import torch

from nanochat.tokenizer import SentencePieceTokenizer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer-model", required=True, help="Path to SentencePiece .model file")
    parser.add_argument("--out-dir", required=True, help="Directory to write token_bytes.pt")
    args = parser.parse_args()
    tok = SentencePieceTokenizer.from_model_file(args.tokenizer_model)
    vocab_size = tok.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int32)

    for i in range(vocab_size):
        piece = tok.id_to_token(i)
        if piece.startswith("<") and piece.endswith(">"):
            token_bytes[i] = 0
            continue
        try:
            decoded = tok.decode([i])
            token_bytes[i] = len(decoded.encode("utf-8"))
        except Exception:
            token_bytes[i] = 0

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "token_bytes.pt")
    with open(out_path, "wb") as f:
        torch.save(token_bytes, f)

    print(f"Saved token_bytes to {out_path}")
    print(f"Vocab size: {vocab_size}")
    nonzero = token_bytes[token_bytes > 0]
    if len(nonzero) > 0:
        print(f"Min nonzero bytes: {int(nonzero.min())}")
        print(f"Max nonzero bytes: {int(nonzero.max())}")
        print(f"Mean nonzero bytes: {float(nonzero.float().mean()):.4f}")

if __name__ == "__main__":
    main()
