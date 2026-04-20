import os

from nanochat.tokenizer import get_tokenizer, get_token_bytes

def main():
    os.environ["NANOCHAT_TOKENIZER_KIND"] = "sentencepiece"

    tok = get_tokenizer()
    print("Tokenizer type:", type(tok).__name__)
    print("Vocab size:", tok.get_vocab_size())
    print("BOS token id:", tok.get_bos_token_id())

    text = "Hello world. This is a Parameter Golf tokenizer smoke test."
    ids = tok.encode(text)
    decoded = tok.decode(ids)

    print("Input text:", text)
    print("Token ids:", ids[:40])
    print("Decoded text:", decoded)

    tb = get_token_bytes(device="cpu")
    print("token_bytes shape:", tb.shape)
    print("First 20 token byte counts:", tb[:20].tolist())

if __name__ == "__main__":
    main()
