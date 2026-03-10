from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()
vocab_size = tokenizer.get_vocab_size()

print(f"Total Vocabulary Size: {vocab_size:,}")
print("-" * 30)

# 1. Print special tokens
print("Special Tokens:")
for name in tokenizer.get_special_tokens():
    token_id = tokenizer.encode_special(name)
    print(f"  {token_id:5d}: {name}")

print("\nSample of Regular Tokens (the first 20):")
for i in range(256, 276):  # Tokens 0-255 are standard bytes, let's look at merges
    token_str = tokenizer.decode([i])
    print(f"  {i:5d}: {repr(token_str)}")

print("\nSample of High-ID Tokens (more complex merges):")
for i in range(vocab_size - 20, vocab_size):
    token_str = tokenizer.decode([i])
    print(f"  {i:5d}: {repr(token_str)}")
