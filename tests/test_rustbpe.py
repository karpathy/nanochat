"""
Comparing the training of:

1. (very slow) Python reference implementation
2. Optimized Python implementation
3. HuggingFace tokenizers training implementation
4. Our own custom RustBPE training implementation

All of these should calculate the same merges and produce
the same vocabulary and tokenizations.

Finally, for inference we will use tiktoken for efficiency.
So we want to make sure we can export our rustbpe tokenizer
into tiktoken and use it for inference with identical results.

Run with:
python -m pytest tests/test_rustbpe.py -v -s
-v is verbose, -s is show prints
"""

import regex as re
from collections import Counter, defaultdict
import time
import rustbpe
import tiktoken
import pytest

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# Reference tokenizer, pretty much copy pasted and pruned a bit from minbpe

def get_stats(ids, counts=None):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    counts = {} if counts is None else counts
    for pair in zip(ids, ids[1:]): # iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    newids = []
    i = 0
    while i < len(ids):
        # if not at the very last position AND the pair matches, replace it
        if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids

class RegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.merges = {} # (int, int) -> int
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # keep track of whether at any point during training the merge is ambiguous (counts of pairs are not unique)
        ambiguous = False

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in text_chunks]

        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            stats = {}
            for chunk_ids in ids:
                # passing in stats will update it in place, adding up counts
                get_stats(chunk_ids, stats)
            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # check if the merge is ambiguous - i.e. the max value is not unique
            pair_count = stats[pair]
            pairs_with_max_count = [pair for pair, count in stats.items() if count == pair_count]
            if len(pairs_with_max_count) > 1:
                # print the top 10 pairs with their counts
                # print(f"{i} Merge is ambiguous! {pair} has {pair_count} occurrences")
                # for print_pair, print_count in sorted(stats.items(), key=lambda x: x[1], reverse=True)[:10]:
                #     print(f"{print_pair}: {print_count}")
                ambiguous = True
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()
        return ambiguous

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# Faster Python tokenizer, optimized version of the reference tokenizer

def fast_merge_inplace(ids, pair, idx):
    """
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx in place
    Example: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    # Find all positions where the pair occurs
    i = 0
    while i < len(ids) - 1:
        if ids[i] == pair[0] and ids[i+1] == pair[1]:
            ids[i] = idx
            ids.pop(i+1)
        else:
            i += 1
    return ids


class FastRegexTokenizer:

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        self.merges = {}
        self.vocab = self._build_vocab()

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def train(self, text, vocab_size, verbose=False):
        """
        A number of optimizations are introduced:
        - delete function call overhead by inlining functions
        - modifying list of ids in place with .pop() instead of creating a new list
        - collapse identical chunks to just the unique ones
        - update counts more cleverly - only around the affected chunks
        """
        assert vocab_size >= 256
        num_merges = vocab_size - 256

        # split the text up into text chunks
        text_chunks = re.findall(self.compiled_pattern, text)

        # many, many chunks are identical, so we can "collapse" them to just the unique ones
        counts = Counter(text_chunks)
        unique_chunks = [ch for ch, count in counts.items()]
        chunk_counts = [count for ch, count in counts.items()]

        # input text preprocessing
        ids = [list(ch.encode("utf-8")) for ch in unique_chunks]
        # iteratively merge the most common pairs to create new tokens
        merges = {} # (int, int) -> int
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes

        # Initial count: build stats and position tracking
        stats = defaultdict(int)
        positions = defaultdict(set)  # pair -> set of chunk indices that contain this pair

        for chunk_idx, (chunk_ids, count) in enumerate(zip(ids, chunk_counts)):
            for pair in zip(chunk_ids, chunk_ids[1:]):
                stats[pair] += count
                positions[pair].add(chunk_idx)

        for i in range(num_merges):
            if not stats:
                break

            # find the pair with the highest count
            pair = max(stats, key=stats.get)
            # mint a new token: assign it the next available id
            idx = 256 + i

            # Get chunks that contain this pair
            affected_chunks = positions[pair]

            # Track count changes for incremental update
            count_changes = defaultdict(int)

            # Replace all occurrences of pair in affected chunks only
            for chunk_idx in affected_chunks:
                chunk_ids = ids[chunk_idx]
                chunk_count = chunk_counts[chunk_idx]
                ix = 0
                while ix < len(chunk_ids) - 1:
                    if chunk_ids[ix] == pair[0] and chunk_ids[ix+1] == pair[1]:
                        # Track what pairs are being removed/added
                        # Remove: (prev, A), (A, B), (B, next)
                        if ix > 0:
                            old_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[old_left] -= chunk_count

                        # The merged pair disappears
                        count_changes[pair] -= chunk_count

                        if ix + 2 < len(chunk_ids):
                            old_right = (chunk_ids[ix+1], chunk_ids[ix+2])
                            count_changes[old_right] -= chunk_count

                        # Apply the merge
                        chunk_ids[ix] = idx
                        chunk_ids.pop(ix+1)

                        # Add: (prev, C), (C, next)
                        if ix > 0:
                            new_left = (chunk_ids[ix-1], chunk_ids[ix])
                            count_changes[new_left] += chunk_count

                        if ix + 1 < len(chunk_ids):
                            new_right = (chunk_ids[ix], chunk_ids[ix+1])
                            count_changes[new_right] += chunk_count
                    else:
                        ix += 1

            # Apply incremental changes to stats and positions
            for changed_pair, delta in count_changes.items():
                if changed_pair == pair:
                    # The merged pair should disappear completely
                    continue

                stats[changed_pair] += delta

                # Update positions for changed pairs - only check affected chunks
                for chunk_idx in affected_chunks:
                    chunk_ids = ids[chunk_idx]
                    contains_pair = any((chunk_ids[j], chunk_ids[j+1]) == changed_pair
                                      for j in range(len(chunk_ids) - 1))
                    if contains_pair:
                        positions[changed_pair].add(chunk_idx)
                    else:
                        positions[changed_pair].discard(chunk_idx)

            # Remove the merged pair completely
            del stats[pair]
            del positions[pair]

            # save the merge
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

        # save class variables
        self.merges = merges # used in encode()
        self.vocab = vocab   # used in decode()

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = fast_merge_inplace(ids, pair, idx)
        return ids

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

# -----------------------------------------------------------------------------
# HuggingFace tokenizer
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import pre_tokenizers, decoders, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # train from an iterator of text
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True, # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None
        # Pre-tokenizer: GPT-4 style
        gpt4_split_regex = Regex(GPT4_SPLIT_PATTERN) # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()
        # Post-processor: None
        tokenizer.post_processor = None
        # Trainer: BPE
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0, # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=[], # no special tokens
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def encode_ordinary(self, text):
        ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        return ids

# -----------------------------------------------------------------------------
# Test all of the above

@pytest.fixture(scope="module")
def enwik8_path():
    """Fixture to download and cache enwik8 dataset."""
    import os
    import zipfile
    from nanochat.common import get_base_dir
    base_dir = get_base_dir()
    # download and unzip enwik8 to .cache directory
    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_dir, "enwik8")
    enwik8_local_path_zip = os.path.join(base_dir, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests
        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_dir)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")
    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """Fixture providing 100KB of enwik8 for quick tests."""
    with open(enwik8_path, "r") as f:
        return f.read(100_000)

@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """Fixture providing 10MB of enwik8 for performance tests."""
    with open(enwik8_path, "r") as f:
        return f.read(10**7)

def time_function(func, *args, **kwargs):
    """Time a function call and return the result and elapsed time"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed

def test_correctness(enwik8_small):
    """Test that all tokenizer implementations produce the same results."""
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20  # 20 merges

    # Train slow reference
    print("\nTraining slow reference...")
    slow_reference_tokenizer = RegexTokenizer()
    ambiguous_flag, slow_reference_train_time = time_function(slow_reference_tokenizer.train, text, vocab_size)
    slow_reference_ids, slow_reference_encode_time = time_function(slow_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Slow reference train time: {slow_reference_train_time:.4f}s")
    print(f"Slow reference encode time: {slow_reference_encode_time:.4f}s")
    print(slow_reference_ids[:20])

    if ambiguous_flag:
        print("‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size")
        print("The implementation could be correct but we might see different results below")
    else:
        print("✅ Merge order is NOT ambiguous")

    # Train fast reference
    print("\nTraining fast reference...")
    fast_reference_tokenizer = FastRegexTokenizer()
    _, fast_reference_train_time = time_function(fast_reference_tokenizer.train, text, vocab_size)
    fast_reference_ids, fast_reference_encode_time = time_function(fast_reference_tokenizer.encode_ordinary, encode_text)
    print(f"Fast reference train time: {fast_reference_train_time:.4f}s")
    print(f"Fast reference encode time: {fast_reference_encode_time:.4f}s")
    print(fast_reference_ids[:20])

    # Assert fast equals slow
    assert fast_reference_ids == slow_reference_ids, "Fast reference should match slow reference"
    print("✅ Fast == Slow")

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    hf_ids, hf_encode_time = time_function(hf_tokenizer.encode_ordinary, encode_text)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    print(f"HuggingFace encode time: {hf_encode_time:.4f}s")
    print(hf_ids[:20])

    # HuggingFace has a different byte order, so we need custom matching
    def custom_match(ids1, ids2):
        perm = {}
        for x, y in zip(ids1, ids2):
            if x < 256:
                if x in perm:
                    if perm[x] != y:
                        return False
                perm[x] = y
            if x >= 256 and x != y:
                return False
        return True

    assert custom_match(hf_ids, fast_reference_ids), "HuggingFace should match fast reference"
    print("✅ HuggingFace == Fast")

    # Finally use our own Rust implementation
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    rustbpe_ids, rustbpe_encode_time = time_function(rustbpe_tokenizer.encode, encode_text)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    print(f"RustBPE encode time: {rustbpe_encode_time:.4f}s")
    print(rustbpe_ids[:20])

    assert rustbpe_ids == fast_reference_ids, "RustBPE should match fast reference"
    print("✅ RustBPE == Fast")

    # Now export rustbpe to tiktoken for more efficient inference
    print("\nTesting tiktoken export...")
    pattern = rustbpe_tokenizer.get_pattern()
    mergeable_ranks_list = rustbpe_tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    enc = tiktoken.Encoding(
        name="rustbpe",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens={},
    )
    tiktoken_ids, tiktoken_encode_time = time_function(enc.encode, encode_text)
    print(f"Tiktoken encode time: {tiktoken_encode_time:.4f}s")
    print(tiktoken_ids[:20])

    assert tiktoken_ids == rustbpe_ids, "Tiktoken should match RustBPE"
    print("✅ Tiktoken == RustBPE")


@pytest.mark.slow
def test_training_performance(enwik8_large):
    """Use a bigger dataset and compare the training speed of the optimized tokenizers (Python, Rust, HuggingFace)."""
    text = enwik8_large
    vocab_size = 2048
    print(f"\nText length: {len(text)}")

    # Commenting out because it's just way too slow to matter
    # Train optimized python version
    # print("Training optimized python version...")
    # optimized_python_tokenizer = FastRegexTokenizer()
    # _, optimized_python_train_time = time_function(optimized_python_tokenizer.train, text, vocab_size)
    # print(f"Optimized python train time: {optimized_python_train_time:.4f}s")

    # Train rustbpe
    print("\nTraining rustbpe...")
    rustbpe_tokenizer = rustbpe.Tokenizer()
    _, rustbpe_train_time = time_function(rustbpe_tokenizer.train_from_iterator, [text], vocab_size)
    print(f"RustBPE train time: {rustbpe_train_time:.4f}s")
    assert rustbpe_train_time > 0, "Training should take some time"

    # Train HuggingFace
    print("\nTraining HuggingFace...")
    hf_tokenizer, hf_train_time = time_function(HuggingFaceTokenizer.train_from_iterator, [text], vocab_size)
    print(f"HuggingFace train time: {hf_train_time:.4f}s")
    assert hf_train_time > 0, "Training should take some time"

    # Print comparison
    print(f"\n📊 Performance comparison:")
    print(f"   RustBPE: {rustbpe_train_time:.4f}s")
    print(f"   HuggingFace: {hf_train_time:.4f}s")
    print(f"   Speedup: {hf_train_time/rustbpe_train_time:.2f}x")

def test_interface(enwik8_small):
    """Test the RustBPETokenizer interface for training, encoding, decoding, and serialization."""
    import tempfile
    from nanochat.tokenizer import RustBPETokenizer

    # Simple train test
    vocab_size = 300
    tok = RustBPETokenizer.train_from_iterator([enwik8_small], vocab_size)
    assert tok.get_vocab_size() == vocab_size, f"Expected vocab size {vocab_size}, got {tok.get_vocab_size()}"
    print(f"✅ Trained tokenizer with vocab size {vocab_size}")

    # Encode/decode text
    encode_text = "Hello world! How are you? 🙃"
    ids = tok.encode(encode_text)
    print(f"\nInput text: {encode_text}")
    print(f"IDs: {ids}")
    decoded = tok.decode(ids)
    print(f"Decoded: {decoded}")
    assert decoded == encode_text, f"Decoded text doesn't match: {decoded} != {encode_text}"
    print("✅ Encode/decode test passed")

    # Encode batch test
    ids_new = tok.encode([encode_text, encode_text])
    assert all(x == ids for x in ids_new), "Batch encoding should produce identical results"
    print("✅ Encode batch OK")

    # append/prepend functionality
    ids_special = tok.encode(encode_text, prepend="<|bos|>", append="<|bos|>")
    bos_token_id = tok.encode_special("<|bos|>")
    assert ids_special == [bos_token_id] + ids + [bos_token_id], "Special tokens not correctly added"
    print("✅ append/prepend OK")

    # Save/load test through a temporary directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        tok.save(tmp_dir)
        tok_reloaded = RustBPETokenizer.from_directory(tmp_dir)
        ids_reloaded = tok_reloaded.encode(encode_text)
        assert ids_reloaded == ids, "Reloaded tokenizer should produce same results"
        print("✅ Save/load through temporary directory OK")
