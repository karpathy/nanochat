# Tokenization: Byte Pair Encoding (BPE)

## Why Tokenization?

Neural networks work with numbers, not text. **Tokenization** converts text into numerical sequences that models can process.

**Why not just use ASCII codes?**
- English uses ~100 common characters
- But common words like "the", "and", "ing" appear constantly
- Better to have single tokens for frequent sequences
- Reduces sequence length and captures semantic meaning

## Tokenization Approaches

1. **Character-level**: Each character is a token
   - Pros: Small vocabulary, handles any text
   - Cons: Very long sequences, doesn't capture word meaning

2. **Word-level**: Each word is a token
   - Pros: Captures semantic meaning
   - Cons: Huge vocabulary, can't handle unknown words

3. **Subword-level** (BPE, WordPiece): Balance between characters and words
   - Pros: Moderate vocabulary, handles rare words, captures common patterns
   - Cons: Slightly complex to implement
   - **This is what nanochat uses!**

## Byte Pair Encoding (BPE) Algorithm

BPE builds a vocabulary by iteratively merging the most frequent pairs of tokens.

### The Training Algorithm

**Input:** Corpus of text, desired vocabulary size $V$

**Output:** Merge rules and vocabulary

**Steps:**

1. **Initialize vocabulary** with all 256 bytes (0-255)
2. **Split text** into chunks using a regex pattern
3. **Convert chunks** to sequences of byte tokens
4. **Repeat** $V - 256$ times:
   - Find the most frequent **pair** of adjacent tokens
   - **Merge** this pair into a new token
   - **Replace** all occurrences of the pair with the new token
5. **Save** the merge rules

### Example by Hand

Let's tokenize "aaabdaaabac" with vocab_size = 259 (256 bytes + 3 merges).

**Initial state:** Convert to bytes
```
text = "aaabdaaabac"
tokens = [97, 97, 97, 98, 100, 97, 97, 97, 98, 97, 99]  # ASCII codes
```

**Iteration 1:** Find most frequent pair
```
Pairs: (97,97) appears 4 times ← most frequent
       (97,98) appears 2 times
       (98,100) appears 1 time
       ...

Merge (97,97) → 256
New tokens: [256, 97, 98, 100, 256, 97, 98, 97, 99]
```

**Iteration 2:**
```
Pairs: (256,97) appears 2 times ← most frequent
       (97,98) appears 2 times (tie-break by lexicographic order)
       ...

Merge (256,97) → 257
New tokens: [257, 98, 100, 257, 98, 97, 99]
```

**Iteration 3:**
```
Pairs: (257,98) appears 2 times ← most frequent
       ...

Merge (257,98) → 258
Final tokens: [258, 100, 258, 97, 99]
```

We've compressed 11 tokens → 5 tokens!

## Implementation in nanochat

nanochat provides **two tokenizer implementations**:

1. **HuggingFaceTokenizer**: Python-based, easy to use but slower
2. **RustBPETokenizer**: High-performance Rust implementation (preferred)

Both implement the same GPT-4 style BPE algorithm.

### File: `nanochat/tokenizer.py`

Let's examine the key components:

#### Special Tokens

```python
SPECIAL_TOKENS = [
    "<|bos|>",           # Beginning of sequence (document delimiter)
    "<|user_start|>",    # User message start
    "<|user_end|>",      # User message end
    "<|assistant_start|>",  # Assistant message start
    "<|assistant_end|>",    # Assistant message end
    "<|python_start|>",  # Python tool call start
    "<|python_end|>",    # Python tool call end
    "<|output_start|>",  # Python output start
    "<|output_end|>",    # Python output end
]
```

These special tokens are added to the vocabulary for chat formatting.

#### Text Splitting Pattern

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

This regex pattern splits text before BPE:
- `'(?i:[sdmt]|ll|ve|re)`: Contractions like 's, 't, 'll, 've, 're
- `[^\r\n\p{L}\p{N}]?+\p{L}+`: Optional non-letter + letters (words)
- `\p{N}{1,2}`: Numbers (1-2 digits, not 3 like GPT-4)
- ` ?[^\s\p{L}\p{N}]++[\r\n]*`: Optional space + punctuation
- `\s*[\r\n]|\s+(?!\S)|\s+`: Whitespace handling

**Why this pattern?**
It groups text into chunks that are semantically meaningful, making BPE more effective.

### RustBPETokenizer Class

The main tokenizer interface (`nanochat/tokenizer.py:155`):

```python
class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token):
        self.enc = enc  # tiktoken.Encoding object
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1) Train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)

        # 2) Construct tiktoken encoding for fast inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}

        # Add special tokens
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}

        # Create tiktoken encoding
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")
```

**Design choice:** Train with Rust (fast), infer with tiktoken (also fast, battle-tested).

#### Encoding Text

```python
def encode(self, text, prepend=None, append=None, num_threads=8):
    # Prepare special tokens
    if prepend is not None:
        prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
    if append is not None:
        append_id = append if isinstance(append, int) else self.encode_special(append)

    if isinstance(text, str):
        # Single string
        ids = self.enc.encode_ordinary(text)
        if prepend is not None:
            ids.insert(0, prepend_id)
        if append is not None:
            ids.append(append_id)
    elif isinstance(text, list):
        # Batch of strings (parallel processing)
        ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
        if prepend is not None:
            for ids_row in ids:
                ids_row.insert(0, prepend_id)
        if append is not None:
            for ids_row in ids:
                ids_row.append(append_id)

    return ids
```

**Key features:**
- Supports single strings or batches
- Optional prepend/append (e.g., BOS token)
- Parallel processing for batches

#### Chat Conversation Rendering

For supervised fine-tuning, we need to convert conversations to tokens:

```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    Tokenize a single Chat conversation.
    Returns:
    - ids: list[int] - token ids
    - mask: list[int] - 1 for tokens to train on, 0 otherwise
    """
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # Get special token IDs
    bos = self.get_bos_token_id()
    user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
    assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")

    # Add BOS token (not trained on)
    add_tokens(bos, 0)

    # Process messages
    for i, message in enumerate(messages):
        if message["role"] == "user":
            # User messages: not trained on
            value_ids = self.encode(message["content"])
            add_tokens(user_start, 0)
            add_tokens(value_ids, 0)
            add_tokens(user_end, 0)
        elif message["role"] == "assistant":
            # Assistant messages: TRAINED ON (mask=1)
            add_tokens(assistant_start, 0)
            value_ids = self.encode(message["content"])
            add_tokens(value_ids, 1)  # ← This is what we train on!
            add_tokens(assistant_end, 1)

    # Truncate if too long
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    return ids, mask
```

**The mask is crucial!** We only compute loss on assistant responses, not user prompts.

## Rust Implementation: `rustbpe/src/lib.rs`

The Rust implementation is highly optimized for speed. Let's examine the core components.

### Data Structures

```rust
type Pair = (u32, u32);  // Pair of token IDs

#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting
    pub pattern: String,
    /// Compiled regex for efficiency
    compiled_pattern: Regex,
}
```

#### Word Representation

```rust
struct Word {
    ids: Vec<u32>,  // Sequence of token IDs
}

impl Word {
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }
}
```

The `pairs()` method generates all adjacent pairs efficiently using sliding windows.

### The Core Training Algorithm

Located at `rustbpe/src/lib.rs:164`:

```rust
fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
    let num_merges = vocab_size - 256;  // 256 base bytes

    // 1. Initial pair counting (parallel!)
    let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

    // 2. Build max-heap of merge candidates
    let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
    for (pair, pos) in where_to_update.drain() {
        let c = *pair_counts.get(&pair).unwrap_or(&0);
        if c > 0 {
            heap.push(MergeJob {
                pair,
                count: c as u64,
                pos,  // Set of word indices where this pair occurs
            });
        }
    }

    // 3. Merge loop
    for merges_done in 0..num_merges {
        // Get highest-count pair
        let Some(mut top) = heap.pop() else { break; };

        // Lazy refresh: check if count is still accurate
        let current = *pair_counts.get(&top.pair).unwrap_or(&0);
        if top.count != current as u64 {
            top.count = current as u64;
            if top.count > 0 {
                heap.push(top);
            }
            continue;
        }

        // Record merge
        let new_id = 256 + merges_done;
        self.merges.insert(top.pair, new_id);

        // Apply merge to all words containing this pair
        let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
        for &word_idx in &top.pos {
            let changes = words[word_idx].merge_pair(top.pair, new_id);

            // Update global pair counts
            for (pair, delta) in changes {
                let delta_total = delta * counts[word_idx];
                if delta_total != 0 {
                    *pair_counts.entry(pair).or_default() += delta_total;
                    if delta > 0 {
                        local_pos_updates.entry(pair).or_default().insert(word_idx);
                    }
                }
            }
        }

        // Re-add updated pairs to heap
        for (pair, pos) in local_pos_updates {
            let cnt = *pair_counts.get(&pair).unwrap_or(&0);
            if cnt > 0 {
                heap.push(MergeJob { pair, count: cnt as u64, pos });
            }
        }
    }
}
```

### Key Optimizations

1. **Parallel Pair Counting:**
```rust
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()  // Parallel iterator!
        .enumerate()
        .map(|(i, w)| {
            // Count pairs in this word
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(/* merge results */)
}
```

Uses **Rayon** for parallel processing across CPU cores.

2. **Efficient Merging:**
```rust
fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
    let (a, b) = pair;
    let mut out: Vec<u32> = Vec::with_capacity(self.ids.len());
    let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

    let mut i = 0;
    while i < self.ids.len() {
        if i + 1 < self.ids.len() && self.ids[i] == a && self.ids[i + 1] == b {
            // Found the pair to merge
            let left = out.last().copied();
            let right = if i + 2 < self.ids.len() { Some(self.ids[i + 2]) } else { None };

            // Track changes in pair counts
            if let Some(x) = left {
                deltas.push(((x, a), -1));      // Remove old pair
                deltas.push(((x, new_id), 1));  // Add new pair
            }
            deltas.push(((a, b), -1));          // Remove merged pair
            if let Some(y) = right {
                deltas.push(((b, y), -1));      // Remove old pair
                deltas.push(((new_id, y), 1));  // Add new pair
            }

            out.push(new_id);
            i += 2;  // Skip both tokens
        } else {
            out.push(self.ids[i]);
            i += 1;
        }
    }

    self.ids = out;
    deltas
}
```

Returns **delta updates** to pair counts, avoiding full recount.

3. **Lazy Heap Updates:**

Instead of updating heap immediately when counts change:
- Pop top element
- Check if count is still valid
- If not, update and re-insert

This avoids expensive heap operations.

4. **Optimized Data Structures:**
- `AHashMap`: Fast hashmap from `ahash` crate
- `OctonaryHeap`: 8-ary heap (better cache locality than binary heap)
- `CompactString`: String optimized for short strings

### Encoding with Trained Tokenizer

```rust
pub fn encode(&self, text: &str) -> Vec<u32> {
    let mut all_ids = Vec::new();

    // Split text using regex pattern
    for m in self.compiled_pattern.find_iter(text) {
        let chunk = m.expect("regex match failed").as_str();

        // Convert to byte tokens
        let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

        // Apply merges iteratively
        while ids.len() >= 2 {
            // Find best pair to merge (lowest token ID = highest priority)
            let mut best_pair: Option<(usize, Pair, u32)> = None;

            for i in 0..ids.len() - 1 {
                let pair: Pair = (ids[i], ids[i + 1]);
                if let Some(&new_id) = self.merges.get(&pair) {
                    if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                        best_pair = Some((i, pair, new_id));
                    }
                }
            }

            // Apply merge if found
            if let Some((idx, _pair, new_id)) = best_pair {
                ids[idx] = new_id;
                ids.remove(idx + 1);
            } else {
                break;  // No more merges
            }
        }

        all_ids.extend(ids);
    }

    all_ids
}
```

**Greedy algorithm:** Always merge the pair with the **lowest token ID** (= earliest in training).

## Training the Tokenizer: `scripts/tok_train.py`

```python
def main():
    # 1. Load data iterator
    shard_size = 250_000_000  # 250M characters per shard
    num_shards = 16           # ~4B characters total
    data_iterator = fineweb_shards_iterator(num_shards, shard_size)

    # 2. Train tokenizer
    tokenizer = RustBPETokenizer.train_from_iterator(
        data_iterator,
        vocab_size=32256,  # Common size for small models
    )

    # 3. Save tokenizer
    tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    tokenizer.save(tokenizer_dir)

    # 4. Save token_bytes tensor for BPB evaluation
    token_bytes = compute_token_bytes(tokenizer)
    torch.save(token_bytes, os.path.join(tokenizer_dir, "token_bytes.pt"))
```

This streams data from FineWeb dataset and trains the tokenizer.

## Usage Example

```python
from nanochat.tokenizer import RustBPETokenizer

# Load trained tokenizer
tokenizer = RustBPETokenizer.from_directory("out/tokenizer")

# Encode text
text = "Hello, world! How are you?"
ids = tokenizer.encode(text, prepend="<|bos|>")
print(ids)  # [32256, 9906, 11, 995, 0, 1374, 389, 345, 30]

# Decode back
decoded = tokenizer.decode(ids)
print(decoded)  # "<|bos|>Hello, world! How are you?"

# Batch encoding (parallel)
texts = ["First sentence.", "Second sentence.", "Third sentence."]
batch_ids = tokenizer.encode(texts, prepend="<|bos|>", num_threads=4)
```

## Why BPE Works

1. **Frequent patterns get single tokens**: "ing", "the", "er"
2. **Rare words split into subwords**: "unhappiness" → ["un", "happiness"]
3. **Can handle any text**: Falls back to bytes for unknown sequences
4. **Compresses sequences**: Fewer tokens = faster training/inference

## Performance Comparison

| Implementation | Training Speed | Inference Speed |
|----------------|----------------|-----------------|
| Python baseline | 1× | 1× |
| HuggingFace | ~2× | ~5× |
| **Rust + tiktoken** | **~20×** | **~50×** |

The Rust implementation in nanochat is **dramatically faster** due to:
- Parallel processing
- Efficient data structures
- No Python overhead
- Compiled to native code

## Next Steps

Now that we understand tokenization, we'll explore the **Transformer Architecture** - the neural network that processes these token sequences.
