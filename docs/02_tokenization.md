# Tokenization: From Text to Numbers

Neural networks can only work with numbers, not text. **Tokenization** is the process of converting text into numbers that a neural network can process. This document explains how tokenization works in nanochat from the ground up.

## Table of Contents
1. [Why We Need Tokenization](#why-we-need-tokenization)
2. [What is Byte Pair Encoding (BPE)?](#what-is-byte-pair-encoding-bpe)
3. [How BPE Training Works](#how-bpe-training-works)
4. [Special Tokens for Conversations](#special-tokens-for-conversations)
5. [Code Walkthrough](#code-walkthrough)
6. [Practical Examples](#practical-examples)

---

## Why We Need Tokenization

### The Naive Approach: Characters

You might think: "Why not just use characters? A=0, B=1, etc.?"

Problems with character-level encoding:
1. **Too granular**: Sequences become very long (each letter is a separate step)
2. **Doesn't capture meaning**: "cat" as three separate characters loses the word's meaning
3. **Inefficient**: The model has to learn that c-a-t often appear together

```python
# Character-level encoding
"Hello" → [H, e, l, l, o] → [7, 4, 11, 11, 14]
# 5 separate tokens for one word!
```

### Another Naive Approach: Words

What about using whole words? "cat"=1, "dog"=2, etc.?

Problems with word-level encoding:
1. **Huge vocabulary**: English has 170,000+ words, plus proper nouns, technical terms...
2. **Out-of-vocabulary**: New words (like "ChatGPT") won't be in the vocabulary
3. **No subword understanding**: "walk", "walking", "walked" are completely different tokens
4. **Multiple languages**: Hard to handle multiple languages in one vocabulary

```python
# Word-level encoding
"walking" → [35821]  # Known word
"ChatGPT" → [UNK]    # Unknown word - problem!
```

### The Solution: Subword Tokenization (BPE)

**Byte Pair Encoding (BPE)** finds a middle ground:
- Breaks words into frequently-occurring pieces (subwords)
- Common words might be one token: "the" → [464]
- Rare words are broken into parts: "ChatGPT" → ["Chat", "G", "PT"] → [13012, 38, 2898]
- Can represent ANY text (falls back to individual bytes if needed)

**Benefits:**
- ✅ Moderate vocabulary size (nanochat uses 65,536 tokens = 2^16)
- ✅ Can represent any text
- ✅ Captures common patterns
- ✅ Learns subword structure

---

## What is Byte Pair Encoding (BPE)?

### The Core Idea

BPE **learns** which character/byte sequences appear frequently and merges them into single tokens.

**Algorithm (simplified):**
1. Start with all individual bytes (256 base tokens: 0-255)
2. Find the most frequent pair of adjacent tokens
3. Merge that pair into a new token
4. Repeat until you have the desired vocabulary size

### Example: Training BPE on a Tiny Corpus

```
Corpus (repeated many times):
"hello hello hello world world"
```

**Initial state:** Each byte is a token
```
h e l l o   h e l l o   h e l l o   w o r l d   w o r l d
```

**Step 1:** Most frequent pair is `l l` (appears 3 times)
- Create new token `ll` (token ID 256)
- Merge all occurrences

```
h e ll o   h e ll o   h e ll o   w o r l d   w o r l d
```

**Step 2:** Most frequent pair is now `h e` (appears 3 times)
- Create new token `he` (token ID 257)

```
he ll o   he ll o   he ll o   w o r l d   w o r l d
```

**Step 3:** Most frequent pair is `he ll` (appears 3 times)
- Create new token `hell` (token ID 258)

```
hell o   hell o   hell o   w o r l d   w o r l d
```

**Step 4:** Continue until reaching desired vocabulary size...

**Result:** Common patterns like "hello" become short sequences:
```
"hello" → [258, o] → [258, 111] (just 2 tokens instead of 5!)
```

### Key Insight

BPE **automatically discovers** useful subwords based on frequency:
- Common words become single tokens
- Common subwords (like "ing", "ed", "un") get their own tokens
- Rare words are broken into known pieces
- You can always fall back to individual bytes for completely unknown text

---

## How BPE Training Works

### The Training Process

In nanochat, tokenizer training happens in `scripts/tok_train.py` using the Rust implementation in `rustbpe/`.

**High-level flow:**

```
1. COLLECT TEXT SAMPLES
   ↓
   Read ~2 billion characters from the FineWeb dataset

2. PRE-TOKENIZE
   ↓
   Split text using a regex pattern (GPT-4 style):
   "Hello, world!" → ["Hello", ",", " world", "!"]

3. COUNT CHUNKS
   ↓
   Count how many times each unique chunk appears
   {"Hello": 1523, ",": 8234, " world": 892, ...}

4. CONVERT TO BYTES
   ↓
   Each chunk becomes a sequence of byte IDs (0-255)
   "Hello" → [72, 101, 108, 108, 111]

5. FIND MOST FREQUENT PAIRS
   ↓
   Across all chunks, find the most common pair of adjacent tokens
   Maybe (108, 108) appears 50,000 times total

6. MERGE THE PAIR
   ↓
   Create a new token (ID 256) for this pair
   Update all chunks: [72, 101, 108, 108, 111] → [72, 101, 256, 111]

7. REPEAT
   ↓
   Continue merging until vocab_size is reached
   Each merge creates token IDs 256, 257, 258, ..., 65535

8. SAVE VOCABULARY
   ↓
   Save the merge rules as "mergeable_ranks"
   This is your trained tokenizer!
```

### The Regex Pre-tokenization Pattern

Before BPE, text is split using a regex pattern:

```python
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

**What this does:**
- Keeps common contractions together: "don't" stays as one chunk
- Separates punctuation from words: "Hello!" → ["Hello", "!"]
- Keeps words as units: "world" stays together
- Groups numbers: "12" stays together (max 2 digits in nanochat's version)

**Why pre-tokenize?**
- Prevents unwanted merges across word boundaries
- Ensures linguistic patterns are respected
- Example: We don't want "Hello, " to merge the "o" and "," together

### The Core Training Algorithm (Incremental BPE)

The training algorithm in `rustbpe/src/lib.rs` is quite clever:

```rust
fn train_core_incremental(&mut self, words, counts, vocab_size) {
    // 1. Count all pairs across all words (parallel!)
    let (pair_counts, where_to_update) = count_pairs_parallel(&words, &counts);

    // 2. Build a max-heap of pairs sorted by frequency
    let mut heap = build_heap(pair_counts, where_to_update);

    // 3. Merge loop: repeat vocab_size - 256 times
    for merge_number in 0..(vocab_size - 256) {
        // Get the most frequent pair
        let top = heap.pop();

        // Create a new token for this pair
        let new_token_id = 256 + merge_number;
        merges.insert(top.pair, new_token_id);

        // Update all words that contain this pair
        for word in words_containing_pair {
            word.merge_pair(top.pair, new_token_id);
            // This affects pair counts! Update them
        }

        // Add affected pairs back to heap with new counts
        heap.push_updated_pairs();
    }
}
```

**Key optimizations:**
1. **Parallel counting**: Uses Rayon to count pairs across CPU cores
2. **Heap for efficiency**: Max-heap gives us the most frequent pair in O(log n)
3. **Incremental updates**: Only update counts for affected pairs, not all pairs
4. **Lazy refresh**: Heap entries may be stale; refresh before using

---

## Special Tokens for Conversations

### What are Special Tokens?

**Special tokens** are tokens that have specific meanings for structuring conversations. They're not learned during BPE training - they're added to the vocabulary afterward.

### Nanochat's Special Tokens

Defined in `nanochat/tokenizer.py`:

```python
SPECIAL_TOKENS = [
    "<|bos|>",            # Beginning of Sequence - starts every document
    "<|user_start|>",     # User's message starts
    "<|user_end|>",       # User's message ends
    "<|assistant_start|>", # Assistant's message starts
    "<|assistant_end|>",   # Assistant's message ends
    "<|python_start|>",    # Assistant calls Python tool
    "<|python_end|>",      # Python tool call ends
    "<|output_start|>",    # Python output starts
    "<|output_end|>",      # Python output ends
]
```

### Why Special Tokens?

Special tokens help the model understand structure:

**Without special tokens:**
```
User: What is 2+2?
Assistant: The answer is 4.
```
How does the model know where the user's text ends and the assistant's begins?

**With special tokens:**
```
<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>The answer is 4.<|assistant_end|>
```
Now the model clearly sees:
- This is a new sequence (`<|bos|>`)
- User spoke from here to here
- Assistant responded from here to here

### Conversation Rendering Example

When fine-tuning, conversations are converted to token sequences:

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"}
    ]
}

# Rendered as tokens (simplified):
[
    65528,  # <|bos|>
    65529,  # <|user_start|>
    9906,   # "Hello"
    0,      # "!"
    65530,  # <|user_end|>
    65531,  # <|assistant_start|>
    13347,  # "Hi"
    612,    # " there"
    0,      # "!"
    65532,  # <|assistant_end|>
]
```

### Training Masks

During fine-tuning, we only train on the **assistant's** responses, not the user's input:

```python
tokens: [<|bos|>, <|user_start|>, "Hello", <|user_end|>, <|assistant_start|>, "Hi", <|assistant_end|>]
mask:   [   0   ,        0      ,    0   ,      0     ,         0         ,  1  ,        1          ]
         ↑ Don't train on this                              ↑ Train on this!
```

**Why?** We want the model to learn to *generate* good assistant responses, not memorize user inputs.

---

## Code Walkthrough

### File Structure

```
nanochat/
├── nanochat/tokenizer.py    # Python wrapper classes
└── rustbpe/
    └── src/lib.rs           # Core BPE training in Rust
```

### 1. Training a Tokenizer (`rustbpe/src/lib.rs`)

**Entry point:**
```rust
#[pymethods]
impl Tokenizer {
    pub fn train_from_iterator(
        &mut self,
        iterator: &PyAny,      // Python iterator of text strings
        vocab_size: u32,       // Target vocabulary size (e.g., 65536)
        pattern: Option<String> // Regex pattern for pre-tokenization
    ) -> PyResult<()> {
        // 1. Set up regex pattern
        self.pattern = pattern.unwrap_or(GPT4_PATTERN.to_string());
        self.compiled_pattern = Regex::new(&self.pattern)?;

        // 2. Stream text from iterator and count chunks
        let mut counts: HashMap<String, i32> = HashMap::new();
        for text in iterator {
            for chunk in self.compiled_pattern.find_iter(text) {
                *counts.entry(chunk).or_default() += 1;
            }
        }

        // 3. Convert chunks to byte sequences
        let words: Vec<Word> = counts.keys()
            .map(|chunk| Word::new(chunk.as_bytes().to_vec()))
            .collect();

        // 4. Run incremental BPE training
        self.train_core_incremental(words, counts.values(), vocab_size);

        Ok(())
    }
}
```

**Key data structures:**

```rust
struct Word {
    ids: Vec<u32>,  // Sequence of token IDs (starts as bytes 0-255)
}

struct MergeJob {
    pair: (u32, u32),           // The pair of tokens to merge
    count: u64,                 // How many times this pair appears
    pos: HashSet<usize>,        // Which words contain this pair
}

struct Tokenizer {
    merges: HashMap<(u32, u32), u32>,  // (token_a, token_b) → new_token_id
    pattern: String,                    // Regex pattern
}
```

### 2. Using a Trained Tokenizer (`nanochat/tokenizer.py`)

**Two implementations:**

#### HuggingFaceTokenizer (slower, more compatible)
```python
class HuggingFaceTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer  # HF Tokenizer object

    def encode(self, text):
        """Convert text to token IDs"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        """Convert token IDs back to text"""
        return self.tokenizer.decode(ids)
```

#### RustBPETokenizer (faster, used in nanochat)
```python
class RustBPETokenizer:
    def __init__(self, enc, bos_token):
        self.enc = enc  # tiktoken.Encoding object
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        # 1. Train using rustbpe (fast Rust code)
        tokenizer = rustbpe.Tokenizer()
        tokenizer.train_from_iterator(text_iterator, vocab_size - len(SPECIAL_TOKENS))

        # 2. Get mergeable ranks (byte sequences → token IDs)
        mergeable_ranks = tokenizer.get_mergeable_ranks()

        # 3. Add special tokens
        special_tokens = {name: offset + i for i, name in enumerate(SPECIAL_TOKENS)}

        # 4. Create tiktoken Encoding for fast inference
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=tokenizer.get_pattern(),
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens
        )

        return cls(enc, "<|bos|>")

    def encode(self, text, prepend=None, append=None):
        """Encode text to token IDs, optionally adding special tokens"""
        ids = self.enc.encode_ordinary(text)  # Fast C++ implementation!

        if prepend is not None:
            ids.insert(0, self.encode_special(prepend))
        if append is not None:
            ids.append(self.encode_special(append))

        return ids
```

### 3. Rendering Conversations

**The `render_conversation` method** is crucial for fine-tuning:

```python
def render_conversation(self, conversation, max_tokens=2048):
    """
    Convert a conversation dict to token IDs with training mask.

    Returns:
        ids: List[int] - Token IDs
        mask: List[int] - 1 for tokens to train on, 0 otherwise
    """
    ids, mask = [], []

    def add_tokens(token_ids, mask_val):
        """Helper to add tokens with corresponding mask"""
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    # Start with <|bos|> token
    add_tokens([self.get_bos_token_id()], 0)

    for i, message in enumerate(conversation["messages"]):
        if message["role"] == "user":
            # User messages: don't train on these
            add_tokens([self.encode_special("<|user_start|>")], 0)
            add_tokens(self.encode(message["content"]), 0)
            add_tokens([self.encode_special("<|user_end|>")], 0)

        elif message["role"] == "assistant":
            # Assistant messages: train on these!
            add_tokens([self.encode_special("<|assistant_start|>")], 0)

            # Content can be text or structured (with tool calls)
            if isinstance(message["content"], str):
                add_tokens(self.encode(message["content"]), 1)  # mask=1!

            elif isinstance(message["content"], list):
                # Handle tool calls
                for part in message["content"]:
                    if part["type"] == "text":
                        add_tokens(self.encode(part["text"]), 1)
                    elif part["type"] == "python":
                        add_tokens([self.encode_special("<|python_start|>")], 1)
                        add_tokens(self.encode(part["text"]), 1)
                        add_tokens([self.encode_special("<|python_end|>")], 1)
                    elif part["type"] == "python_output":
                        # Don't train on outputs (they come from Python)
                        add_tokens([self.encode_special("<|output_start|>")], 0)
                        add_tokens(self.encode(part["text"]), 0)
                        add_tokens([self.encode_special("<|output_end|>")], 0)

            add_tokens([self.encode_special("<|assistant_end|>")], 1)

    # Truncate if too long
    return ids[:max_tokens], mask[:max_tokens]
```

---

## Practical Examples

### Example 1: Training a Tokenizer

```python
# From scripts/tok_train.py (simplified)

# 1. Load training data
from nanochat.dataset import get_text_iterator
text_iter = get_text_iterator(num_chars=2_000_000_000)  # 2B characters

# 2. Train tokenizer
from nanochat.tokenizer import RustBPETokenizer
tokenizer = RustBPETokenizer.train_from_iterator(
    text_iter,
    vocab_size=65536  # 2^16 tokens
)

# 3. Save tokenizer
tokenizer.save("out/tokenizer")
```

### Example 2: Encoding and Decoding

```python
from nanochat.tokenizer import get_tokenizer

tokenizer = get_tokenizer()

# Encode text to token IDs
text = "Hello, world!"
ids = tokenizer.encode(text)
print(ids)  # [9906, 11, 995, 0]

# Decode token IDs back to text
decoded = tokenizer.decode(ids)
print(decoded)  # "Hello, world!"

# Encode with special tokens
ids = tokenizer.encode(text, prepend="<|bos|>")
print(ids)  # [65528, 9906, 11, 995, 0]
```

### Example 3: Tokenizing a Conversation

```python
conversation = {
    "messages": [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "The answer is 4."}
    ]
}

ids, mask = tokenizer.render_conversation(conversation)

# Visualize what gets trained on
print(tokenizer.visualize_tokenization(ids, mask))
# Red text = not trained on (user input, special tokens)
# Green text = trained on (assistant responses)
```

### Example 4: Tokenizing a Tool Use

```python
conversation = {
    "messages": [
        {"role": "user", "content": "Calculate 123 * 456"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Let me calculate that."},
                {"type": "python", "text": "123 * 456"},
                {"type": "python_output", "text": "56088"},
                {"type": "text", "text": "The answer is 56088."}
            ]
        }
    ]
}

ids, mask = tokenizer.render_conversation(conversation)

# Token sequence looks like:
# <|bos|><|user_start|>Calculate 123 * 456<|user_end|>
# <|assistant_start|>Let me calculate that.<|python_start|>123 * 456<|python_end|>
# <|output_start|>56088<|output_end|>The answer is 56088.<|assistant_end|>

# Mask = 1 for:
# - "Let me calculate that."
# - "123 * 456" (inside <|python_start|> tags)
# - "The answer is 56088."
# Mask = 0 for:
# - User input
# - Python output (since it comes from the interpreter, not the model)
```

---

## Key Takeaways

1. **Tokenization converts text to numbers** that neural networks can process

2. **BPE learns subwords** by iteratively merging the most frequent pairs of tokens

3. **Nanochat uses a two-stage approach:**
   - Training: Fast Rust implementation (`rustbpe`)
   - Inference: Fast C++ implementation (`tiktoken`)

4. **Special tokens provide structure** for conversations and tool use

5. **Training masks** ensure we only train on assistant responses, not user inputs

6. **Pre-tokenization with regex** prevents unwanted merges across word boundaries

7. **Vocabulary size** is a trade-off:
   - Larger = better compression, longer training
   - Smaller = worse compression, faster training
   - Nanochat uses 65,536 (2^16) as a sweet spot

---

## What's Next?

Now that you understand how text becomes numbers, let's see what the model does with those numbers!

**→ Next: [Document 3: The GPT Architecture](03_architecture.md)**

You'll learn:
- What is a Transformer?
- How attention mechanisms work
- Layer-by-layer breakdown of the GPT model
- How the model processes tokens to make predictions

---

## Self-Check

Before moving on, make sure you understand:

- [ ] Why we use BPE instead of characters or words
- [ ] How BPE training finds frequent pairs and merges them
- [ ] What special tokens are and why we need them
- [ ] The difference between training on user input vs assistant responses
- [ ] How a conversation is converted to token IDs
- [ ] Where the tokenizer code is (`tokenizer.py` and `rustbpe/src/lib.rs`)
