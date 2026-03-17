"""RustBPE + tiktoken tokenizer: trains with rustbpe, runs inference with tiktoken."""

import copy
import os
import pickle
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import rustbpe
import tiktoken

from nanochat.tokenizer.constants import SPECIAL_TOKENS, SPLIT_PATTERN


class RustBPETokenizer:
    """Trains with rustbpe, runs inference with tiktoken.

    Training uses the rustbpe Rust extension for fast BPE merges.
    The resulting vocabulary is wrapped in a tiktoken.Encoding for
    efficient batched inference. Special tokens are appended after
    training so they never appear in the merge table.
    """

    def __init__(self, enc: tiktoken.Encoding, bos_token: str) -> None:
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def train_from_iterator(cls, text_iterator: object, vocab_size: int) -> "RustBPETokenizer":
        """Train a BPE tokenizer from a text iterator."""
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "RustBPETokenizer":
        """Load a previously saved tokenizer from disk."""
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> "RustBPETokenizer":
        """Load a pretrained tiktoken encoding (e.g. 'gpt2', 'cl100k_base').

        tiktoken uses '<|endoftext|>' as the BOS token; nanochat maps it to '<|bos|>'.
        """
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_special_tokens(self) -> set[str]:
        return self.enc.special_tokens_set

    def id_to_token(self, id: int) -> str:
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: Union[str, List[str]],
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
        num_threads: int = 8,
    ) -> Union[List[int], List[List[int]]]:
        """Encode one string or a batch of strings.

        Args:
            text: A single string or list of strings.
            prepend: Optional token (id or special-token string) to prepend to each sequence.
            append: Optional token (id or special-token string) to append to each sequence.
            num_threads: Worker threads for batch encoding.
        """
        prepend_id = (
            (prepend if isinstance(prepend, int) else self.encode_special(prepend)) if prepend is not None else None
        )
        append_id = (append if isinstance(append, int) else self.encode_special(append)) if append is not None else None

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
        else:
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append_id is not None:
                for ids_row in ids:
                    ids_row.append(append_id)

        return ids

    def __call__(self, *args: object, **kwargs: object) -> Union[List[int], List[List[int]]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: List[int]) -> str:
        return self.enc.decode(ids)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, tokenizer_dir: str) -> None:
        """Save the tiktoken encoding to disk as a pickle."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    # ------------------------------------------------------------------
    # Chat rendering
    # ------------------------------------------------------------------

    def render_conversation(self, conversation: Dict[str, Any], max_tokens: int = 2048) -> Tuple[List[int], List[int]]:
        """Tokenize a chat conversation into token ids and a supervision mask.

        Returns:
            ids: Token ids for the full conversation.
            mask: Same length as ids; 1 for assistant tokens to train on, 0 otherwise.
        """
        ids: List[int] = []
        mask: List[int] = []

        def add_tokens(token_ids: Union[int, List[int]], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # Merge leading system message into the first user message
        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                add_tokens(user_start, 0)
                add_tokens(self.encode(content), 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    add_tokens(self.encode(content), 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids: List[int], mask: List[int], with_token_id: bool = False) -> str:
        """Visualize token ids and supervision mask with ANSI colors (for debugging)."""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation: Dict[str, Any]) -> List[int]:
        """Render a conversation primed for assistant completion (used in RL).

        Strips the last assistant message and appends <|assistant_start|> to
        prompt the model for a completion.
        """
        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()
        ids, _ = self.render_conversation(conversation)
        ids.append(self.encode_special("<|assistant_start|>"))
        return ids
