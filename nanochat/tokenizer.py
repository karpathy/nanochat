"""BPE Tokenizer in the style of GPT-4.

Two implementations are available:
1. HuggingFace Tokenizer that can do both training and inference but is really confusing
2. Our own RustBPE Tokenizer for training and tiktoken for efficient inference
"""

from __future__ import annotations

import copy
import os
import pickle
from collections.abc import Iterator
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import rustbpe
import tiktoken
from tokenizers import Regex
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders
from tokenizers import pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

if TYPE_CHECKING:
    import torch

SPECIAL_TOKENS: list[str] = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",      # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",    # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",    # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# I did this because I didn't want to "waste" too many tokens on numbers for smaller vocab sizes.
# I verified that 2 is the sweet spot for vocab size of 32K. 1 is a bit worse, 3 was worse still.
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


# -----------------------------------------------------------------------------
# Generic GPT-4-style tokenizer based on HuggingFace Tokenizer


class HuggingFaceTokenizer:
    """Light wrapper around HuggingFace Tokenizer for some utilities.

    Attributes:
        tokenizer: The underlying HuggingFace Tokenizer instance.
    """

    def __init__(self, tokenizer: HFTokenizer) -> None:
        """Initializes the HuggingFace tokenizer wrapper.

        Args:
            tokenizer: A HuggingFace Tokenizer instance.
        """
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> HuggingFaceTokenizer:
        """Creates a tokenizer from a HuggingFace pretrained model.

        Args:
            hf_path: The HuggingFace model path (e.g., "gpt2").

        Returns:
            A new HuggingFaceTokenizer instance.
        """
        tokenizer = HFTokenizer.from_pretrained(hf_path)
        return cls(tokenizer)

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> HuggingFaceTokenizer:
        """Creates a tokenizer from a local directory on disk.

        Args:
            tokenizer_dir: Path to the directory containing tokenizer.json
                (e.g., "out/tokenizer").

        Returns:
            A new HuggingFaceTokenizer instance.
        """
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        tokenizer = HFTokenizer.from_file(tokenizer_path)
        return cls(tokenizer)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterator[str],
        vocab_size: int,
    ) -> HuggingFaceTokenizer:
        """Trains a tokenizer from an iterator of text.

        Configures a GPT-4 style BPE tokenizer with byte fallback and
        trains it on the provided text data.

        Args:
            text_iterator: An iterator yielding text strings for training.
            vocab_size: The target vocabulary size.

        Returns:
            A new HuggingFaceTokenizer instance with the trained tokenizer.
        """
        # Configure the HuggingFace Tokenizer
        tokenizer = HFTokenizer(BPE(
            byte_fallback=True,  # needed!
            unk_token=None,
            fuse_unk=False,
        ))
        # Normalizer: None
        tokenizer.normalizer = None  # type: ignore[assignment]
        # Pre-tokenizer: GPT-4 style
        # the regex pattern used by GPT-4 to split text into groups before BPE
        # NOTE: The pattern was changed from \p{N}{1,3} to \p{N}{1,2} because I suspect it is harmful to
        # very small models and smaller vocab sizes, because it is a little bit wasteful in the token space.
        # (but I haven't validated this! TODO)
        gpt4_split_regex = Regex(SPLIT_PATTERN)  # huggingface demands that you wrap it in Regex!!
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([  # type: ignore[assignment]
            pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        # Decoder: ByteLevel (it pairs together with the ByteLevel pre-tokenizer)
        tokenizer.decoder = decoders.ByteLevel()  # type: ignore[assignment]
        # Post-processor: None
        tokenizer.post_processor = None  # type: ignore[assignment]
        # Trainer: BPE
        # type stubs for tokenizers library are incomplete
        trainer = BpeTrainer(
            vocab_size=vocab_size,  # type: ignore[call-arg]
            show_progress=True,  # type: ignore[call-arg]
            min_frequency=0,  # type: ignore[call-arg]  # no minimum frequency
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # type: ignore[call-arg]
            special_tokens=SPECIAL_TOKENS,  # type: ignore[call-arg]
        )
        # Kick off the training
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Returns:
            The number of tokens in the vocabulary.
        """
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> list[str]:
        """Returns the list of special tokens.

        Returns:
            A list of special token strings.
        """
        special_tokens_map = self.tokenizer.get_added_tokens_decoder()
        special_tokens = [w.content for w in special_tokens_map.values()]
        return special_tokens

    def id_to_token(self, id: int) -> str | None:
        """Converts a token ID to its string representation.

        Args:
            id: The token ID.

        Returns:
            The token string, or None if the ID is invalid.
        """
        return self.tokenizer.id_to_token(id)

    def _encode_one(
        self,
        text: str,
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int | None = None,
    ) -> list[int]:
        """Encodes a single string to token IDs.

        Args:
            text: The text to encode.
            prepend: Optional token to prepend. Can be a special token string
                or a token ID directly.
            append: Optional token to append. Can be a special token string
                or a token ID directly.
            num_threads: Ignored (only used by the nanochat Tokenizer for
                parallel encoding).

        Returns:
            A list of token IDs.
        """
        assert isinstance(text, str)
        ids: list[int] = []
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
            if prepend_id is not None:
                ids.append(prepend_id)
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)
            if append_id is not None:
                ids.append(append_id)
        return ids

    def encode_special(self, text: str) -> int | None:
        """Encodes a single special token via exact match.

        Args:
            text: The special token string.

        Returns:
            The token ID, or None if not found.
        """
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self) -> int:
        """Gets the Beginning of Sequence (BOS) token ID.

        Different HuggingFace models use different BOS tokens and there
        is little consistency. This method attempts to find a suitable
        BOS token.

        Returns:
            The BOS token ID.

        Raises:
            AssertionError: If no BOS token is found in the tokenizer.
        """
        # 1) attempt to find a <|bos|> token
        bos = self.encode_special("<|bos|>")
        # 2) if that fails, attempt to find a <|endoftext|> token (e.g. GPT-2 models)
        if bos is None:
            bos = self.encode_special("<|endoftext|>")
        # 3) if these fail, it's better to crash than to silently return None
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def encode(
        self,
        text: str | list[str],
        *args: Any,
        **kwargs: Any,
    ) -> list[int] | list[list[int]]:
        """Encodes text to token IDs.

        Args:
            text: A string or list of strings to encode.
            *args: Additional positional arguments passed to _encode_one.
            **kwargs: Additional keyword arguments passed to _encode_one.

        Returns:
            A list of token IDs if text is a string, or a list of lists
            of token IDs if text is a list of strings.

        Raises:
            ValueError: If text is neither a string nor a list.
        """
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        elif isinstance(text, list):
            return [self._encode_one(t, *args, **kwargs) for t in text]
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

    def __call__(self, *args: Any, **kwargs: Any) -> list[int] | list[list[int]]:
        """Encodes text to token IDs (callable interface).

        Args:
            *args: Positional arguments passed to encode.
            **kwargs: Keyword arguments passed to encode.

        Returns:
            The encoded token IDs.
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs to text.

        Args:
            ids: A list of token IDs.

        Returns:
            The decoded text string.
        """
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        """Saves the tokenizer to disk.

        Args:
            tokenizer_dir: The directory to save the tokenizer to.
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")


# -----------------------------------------------------------------------------
# Tokenizer based on rustbpe + tiktoken combo


class RustBPETokenizer:
    """Light wrapper around tiktoken for efficient inference with rustbpe training.

    Attributes:
        enc: The tiktoken Encoding instance.
        bos_token_id: The Beginning of Sequence token ID.
    """

    def __init__(self, enc: tiktoken.Encoding, bos_token: str) -> None:
        """Initializes the RustBPE tokenizer.

        Args:
            enc: A tiktoken Encoding instance.
            bos_token: The special token string to use as BOS.
        """
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

    @classmethod
    def train_from_iterator(
        cls,
        text_iterator: Iterator[str],
        vocab_size: int,
    ) -> RustBPETokenizer:
        """Trains a tokenizer from an iterator of text using rustbpe.

        Args:
            text_iterator: An iterator yielding text strings for training.
            vocab_size: The target vocabulary size.

        Returns:
            A new RustBPETokenizer instance with the trained tokenizer.

        Raises:
            AssertionError: If vocab_size_no_special is less than 256.
        """
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()  # type: ignore[attr-defined]
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,    # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> RustBPETokenizer:
        """Creates a tokenizer from a local directory on disk.

        Args:
            tokenizer_dir: Path to the directory containing tokenizer.pkl.

        Returns:
            A new RustBPETokenizer instance.
        """
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name: str) -> RustBPETokenizer:
        """Creates a tokenizer from a pretrained tiktoken encoding.

        See: https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py

        Note: tiktoken calls the special document delimiter token "<|endoftext|>".
        This is confusing because this token is almost always PREPENDED to the
        beginning of the document. It is most often used to signal the start of
        a new sequence to the LLM during inference. In nanoChat we always use
        "<|bos|>" short for "beginning of sequence", but historically it is
        often called "<|endoftext|>".

        Args:
            tiktoken_name: The name of the tiktoken encoding.

        Returns:
            A new RustBPETokenizer instance.
        """
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self) -> int:
        """Returns the vocabulary size of the tokenizer.

        Returns:
            The number of tokens in the vocabulary.
        """
        return self.enc.n_vocab

    def get_special_tokens(self) -> set[str]:
        """Returns the set of special tokens.

        Returns:
            A set of special token strings.
        """
        return self.enc.special_tokens_set

    def id_to_token(self, id: int) -> str:
        """Converts a token ID to its string representation.

        Args:
            id: The token ID.

        Returns:
            The decoded token string.
        """
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text: str) -> int:
        """Encodes a single special token via exact match.

        Results are cached for efficiency.

        Args:
            text: The special token string.

        Returns:
            The token ID.
        """
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self) -> int:
        """Gets the Beginning of Sequence (BOS) token ID.

        Returns:
            The BOS token ID.
        """
        return self.bos_token_id

    def encode(
        self,
        text: str | list[str],
        prepend: str | int | None = None,
        append: str | int | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        """Encodes text to token IDs.

        Args:
            text: A string or list of strings to encode.
            prepend: Optional token to prepend. Can be a special token string
                or a token ID directly.
            append: Optional token to append. Can be a special token string
                or a token ID directly.
            num_threads: Number of threads for parallel batch encoding.
                Defaults to 8.

        Returns:
            A list of token IDs if text is a string, or a list of lists
            of token IDs if text is a list of strings.

        Raises:
            ValueError: If text is neither a string nor a list.
        """
        prepend_id: int | None = None
        append_id: int | None = None

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)  # TODO: slightly inefficient here? :( hmm
            if append_id is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)  # TODO: same
            if append_id is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args: Any, **kwargs: Any) -> list[int] | list[list[int]]:
        """Encodes text to token IDs (callable interface).

        Args:
            *args: Positional arguments passed to encode.
            **kwargs: Keyword arguments passed to encode.

        Returns:
            The encoded token IDs.
        """
        return self.encode(*args, **kwargs)

    def decode(self, ids: list[int]) -> str:
        """Decodes token IDs to text.

        Args:
            ids: A list of token IDs.

        Returns:
            The decoded text string.
        """
        return self.enc.decode(ids)

    def save(self, tokenizer_dir: str) -> None:
        """Saves the encoding object to disk.

        Args:
            tokenizer_dir: The directory to save the tokenizer to.
        """
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(
        self,
        conversation: dict[str, Any],
        max_tokens: int = 2048,
    ) -> tuple[list[int], list[int]]:
        """Tokenizes a single Chat conversation.

        Args:
            conversation: A dictionary containing the conversation with a
                "messages" key. Each message has "role" and "content" keys.
            max_tokens: Maximum number of tokens to return. Defaults to 2048.

        Returns:
            A tuple containing:
                - ids: A list of token IDs for the rendered conversation.
                - mask: A list of integers of same length where mask=1 for
                  tokens that the Assistant is expected to train on.

        Raises:
            AssertionError: If message roles don't alternate correctly or
                if the conversation structure is invalid.
            ValueError: If an unknown content or part type is encountered.
        """
        # ids, masks that we will return and a helper function to help build them up.
        ids: list[int] = []
        mask: list[int] = []

        def add_tokens(token_ids: int | list[int], mask_val: int) -> None:
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation)  # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start = self.encode_special("<|user_start|>")
        user_end = self.encode_special("<|user_end|>")
        assistant_start = self.encode_special("<|assistant_start|>")
        assistant_end = self.encode_special("<|assistant_end|>")
        python_start = self.encode_special("<|python_start|>")
        python_end = self.encode_special("<|python_end|>")
        output_start = self.encode_special("<|output_start|>")
        output_end = self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = cast(list[int], self.encode(content))
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = cast(list[int], self.encode(content))
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = cast(list[int], self.encode(part["text"]))
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(
        self,
        ids: list[int],
        mask: list[int],
        with_token_id: bool = False,
    ) -> str:
        """Visualizes the tokenization of a rendered conversation.

        Useful for debugging. Tokens are color-coded: green for masked
        tokens (to be trained on), red for unmasked tokens.

        Args:
            ids: A list of token IDs.
            mask: A list of mask values (1 for train, 0 for ignore).
            with_token_id: Whether to include token IDs in the output.
                Defaults to False.

        Returns:
            A string with color-coded token visualization.
        """
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens: list[str] = []
        for token_id, mask_val in zip(ids, mask):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation: dict[str, Any]) -> list[int]:
        """Renders a conversation priming the Assistant for completion.

        Used during Reinforcement Learning. Unlike the Chat SFT case,
        we don't need to return the mask. The last assistant message
        is removed and the assistant start token is appended.

        Args:
            conversation: A dictionary containing the conversation with a
                "messages" key. The last message must be from the assistant.

        Returns:
            A list of token IDs ready for completion generation.

        Raises:
            AssertionError: If the last message is not from the Assistant.
        """
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation)  # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop()  # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


# -----------------------------------------------------------------------------
# nanochat-specific convenience functions


def get_tokenizer() -> RustBPETokenizer:
    """Gets the default nanochat tokenizer from the cache directory.

    Returns:
        A RustBPETokenizer instance loaded from the tokenizer directory.
    """
    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    # return HuggingFaceTokenizer.from_directory(tokenizer_dir)
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device: str = "cpu") -> "torch.Tensor":
    """Gets the cached token-to-bytes mapping tensor.

    This mapping is used for efficient evaluation of bits per byte.
    Unlike the typical mean loss, this allows reporting a loss that
    is invariant to the vocab size of the tokenizer.

    Args:
        device: The device to load the tensor to. Defaults to "cpu".

    Returns:
        A tensor mapping token IDs to their byte lengths.

    Raises:
        AssertionError: If the token_bytes.pt file is not found.
    """
    import torch

    from nanochat.common import get_base_dir

    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? It gets written by tok_train.py"
    )
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes
