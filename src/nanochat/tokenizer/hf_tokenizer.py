"""HuggingFace-based BPE tokenizer (experimental alternative to RustBPETokenizer)."""

import os
from typing import List, Optional, Union

from tokenizers import Regex, decoders, pre_tokenizers
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from nanochat.tokenizer.constants import SPECIAL_TOKENS, SPLIT_PATTERN


class HuggingFaceTokenizer:
    """GPT-4-style BPE tokenizer backed by the HuggingFace tokenizers library.

    Supports both training and inference. Kept as an experimental alternative
    to RustBPETokenizer — not used in the production training pipeline.
    """

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, hf_path: str) -> "HuggingFaceTokenizer":
        """Load from a HuggingFace pretrained tokenizer (e.g. 'gpt2')."""
        return cls(HFTokenizer.from_pretrained(hf_path))

    @classmethod
    def from_directory(cls, tokenizer_dir: str) -> "HuggingFaceTokenizer":
        """Load from a local directory containing a tokenizer.json file."""
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        return cls(HFTokenizer.from_file(tokenizer_path))

    @classmethod
    def train_from_iterator(cls, text_iterator: object, vocab_size: int) -> "HuggingFaceTokenizer":
        """Train a GPT-4-style BPE tokenizer from a text iterator."""
        tokenizer = HFTokenizer(BPE(byte_fallback=True, unk_token=None, fuse_unk=False))
        tokenizer.normalizer = None
        tokenizer.post_processor = None
        tokenizer.decoder = decoders.ByteLevel()
        # NOTE: pattern changed from \p{N}{1,3} to \p{N}{1,2} vs GPT-4
        gpt4_split_regex = Regex(SPLIT_PATTERN)
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(pattern=gpt4_split_regex, behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
            ]
        )
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            min_frequency=0,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=SPECIAL_TOKENS,
        )
        tokenizer.train_from_iterator(text_iterator, trainer)
        return cls(tokenizer)

    def get_vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def get_special_tokens(self) -> List[str]:
        return [w.content for w in self.tokenizer.get_added_tokens_decoder().values()]

    def id_to_token(self, id: int) -> str:
        return self.tokenizer.id_to_token(id)

    def encode_special(self, text: str) -> int | None:
        return self.tokenizer.token_to_id(text)

    def get_bos_token_id(self) -> int:
        bos = self.encode_special("<|bos|>") or self.encode_special("<|endoftext|>")
        assert bos is not None, "Failed to find BOS token in tokenizer"
        return bos

    def _encode_one(
        self,
        text: str,
        prepend: Optional[Union[str, int]] = None,
        append: Optional[Union[str, int]] = None,
        num_threads: Optional[int] = None,  # ignored, kept for API compat with RustBPETokenizer
    ) -> List[int]:
        assert isinstance(text, str)
        ids: List[int] = []
        if prepend is not None:
            ids.append(prepend if isinstance(prepend, int) else self.encode_special(prepend))
        ids.extend(self.tokenizer.encode(text, add_special_tokens=False).ids)
        if append is not None:
            ids.append(append if isinstance(append, int) else self.encode_special(append))
        return ids

    def encode(self, text: Union[str, List[str]], *args: object, **kwargs: object) -> Union[List[int], List[List[int]]]:
        if isinstance(text, str):
            return self._encode_one(text, *args, **kwargs)
        return [self._encode_one(t, *args, **kwargs) for t in text]

    def __call__(self, *args: object, **kwargs: object) -> Union[List[int], List[List[int]]]:
        return self.encode(*args, **kwargs)

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=False)

    def save(self, tokenizer_dir: str) -> None:
        """Save the tokenizer to disk as tokenizer.json."""
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer_path = os.path.join(tokenizer_dir, "tokenizer.json")
        self.tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")
