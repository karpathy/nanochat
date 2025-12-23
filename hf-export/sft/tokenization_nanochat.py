# hf-export/sft/tokenization_nanochat.py
import os
import pickle
from typing import List, Optional

from transformers import PreTrainedTokenizer

class NanoChatTokenizer(PreTrainedTokenizer):
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer_file: str = "tokenizer.pkl", **kwargs):
        # 1) 先加载 enc，先把 vocab size 准备好（一定要在 super() 之前）
        if tokenizer_file is None:
            tokenizer_file = "tokenizer.pkl"
        # 关键：优先用 HF 传进来的“模型目录”找文件，而不是 __file__
        # 常见字段：pretrained_model_name_or_path / name_or_path
        base_dir = (
            kwargs.pop("pretrained_model_name_or_path", None)
            or kwargs.get("name_or_path", None)
        )
        if base_dir and os.path.isdir(base_dir):
            path = os.path.join(base_dir, tokenizer_file)
        else:
            # 兜底：同目录（仅在不走 dynamic module 时才可能对）
            path = os.path.join(os.path.dirname(__file__), tokenizer_file)

        with open(path, "rb") as f:
            self.enc = pickle.load(f)
        self._vocab_size = int(self.enc.n_vocab)

        # Avoid letting HF create new token ids for specials; we'll bind to existing ids from the pickle.
        bos = "<|bos|>"
        eos = "<|assistant_end|>"
        # Drop potential duplicates coming from tokenizer_config.json to avoid double kwargs.
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("pad_token", None)
        # Call parent without special tokens to prevent added_tokens growth.
        super().__init__(bos_token=None, eos_token=None, pad_token=None, **kwargs)
        # Reset any auto-added tokens (should be empty, but be safe)
        self.added_tokens_encoder.clear()
        self.added_tokens_decoder.clear()
        # Bind specials to existing ids from the exported encoder
        self._bos_token = bos
        self._eos_token = eos
        self._pad_token = bos
        self._bos_token_id = self.enc._special_tokens[bos]
        self._eos_token_id = self.enc._special_tokens[eos]
        self._pad_token_id = self.enc._special_tokens[bos]

    # HF 常用：len(tokenizer) 会被调用
    def __len__(self):
        return self._vocab_size

    # 有些地方会访问 tokenizer.vocab_size
    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    # Override special token accessors to bind to existing ids (no new tokens added).
    @property
    def bos_token(self) -> str:
        return "<|bos|>"

    @property
    def eos_token(self) -> str:
        return "<|assistant_end|>"

    @property
    def pad_token(self) -> str:
        return "<|bos|>"

    @property
    def bos_token_id(self) -> int:
        return self.enc._special_tokens["<|bos|>"]

    @property
    def eos_token_id(self) -> int:
        return self.enc._special_tokens["<|assistant_end|>"]

    @property
    def pad_token_id(self) -> int:
        return self.enc._special_tokens["<|bos|>"]

    def get_vocab(self):
        # 注意：这里不要用 self.vocab_size（可能触发基类 __getattr__ 的时序坑）
        return {str(i): i for i in range(self._vocab_size)}

    def _tokenize(self, text: str) -> List[str]:
        # Allow special tokens like <|assistant_start|> to pass through with their ids.
        ids = self.enc.encode(text, allowed_special=self.enc.special_tokens_set)
        return [str(i) for i in ids]

    def _convert_token_to_id(self, token: str) -> int:
        if isinstance(token, str) and token in self.enc._special_tokens:
            return self.enc._special_tokens[token]
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        # Translate special token ids back to their string form.
        specials = {v: k for k, v in self.enc._special_tokens.items()}
        if index in specials:
            return specials[index]
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        # Preserve token order instead of front-loading specials.
        specials = {v: k for k, v in self.enc._special_tokens.items()}
        parts: List[str] = []
        pending_ids: List[int] = []

        def flush_pending():
            nonlocal pending_ids
            if pending_ids:
                parts.append(self.enc.decode(pending_ids))
                pending_ids = []

        for t in tokens:
            # pass through known special token strings
            if isinstance(t, str) and t in self.enc._special_tokens:
                flush_pending()
                parts.append(t)
                continue
            # or map special ids back to strings
            try:
                tid = int(t)
            except (ValueError, TypeError):
                continue
            if tid in specials:
                flush_pending()
                parts.append(specials[tid])
            else:
                pending_ids.append(tid)

        flush_pending()
        return "".join(parts)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ):
        return token_ids_0 if token_ids_1 is None else token_ids_0 + token_ids_1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None):
        return ()
