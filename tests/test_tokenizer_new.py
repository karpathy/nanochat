import os
import pickle
from types import SimpleNamespace

import pytest
import torch

import nanochat.tokenizer as tokmod


class _FakeHFCore:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.saved = None
        self.normalizer = None
        self.pre_tokenizer = None
        self.decoder = None
        self.post_processor = None
        self.trained = False
        self.special = {"<|bos|>": 100, "<|endoftext|>": 101}

    @classmethod
    def from_pretrained(cls, _path):
        return cls()

    @classmethod
    def from_file(cls, _path):
        return cls()

    def train_from_iterator(self, text_iterator, trainer):
        del trainer
        list(text_iterator)
        self.trained = True

    def get_vocab_size(self):
        return 123

    def get_added_tokens_decoder(self):
        return {0: SimpleNamespace(content="<|bos|>"), 1: SimpleNamespace(content="<|assistant_start|>")}

    def id_to_token(self, id):
        return f"tok-{id}"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return SimpleNamespace(ids=[len(text), len(text) + 1])

    def token_to_id(self, text):
        return self.special.get(text)

    def decode(self, ids, skip_special_tokens=False):
        del skip_special_tokens
        return "|".join(map(str, ids))

    def save(self, path):
        self.saved = path
        with open(path, "w", encoding="utf-8") as f:
            f.write("{}")


def test_huggingface_tokenizer_wrapper(monkeypatch, tmp_path):
    monkeypatch.setattr(tokmod, "HFTokenizer", _FakeHFCore)
    monkeypatch.setattr(tokmod, "BPE", lambda **kwargs: kwargs)
    monkeypatch.setattr(tokmod, "Regex", lambda x: f"re:{x}")

    class _ByteLevel:
        @staticmethod
        def alphabet():
            return [b"a"]

        def __init__(self, **kwargs):
            self.kw = kwargs

    fake_pre = SimpleNamespace(
        Split=lambda **kwargs: ("split", kwargs),
        ByteLevel=_ByteLevel,
        Sequence=lambda xs: ("seq", xs),
    )
    monkeypatch.setattr(tokmod, "pre_tokenizers", fake_pre)
    monkeypatch.setattr(tokmod, "decoders", SimpleNamespace(ByteLevel=lambda: "decoder"))
    monkeypatch.setattr(tokmod, "BpeTrainer", lambda **kwargs: ("trainer", kwargs))

    h1 = tokmod.HuggingFaceTokenizer.from_pretrained("gpt2")
    assert isinstance(h1, tokmod.HuggingFaceTokenizer)

    d = tmp_path / "hf"
    d.mkdir()
    (d / "tokenizer.json").write_text("{}", encoding="utf-8")
    h2 = tokmod.HuggingFaceTokenizer.from_directory(str(d))
    assert isinstance(h2, tokmod.HuggingFaceTokenizer)

    h3 = tokmod.HuggingFaceTokenizer.train_from_iterator(iter(["hello", "world"]), vocab_size=300)
    assert isinstance(h3, tokmod.HuggingFaceTokenizer)
    assert h3.tokenizer.trained is True

    assert h3.get_vocab_size() == 123
    assert "<|bos|>" in h3.get_special_tokens()
    assert h3.id_to_token(7) == "tok-7"
    assert h3.encode_special("<|bos|>") == 100
    assert h3.get_bos_token_id() == 100

    h3.tokenizer.special["<|bos|>"] = None
    assert h3.get_bos_token_id() == 101
    h3.tokenizer.special["<|endoftext|>"] = None
    with pytest.raises(AssertionError):
        h3.get_bos_token_id()
    h3.tokenizer.special["<|bos|>"] = 100

    one = h3._encode_one("abc", prepend="<|bos|>", append=999)
    assert one[0] == 100 and one[-1] == 999
    assert isinstance(h3.encode("abc"), list)
    assert isinstance(h3.encode(["a", "bb"]), list)
    with pytest.raises(ValueError):
        h3.encode(123)  # type: ignore[arg-type]

    assert h3("abc") == h3.encode("abc")
    assert h3.decode([1, 2]) == "1|2"
    save_dir = tmp_path / "save_hf"
    h3.save(str(save_dir))
    assert (save_dir / "tokenizer.json").exists()


class _FakeRustTokenizerCore:
    def __init__(self):
        self.pattern = "pat"
        self.mergeable = [(b"a", 0), (b"b", 1)]
        self.trained = None

    def train_from_iterator(self, it, vocab_size_no_special, pattern):
        self.trained = (list(it), vocab_size_no_special, pattern)

    def get_pattern(self):
        return self.pattern

    def get_mergeable_ranks(self):
        return self.mergeable


class _FakeEnc:
    def __init__(self):
        self.n_vocab = 300
        self.special_tokens_set = set(tokmod.SPECIAL_TOKENS)
        self.special = {name: 256 + i for i, name in enumerate(tokmod.SPECIAL_TOKENS)}
        self.special["<|endoftext|>"] = 999

    def encode_single_token(self, text):
        return self.special[text]

    def encode_ordinary(self, text):
        return [ord(c) % 50 for c in text]

    def encode_ordinary_batch(self, text, num_threads=8):
        del num_threads
        return [self.encode_ordinary(t) for t in text]

    def decode(self, ids):
        return ",".join(str(x) for x in ids)


def test_rust_bpe_wrapper_and_helpers(monkeypatch, tmp_path):
    monkeypatch.setattr(tokmod.rustbpe, "Tokenizer", _FakeRustTokenizerCore)
    monkeypatch.setattr(tokmod.tiktoken, "Encoding", lambda **kwargs: _FakeEnc())
    monkeypatch.setattr(tokmod.tiktoken, "get_encoding", lambda _name: _FakeEnc())

    r1 = tokmod.RustBPETokenizer.train_from_iterator(iter(["hello", "world"]), vocab_size=300)
    assert isinstance(r1, tokmod.RustBPETokenizer)

    with pytest.raises(AssertionError):
        tokmod.RustBPETokenizer.train_from_iterator(iter(["x"]), vocab_size=10)

    assert r1.get_vocab_size() == 300
    assert "<|bos|>" in r1.get_special_tokens()
    assert isinstance(r1.id_to_token(3), str)
    assert r1.get_bos_token_id() == r1.encode_special("<|bos|>")
    assert r1.encode_special("<|bos|>") == r1.encode_special("<|bos|>")  # cache path

    s = r1.encode("abc", prepend="<|bos|>", append="<|assistant_end|>")
    assert s[0] == r1.encode_special("<|bos|>")
    assert s[-1] == r1.encode_special("<|assistant_end|>")
    b = r1.encode(["a", "bc"], prepend=1, append=2, num_threads=1)
    assert b[0][0] == 1 and b[0][-1] == 2
    with pytest.raises(ValueError):
        r1.encode(123)  # type: ignore[arg-type]
    assert r1("abc") == r1.encode("abc")
    assert isinstance(r1.decode([1, 2]), str)

    save_dir = tmp_path / "save_rust"
    r1.save(str(save_dir))
    assert (save_dir / "tokenizer.pkl").exists()
    r2 = tokmod.RustBPETokenizer.from_directory(str(save_dir))
    assert isinstance(r2, tokmod.RustBPETokenizer)
    r3 = tokmod.RustBPETokenizer.from_pretrained("gpt2")
    assert isinstance(r3, tokmod.RustBPETokenizer)


def test_render_conversation_and_completion(monkeypatch):
    r = tokmod.RustBPETokenizer(_FakeEnc(), "<|bos|>")

    conv = {
        "messages": [
            {"role": "system", "content": "system msg"},
            {"role": "user", "content": "hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi"},
                    {"type": "python", "text": "1+1"},
                    {"type": "python_output", "text": "2"},
                ],
            },
        ]
    }
    ids, mask = r.render_conversation(conv, max_tokens=128)
    assert len(ids) == len(mask)
    assert any(m == 1 for m in mask)
    vis = r.visualize_tokenization(ids[:5], mask[:5], with_token_id=True)
    assert "|" in vis

    conv_str_assistant = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "plain answer"},
        ]
    }
    ids2, mask2 = r.render_conversation(conv_str_assistant, max_tokens=64)
    assert len(ids2) == len(mask2)
    assert any(m == 1 for m in mask2)

    comp_conv = {
        "messages": [
            {"role": "user", "content": "Q"},
            {"role": "assistant", "content": "A"},
        ]
    }
    out = r.render_for_completion(comp_conv)
    assert out[-1] == r.encode_special("<|assistant_start|>")

    bad_role = {"messages": [{"role": "assistant", "content": "x"}]}
    with pytest.raises(AssertionError):
        r.render_conversation(bad_role)

    bad_content = {
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": 123},
        ]
    }
    with pytest.raises(ValueError):
        r.render_conversation(bad_content)

    bad_part = {
        "messages": [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": [{"type": "bad", "text": "x"}]},
        ]
    }
    with pytest.raises(ValueError):
        r.render_conversation(bad_part)

    with pytest.raises(AssertionError):
        r.render_for_completion({"messages": [{"role": "user", "content": "x"}]})


def test_get_tokenizer_and_token_bytes(monkeypatch, tmp_path):
    tok_dir = tmp_path / "tokenizer"
    tok_dir.mkdir()
    token_bytes = torch.tensor([0, 1, 2], dtype=torch.int64)
    torch.save(token_bytes, tok_dir / "token_bytes.pt")

    monkeypatch.setattr("nanochat.common.get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(tokmod.RustBPETokenizer, "from_directory", classmethod(lambda cls, d: ("tok", d)))
    got = tokmod.get_tokenizer()
    assert got[0] == "tok"
    assert got[1].endswith("tokenizer")

    tb = tokmod.get_token_bytes(device="cpu")
    assert torch.equal(tb, token_bytes)

    os.remove(tok_dir / "token_bytes.pt")
    with pytest.raises(AssertionError):
        tokmod.get_token_bytes()
