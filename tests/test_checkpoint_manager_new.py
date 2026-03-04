import json
import os
from types import SimpleNamespace

import pytest
import torch

import nanochat.checkpoint_manager as ckpt


def test_log0_respects_rank(monkeypatch):
    seen = {"n": 0}
    monkeypatch.setattr(ckpt.logger, "info", lambda _m: seen.__setitem__("n", seen["n"] + 1))
    monkeypatch.setenv("RANK", "0")
    ckpt.log0("hi")
    monkeypatch.setenv("RANK", "1")
    ckpt.log0("skip")
    assert seen["n"] == 1


def test_patch_missing_helpers():
    cfg = {"vocab_size": 32}
    ckpt._patch_missing_config_keys(cfg)
    assert cfg["window_pattern"] == "L"
    ckpt._patch_missing_config_keys(cfg)
    assert cfg["window_pattern"] == "L"

    model_data = {}
    model_cfg = SimpleNamespace(n_layer=3)
    ckpt._patch_missing_keys(model_data, model_cfg)
    assert torch.all(model_data["resid_lambdas"] == 1)
    assert torch.all(model_data["x0_lambdas"] == 0)

    ckpt._patch_missing_keys(model_data, model_cfg)
    assert model_data["resid_lambdas"].numel() == 3


def test_save_and_load_checkpoint(tmp_path):
    model_data = {"w": torch.tensor([1, 2])}
    optim_data = {"g": torch.tensor([3, 4])}
    meta_data = {"model_config": {"vocab_size": 8, "n_layer": 1, "n_head": 1, "n_kv_head": 1, "n_embd": 4, "sequence_len": 4}}
    ckpt.save_checkpoint(str(tmp_path), 7, model_data, optim_data, meta_data, rank=0)
    ckpt.save_checkpoint(str(tmp_path), 8, model_data, None, meta_data, rank=0)
    got_model, got_optim, got_meta = ckpt.load_checkpoint(str(tmp_path), 7, "cpu", load_optimizer=True, rank=0)
    assert torch.equal(got_model["w"], model_data["w"])
    assert torch.equal(got_optim["g"], optim_data["g"])
    assert got_meta["model_config"]["vocab_size"] == 8

    got_model2, got_optim2, got_meta2 = ckpt.load_checkpoint(str(tmp_path), 8, "cpu", load_optimizer=False, rank=0)
    assert got_optim2 is None
    assert got_meta2 == meta_data
    assert torch.equal(got_model2["w"], model_data["w"])


def test_build_model_cpu_and_train_eval(monkeypatch):
    model_data = {"_orig_mod.w": torch.ones(1, dtype=torch.bfloat16)}
    meta_data = {"model_config": {"vocab_size": 16, "n_layer": 2, "n_head": 1, "n_kv_head": 1, "n_embd": 4, "sequence_len": 8}}

    class FakeTokenizer:
        def get_vocab_size(self):
            return 16

    class FakeModel:
        def __init__(self, cfg):
            self.cfg = cfg
            self.mode = None
            self.loaded = None

        def to_empty(self, device):
            self.device = device

        def init_weights(self):
            self.init_called = True

        def load_state_dict(self, data, strict, assign):
            self.loaded = (data, strict, assign)

        def eval(self):
            self.mode = "eval"

        def train(self):
            self.mode = "train"

    monkeypatch.setattr(ckpt, "load_checkpoint", lambda *a, **k: (model_data.copy(), None, meta_data))
    monkeypatch.setattr(ckpt, "GPTConfig", lambda **kw: SimpleNamespace(**kw))
    monkeypatch.setattr(ckpt, "GPT", lambda cfg: FakeModel(cfg))
    monkeypatch.setattr(ckpt, "get_tokenizer", lambda: FakeTokenizer())

    m, tok, meta = ckpt.build_model("/x", 1, torch.device("cpu"), phase="eval")
    assert m.mode == "eval"
    assert tok.get_vocab_size() == 16
    assert meta == meta_data
    assert "w" in m.loaded[0]
    assert m.loaded[0]["w"].dtype == torch.float32

    m2, _, _ = ckpt.build_model("/x", 1, torch.device("cpu"), phase="train")
    assert m2.mode == "train"

    with pytest.raises(AssertionError):
        ckpt.build_model("/x", 1, torch.device("cpu"), phase="bad")

    class BadTok(FakeTokenizer):
        def get_vocab_size(self):
            return 99

    monkeypatch.setattr(ckpt, "get_tokenizer", lambda: BadTok())
    with pytest.raises(AssertionError):
        ckpt.build_model("/x", 1, torch.device("cpu"), phase="eval")


def test_find_largest_model_and_last_step(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError):
        ckpt.find_largest_model(str(tmp_path))

    (tmp_path / "d3").mkdir()
    (tmp_path / "d10").mkdir()
    assert ckpt.find_largest_model(str(tmp_path)) == "d10"

    # Fallback path: no d<number> tags, choose newest by mtime.
    other = tmp_path / "abc"
    newest = tmp_path / "zzz"
    other.mkdir(exist_ok=True)
    newest.mkdir(exist_ok=True)
    monkeypatch.setattr(ckpt.re, "match", lambda _pat, _s: None)
    assert ckpt.find_largest_model(str(tmp_path)) == "zzz"

    step_dir = tmp_path / "steps"
    step_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        ckpt.find_last_step(str(step_dir))
    (step_dir / "model_000012.pt").write_bytes(b"")
    (step_dir / "model_000007.pt").write_bytes(b"")
    assert ckpt.find_last_step(str(step_dir)) == 12


def test_load_model_from_dir_and_load_model(monkeypatch):
    calls = {}

    monkeypatch.setattr(ckpt, "find_largest_model", lambda d: "d12")
    monkeypatch.setattr(ckpt, "find_last_step", lambda d: 42)

    def fake_build_model(checkpoint_dir, step, device, phase):
        calls["args"] = (checkpoint_dir, step, device, phase)
        return "m", "t", {"k": 1}

    monkeypatch.setattr(ckpt, "build_model", fake_build_model)
    out = ckpt.load_model_from_dir("/root/checkpoints", device="cpu", phase="eval")
    assert out == ("m", "t", {"k": 1})
    assert calls["args"][0].endswith("/root/checkpoints/d12")
    assert calls["args"][1] == 42

    monkeypatch.setattr(ckpt, "get_base_dir", lambda: "/base")
    monkeypatch.setattr(ckpt, "load_model_from_dir", lambda checkpoints_dir, *a, **k: ("ok", checkpoints_dir, a, k))
    got = ckpt.load_model("base", "cpu", phase="eval")
    assert got[0] == "ok"
    assert got[1].endswith("/base/base_checkpoints")
    assert ckpt.load_model("sft", "cpu", phase="eval")[1].endswith("/base/chatsft_checkpoints")
    assert ckpt.load_model("rl", "cpu", phase="eval")[1].endswith("/base/chatrl_checkpoints")


def test_load_optimizer_state(monkeypatch, tmp_path):
    model_root = tmp_path / "base_checkpoints" / "d7"
    model_root.mkdir(parents=True)
    opt_path = model_root / "optim_000003_rank2.pt"
    torch.save({"x": torch.tensor([1])}, opt_path)

    monkeypatch.setattr(ckpt, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(ckpt, "find_largest_model", lambda _d: "d7")
    monkeypatch.setattr(ckpt, "find_last_step", lambda _d: 3)

    got = ckpt.load_optimizer_state("base", "cpu", rank=2, model_tag=None, step=None)
    assert torch.equal(got["x"], torch.tensor([1]))

    missing = ckpt.load_optimizer_state("base", "cpu", rank=99, model_tag="d7", step=3)
    assert missing is None
