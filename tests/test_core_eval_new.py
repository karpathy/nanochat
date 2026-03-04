import random
from dataclasses import dataclass

import pytest
import torch

import nanochat.core_eval as core_eval


@dataclass
class ItemMC:
    query: str
    choices: list
    gold: int


@dataclass
class ItemSchema:
    context_options: list
    continuation: str
    gold: int


@dataclass
class ItemLM:
    context: str
    continuation: str
    gold: int = 0


class TinyTokenizer:
    def __init__(self, bos=99):
        self._bos = bos

    def get_bos_token_id(self):
        return self._bos

    def __call__(self, prompts, prepend=None):
        return self.encode(prompts, prepend=prepend)

    def encode(self, prompts, prepend=None):
        if isinstance(prompts, str):
            prompts = [prompts]
        out = []
        for p in prompts:
            ids = [len(x) % 11 + 1 for x in p.split()]
            if prepend is not None:
                ids = [prepend] + ids
            out.append(ids)
        return out


def test_prompt_renderers():
    mc_item = {"query": "Q?", "choices": ["A", "B"], "gold": 1}
    mc_shots = [ItemMC(query="Q0", choices=["x", "y"], gold=0)]
    prompts_mc = core_eval.render_prompts_mc(mc_item, " -> ", mc_shots)
    assert len(prompts_mc) == 2
    assert "Q0 -> x" in prompts_mc[0]

    schema_item = {"context_options": ["ctx1", "ctx2"], "continuation": "cont", "gold": 0}
    schema_shots = [ItemSchema(context_options=["ca", "cb"], continuation="x", gold=1)]
    prompts_schema = core_eval.render_prompts_schema(schema_item, " || ", schema_shots)
    assert len(prompts_schema) == 2
    assert "cb || x" in prompts_schema[0]

    lm_item = {"context": "c1  ", "continuation": "end", "gold": 0}
    lm_shots = [ItemLM(context="ctx ", continuation="tail")]
    prompts_lm = core_eval.render_prompts_lm(lm_item, " ## ", lm_shots)
    assert len(prompts_lm) == 2
    assert prompts_lm[0].endswith("##")
    assert prompts_lm[1].endswith("## end")


def test_sequence_helpers():
    assert core_eval.find_common_length([[1, 2, 3], [1, 2, 4]], direction="left") == 2
    assert core_eval.find_common_length([[1, 2, 3], [0, 2, 3]], direction="right") == 2
    assert core_eval.find_common_length([[7, 8], [7, 8]], direction="left") == 2

    stacked = core_eval.stack_sequences([[1, 2], [3]], pad_token_id=0)
    assert stacked.tolist() == [[1, 2], [3, 0]]


def test_batch_sequence_helpers():
    tok = TinyTokenizer(bos=5)
    prompts = ["aa bb", "aa bb cc"]
    tokens_mc, s_mc, e_mc = core_eval.batch_sequences_mc(tok, prompts)
    assert len(tokens_mc) == 2
    assert len(s_mc) == 2 and len(e_mc) == 2

    tokens_schema, s_schema, e_schema = core_eval.batch_sequences_schema(tok, prompts)
    assert len(tokens_schema) == 2
    assert len(s_schema) == 2 and len(e_schema) == 2

    lm_prompts = ["a b", "a b c d"]
    tokens_lm, s_lm, e_lm = core_eval.batch_sequences_lm(tok, lm_prompts)
    assert len(tokens_lm) == 1
    assert s_lm[0] < e_lm[0]

    bad = [[1, 2], [8, 9]]
    class BadTok:
        def get_bos_token_id(self):
            return 1

        def __call__(self, *args, **kwargs):
            del args, kwargs
            return bad

    with pytest.raises(AssertionError):
        core_eval.batch_sequences_lm(BadTok(), ["x", "y"])


def test_forward_model():
    class M:
        def __call__(self, input_ids):
            b, t = input_ids.shape
            v = 6
            logits = torch.zeros((b, t, v), dtype=torch.float32)
            for i in range(b):
                for j in range(t):
                    logits[i, j, (j + 1) % v] = 10.0
            return logits

    input_ids = torch.tensor([[1, 2, 3], [2, 1, 0]], dtype=torch.long)
    losses, preds = core_eval.forward_model(M(), input_ids)
    assert losses.shape == input_ids.shape
    assert torch.isnan(losses[:, -1]).all()
    assert preds.shape == input_ids.shape


def _simple_model_for_eval():
    class M:
        max_seq_len = 3

        def __call__(self, input_ids):
            b, t = input_ids.shape
            # Prefer token 3 always to create deterministic MC scores.
            logits = torch.zeros((b, t, 8), dtype=torch.float32)
            logits[..., 3] = 3.0
            return logits

    return M()


def _simple_model_no_crop():
    class M:
        max_seq_len = None

        def __call__(self, input_ids):
            b, t = input_ids.shape
            logits = torch.zeros((b, t, 8), dtype=torch.float32)
            logits[..., 3] = 3.0
            return logits

    return M()


def test_evaluate_example_main_paths(monkeypatch):
    tok = TinyTokenizer(bos=1)
    data_mc = [
        {"query": "q0", "choices": ["a", "b"], "gold": 0},
        {"query": "q1", "choices": ["c", "d"], "gold": 1},
        {"query": "q2", "choices": ["e", "f"], "gold": 0},
    ]
    meta_mc = {"task_type": "multiple_choice", "num_fewshot": 1, "continuation_delimiter": " => "}
    out_mc = core_eval.evaluate_example(0, _simple_model_for_eval(), tok, data_mc, "cpu", meta_mc)
    assert isinstance(out_mc, bool)

    data_schema = [
        {"context_options": ["ctx0", "ctx1"], "continuation": "next", "gold": 0},
        {"context_options": ["ctx2", "ctx3"], "continuation": "next", "gold": 1},
    ]
    meta_schema = {"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " -> "}
    out_schema = core_eval.evaluate_example(0, _simple_model_no_crop(), tok, data_schema, "cpu", meta_schema)
    assert isinstance(out_schema, bool)

    # LM path with explicit prefix check and correctness.
    class LMTokenizer(TinyTokenizer):
        def encode(self, prompts, prepend=None):
            if prompts == ["p0", "p1"]:
                return [[1, 2], [1, 2, 3]]
            return super().encode(prompts, prepend=prepend)

    lm_data = [{"context": "x", "continuation": "y", "gold": 0}]
    meta_lm = {"task_type": "language_modeling", "num_fewshot": 0, "continuation_delimiter": ""}
    monkeypatch.setattr(core_eval, "render_prompts_lm", lambda *_a, **_k: ["p0", "p1"])
    out_lm = core_eval.evaluate_example(0, _simple_model_for_eval(), LMTokenizer(), lm_data, "cpu", meta_lm)
    assert isinstance(out_lm, bool)

    with pytest.raises(ValueError):
        core_eval.evaluate_example(0, _simple_model_for_eval(), tok, data_mc, "cpu", {"task_type": "bad", "num_fewshot": 0, "continuation_delimiter": ""})


def test_evaluate_example_reaches_final_else(monkeypatch):
    # Make first task-type comparison pass once, then fail later so final else is executed.
    class FlakyType:
        def __init__(self):
            self.first = True

        def __eq__(self, other):
            if self.first and other == "multiple_choice":
                self.first = False
                return True
            return False

    tok = TinyTokenizer(bos=1)
    data_mc = [{"query": "q", "choices": ["a", "b"], "gold": 0}]
    meta = {"task_type": FlakyType(), "num_fewshot": 0, "continuation_delimiter": " :: "}
    with pytest.raises(ValueError):
        core_eval.evaluate_example(0, _simple_model_for_eval(), tok, data_mc, "cpu", meta)


def test_evaluate_task(monkeypatch):
    data = [{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}]
    task_meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": ""}

    monkeypatch.setattr(core_eval, "evaluate_example", lambda idx, *_a, **_k: (idx % 2) == 0)
    monkeypatch.setattr(core_eval.dist, "is_initialized", lambda: False)
    out = core_eval.evaluate_task(model=None, tokenizer=None, data=data, device="cpu", task_meta=task_meta)
    assert 0.0 <= out <= 1.0

    # Distributed branch.
    calls = {"barrier": 0, "reduce": 0}
    monkeypatch.setattr(core_eval.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(core_eval.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(core_eval.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(core_eval.dist, "barrier", lambda: calls.__setitem__("barrier", calls["barrier"] + 1))
    monkeypatch.setattr(core_eval.dist, "all_reduce", lambda *_a, **_k: calls.__setitem__("reduce", calls["reduce"] + 1))
    out2 = core_eval.evaluate_task(model=None, tokenizer=None, data=data, device="cpu", task_meta=task_meta)
    assert 0.0 <= out2 <= 1.0
    assert calls["barrier"] == 1
    assert calls["reduce"] == 1
