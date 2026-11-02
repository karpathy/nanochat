"""CORE continuation spans, using offline RustBPE and a real tiny CPU GPT.

Run: TORCH_COMPILE_DISABLE=1 python -m pytest tests/test_core_eval.py -q
The ARC fixture is from nanochat's public eval_bundle.zip, row 254 and the
exact ten examples selected by Random(1234 + 254). The schema fixtures are
Winograd rows 26 and 40. Fixture sources and licenses are recorded alongside
the data; no model download is used.
"""
import json
import random
from pathlib import Path

import pytest
import torch

from nanochat import core_eval
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer


@pytest.fixture(scope="module")
def native():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    tokenizer = RustBPETokenizer.train_from_iterator(
        iter(['Which sentence? a a b phrase more sentence text ' * 20]), 272,
    )
    with torch.random.fork_rng():
        torch.manual_seed(3)
        model = GPT(GPTConfig(vocab_size=tokenizer.get_vocab_size(), n_layer=2,
                              n_head=2, n_kv_head=2, n_embd=32, sequence_len=2048))
        model.init_weights()
    model.eval()
    yield model, tokenizer
    torch.set_num_threads(previous_threads)


def capture_example(monkeypatch, native, data, index=0, mode='multiple_choice', shots=0, delimiter=' '):
    model, tokenizer = native
    captured = {}
    original_batch = getattr(core_eval, {'multiple_choice': 'batch_sequences_mc',
                                        'schema': 'batch_sequences_schema',
                                        'language_modeling': 'batch_sequences_lm'}[mode])
    original_forward = core_eval.forward_model

    def batch(*args, **kwargs):
        captured['batch'] = original_batch(*args, **kwargs)
        return captured['batch']

    def forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        captured['losses'] = result[0]
        return result

    monkeypatch.setattr(core_eval, original_batch.__name__, batch)
    monkeypatch.setattr(core_eval, 'forward_model', forward)
    core_eval.evaluate_example(index, model, tokenizer, data, torch.device('cpu'), {
        'task_type': mode, 'num_fewshot': shots, 'continuation_delimiter': delimiter,
    })
    return captured


@pytest.mark.parametrize('choices', [['a', 'a b'], ['a cat', 'a dog']])
def test_mc_scores_the_whole_answer_not_only_the_different_suffix(monkeypatch, native, choices):
    result = capture_example(monkeypatch, native, [
        {'query': 'Which sentence?', 'choices': choices, 'gold': 0},
    ])
    _, tokenizer = native
    tokens, starts, ends = result['batch']
    for i, (ids, start, end, choice) in enumerate(zip(tokens, starts, ends, choices)):
        assert ids[0] == tokenizer.get_bos_token_id()
        assert start > 0 and end > start
        assert tokenizer.decode(ids[start:end]).lstrip() == choice
        assert torch.isfinite(result['losses'][i, start-1:end-1].mean())


def test_mc_keeps_joint_tokens_when_the_answer_merges_with_the_delimiter(monkeypatch, native):
    item = {'query': 'Which sentence?', 'choices': ['a', 'a b'], 'gold': 0}
    _, tokenizer = native
    prompts = core_eval.render_prompts_mc(item, ' ')
    context_ids = tokenizer('Which sentence? ', prepend=tokenizer.get_bos_token_id())
    expected = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    assert expected[0][:len(context_ids)] != context_ids
    result = capture_example(monkeypatch, native, [item])
    tokens, starts, ends = result['batch']
    assert tokens == expected
    assert tokenizer.decode(tokens[0][starts[0]:ends[0]]) == ' a'


def test_arc_easy_actual_ten_shot_row_has_no_empty_answer_span(monkeypatch, native):
    fixture = json.loads((Path(__file__).parent / 'fixtures/core_arc_easy_prefix.json').read_text())
    index = fixture['index']
    selected = random.Random(1234 + index).sample(
        [i for i in range(fixture['dataset_size']) if i != index], 10,
    )
    assert selected == fixture['fewshot_indices']
    data = [None] * fixture['dataset_size']
    data[index] = fixture['item']
    for position, example in zip(selected, fixture['fewshot_examples']):
        data[position] = example
    result = capture_example(monkeypatch, native, data, index=index, shots=10, delimiter=fixture['delimiter'])
    _, tokenizer = native
    tokens, starts, ends = result['batch']
    expected = tokenizer(core_eval.render_prompts_mc(fixture['item'], fixture['delimiter'],
                                                   fixture['fewshot_examples']),
                         prepend=tokenizer.get_bos_token_id())
    assert tokens == expected
    for ids, start, end, choice in zip(tokens, starts, ends, fixture['item']['choices']):
        assert end > start > 0
        assert tokenizer.decode(ids[start:end]).lstrip() == choice
    assert all(torch.isfinite(result['losses'][i, start-1:end-1].mean())
               for i, (start, end) in enumerate(zip(starts, ends)))


@pytest.mark.parametrize('mode,item,answer', [
    ('schema', {'context_options': ['The cat', 'The dog'], 'continuation': 'hello', 'gold': 0}, 'hello'),
    ('language_modeling', {'context': 'The answer is:', 'continuation': '42'}, '42'),
])
def test_other_core_modes_keep_their_continuation(monkeypatch, native, mode, item, answer):
    result = capture_example(monkeypatch, native, [item], mode=mode)
    _, tokenizer = native
    tokens, starts, ends = result['batch']
    for ids, start, end in zip(tokens, starts, ends):
        assert start > 0 and end > start
        assert tokenizer.decode(ids[start:end]).lstrip() == answer


def test_mc_does_not_score_unmerged_context_for_a_different_answer(monkeypatch, native):
    # This native tokenizer merges the space into " a", but encodes " x" as
    # two tokens. Each answer is one joint token; the latter's space is context.
    item = {'query': 'Which sentence?', 'choices': ['a', 'x'], 'gold': 0}
    result = capture_example(monkeypatch, native, [item])
    tokens, starts, ends = result['batch']
    _, tokenizer = native
    assert [tokenizer.decode(ids[-1:]) for ids in tokens] == [' a', 'x']
    for i, (ids, start, end) in enumerate(zip(tokens, starts, ends)):
        assert end - start == 1
        expected_loss = result['losses'][i, len(ids) - 2]
        actual_loss = result['losses'][i, start - 1:end - 1].mean()
        torch.testing.assert_close(actual_loss, expected_loss)


@pytest.mark.parametrize('row', [0, 1])
def test_actual_schema_rows_do_not_score_shared_context_words(monkeypatch, native, row):
    fixture = json.loads((Path(__file__).parent / 'fixtures/core_schema_shared_context.json').read_text())
    item = fixture['items'][row]['item']
    result = capture_example(monkeypatch, native, [item], mode='schema',
                             shots=fixture['configured_num_fewshot'], delimiter=fixture['delimiter'])
    _, tokenizer = native
    tokens, starts, ends = result['batch']
    expected_inputs = tokenizer(core_eval.render_prompts_schema(item, fixture['delimiter']),
                                prepend=tokenizer.get_bos_token_id())
    assert tokens == expected_inputs
    for ids, start, end in zip(tokens, starts, ends):
        assert start > 0 and end > start
        assert tokenizer.decode(ids[start:end]).lstrip() == item['continuation']
