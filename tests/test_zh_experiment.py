import pyarrow as pa
import pyarrow.parquet as pq

from nanochat.dataloader import _document_batches
from nanochat.tokenizer import IncrementalTextDecoder
from scripts.prepare_zh_experiment_data import cjk_language_ratio, normalize_messages
from scripts.zh_eval import response_metrics, summarize


class ByteTokenizer:
    def decode(self, token_ids):
        return bytes(token_ids).decode("utf-8", errors="replace")


def test_incremental_decoder_hides_partial_utf8():
    decoder = IncrementalTextDecoder(ByteTokenizer())
    encoded = list("你".encode("utf-8"))
    assert decoder.push(encoded[0]) == ""
    assert decoder.push(encoded[1]) == ""
    assert decoder.push(encoded[2]) == "你"


def test_normalize_alpaca_messages():
    messages = normalize_messages({
        "instruction": "介绍机器学习",
        "input": "面向初学者",
        "output": "机器学习让计算机从数据中学习。",
    })
    assert messages == [
        {"role": "user", "content": "介绍机器学习\n\n面向初学者"},
        {"role": "assistant", "content": "机器学习让计算机从数据中学习。"},
    ]


def test_normalize_conversations_messages():
    messages = normalize_messages({
        "conversations": [
            {"from": "human", "value": "你好"},
            {"from": "gpt", "value": "你好！"},
        ]
    })
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_response_metrics_language_ratio():
    metrics = response_metrics("你好，this is a test")
    assert metrics["cjk_chars"] == 2
    assert metrics["latin_chars"] == 11
    assert not metrics["contains_replacement_character"]
    assert cjk_language_ratio("这是中文回答") == 1.0


def test_empty_response_does_not_pass_language_checks():
    records = [
        {"language": "zh", "metrics": response_metrics("")},
        {"language": "en", "metrics": response_metrics("")},
    ]
    summary = summarize(records)
    assert summary["zh_response_rate"] == 0.0
    assert summary["en_response_rate"] == 0.0


def test_document_batches_accept_custom_data_dir(tmp_path):
    train_path = tmp_path / "shard_00000.parquet"
    val_path = tmp_path / "shard_99999.parquet"
    pq.write_table(pa.table({"text": ["train-a", "train-b"]}), train_path)
    pq.write_table(pa.table({"text": ["val-a"]}), val_path)

    train_batches = _document_batches("train", None, tokenizer_batch_size=8, data_dir=tmp_path)
    val_batches = _document_batches("val", None, tokenizer_batch_size=8, data_dir=tmp_path)
    assert next(train_batches)[0] == ["train-a", "train-b"]
    assert next(val_batches)[0] == ["val-a"]


def test_custom_data_dir_requires_train_and_validation_shards(tmp_path):
    pq.write_table(pa.table({"text": ["only-shard"]}), tmp_path / "shard_00000.parquet")

    try:
        next(_document_batches("train", None, tokenizer_batch_size=8, data_dir=tmp_path))
    except AssertionError as error:
        assert "Custom data directory" in str(error)
    else:
        raise AssertionError("Expected a custom data directory with one shard to be rejected")
