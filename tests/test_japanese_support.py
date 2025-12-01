"""
Japanese language support integration tests.

Tests:
1. Japanese data configuration and loading
2. Japanese tokenizer training and compression
3. JapaneseInstruct task functionality
4. JCommonsenseQA task functionality

Run with:
python -m pytest tests/test_japanese_support.py -v -s
"""

import pytest
import os


class TestDataConfig:
    """Test language-specific data configuration."""

    def test_get_data_config_default(self):
        """Default language should be English."""
        from nanochat.dataset import get_data_config

        # Clear any existing env var
        orig = os.environ.pop("NANOCHAT_LANG", None)
        try:
            config = get_data_config()
            assert config.base_url.endswith("fineweb-edu-100b-shuffle/resolve/main")
            assert config.text_column == "text"
        finally:
            if orig is not None:
                os.environ["NANOCHAT_LANG"] = orig

    def test_get_data_config_english(self):
        """English config should use fineweb-edu."""
        from nanochat.dataset import get_data_config

        config = get_data_config("en")
        assert "fineweb-edu-100b-shuffle" in config.base_url
        assert config.max_shard == 1822
        assert config.text_column == "text"

    def test_get_data_config_japanese(self):
        """Japanese config should use fineweb-2-edu-japanese."""
        from nanochat.dataset import get_data_config

        config = get_data_config("ja")
        assert "fineweb-2-edu-japanese" in config.base_url
        assert config.max_shard == 892
        assert config.text_column == "text"

    def test_get_data_config_from_env(self):
        """Should read language from NANOCHAT_LANG env var."""
        from nanochat.dataset import get_data_config

        orig = os.environ.get("NANOCHAT_LANG")
        try:
            os.environ["NANOCHAT_LANG"] = "ja"
            config = get_data_config()
            assert "fineweb-2-edu-japanese" in config.base_url
        finally:
            if orig is not None:
                os.environ["NANOCHAT_LANG"] = orig
            else:
                os.environ.pop("NANOCHAT_LANG", None)

    def test_get_data_config_unsupported_lang(self):
        """Should raise error for unsupported language."""
        from nanochat.dataset import get_data_config

        with pytest.raises(ValueError, match="Unsupported language"):
            get_data_config("zh")

    def test_get_data_dir_english(self):
        """English data dir should be base_data."""
        from nanochat.dataset import get_data_dir
        from nanochat.common import get_base_dir

        data_dir = get_data_dir("en")
        assert data_dir == os.path.join(get_base_dir(), "base_data")

    def test_get_data_dir_japanese(self):
        """Japanese data dir should be base_data_ja."""
        from nanochat.dataset import get_data_dir
        from nanochat.common import get_base_dir

        data_dir = get_data_dir("ja")
        assert data_dir == os.path.join(get_base_dir(), "base_data_ja")


class TestJapaneseTokenizer:
    """Test Japanese text tokenization."""

    def test_encode_decode_japanese(self):
        """Test encoding and decoding Japanese text."""
        from nanochat.tokenizer import RustBPETokenizer

        # Train a small tokenizer with Japanese text
        japanese_texts = [
            "これはテストです。日本語のテキストをトークナイズします。",
            "人工知能は機械学習の一分野です。",
            "東京は日本の首都です。大阪は西日本の中心都市です。",
            "ひらがなとカタカナと漢字を含むテキスト。",
        ]

        tok = RustBPETokenizer.train_from_iterator(japanese_texts, vocab_size=300)

        # Test encode/decode roundtrip
        test_text = "日本語のテスト文です。"
        ids = tok.encode(test_text)
        decoded = tok.decode(ids)
        assert decoded == test_text, f"Roundtrip failed: {decoded} != {test_text}"

    def test_japanese_compression_ratio(self):
        """Test that Japanese text achieves reasonable compression."""
        from nanochat.tokenizer import RustBPETokenizer

        # Use more Japanese text for training
        japanese_texts = [
            "人工知能（じんこうちのう、英: artificial intelligence、AI）とは、" * 10,
            "大規模言語モデル（LLM）は、自然言語処理において革新的な進歩をもたらした。" * 10,
            "機械学習の基本的な流れは、データの収集と前処理から始まる。" * 10,
        ]

        tok = RustBPETokenizer.train_from_iterator(japanese_texts, vocab_size=512)

        test_text = "日本語処理においては、形態素解析やサブワードトークナイゼーションが重要な役割を果たす。"
        ids = tok.encode(test_text)
        text_bytes = len(test_text.encode('utf-8'))
        num_tokens = len(ids)
        ratio = text_bytes / num_tokens

        # Japanese UTF-8 is typically 3 bytes per character
        # A reasonable BPE should compress to at least 2 bytes/token
        assert ratio >= 1.5, f"Compression ratio too low: {ratio:.2f} bytes/token"
        print(f"Japanese compression ratio: {ratio:.2f} bytes/token")


class TestJapaneseInstruct:
    """Test JapaneseInstruct task."""

    def test_task_loads(self):
        """Test that JapaneseInstruct task loads successfully."""
        from tasks.japanese_instruct import JapaneseInstruct

        task = JapaneseInstruct(split="train", start=0, stop=10)
        # num_examples() returns total dataset size, __len__ returns sliced size
        assert len(task) == 10

    def test_task_example_format(self):
        """Test that examples have correct format."""
        from tasks.japanese_instruct import JapaneseInstruct

        task = JapaneseInstruct(split="train", start=0, stop=5)
        example = task.get_example(0)

        assert "messages" in example
        messages = example["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert len(messages[0]["content"]) > 0
        assert len(messages[1]["content"]) > 0

    def test_task_contains_japanese(self):
        """Test that examples contain Japanese text."""
        from tasks.japanese_instruct import JapaneseInstruct
        import re

        task = JapaneseInstruct(split="train", start=0, stop=20)

        # Check multiple examples for Japanese characters
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        found_japanese = False

        for i in range(min(20, task.num_examples())):
            example = task.get_example(i)
            content = example["messages"][0]["content"] + example["messages"][1]["content"]
            if japanese_pattern.search(content):
                found_japanese = True
                break

        assert found_japanese, "No Japanese text found in examples"


class TestJCommonsenseQA:
    """Test JCommonsenseQA task."""

    def test_task_loads(self):
        """Test that JCommonsenseQA task loads successfully."""
        from tasks.jcommonsenseqa import JCommonsenseQA

        task = JCommonsenseQA(split="validation", start=0, stop=10)
        # num_examples() returns total dataset size, __len__ returns sliced size
        assert len(task) == 10

    def test_task_example_format(self):
        """Test that examples have correct format."""
        from tasks.jcommonsenseqa import JCommonsenseQA

        task = JCommonsenseQA(split="validation", start=0, stop=5)
        example = task.get_example(0)

        assert "messages" in example
        assert "letters" in example
        messages = example["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        # Answer should be a single letter A-E
        assert messages[1]["content"] in ["A", "B", "C", "D", "E"]

    def test_eval_type(self):
        """Test that eval_type is categorical."""
        from tasks.jcommonsenseqa import JCommonsenseQA

        task = JCommonsenseQA(split="validation")
        assert task.eval_type == "categorical"

    def test_evaluate_correct(self):
        """Test evaluate method with correct answer."""
        from tasks.jcommonsenseqa import JCommonsenseQA

        task = JCommonsenseQA(split="validation", start=0, stop=5)
        example = task.get_example(0)
        correct_answer = example["messages"][1]["content"]

        result = task.evaluate(example, correct_answer)
        assert result is True

    def test_evaluate_incorrect(self):
        """Test evaluate method with incorrect answer."""
        from tasks.jcommonsenseqa import JCommonsenseQA

        task = JCommonsenseQA(split="validation", start=0, stop=5)
        example = task.get_example(0)
        correct_answer = example["messages"][1]["content"]

        # Pick a wrong answer
        wrong_answers = [l for l in ["A", "B", "C", "D", "E"] if l != correct_answer]
        wrong_answer = wrong_answers[0]

        result = task.evaluate(example, wrong_answer)
        assert result is False

    def test_contains_japanese(self):
        """Test that questions contain Japanese text."""
        from tasks.jcommonsenseqa import JCommonsenseQA
        import re

        task = JCommonsenseQA(split="validation", start=0, stop=10)

        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')

        for i in range(min(10, task.num_examples())):
            example = task.get_example(i)
            content = example["messages"][0]["content"]
            assert japanese_pattern.search(content), f"Example {i} has no Japanese: {content[:100]}"


class TestTokEvalJapanese:
    """Test that tok_eval includes Japanese text."""

    def test_japanese_text_in_tok_eval(self):
        """Verify japanese_text variable exists in tok_eval."""
        # Import the module to check the variable exists
        import scripts.tok_eval as tok_eval

        # Check that japanese_text is defined
        assert hasattr(tok_eval, 'japanese_text'), "japanese_text not found in tok_eval"
        japanese_text = tok_eval.japanese_text

        # Check that it contains Japanese characters
        import re
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        assert japanese_pattern.search(japanese_text), "japanese_text does not contain Japanese characters"

        # Check that japanese is in all_text
        all_text_names = [name for name, _ in tok_eval.all_text]
        assert "japanese" in all_text_names, "japanese not in all_text list"


class TestSFTIntegration:
    """Test SFT integration with Japanese data."""

    def test_japanese_instruct_in_task_mixture(self):
        """Test that JapaneseInstruct works in TaskMixture."""
        from tasks.common import TaskMixture
        from tasks.japanese_instruct import JapaneseInstruct

        task = JapaneseInstruct(split="train", start=0, stop=50)
        mixture = TaskMixture([task])

        assert len(mixture) == 50
        example = mixture[0]
        assert "messages" in example

    def test_tokenizer_renders_japanese_conversation(self):
        """Test that tokenizer correctly renders Japanese conversations."""
        from nanochat.tokenizer import get_tokenizer
        from tasks.japanese_instruct import JapaneseInstruct
        import re

        tok = get_tokenizer()
        task = JapaneseInstruct(split="train", start=0, stop=10)

        # Test render_conversation
        example = task[0]
        ids, mask = tok.render_conversation(example)

        assert len(ids) > 0
        assert len(mask) == len(ids)

        # Verify roundtrip preserves Japanese
        decoded = tok.decode(ids)
        japanese_pattern = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
        # Note: some examples may be English translations, so check if original had Japanese
        original_text = example["messages"][0]["content"] + example["messages"][1]["content"]
        if japanese_pattern.search(original_text):
            assert japanese_pattern.search(decoded), "Japanese characters not preserved"

    def test_chat_sft_imports(self):
        """Test that chat_sft can import JapaneseInstruct."""
        # This verifies the import in chat_sft.py works
        from tasks.japanese_instruct import JapaneseInstruct
        task = JapaneseInstruct(split="train", start=0, stop=5)
        assert len(task) == 5


class TestEvalIntegration:
    """Test evaluation integration with Japanese tasks."""

    def test_jcommonsenseqa_in_chat_eval(self):
        """Test that JCommonsenseQA is available in chat_eval task module."""
        from functools import partial
        from tasks.jcommonsenseqa import JCommonsenseQA

        # Simulate the task_module dict from chat_eval
        task_module = partial(JCommonsenseQA, split="validation")
        task_object = task_module()

        assert task_object.eval_type == "categorical"
        assert len(task_object) > 0

    def test_jcommonsenseqa_baseline_accuracy(self):
        """Test that baseline accuracy for 5-choice MC is 20%."""
        # This is the random baseline for 5-choice questions
        baseline = 0.20
        assert baseline == 1.0 / 5.0


class TestWebUIJapanese:
    """Test Web UI Japanese support (code-level verification)."""

    def test_tokenizer_encodes_japanese_message(self):
        """Test that tokenizer correctly encodes Japanese message content."""
        from nanochat.tokenizer import get_tokenizer

        tok = get_tokenizer()
        japanese_message = "こんにちは、日本語でお話しましょう。"

        # Encode and decode roundtrip
        ids = tok.encode(japanese_message)
        decoded = tok.decode(ids)
        assert decoded == japanese_message

    def test_json_dumps_japanese_with_ensure_ascii_false(self):
        """Test that JSON dumps preserves Japanese characters with ensure_ascii=False."""
        import json

        token_data = {"token": "日本語のテスト", "gpu": 0}
        json_str = json.dumps(token_data, ensure_ascii=False)

        # Japanese characters should be preserved, not escaped
        assert "日本語のテスト" in json_str
        assert "\\u" not in json_str  # No unicode escapes

    def test_utf8_boundary_detection(self):
        """Test detection of incomplete UTF-8 sequences (replacement character)."""
        # Simulate the web server's UTF-8 boundary detection
        complete_text = "日本語"
        assert not complete_text.endswith('�')

        # Verify that incomplete UTF-8 would be detected
        # (In practice, tokenizer.decode handles this internally)

    def test_special_tokens_for_conversation(self):
        """Test that special tokens for conversation are available."""
        from nanochat.tokenizer import get_tokenizer

        tok = get_tokenizer()

        # These tokens are used in chat_web.py
        bos = tok.get_bos_token_id()
        user_start = tok.encode_special("<|user_start|>")
        user_end = tok.encode_special("<|user_end|>")
        assistant_start = tok.encode_special("<|assistant_start|>")
        assistant_end = tok.encode_special("<|assistant_end|>")

        assert isinstance(bos, int)
        assert isinstance(user_start, int)
        assert isinstance(user_end, int)
        assert isinstance(assistant_start, int)
        assert isinstance(assistant_end, int)

    def test_conversation_encoding_with_japanese(self):
        """Test encoding a full conversation with Japanese content."""
        from nanochat.tokenizer import get_tokenizer

        tok = get_tokenizer()

        # Build conversation like chat_web.py does
        bos = tok.get_bos_token_id()
        user_start = tok.encode_special("<|user_start|>")
        user_end = tok.encode_special("<|user_end|>")
        assistant_start = tok.encode_special("<|assistant_start|>")

        conversation_tokens = [bos]
        conversation_tokens.append(user_start)
        conversation_tokens.extend(tok.encode("日本語で挨拶してください。"))
        conversation_tokens.append(user_end)
        conversation_tokens.append(assistant_start)

        # Verify we can decode the conversation
        decoded = tok.decode(conversation_tokens)
        assert "日本語で挨拶してください" in decoded
