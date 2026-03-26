from nanochat.tokenizer import RustBPETokenizer


class TinyTokenizer(RustBPETokenizer):
    def __init__(self):
        pass

    def encode_special(self, text):
        return {
            "<|bos|>": 1000,
            "<|user_start|>": 1001,
            "<|user_end|>": 1002,
            "<|assistant_start|>": 1003,
            "<|assistant_end|>": 1004,
            "<|python_start|>": 1005,
            "<|python_end|>": 1006,
            "<|output_start|>": 1007,
            "<|output_end|>": 1008,
        }[text]

    def get_bos_token_id(self):
        return self.encode_special("<|bos|>")

    def encode(self, text, *args, **kwargs):
        return [ord(ch) for ch in text]


def test_render_for_assistant_reply_respects_max_tokens():
    tokenizer = TinyTokenizer()
    conversation = {
        "messages": [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "abcdefghij"},
        ]
    }

    ids = tokenizer.render_for_assistant_reply(conversation, max_tokens=6)

    assert len(ids) == 6
    assert ids[-1] == tokenizer.encode_special("<|assistant_start|>")
