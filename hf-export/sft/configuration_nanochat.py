
from transformers import PretrainedConfig

class NanoChatHFConfig(PretrainedConfig):
    model_type = "nanochat"
    def __init__(self, sequence_len=1024, vocab_size=50304, n_layer=12, n_head=6, n_kv_head=6, n_embd=768, **kwargs):
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd
        # HF compatibility aliases
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = sequence_len
