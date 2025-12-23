
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_nanochat import NanoChatHFConfig
from .gpt import GPT, GPTConfig

class NanoChatHFForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = NanoChatHFConfig

    def __init__(self, config: NanoChatHFConfig):
        super().__init__(config)
        gpt_cfg = GPTConfig(
            sequence_len=config.sequence_len,
            vocab_size=config.vocab_size,
            n_layer=config.n_layer,
            n_head=config.n_head,
            n_kv_head=config.n_kv_head,
            n_embd=config.n_embd,
        )
        self.model = GPT(gpt_cfg)

    def get_input_embeddings(self):
        return self.model.transformer.wte

    def set_input_embeddings(self, value):
        self.model.transformer.wte = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def tie_weights(self):
        return

    def forward(self, input_ids=None, attention_mask=None, labels=None, past_key_values=None, **_):
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None, hidden_states=None, attentions=None)

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}
