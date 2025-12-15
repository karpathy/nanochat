"""
Convert a nanochat checkpoint into a HuggingFace-style folder.

Usage (example):
python -m nanochat.to_hf --source base --output hf-export/base

Notes
- Assumes checkpoints live under ~/.cache/nanochat/<source>_checkpoints/ (same as training scripts).
- The exported model can be loaded with transformers via:
    AutoModelForCausalLM.from_pretrained(<export_dir>, trust_remote_code=True)
- KV cache is not implemented in the HF wrapper; generation works but is not incremental.
"""
import argparse
import os
import shutil
from typing import Optional

import torch
import torch.nn.functional as F
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError as exc:
    raise SystemExit(
        "transformers is required for HF export. Run `uv sync` (with the hf extra) first."
    ) from exc

from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer


class NanoChatHFConfig(PretrainedConfig):
    model_type = "nanochat"

    def __init__(
        self,
        sequence_len: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 6,
        n_kv_head: int = 6,
        n_embd: int = 768,
        **kwargs,
    ):
        # Don't tie embeddings; nanochat uses untied wte/lm_head
        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)
        self.sequence_len = sequence_len
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        self.n_embd = n_embd


class NanoChatHFForCausalLM(PreTrainedModel):
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
        # nanochat uses untied embeddings; override to no-op
        return

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,  # unused
        labels: Optional[torch.LongTensor] = None,
        past_key_values=None,  # not implemented
        **_: dict,
    ) -> CausalLMOutputWithPast:
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        logits = self.model(input_ids)
        loss = None
        if labels is not None:
            # Align shapes for CE: shift labels to match logits
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1
            )
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, "attention_mask": kwargs.get("attention_mask", None)}


def copy_tokenizer_files(output_dir: str):
    base_dir = get_base_dir()
    tokenizer_dir = os.path.join(base_dir, "tokenizer")
    if not os.path.isdir(tokenizer_dir):
        print(f"[to_hf] tokenizer directory not found at {tokenizer_dir}, skipping tokenizer export")
        return
    for name in os.listdir(tokenizer_dir):
        src = os.path.join(tokenizer_dir, name)
        dst = os.path.join(output_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    print(f"[to_hf] Copied tokenizer files from {tokenizer_dir} to {output_dir}")


def export_to_hf(source: str, output_dir: str, model_tag: Optional[str], step: Optional[int]):
    device = torch.device("cpu")
    model, tokenizer, meta = load_model(source, device=device, phase="eval", model_tag=model_tag, step=step)
    cfg_kwargs = meta["model_config"]
    hf_config = NanoChatHFConfig(**cfg_kwargs)
    hf_model = NanoChatHFForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict(), strict=True)

    os.makedirs(output_dir, exist_ok=True)
    hf_model.save_pretrained(output_dir, safe_serialization=False)
    # Best effort: drop tokenizer files alongside weights
    copy_tokenizer_files(output_dir)
    print(f"[to_hf] Exported {source} checkpoint to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Export nanochat checkpoint to HuggingFace format")
    parser.add_argument("--source", choices=["base", "mid", "sft", "rl"], default="base", help="Which checkpoint family to export")
    parser.add_argument("--model-tag", type=str, default=None, help="Model tag (e.g., d20). Defaults to largest available.")
    parser.add_argument("--step", type=int, default=None, help="Checkpoint step. Defaults to latest.")
    parser.add_argument("--output", type=str, default="hf-export", help="Output directory for HF files")
    args = parser.parse_args()

    export_to_hf(args.source, args.output, args.model_tag, args.step)


if __name__ == "__main__":
    main()
