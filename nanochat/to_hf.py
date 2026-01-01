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
import json
import shutil
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
try:
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.generation.utils import GenerationMixin
except ImportError as exc:
    raise SystemExit(
        "transformers is required for HF export. Run `uv sync` (with the hf extra) first."
    ) from exc
    

from nanochat.checkpoint_manager import load_model
from nanochat.gpt import GPT, GPTConfig
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer

CHAT_TEMPLATE = (
    "<|bos|>{% for message in messages %}{% if message['role'] == 'user' %}"
    "<|user_start|>{{ message['content'] }}<|user_end|>"
    "{% elif message['role'] == 'assistant' %}"
    "<|assistant_start|>{{ message['content'] }}<|assistant_end|>"
    "{% endif %}{% endfor %}<|assistant_start|>"
)


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
        # HF compatibility aliases
        self.num_hidden_layers = n_layer
        self.hidden_size = n_embd
        self.num_attention_heads = n_head
        self.num_key_value_heads = n_kv_head
        self.max_position_embeddings = sequence_len


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

def write_hf_code(output_dir: str):
    """
    Copy the HF-compatible python modules into the export dir.
    Prefer the checked-in template under hf-export/std; fallback to writing nothing
    if not present (export will still work if trust_remote_code imports from local FS).
    """
    template_dir = Path(__file__).resolve().parent.parent / "hf-export" / "std"
    targets = ["configuration_nanochat.py", "modeling_nanochat.py", "tokenization_nanochat.py", "gpt.py"]
    if template_dir.is_dir():
        for name in targets:
            src = template_dir / name
            if src.exists():
                shutil.copy2(src, Path(output_dir) / name)
    else:
        print(f"[to_hf] Template dir not found at {template_dir}, skipping python module copy")


def export_to_hf(source: str, output_dir: str, model_tag: Optional[str], step: Optional[int]):
    device = torch.device("cpu")
    model, tokenizer, meta = load_model(source, device=device, phase="eval", model_tag=model_tag, step=step)
    cfg_kwargs = meta["model_config"]
    hf_config = NanoChatHFConfig(**cfg_kwargs)
    hf_model = NanoChatHFForCausalLM(hf_config)
    hf_model.model.load_state_dict(model.state_dict(), strict=True)

    os.makedirs(output_dir, exist_ok=True)
    # Tell transformers how to load custom code when trust_remote_code=True
    hf_model.config.auto_map = {
        "AutoConfig": "configuration_nanochat.NanoChatHFConfig",
        "AutoModelForCausalLM": "modeling_nanochat.NanoChatHFForCausalLM",
        "AutoTokenizer": ["tokenization_nanochat.NanoChatTokenizer", None],
    }
    hf_model.config.tokenizer_class = "NanoChatTokenizer"
    hf_model.config.architectures = ["NanoChatHFForCausalLM"]
    # Bind special token ids/strings from tokenizer
    bos_id = tokenizer.encode_special("<|bos|>")
    eos_id = tokenizer.encode_special("<|assistant_end|>")
    hf_model.config.bos_token_id = bos_id
    hf_model.config.eos_token_id = eos_id
    hf_model.config.pad_token_id = bos_id

    hf_model.save_pretrained(output_dir, safe_serialization=False)
    # Best effort: drop tokenizer files alongside weights
    copy_tokenizer_files(output_dir)
    # Write tokenizer_config with chat template and special tokens
    tok_cfg = {
        "chat_template": CHAT_TEMPLATE,
        "bos_token": "<|bos|>",
        "eos_token": "<|assistant_end|>",
        "pad_token": "<|bos|>",
    }
    with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
        json.dump(tok_cfg, f, indent=2)
    print(f"[to_hf] Exported {source} checkpoint to {output_dir}")
    write_hf_code(output_dir)

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
