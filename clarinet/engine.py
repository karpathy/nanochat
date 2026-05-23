"""
Clarinet inference engine — dual-pass IV-style token generation.

At every decode step we run the model twice:
  - Conditional pass: prompt prefixed with [BOS, <|src_reasoning|>, ...]
  - Unconditional pass: prompt prefixed with [BOS, <|src_unknown|>, ...]

Then combine logits as

    logit_iv = logit_uncond + w * s * (logit_cond - logit_uncond)

where
  - w is the standard guidance weight (`iv_weight`)
  - s is the Wald-style scale factor (`wald_scale`); s=1 reduces to vanilla CFG.
    A learned-from-hidden-states estimator for s is left as future work
    (would require exposing pre-lm_head activations from gpt.py).

Two parallel KVCaches are maintained — one per conditioning. Both consume the
same sampled-token sequence after combination, so they stay in lock-step.
"""

import torch

from nanochat.engine import Engine, KVCache, RowState, sample_next_token, use_calculator


class ClarinetEngine(Engine):

    SRC_REASONING = "<|src_reasoning|>"
    SRC_UNKNOWN = "<|src_unknown|>"

    def _prefix_with_marker(self, tokens, marker_id, bos_id):
        # Splice marker into the prompt to mirror training-time layout
        # ([BOS, marker, ...]). If the prompt doesn't already start with BOS
        # (uncommon — chat/render_conversation always prepends one), add it.
        if tokens and tokens[0] == bos_id:
            return [tokens[0], marker_id, *tokens[1:]]
        return [bos_id, marker_id, *tokens]

    @staticmethod
    def combine_logits(logit_cond, logit_uncond, iv_weight, wald_scale):
        """
        Wald-scaled IV combine: logit_uncond + w*s*(logit_cond - logit_uncond).
        w=0 returns the unconditional logits; w=1, s=1 returns the conditional
        logits exactly. Extracted as a static method for unit-testability.
        """
        return logit_uncond + iv_weight * wald_scale * (logit_cond - logit_uncond)

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0,
                 top_k=None, seed=42, iv_weight=1.5, wald_scale=1.0):
        """
        IV-conditioned generation. iv_weight=0 recovers the unconditional
        distribution; iv_weight=1, wald_scale=1 recovers the conditional
        distribution exactly. iv_weight in [1.5, 2.5] is the expected
        useful range.

        Forwards everything else (tool-use state machine, forced tokens,
        completion detection) from the upstream Engine.generate().
        """
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        src_reasoning_id = get_special(self.SRC_REASONING)
        src_unknown_id = get_special(self.SRC_UNKNOWN)

        cond_tokens = self._prefix_with_marker(tokens, src_reasoning_id, bos)
        uncond_tokens = self._prefix_with_marker(tokens, src_unknown_id, bos)
        # The two variants differ only at position 1 (the marker), so they have
        # equal length; the decode loop relies on that.
        assert len(cond_tokens) == len(uncond_tokens)
        prompt_len = len(cond_tokens)

        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}

        def run_prefill(prompt_tokens):
            cache = KVCache(batch_size=1, seq_len=prompt_len, device=device, dtype=dtype, **kv_model_kwargs)
            ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
            logits = self.model.forward(ids, kv_cache=cache)
            logits = logits[:, -1, :].expand(num_samples, -1)
            return cache, logits

        cond_prefill, cond_logits = run_prefill(cond_tokens)
        uncond_prefill, uncond_logits = run_prefill(uncond_tokens)

        kv_length_hint = (prompt_len + max_tokens) if max_tokens is not None else self.model.config.sequence_len

        def expand_to_decode_cache(prefill_cache):
            decode_cache = KVCache(batch_size=num_samples, seq_len=kv_length_hint, device=device, dtype=dtype, **kv_model_kwargs)
            decode_cache.prefill(prefill_cache)
            return decode_cache

        cond_cache = expand_to_decode_cache(cond_prefill)
        uncond_cache = expand_to_decode_cache(uncond_prefill)
        del cond_prefill, uncond_prefill

        logits = self.combine_logits(cond_logits, uncond_logits, iv_weight, wald_scale)

        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        num_generated = 0
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break
            if all(state.completed for state in row_states):
                break

            next_ids = sample_next_token(logits, rng, temperature, top_k)
            sampled_tokens = next_ids[:, 0].tolist()

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                is_forced = len(state.forced_tokens) > 0
                token_masks.append(0 if is_forced else 1)
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                state.current_tokens.append(next_token)
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            yield token_column, token_masks
            num_generated += 1

            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            cond_logits = self.model.forward(ids, kv_cache=cond_cache)[:, -1, :]
            uncond_logits = self.model.forward(ids, kv_cache=uncond_cache)[:, -1, :]
            logits = self.combine_logits(cond_logits, uncond_logits, iv_weight, wald_scale)
