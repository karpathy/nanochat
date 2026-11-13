"""
Memory preflight: estimate the peak of a training step and shrink the device batch size
until it fits, before anything large is allocated.

Aimed at MPS, where Apple unified memory OOMs on the (B, T, vocab) logits tensor that
nanochat/gpt.py builds in the lm_head. Perfect accounting is not possible there
(unified memory, fragmentation, compile caches), so the estimate is deliberately
pessimistic: it is cheaper to run a smaller batch than to lose an hour to an OOM.

Nothing here prints. Callers echo the returned report with the formatters below, so the
[auto-fit] lines stay identical across base_train / base_eval / chat_sft.
"""
import os

import torch

from nanochat.defaults import VOCAB_SIZE, DEFAULT_SEQ_LEN, MEMORY_FIT_MARGIN, MEMORY_ACTIVATION_FUDGE

GIB = 1024 ** 3
BYTES_PER_FP32 = 4

# Per parameter: weight + gradient + 2 optimizer moments, all fp32. Muon keeps a single
# momentum buffer, so budgeting 2 moments for every parameter is the conservative choice.
TRAIN_BYTES_PER_PARAM = 4 * BYTES_PER_FP32

# nanochat model geometry, mirrors build_model_meta() in scripts/base_train.py
ASPECT_RATIO = 64
HEAD_DIM = 128

# CPU has no device cap worth fitting against: hand back a cap no estimate can reach,
# which makes auto-fit a no-op there instead of a special case at every call site.
CPU_SOFT_CAP_BYTES = 1 << 60

# torch < 2.3 has no torch.mps.recommended_max_memory(); fall back to a fraction of system
# RAM, low enough to leave the OS and the rest of the machine their share of unified memory.
MPS_SYSTEM_RAM_FRACTION = 0.7
FALLBACK_SYSTEM_RAM_BYTES = 8 * GIB

MIN_DEVICE_BATCH_SIZE = 1
SHRINK_REASON = "estimated peak exceeds margin * available memory"

MARGIN_ENV_VAR = "NANOCHAT_MEMORY_FIT_MARGIN"

# Both allocators say this in the message; MPS has no dedicated exception class to catch.
OOM_MESSAGE_MARKER = "out of memory"


def _device_type(device):
    """Accept either a torch.device or a string like "mps" / "cuda:0"."""
    return getattr(device, "type", None) or str(device).split(":")[0]


def resolve_fit_margin():
    """The fraction of available memory a fit may use: the run-time override, else the default."""
    raw = os.environ.get(MARGIN_ENV_VAR)
    if raw is None:
        return MEMORY_FIT_MARGIN
    try:
        margin = float(raw)
    except ValueError:
        raise ValueError(
            f"{MARGIN_ENV_VAR}={raw!r} is not a number, so the memory budget cannot be computed. "
            f"Set it to a fraction of available device memory, e.g. {MARGIN_ENV_VAR}=0.5, or unset "
            f"it to use the default of {MEMORY_FIT_MARGIN} (see docs/RUNNING.md)."
        ) from None
    if not 0 < margin <= 1:
        raise ValueError(
            f"{MARGIN_ENV_VAR}={raw} is not a fraction of available device memory, so it cannot be "
            f"a memory budget. Set it above 0 and at most 1, e.g. {MARGIN_ENV_VAR}=0.5 "
            f"(see docs/RUNNING.md)."
        )
    return margin


def auto_fit_enabled(requested, device_type):
    """
    Resolve the --auto-fit-batch tri-state, where None means "decide from the device".

    On by default on mps, where the unified-memory OOM is the failure this exists for; off on
    cuda and cpu, so an unattended GPU run keeps exactly the batch size it was given.
    """
    if requested is None:
        return _device_type(device_type) == "mps"
    return requested


def _system_ram_bytes():
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return FALLBACK_SYSTEM_RAM_BYTES


def available_memory_bytes(device):
    """Bytes we are willing to treat as available on this device."""
    device_type = _device_type(device)
    if device_type == "cuda":
        free_bytes, _total_bytes = torch.cuda.mem_get_info()
        return free_bytes
    if device_type == "mps":
        recommended_max_memory = getattr(torch.mps, "recommended_max_memory", None)
        if recommended_max_memory is not None:
            return recommended_max_memory()
        return int(MPS_SYSTEM_RAM_FRACTION * _system_ram_bytes())
    return CPU_SOFT_CAP_BYTES


def _num_params(depth, vocab_size):
    """Parameter count at this depth, following build_model_meta() in scripts/base_train.py."""
    base_dim = depth * ASPECT_RATIO
    model_dim = ((base_dim + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
    per_layer = 12 * model_dim ** 2  # attn q,k,v,proj = 4*d^2; mlp 4x up and down = 8*d^2
    n_value_embeds = (depth + 1) // 2  # value embeddings sit on alternating layers, last always included
    embeddings = (2 + n_value_embeds) * vocab_size * model_dim  # wte, lm_head, value embeds
    return depth * per_layer + embeddings


def estimate_train_peak_bytes(depth, batch, seq_len=DEFAULT_SEQ_LEN, vocab_size=VOCAB_SIZE, fudge=MEMORY_ACTIVATION_FUDGE):
    """
    Conservative peak of one training step: weights + gradients + optimizer state, plus the
    fp32 (B, T, vocab) logits and `fudge` further copies of them.

    The fudge of 1.5 stands for the full-size logits buffers the forward holds alive at the
    same time - the fp32 upcast, the tanh softcap result and the cross-entropy workspace.
    Hidden-state activations are small next to those and are covered by the same slack.
    """
    weights_bytes = _num_params(depth, vocab_size) * TRAIN_BYTES_PER_PARAM
    logits_bytes = batch * seq_len * vocab_size * BYTES_PER_FP32
    return int(weights_bytes + (1 + fudge) * logits_bytes)


def fit_device_batch_size(requested_batch_size, depth, seq_len=DEFAULT_SEQ_LEN, device_type="cpu",
                          vocab_size=VOCAB_SIZE, available_bytes=None, margin=None):
    """
    Halve the device batch size until the estimated peak fits `margin` of available memory.

    Floors at 1: when even that does not fit, the report says so rather than claiming a fit.
    Returns (batch_size, report), the report carrying everything the [auto-fit] echo needs.

    `margin=None` resolves the run-time override, so every call site honours
    NANOCHAT_MEMORY_FIT_MARGIN without repeating the lookup.
    """
    margin = resolve_fit_margin() if margin is None else margin
    avail = available_memory_bytes(device_type) if available_bytes is None else available_bytes
    budget = margin * avail

    def estimate(batch):
        return estimate_train_peak_bytes(depth, batch, seq_len, vocab_size)

    batch_size = max(MIN_DEVICE_BATCH_SIZE, requested_batch_size)
    while estimate(batch_size) > budget and batch_size > MIN_DEVICE_BATCH_SIZE:
        batch_size = max(MIN_DEVICE_BATCH_SIZE, batch_size // 2)
    adjusted_peak = estimate(batch_size)
    report = {
        "original": requested_batch_size,
        "adjusted": batch_size,
        "est_peak": estimate(requested_batch_size),  # at the requested size: the number that forces a shrink
        "est_peak_adjusted": adjusted_peak,
        "avail": avail,
        "margin": margin,
        "fits": adjusted_peak <= budget,
        "reason": None if batch_size == requested_batch_size else SHRINK_REASON,
    }
    return batch_size, report


def _gib(num_bytes):
    return f"{num_bytes / GIB:.1f}GiB"


def _over_budget(est_peak, report):
    return f"est_peak={_gib(est_peak)} > margin*avail={report['margin']:g}*{_gib(report['avail'])}"


def format_auto_fit_line(report):
    """The contract line for a fit, in the frozen [auto-fit] shape."""
    if report["adjusted"] != report["original"]:
        return (f"[auto-fit] device_batch_size: original/default={report['original']} → adjusted={report['adjusted']} "
                f"(reason: {_over_budget(report['est_peak'], report)})")
    if report["fits"]:
        return f"[auto-fit] device_batch_size={report['original']} (unchanged; estimate fits)"
    return (f"[auto-fit] device_batch_size={report['original']} (unchanged; already at the minimum, "
            f"{_over_budget(report['est_peak'], report)}; OOM is likely)")


def format_memory_fit_hint(report):
    """Plain-language enrichment line: what the budget is and what it cost."""
    budget = f"≤{report['margin'] * 100:g}% of ~{report['avail'] / GIB:.0f}GiB"
    if report["adjusted"] != report["original"]:
        return f"memory: aiming to use {budget}; batch {report['original']}→{report['adjusted']} so logits fit"
    return f"memory: aiming to use {budget}; batch {report['original']} fits as-is"


def format_batch_size_precedence_line(sources):
    """
    Where the device batch size ended up coming from, as (layer, value) pairs ordered from
    the weakest layer to the strongest, e.g. [("CLI_default", 32), ("profile", 8), ("fitted", 2)].
    """
    chain = " → ".join(f"{layer}={value}" for layer, value in sources)
    return f"[auto-fit] device_batch_size precedence: {chain}"


def auto_fit_echo_lines(report, sources):
    """
    Every line a call site echoes after a fit, in emission order: the [auto-fit] contract line,
    the precedence chain when the layers disagree, then the single plain-language enrichment.
    """
    lines = [format_auto_fit_line(report)]
    if len({value for _layer, value in sources}) > 1:
        lines.append(format_batch_size_precedence_line(sources))
    lines.append(format_memory_fit_hint(report))
    return lines


def is_out_of_memory_error(exception):
    """
    True for the allocator exhaustion a smaller batch can recover from.

    CUDA raises torch.cuda.OutOfMemoryError, MPS only a plain RuntimeError, and the CPU
    allocator a MemoryError, so the message is the one signal all three share.
    """
    if isinstance(exception, (MemoryError, torch.cuda.OutOfMemoryError)):
        return True
    return isinstance(exception, RuntimeError) and OOM_MESSAGE_MARKER in str(exception).lower()


def format_oom_retry_line(original_batch_size, adjusted_batch_size):
    return (f"[auto-fit] OOM recovery: device_batch_size: original/default={original_batch_size} → "
            f"adjusted={adjusted_batch_size} (reason: out of memory after fit; cache emptied, retrying once)")


def empty_device_cache(device):
    """Release cached blocks so the next allocation sees the freed memory (no-op on cpu)."""
    device_type = _device_type(device)
    if device_type == "cuda":
        torch.cuda.empty_cache()
    if device_type == "mps":
        torch.mps.empty_cache()


def oom_rehalve_once(batch_size, device_type="cpu"):
    """
    Recovery for an OOM the estimate did not predict: free cached blocks and halve once more,
    so the caller can retry a single time. Raises at the floor, where nothing is left to halve.
    """
    if batch_size <= MIN_DEVICE_BATCH_SIZE:
        raise MemoryError(
            "out of memory at device_batch_size=1: this model does not fit this device. "
            "Lower --max-seq-len or --depth, close other GPU/MPS processes, or set "
            f"NANOCHAT_MEMORY_FIT_MARGIN below {MEMORY_FIT_MARGIN} to fit more conservatively "
            "(see docs/RUNNING.md)."
        )
    empty_device_cache(device_type)
    return max(MIN_DEVICE_BATCH_SIZE, batch_size // 2)


def training_module_after_oom_rehalve(uncompiled_model):
    """
    After an OOM rehalve changes device_batch_size mid-run, training must continue on the
    uncompiled module. torch.compile(..., dynamic=False) assumes fixed input shapes; the
    same pattern is used for eval/sample when shapes vary.
    """
    return uncompiled_model


def oom_after_rehalve_error(batch_size):
    """Actionable MemoryError when the single post-rehalve retry still OOMs (any batch size)."""
    return MemoryError(
        f"out of memory again after rehalve at device_batch_size={batch_size}: "
        "this model still does not fit this device. "
        "Lower --max-seq-len or --depth, close other GPU/MPS processes, or see docs/RUNNING.md."
    )
