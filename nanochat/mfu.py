import torch


def get_promised_flops_per_gpu():
    """
    Return best-effort dense BF16 Tensor/Matrix peak FLOPs for the active GPU.
    Returns:
        tuple[str, float, bool]: (device_name, flops_per_gpu, is_estimated)
            device_name is the CUDA-reported name for the active device.
            flops_per_gpu is the per-device BF16 dense peak in FLOPs.
            is_estimated is True when we fall back to heuristic defaults.
    """
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    name = props.name.lower()
    device_name = props.name

    def result(flops, estimated):
        return device_name, flops, estimated

    def has(*keywords: str) -> bool:
        return any(k in name for k in keywords)

    # --- NVIDIA Blackwell ---
    if has("gb200", "grace blackwell"):
        return result(2.5e15, False)       # GB200 dense BF16 ≈ 2.5 PFLOPS (5.0 PFLOPS sparse)
    if has("b200"):
        return result(2.25e15, False)      # B200 dense BF16 ≈ 2.25 PFLOPS (4.5 PFLOPS sparse)
    if has("b100"):
        return result(1.8e15, False)       # B100 dense BF16 ≈ 1.8 PFLOPS (3.5 PFLOPS sparse)

    # --- NVIDIA Hopper (H100/H200/H800) ---
    if has("h200"):
        if has("nvl", "pcie"):
            return result(836e12, False)   # H200 NVL/PCIe dense BF16 ≈ 836 TFLOPS
        return result(989e12, False)       # H200 SXM dense BF16 ≈ 989 TFLOPS

    if has("h100"):
        if has("nvl"):
            return result(835e12, False)   # H100 NVL dense BF16 ≈ 835 TFLOPS
        if has("pcie"):
            return result(756e12, False)   # H100 PCIe dense BF16 ≈ 756 TFLOPS
        return result(989e12, False)       # H100 SXM dense BF16 ≈ 989 TFLOPS

    if has("h800"):
        if has("nvl"):
            return result(989e12, False)   # H800 NVLink dense BF16 ≈ 989 TFLOPS
        return result(756e12, False)       # H800 PCIe dense BF16 ≈ 756 TFLOPS

    # --- NVIDIA Ampere data center / export variants ---
    if has("a100", "pg506"):
        return result(312e12, False)       # A100 SXM dense BF16 = 312 TFLOPS
    if has("a800"):
        return result(312e12, False)       # A800 dense BF16 ≈ 312 TFLOPS (Ampere-class)

    # Useful Ada data-center cards
    if has("l40s", "l40-s", "l40 s"):
        return result(362e12, False)       # L40S dense BF16 ≈ 362 TFLOPS
    if has(" l4", " l4 ", "nvidia l4", " l4-") or name.endswith(" l4"):
        return result(121e12, False)       # L4 dense BF16 ≈ 121 TFLOPS

    # Other widely used Ampere data-center cards
    if has("a30"):
        return result(165e12, False)       # A30 dense BF16 ≈ 165 TFLOPS
    if has("a40"):
        return result(149.7e12, False)     # A40 dense BF16 ≈ 149.7 TFLOPS

    # --- AMD CDNA accelerators ---
    if has("mi355"):
        return result(2.5e15, False)       # MI355X dense BF16 ≈ 2.5 PFLOPS (5.0 PFLOPS sparse)
    if has("mi325"):
        return result(1.3074e15, False)    # MI325X dense BF16 ≈ 1.3074 PFLOPS
    if has("mi300x"):
        return result(1.3074e15, False)    # MI300X dense BF16 ≈ 1.3074 PFLOPS
    if has("mi300a"):
        return result(980.6e12, False)     # MI300A dense BF16 ≈ 980.6 TFLOPS
    if has("mi250x"):
        return result(383e12, False)       # MI250X dense BF16 ≈ 383 TFLOPS
    if has("mi250"):
        return result(362.1e12, False)     # MI250 dense BF16 ≈ 362.1 TFLOPS

    # --- Consumer RTX ---
    if has("5090"):
        return result(209.5e12, False)     # RTX 5090 dense BF16 ≈ 209.5 TFLOPS (w/ FP32 accumulate)
    if has("4090"):
        return result(165.2e12, False)     # RTX 4090 dense BF16 ≈ 165.2 TFLOPS (w/ FP32 accumulate)
    if has("3090"):
        return result(71e12, False)        # RTX 3090 dense BF16 ≈ 71 TFLOPS (w/ FP32 accumulate)

    # unknown
    return result(1e12, True)
