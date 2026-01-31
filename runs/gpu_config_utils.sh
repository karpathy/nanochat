#!/bin/bash
# Auto-detect PyTorch flavour and set optimal environment variables
# Returns: cuda, rocm, or cpu - matching pyproject.toml optional-dependencies
# Note: 'gpu' is an alias for 'cuda' (backward compatibility)

detect_torch_flavour() {
    echo "Auto-detecting PyTorch flavour..." >&2
    
    # Check for ROCm first (AMD GPUs)
    if command -v rocminfo &> /dev/null && rocminfo &> /dev/null 2>&1; then
        echo "  → Detected ROCm (AMD GPU via rocminfo)" >&2
        echo "rocm"
        return 0
    fi
    
    # Check for CUDA (NVIDIA GPUs)
    if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null 2>&1; then
        echo "  → Detected CUDA (NVIDIA GPU via nvidia-smi)" >&2
        echo "cuda"
        return 0
    fi
    
    # Fallback to CPU
    echo "  → No GPU detected, using CPU backend" >&2
    echo "cpu"
    return 0
}

detect_num_gpus_pytorch() {
    # Use PyTorch to detect GPUs (only call this after PyTorch is installed)
    python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo "0"
}

set_backend_env_vars() {
    local torch_flavour=$1
    
    # Normalize 'gpu' to 'cuda' (gpu is an alias)
    if [ "$torch_flavour" = "gpu" ]; then
        torch_flavour="cuda"
    fi
    
    if [ "$torch_flavour" = "rocm" ]; then
        echo "Setting ROCm-optimized environment variables..."
        export FLASH_ATTENTION_TRITON_AMD_ENABLE="FALSE"
        export USE_ROCM_CK_SDPA=1
        export USE_ROCM_CK_GEMM=1
        export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
        # ROCm-specific optimizations
        export HIP_LAUNCH_BLOCKING=0
    elif [ "$torch_flavour" = "cuda" ]; then
        echo "Setting CUDA-optimized environment variables..."
    else
        echo "Using CPU backend (no GPU-specific optimizations)"
    fi
    
    # Common optimizations for all backends
    export OMP_NUM_THREADS=1
    export TOKENIZERS_PARALLELISM=false
}

verify_backend_after_install() {
    echo "Verifying PyTorch backend after installation..."
    python3 -c "
import torch
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f'✓ ROCm backend active (HIP version: {torch.version.hip})')
    else:
        print(f'✓ CUDA backend active (CUDA version: {torch.version.cuda})')
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Device count: {torch.cuda.device_count()}')
else:
    print('✓ CPU backend active (no GPU detected)')
" 2>/dev/null || echo "Warning: Could not verify backend"
}

show_gpu_info_pytorch() {
    echo ""
    echo "PyTorch GPU Detection (after installation):"
    echo "==========================================="
    
    python3 -c "
import torch
if torch.cuda.is_available():
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        print(f'Backend: ROCm (HIP version: {torch.version.hip})')
    else:
        print(f'Backend: CUDA (CUDA version: {torch.version.cuda})')
    device_count = torch.cuda.device_count()
    print(f'Number of GPUs detected by PyTorch: {device_count}')
    for i in range(device_count):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'    Total memory: {props.total_memory / 1024**3:.2f} GB')
else:
    print('Backend: CPU (no GPU detected by PyTorch)')
    print('Number of processes: 1')
" 2>/dev/null || echo "Warning: Could not get GPU info from PyTorch"
    echo ""
}

# Main execution if script is run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    TORCH_FLAVOUR=$(detect_torch_flavour)
    set_backend_env_vars "$TORCH_FLAVOUR"
    verify_backend_after_install
    show_gpu_info_pytorch
    echo "Detected PyTorch flavour: $TORCH_FLAVOUR"
fi
