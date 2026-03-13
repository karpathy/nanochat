#!/bin/bash
#
# Nanochat Setup and Run Helper
#
# This script helps set up the environment and run experiments with sensible defaults.
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_env() {
    print_header "Setting Up Environment"
    
    # Set NANOCHAT_BASE_DIR
    export NANOCHAT_BASE_DIR="$SCRIPT_DIR/local"
    echo "export NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR" >> ~/.bashrc 2>/dev/null || true
    print_success "NANOCHAT_BASE_DIR set to: $NANOCHAT_BASE_DIR"
    
    # Create local directories
    mkdir -p local/{data,models,checkpoints,experiments,docs}
    print_success "Created local directories"
    
    # Load config if exists
    if [ -f "local/.env" ]; then
        source local/.env
        print_success "Loaded local/.env"
    else
        print_warning "No local/.env found. Copy config.default to local/.env to customize."
    fi
    
    # Check GPU
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        print_success "GPU detected: $GPU_INFO"
    else
        print_warning "No NVIDIA GPU detected. Will use CPU/MPS."
    fi
}

# ============================================================================
# Data Setup
# ============================================================================

download_data() {
    print_header "Downloading Training Data"
    
    local SHARDS=${1:-8}
    
    if [ -d "$NANOCHAT_BASE_DIR/data" ] && [ "$(ls -A $NANOCHAT_BASE_DIR/data)" ]; then
        print_warning "Data directory already exists. Skipping download."
        return
    fi
    
    print_warning "Downloading $SHARDS shards (~$((SHARDS * 1200 / 1000))GB). This may take a while..."
    uv run python -m nanochat.data.dataset -n "$SHARDS"
    print_success "Data downloaded to $NANOCHAT_BASE_DIR/data/"
}

train_tokenizer() {
    print_header "Training Tokenizer"
    
    if [ -f "$NANOCHAT_BASE_DIR/tok32768.model" ]; then
        print_warning "Tokenizer already exists. Skipping training."
        return
    fi
    
    print_warning "Training tokenizer. This may take 30-60 minutes..."
    uv run python -m nanochat.scripts.tok_train \
        --max-chars=2000000000 \
        --vocab-size=32768
    print_success "Tokenizer trained: $NANOCHAT_BASE_DIR/tok32768.model"
}

# ============================================================================
# Experiment Runners
# ============================================================================

run_quick_test() {
    print_header "Running Quick Test (d12, 100 steps, ~10 minutes)"
    
    uv run python -m nanochat.scripts.base_train \
        --depth=12 \
        --num-iterations=100 \
        --track-compression \
        --compression-log-every=10 \
        --eval-every=50 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --run=quick-test-$(date +%Y%m%d-%H%M%S)
    
    print_success "Quick test complete!"
}

run_compression_validation() {
    print_header "Running Compression Validation (d12, 2000 steps, ~1-2 hours)"
    
    local DEPTH=${1:-12}
    local ITERATIONS=${2:-2000}
    
    uv run python -m nanochat.scripts.base_train \
        --depth="$DEPTH" \
        --num-iterations="$ITERATIONS" \
        --track-compression \
        --compression-log-every=50 \
        --eval-every=250 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --run=compression-validation-d${DEPTH}-$(date +%Y%m%d-%H%M%S)
    
    print_success "Compression validation complete!"
}

run_full_training() {
    print_header "Running Full Training (d24, multi-GPU)"
    
    local DEPTH=${1:-24}
    local GPUS=${2:-8}
    
    torchrun --nproc_per_node="$GPUS" -m nanochat.scripts.base_train \
        --depth="$DEPTH" \
        --track-compression \
        --compression-log-every=100 \
        --eval-every=500 \
        --run=full-training-d${DEPTH}-$(date +%Y%m%d-%H%M%S)
    
    print_success "Full training complete!"
}

# ============================================================================
# Main Menu
# ============================================================================

show_menu() {
    echo ""
    print_header "Nanochat Helper Menu"
    echo ""
    echo "Setup:"
    echo "  1) Setup environment"
    echo "  2) Download data (8 shards, ~10GB)"
    echo "  3) Download data (100 shards, ~120GB)"
    echo "  4) Train tokenizer"
    echo ""
    echo "Experiments:"
    echo "  5) Quick test (d12, 100 steps, ~10 min)"
    echo "  6) Compression validation (d12, 2000 steps, ~1-2 hours)"
    echo "  7) Compression validation (d16, 5000 steps, ~4-6 hours)"
    echo "  8) Full training (d24, multi-GPU, ~8-12 hours)"
    echo ""
    echo "Utilities:"
    echo "  9) Check status"
    echo "  0) Exit"
    echo ""
}

check_status() {
    print_header "Status Check"
    
    echo "Environment:"
    echo "  NANOCHAT_BASE_DIR: ${NANOCHAT_BASE_DIR:-NOT SET}"
    
    echo ""
    echo "Data:"
    if [ -d "$NANOCHAT_BASE_DIR/data" ] && [ "$(ls -A $NANOCHAT_BASE_DIR/data)" ]; then
        DATA_SIZE=$(du -sh "$NANOCHAT_BASE_DIR/data" 2>/dev/null | cut -f1)
        print_success "Data exists ($DATA_SIZE)"
    else
        print_warning "No data found"
    fi
    
    echo ""
    echo "Tokenizer:"
    if [ -f "$NANOCHAT_BASE_DIR/tok32768.model" ]; then
        print_success "Tokenizer exists"
    else
        print_warning "No tokenizer found"
    fi
    
    echo ""
    echo "GPU:"
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv
    else
        print_warning "No NVIDIA GPU detected"
    fi
    
    echo ""
    echo "Tests:"
    uv run pytest tests/ -v --tb=short 2>&1 | tail -5
}

# ============================================================================
# Main
# ============================================================================

main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Select option: " choice
            case $choice in
                1) setup_env ;;
                2) download_data 8 ;;
                3) download_data 100 ;;
                4) train_tokenizer ;;
                5) run_quick_test ;;
                6) run_compression_validation 12 2000 ;;
                7) run_compression_validation 16 5000 ;;
                8) run_full_training 24 8 ;;
                9) check_status ;;
                0) exit 0 ;;
                *) print_error "Invalid option" ;;
            esac
            echo ""
            read -p "Press Enter to continue..."
        done
    else
        # Command line mode
        case "$1" in
            setup) setup_env ;;
            data) download_data "${2:-8}" ;;
            tokenizer) train_tokenizer ;;
            test) run_quick_test ;;
            validate) run_compression_validation "${2:-12}" "${3:-2000}" ;;
            train) run_full_training "${2:-24}" "${3:-8}" ;;
            status) check_status ;;
            *)
                echo "Usage: $0 [command] [args]"
                echo ""
                echo "Commands:"
                echo "  setup              - Setup environment"
                echo "  data [shards]      - Download data (default: 8 shards)"
                echo "  tokenizer          - Train tokenizer"
                echo "  test               - Quick test"
                echo "  validate [d] [n]   - Compression validation (depth, iterations)"
                echo "  train [d] [gpus]   - Full training (depth, num_gpus)"
                echo "  status             - Check status"
                echo ""
                echo "Or run without arguments for interactive menu."
                exit 1
                ;;
        esac
    fi
}

main "$@"
