#!/bin/bash
# Cleanup script for nanochat on UPPMAX
# Use this to free up storage after experiments

echo "=== nanochat Storage Usage ==="
echo ""
du -sh ~/.cache/nanochat 2>/dev/null || echo "No nanochat cache found"
du -sh ~/.cache/nanochat/*/ 2>/dev/null || true
echo ""
du -sh ~/nanochat/.venv 2>/dev/null || echo "No venv found"
echo ""
echo "Total home usage:"
du -sh ~ 2>/dev/null
echo ""

echo "=== Cleanup Options ==="
echo "1) Remove downloaded data only (keeps checkpoints)"
echo "2) Remove everything except venv"
echo "3) Remove EVERYTHING (full reset)"
echo "4) Cancel"
echo ""
read -p "Choose [1-4]: " choice

case $choice in
    1)
        echo "Removing data shards..."
        rm -rf ~/.cache/nanochat/data/
        echo "Done. Checkpoints preserved."
        ;;
    2)
        echo "Removing data, tokenizer, and checkpoints..."
        rm -rf ~/.cache/nanochat/
        echo "Done. Venv preserved - no need to re-run setup."
        ;;
    3)
        echo "Removing everything..."
        rm -rf ~/.cache/nanochat/
        rm -rf ~/nanochat/.venv
        echo "Done. Run 'bash runs/uppmax/setup.sh' to start fresh."
        ;;
    4)
        echo "Cancelled."
        ;;
    *)
        echo "Invalid choice."
        ;;
esac

echo ""
echo "Current usage:"
du -sh ~ 2>/dev/null
