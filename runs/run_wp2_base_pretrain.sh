#!/bin/bash
#
# WP2: 8卡基座预训练 (Base Pretraining)
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 24层 Transformer (等效 GPT-2 规模)
# 数据: ~24B token (Chinchilla 比例 = 8)
# 优化器: DistMuonAdamW 混合策略 + FP8 tensorwise 量化
#
# 前置条件:
#   - WP1 已完成 (tokenizer + 171 shard 语料已就绪)
#   - rustbpe 已安装 (pip install rustbpe --index-url https://pypi.org/simple/)
#
# 预估耗时: H20-3e × 8 约 3~4.5 小时 (标称 8×H100 的 ~1.5h × 2~3x)
#
# 用法:
#   bash runs/run_wp2_base_pretrain.sh
#
# 输出:
#   - Checkpoint: cache/base_checkpoints/d24/
#   - 日志: /tmp/base_train.log

set -euo pipefail

# ===== 环境配置 =====
export NANOCHAT_BASE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export OMP_NUM_THREADS=1
cd /volume/posttrain/users/lqiu/src/nanochat
alias python=/opt/venv/bin/python
export PATH=/opt/venv/bin:$PATH

# ===== 打印配置概览 =====
echo "============================================"
echo " WP2: 8卡基座预训练"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: --depth=24 --fp8"
echo " 数据: --target-param-data-ratio=8"
echo " Batch: --device-batch-size=16 (稳定值, 每 500 步保存 checkpoint)"
echo " Checkpoint 目录: cache/base_checkpoints/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/4] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/4] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/4] 检查 tokenizer 和数据..."
ls -1 cache/tokenizer/tokenizer.pkl cache/tokenizer/token_bytes.pt 2>/dev/null && echo "  [OK] Tokenizer" || { echo "  [FAIL] Tokenizer 缺失, 请先完成 WP1"; exit 1; }
SHARD_COUNT=$(ls cache/base_data_climbmix/*.parquet 2>/dev/null | wc -l)
echo "  数据 shard: ${SHARD_COUNT}/171"
[ "$SHARD_COUNT" -ge 1 ] && echo "  [OK] 数据就绪" || { echo "  [FAIL] 数据缺失, 请先完成 WP1"; exit 1; }

echo "[4/4] 检查 rustbpe..."
/opt/venv/bin/python -c "import rustbpe" 2>/dev/null && echo "  [OK] rustbpe" || {
    echo "  [WARN] rustbpe 未安装, 正在安装..."
    pip install rustbpe --index-url https://pypi.org/simple/ 2>&1 | tail -1
}

echo ""
echo "============================================"
echo " 启动 8 卡预训练..."
echo " 日志: /tmp/base_train.log"
echo "============================================"
echo ""

# ===== 启动训练 =====
echo ""
echo "============================================"
echo " 训练开始，直接输出到终端..."
echo " 训练结束后，执行评测:"
echo "   /opt/venv/bin/python -m torch.distributed.run"
echo "     --standalone --nproc_per_node=8"
echo "     -m scripts.base_eval -- --device-batch-size=16"
echo "============================================"
echo ""

/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.base_train \
  -- \
  --depth=24 \
  --target-param-data-ratio=8 \
  --device-batch-size=16 \
  --save-every=500 \
  --fp8 \
  --run=dummy