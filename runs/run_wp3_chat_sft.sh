#!/bin/bash
#
# WP3: 8卡 SFT 微调 (Supervised Fine-Tuning)
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 加载 WP2 基座 checkpoint (d24, step 5568, val_bpb=0.7177)
# 数据: SmolTalk + MMLU(×3 epochs) + GSM8K(×4 epochs) 多任务混合
# 优化器: 继承预训练优化器状态 + learning rate warmup
#
# 前置条件:
#   - WP2 已完成 (cache/base_checkpoints/d24/ 有 model_005568.pt)
#   - NANOCHAT_DATASET_URL 已设置 (hf-mirror.com)
#
# 预估耗时: ~1-2 小时 (依数据量和 eval 频率而定)
#
# 用法:
#   bash runs/run_wp3_chat_sft.sh
#
# 输出:
#   - Checkpoint: cache/chatsft_checkpoints/d24/
#   - 训练日志: 直接输出到终端

set -euo pipefail

# ===== 环境配置 =====
export NANOCHAT_BASE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache"
export NANOCHAT_DATASET_URL="https://hf-mirror.com"
export OMP_NUM_THREADS=1
cd /volume/posttrain/users/lqiu/src/nanochat
export PATH=/opt/venv/bin:$PATH

# ===== 打印配置概览 =====
echo "============================================"
echo " WP3: 8卡 SFT 微调"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: 加载 WP2 基座 d24 (step 5568)"
echo " 数据: SmolTalk + MMLU(x3) + GSM8K(x4)"
echo " Batch: 继承预训练配置 (device-batch-size=16)"
echo " Checkpoint 目录: cache/chatsft_checkpoints/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/3] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/3] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/3] 检查 WP2 checkpoint..."
ls -1 cache/base_checkpoints/d24/model_005568.pt 2>/dev/null && echo "  [OK] 基座 checkpoint 就绪" || {
    echo "  [FAIL] 未找到基座 checkpoint, 请先完成 WP2"
    exit 1
}
ls -1 cache/tokenizer/tokenizer.pkl 2>/dev/null && echo "  [OK] Tokenizer 就绪" || {
    echo "  [FAIL] Tokenizer 缺失, 请先完成 WP1"
    exit 1
}

echo ""
echo "============================================"
echo " 启动 8 卡 SFT 微调..."
echo " 注意: 首次运行会下载 SmolTalk/MMLU/GSM8K 数据"
echo "       (hf-mirror.com, 约需几分钟)"
echo " 注意: SFT 仅在训练结束时保存 checkpoint"
echo "       (无 --save-every 选项)"
echo "============================================"
echo ""

/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.chat_sft \
  -- \
  --run=dummy