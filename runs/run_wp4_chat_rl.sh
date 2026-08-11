#!/bin/bash
#
# WP4: 8卡 GSM8K 强化学习 (Chat RL - REINFORCE/GRPO 风格)
#
# 硬件: 8× NVIDIA H20-3e (Hopper SM 9.0, 96GB 每卡)
# 模型: 加载 WP3 SFT checkpoint (d24, step 466)
# 算法: on-policy REINFORCE (token-level advantage r - mu, DAPO 风格)
#        无 KL 正则, 无 PPO ratio+clip (on-policy 不需要)
# 数据: GSM8K 训练集 (rollout 采样 + 沙盒验证)
#
# 前置条件:
#   - WP3 已完成 (cache/chatsft_checkpoints/d24/ 有 checkpoint)
#   - NANOCHAT_DATASET_URL 已设置 (hf-mirror.com)
#
# 预估耗时: ~1-3 小时 (依 num_steps 与 eval 频率而定)
#
# 用法:
#   bash runs/run_wp4_chat_rl.sh
#
# 输出:
#   - Checkpoint: cache/chatrl_checkpoints/d24/
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
echo " WP4: 8卡 GSM8K 强化学习 (Chat RL)"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e"
echo " 模型: 加载 WP3 SFT checkpoint (d24)"
echo " 数据: GSM8K train (rollout) + test (pass@k eval)"
echo " Batch: examples-per-step=16, num-samples=16"
echo " Checkpoint 目录: cache/chatrl_checkpoints/"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/3] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/3] 检查 GPU 可用性..."
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>&1 | head -8

echo "[3/3] 检查 WP3 SFT checkpoint..."
ls -1 cache/chatsft_checkpoints/d24/model_*.pt 2>/dev/null | tail -1 && echo "  [OK] SFT checkpoint 就绪" || {
    echo "  [FAIL] 未找到 SFT checkpoint, 请先完成 WP3"
    exit 1
}
ls -1 cache/tokenizer/tokenizer.pkl 2>/dev/null && echo "  [OK] Tokenizer 就绪" || {
    echo "  [FAIL] Tokenizer 缺失, 请先完成 WP1"
    exit 1
}

echo ""
echo "============================================"
echo " 启动 8 卡 GSM8K 强化学习..."
echo " 注意: 首次运行会下载 GSM8K 数据 (hf-mirror.com)"
echo " 注意: 每个 step 含 rollout 采样 + 梯度更新, 速度较慢"
echo " 注意: save_every=60, 每 60 步保存一次 checkpoint"
echo "============================================"
echo ""

/opt/venv/bin/python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=8 \
  -m scripts.chat_rl \
  -- \
  --run=dummy \
  --device-batch-size=8 \
  --examples-per-step=16 \
  --num-samples=16 \
  --save-every=60