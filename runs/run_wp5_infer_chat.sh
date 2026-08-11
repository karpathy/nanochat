#!/bin/bash
#
# WP5: 推理基准测试 + CLI 对话演示
#
# 硬件: 8× NVIDIA H20-3e (实际仅用 1 卡推理)
# 模型: 加载 WP4 RL checkpoint (d24, step 466)
# 阶段:
#   1. infer_bench - 单卡推理基准 (延迟/吞吐/显存)
#   2. chat_cli    - 交互式对话 (带心算/数学题验证)
#
# 前置条件:
#   - WP4 已完成 (cache/chatrl_checkpoints/d24/ 有 checkpoint)
#
# 用法:
#   bash runs/run_wp5_infer_chat.sh
#
# 输出:
#   - 推理基准结果
#   - 交互式 CLI 对话

set -euo pipefail

# ===== 环境配置 =====
export NANOCHAT_BASE_DIR="/volume/posttrain/users/lqiu/src/nanochat/cache"
export OMP_NUM_THREADS=1
cd /volume/posttrain/users/lqiu/src/nanochat
export PATH=/opt/venv/bin:$PATH

echo "============================================"
echo " WP5: 推理基准测试 + CLI 对话"
echo "============================================"
echo " 时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo " 硬件: 8× NVIDIA H20-3e (推理用 1 卡)"
echo " 模型: WP4 RL checkpoint (d24)"
echo "============================================"
echo ""

# ===== 环境检查 =====
echo "[1/3] 检查 Python 环境..."
/opt/venv/bin/python --version
/opt/venv/bin/python -c "import torch; print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')" 2>&1 | grep -v FutureWarning

echo "[2/3] 检查 RL checkpoint..."
ls -1 cache/chatrl_checkpoints/d24/model_*.pt 2>/dev/null | tail -3 && echo "  [OK] RL checkpoint 就绪" || echo "  [WARN] 无 RL checkpoint, 回退到 SFT"
ls -1 cache/chatsft_checkpoints/d24/model_*.pt 2>/dev/null | tail -3 && echo "  [OK] SFT checkpoint 就绪" || {
    echo "  [FAIL] 无可用 checkpoint"
    exit 1
}

echo "[3/3] 检查 tokenizer..."
ls -1 cache/tokenizer/tokenizer.pkl 2>/dev/null && echo "  [OK]" || {
    echo "  [FAIL] Tokenizer 缺失"
    exit 1
}

echo ""
echo "============================================"
echo " 阶段 1: 推理基准测试 (infer_bench)"
echo " 模型: RL checkpoint"
echo "============================================"
echo ""

/opt/venv/bin/python -m scripts.infer_bench \
  -i rl -s 466 2>&1 | tee /tmp/infer_bench_rl.log

echo ""
echo "============================================"
echo " 阶段 2: CLI 对话 (单轮测试)"
echo " 模型: RL checkpoint"
echo "============================================"
echo ""

PROMPTS=(
    "What is the capital of France?"
    "What is 24 * 37?"
    "If I have 12 apples and give away 4, then buy 8 more, how many do I have?"
    "What is the derivative of x^3 + 2x?"
)

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "--------------------------------------------"
    echo " Q: $prompt"
    echo "--------------------------------------------"
    /opt/venv/bin/python -m scripts.chat_cli \
      -i rl -s 466 \
      -p "$prompt" \
      -t 0.6 2>&1
    echo ""
done

echo ""
echo "============================================"
echo " WP5 完成!"
echo " 推理基准日志: /tmp/infer_bench_rl.log"
echo "============================================"