#!/bin/bash
# -----------------------------------------------------------------------------
# 变量设置
NPROC_PER_NODE=1
NANOCHAT_BASE_DIR="/media/data/liujiang/data/datasets/nanochat_base_dir"
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_HUB_CACHE=true
export HF_HUB_CACHE="${NANOCHAT_BASE_DIR}/hf_hub_cache"
export HF_DATASETS_CACHE="${NANOCHAT_BASE_DIR}/hf_datasets_cache"

# -----------------------------------------------------------------------------
# 环境搭建

# install uv (if not already installed)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install the repo dependencies
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 日志系统设置
# 如果使用 wandb 进行日志记录：
# 1) 在环境变量中设置 `WANDB_API_KEY`
# 2) 在本地先运行 `wandb login`
# 3) 运行脚本时设置 WANDB_RUN 环境变量，例如：`WANDB_RUN=d26 bash speedrun.sh`

if [ -z "$WANDB_RUN" ]; then
    # 默认使用 "dummy"：这是一个特殊情况，会跳过 wandb 日志记录
    WANDB_RUN=dummy
else
    # 如果设置了 WANDB_RUN 且不等于 dummy，则运行 online 模式
    if [ "$WANDB_RUN" != "dummy" ]; then
        # 检查是否提供了 API KEY
        if [ -z "$WANDB_API_KEY" ]; then
            echo "错误: 检测到 WANDB_RUN=$WANDB_RUN 为 Online 模式，但未检测到 WANDB_API_KEY"
            exit 1
        fi
    fi
fi

# -----------------------------------------------------------------------------
# 日志系统重置
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 分词器训练
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 等待预训练数据下载完成，然后进行预训练
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=18 --target-param-data-ratio=8.25 --device-batch-size=1 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval -- --device-batch-size=1

# -----------------------------------------------------------------------------
# 指令微调数据集下载，并进行指令微调和评测
IDENTITY_FILE="$NANOCHAT_BASE_DIR/identity_conversations.jsonl"
if [ ! -f "$IDENTITY_FILE" ]; then
    echo "文件不存在，正在从 S3 下载..."
    curl -L -o "$IDENTITY_FILE" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
else
    echo "文件已存在，跳过下载: $IDENTITY_FILE"
fi
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --device-batch-size=1 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# 命令行聊天测试
python -m scripts.chat_cli -p "Why is the sky blue?"

# web聊天测试
python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 生成报告
python -m nanochat.report generate
