#!/bin/bash
# 【核心逻辑】nanochat核心启动脚本：一站式完成环境配置、数据下载、模型训练/微调全流程
# 【代码规范】遵循Shell脚本规范：首行指定bash解释器，注释统一用#开头，关键步骤用分隔线区分
# 【设计定位】低成本复刻ChatGPT：适配8XH100节点，4小时内完成训练，总成本约$100

# 【使用场景说明】提供3种启动方式，覆盖不同使用需求
# 1) Example launch (simplest):
# 【场景1】基础启动：适合短时间测试，无后台运行/日志记录，操作最简
# bash speedrun.sh
# 2) Example launch in a screen session (because the run takes ~4 hours):
# 【场景2】Screen会话启动：解决训练耗时久的问题，断开连接后仍能后台运行，同时记录日志
# screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# 【场景3】Wandb日志启动：集成训练可视化工具，需提前配置wandb账号，便于监控训练过程
# WANDB_RUN=speedrun screen -L -Logfile speedrun.log -S speedrun bash speedrun.sh

# 【环境配置】设置OMP线程数为1，避免多线程冲突；指定中间产物缓存目录，避免重复下载
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
# 【容错处理】创建缓存目录：若目录不存在则自动创建，防止后续步骤因目录缺失失败
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# 【模块划分】Python虚拟环境配置模块：使用uv工具管理依赖，保证环境隔离与版本一致性
# Python venv setup with uv

# install uv (if not already installed)
# 【自动化依赖安装】检查uv是否安装：未安装则自动下载安装，降低环境配置门槛
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
# 【环境隔离】创建虚拟环境：仅当.venv目录不存在时创建，避免重复操作
[ -d ".venv" ] || uv venv
# install the repo dependencies
# 【依赖管理】安装GPU版本依赖：通过--extra gpu指定GPU相关依赖，适配训练硬件
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
# 【环境激活】激活虚拟环境：确保后续Python命令使用项目专属环境，避免系统环境污染
source .venv/bin/activate

# -----------------------------------------------------------------------------
# 【模块划分】Wandb日志配置模块：可选功能，支持训练过程可视化监控
# wandb setup
# If you wish to use wandb for logging (it's nice!, recommended).
# 1) Make sure to first log in to wandb, e.g. run:
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
#    `WANDB_RUN=d26 bash speedrun.sh`
# 【容错处理】默认日志模式：未指定WANDB_RUN时，设置为dummy模式跳过日志记录，避免启动失败
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# 【模块划分】训练报告初始化模块：清空历史报告，记录启动信息与系统参数
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
# 【报告管理】重置报告：清空报告目录，写入启动时间戳和系统信息，为后续报告生成做准备
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# 【模块划分】Tokenizer（分词器）配置模块：安装依赖、训练分词器、下载预处理数据
# Tokenizer

# Install Rust / Cargo
# 【依赖安装】安装Rust环境：分词器基于Rust开发，需先配置编译环境
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

# Build the rustbpe Tokenizer
# 【编译优化】编译Rust版分词器：使用maturin编译release版本，提升分词性能
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# 【数据准备】下载基础预训练数据：先下载8个分片（约20亿字符），满足分词器训练需求
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
# 【性能优化】后台下载更多数据：分词器训练时异步下载240个分片，提升整体流程效率
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
# 【核心逻辑】训练分词器：基于20亿字符数据，训练词汇量为65536的分词器
python -m scripts.tok_train --max_chars=2000000000
# evaluate the tokenizer (report compression ratio etc.)
# 【效果验证】评估分词器：输出压缩率等指标，验证分词器效果
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# 【模块划分】基础模型预训练模块：基于下载的数据集训练561M参数的基础模型
# Base model (pretraining)

# The d20 model is 561M parameters.
# Chinchilla says #tokens = 20X #params, so we need 561e6 * 20 = 11.2B tokens.
# Assume our tokenizer is 4.8 chars/token, this is 11.2B * 4.8 ~= 54B chars.
# At 250M chars/shard, this is 54B / 250M ~= 216 shards needed for pretraining.
# Round up to 240 for safety. At ~100MB/shard, this downloads ~24GB of data to disk.
# (The total number of shards available in the entire dataset is 1822.)
# 【数据校验】等待数据集下载完成：确保240个分片全部下载，满足预训练数据需求
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# Number of processes/GPUs to use
# 【硬件适配】设置GPU数量：指定8个GPU并行训练，适配8XH100节点
NPROC_PER_NODE=8

# pretrain the d20 model
# 【核心逻辑】预训练d20模型：基于8卡并行，训练561M参数的基础模型
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- --depth=20 --run=$WANDB_RUN
# evaluate the model on a larger chunk of train/val data and draw some samples
# 【效果验证】评估模型损失：在更多训练/验证数据上评估，输出样本结果
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
# evaluate the model on CORE tasks
# 【效果验证】CORE任务评估：在标准CORE任务上验证模型性能
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# 【模块划分】中期训练模块：教会模型对话特殊令牌、工具使用、多选任务
# Midtraining (teach the model conversation special tokens, tool use, multiple choice)

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_sft_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# 【数据准备】下载对话数据：获取2.3MB合成身份对话数据，赋予模型对话人格
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run midtraining and eval the model
# 【核心逻辑】执行中期训练：训练模型掌握对话特殊令牌、工具使用等能力
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- --run=$WANDB_RUN
# 【效果验证】评估中期训练效果：验证对话能力是否达标
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# 【模块划分】监督微调模块：针对单条序列做领域适配，提升模型对话效果
# Supervised Finetuning (domain adaptation to each sequence all by itself per row)

# train sft and re-eval right away (should see a small bump)
# 【核心逻辑】执行监督微调：进一步适配对话场景，提升模型效果
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- --run=$WANDB_RUN
# 【效果验证】评估微调效果：验证微调后模型性能是否提升
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# 【功能体验】CLI交互对话：支持命令行与模型交互，-p参数指定预设问题
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# 【功能体验】WebUI交互对话：提供ChatGPT风格的可视化界面，更友好的交互体验
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# 【模块划分】强化学习模块（可选）：针对GSM8K任务优化模型
# Reinforcement Learning. Optional, and currently only on GSM8K
# (optional)

# run reinforcement learning
# 【可选逻辑】执行强化学习：针对GSM8K数学任务优化模型（可选步骤）
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_rl -- --run=$WANDB_RUN
# eval the RL model only on GSM8K
# 【可选验证】评估强化学习效果：仅验证GSM8K任务上的性能提升
# torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i rl -a GSM8K

# -----------------------------------------------------------------------------
# 【模块划分】报告生成模块：整合所有训练阶段结果，生成完整报告
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
# 【结果汇总】生成完整报告：整合各阶段日志/评估结果，输出report.md到当前目录
python -m nanochat.report generate
