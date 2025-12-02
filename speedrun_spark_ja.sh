#!/usr/bin/env bash
# speedrun_spark_ja.sh — 日本語学習用スクリプト (DGX Spark 単GPU版)
# 日本語データセット (fineweb-2-edu-japanese) で学習を実行
set -euo pipefail

# ===== ユーザー設定 =====
DEPTH=20
DEVICE_BATCH_SIZE=16
DATA_SHARDS=30
NUM_ITERATIONS=1000
#NUM_ITERATIONS=10
CACHE_DIR="$HOME/.cache/nanochat"
# ========================

# --- 日本語言語設定 ---
export NANOCHAT_LANG=ja

# --- 実行環境・OOM対策 ---
export PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:256"
export TORCHDYNAMO_DISABLE=1
export TORCHINDUCTOR_DISABLE=1

# ---- 計測開始 ----
T0=$(date +%s)

echo "=== nanochat 日本語学習 speedrun (single GPU on DGX Spark) ==="
echo "DEPTH=${DEPTH}, DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE}, LANG=${NANOCHAT_LANG}"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("gpu", torch.cuda.get_device_name(0), "cc", torch.cuda.get_device_capability(0))
PY

echo "== 1) 日本語データ準備 =="
python -m nanochat.dataset -n "${DATA_SHARDS}" --lang ja

echo "== 2) 日本語トークナイザ学習 =="
python -m scripts.tok_train --max_chars=500000000
python -m scripts.tok_eval || true
ls -l "${CACHE_DIR}/tokenizer" || true

echo "== 3) BASE (pretrain) =="
python -m scripts.base_train \
  --depth="${DEPTH}" \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

echo "== 4) MID =="
python -m scripts.mid_train \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

echo "== 5) SFT =="
python -m scripts.chat_sft \
  --device_batch_size="${DEVICE_BATCH_SIZE}" \
  --num_iterations="${NUM_ITERATIONS}"

# echo "== 6) 日本語評価 =="
# python -m scripts.chat_eval -i sft

# ---- 計測終了＆表示 ----
T1=$(date +%s)
ELAPSED=$((T1 - T0))
printf "\n== SUMMARY ==\nTotal elapsed: %d s (%02d:%02d:%02d)\n" \
  "$ELAPSED" "$((ELAPSED/3600))" "$(((ELAPSED%3600)/60))" "$((ELAPSED%60))"

echo "✅ 日本語学習完了：Web UI →  python -m scripts.chat_web"
