# bench_train_step.py 使用说明

`scripts/bench_train_step.py` 是一个训练 step 微基准脚本，用来在不跑完整训练的情况下，快速测量一次优化 step 的性能拆分。默认结果会追加写入：

```text
docs/train_step_bench.csv
```

它不会训练出有用模型，也不会读真实数据集、不会 eval、不会保存 checkpoint。脚本使用随机 token 跑少量 step，主要回答：

- 这个实验的单步训练是否更快
- 快或慢发生在 forward、backward、optimizer 还是 zero_grad
- tok/sec、TFLOP/s、MFU 是否有变化
- 理论训练 FLOPs 是否因为模型结构变化而变化

## 基本用法

在 repo 根目录运行：

```bash
cd /data/home/zzq/nanochat
source .venv/bin/activate
```

单卡小规模试跑：

```bash
python -m scripts.bench_train_step \
  --label smoke_d16 \
  --depth=16 \
  --device-batch-size=16 \
  --steps=20 \
  --warmup-steps=5
```

2xH20 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scripts.bench_train_step \
  --label baseline_d16 \
  --depth=16 \
  --n-kv-head=-1 \
  --device-batch-size=16 \
  --steps=20 \
  --warmup-steps=5
```

FP8 示例：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scripts.bench_train_step \
  --label fp8_d16 \
  --depth=16 \
  --n-kv-head=-1 \
  --device-batch-size=16 \
  --fp8 \
  --steps=20 \
  --warmup-steps=5
```

## 对齐完整训练配置

为了让 benchmark 和 `runs/speedrun_2xH20.sh` 更可比，关键参数要保持一致：

- `--depth`
- `--n-kv-head`
- `--device-batch-size`
- `--max-seq-len`
- `--window-pattern`
- `--fp8`
- GPU 数量和 `CUDA_VISIBLE_DEVICES`
- dtype 环境变量，例如 `NANOCHAT_DTYPE`

默认情况下，脚本不做 gradient accumulation：

```text
global_tokens_per_step = device_batch_size * max_seq_len * world_size
```

如果要模拟完整训练里的全局 batch size，需要显式传 `--total-batch-size`。例如 2 卡、`device_batch_size=16`、`max_seq_len=2048` 时，单个 micro step 是：

```text
16 * 2048 * 2 = 65,536 tokens
```

如果传：

```bash
--total-batch-size=524288
```

则：

```text
grad_accum_steps = 524288 / 65536 = 8
```

## 推荐实验流程

先在 baseline 代码上跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scripts.bench_train_step \
  --label baseline_d16 \
  --depth=16 \
  --n-kv-head=-1 \
  --device-batch-size=16 \
  --steps=30 \
  --warmup-steps=10
```

切到实验代码或打开实验 flag 后跑：

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  -m scripts.bench_train_step \
  --label my_experiment_d16 \
  --depth=16 \
  --n-kv-head=-1 \
  --device-batch-size=16 \
  --steps=30 \
  --warmup-steps=10
```

两个结果会 append 到 `docs/train_step_bench.csv`，后续可以直接对比。

## 输出怎么看

终端会输出类似：

```text
Benchmark summary for fp8_d16 (mean over measured steps; max-reduced across ranks):
  forward:       90.21 ms ( 28.1%)
  backward:     200.12 ms ( 62.4%)
  optimizer:     31.81 ms (  9.9%)
  zero_grad:      0.60 ms (  0.2%)
  step:         320.52 ms
  tok/sec:      204,469
  TFLOP/s:       324.18
  fwd TF/s:      383.95 (approx)
  bwd TF/s:      346.14 (approx)
  MFU:           109.52%
```

主要看这些字段：

| 字段 | 含义 |
| --- | --- |
| `forward_ms` | 前向耗时，包括模型 forward 和 loss |
| `backward_ms` | 反向耗时，包括 autograd，分布式时也会包含相关同步开销 |
| `optimizer_ms` | `optimizer.step()` 耗时，包含 Muon/AdamW 更新和分布式 optimizer 通信 |
| `zero_grad_ms` | `model.zero_grad(set_to_none=True)` 耗时 |
| `step_ms` | 上面几段相加后的单个 optimizer step 耗时 |
| `tok_per_sec` | 每秒处理 token 数，越高越好 |
| `flops_per_step` | 估算的每个 optimizer step 训练 FLOPs |
| `flops_per_sec` | 估算训练 FLOPs / step 时间 |
| `mfu_pct` | Model FLOPS Utilization，按 GPU 理论 BF16 峰值估算 |
| `forward_pct` | forward 占总 step 时间比例 |
| `backward_pct` | backward 占总 step 时间比例 |
| `optimizer_pct` | optimizer 占总 step 时间比例 |

`approx_forward_flops_per_sec` 和 `approx_backward_flops_per_sec` 是粗略拆分值。脚本使用常见训练 FLOPs 近似：

```text
forward ~= 1/3 total training FLOPs
backward ~= 2/3 total training FLOPs
```

optimizer 部分没有硬算理论 FLOPs，因为这里混合了 Muon、AdamW、状态更新、kernel launch 和分布式通信，更适合直接看耗时。

## CSV 对比

默认情况下，脚本会 append 一行结果到 `docs/train_step_bench.csv`：

```bash
column -t -s, docs/train_step_bench.csv
```

如果想写到其他位置，可以传 `--output-csv path/to/file.csv`。如果只想看终端输出、不写 CSV，可以传空字符串：

```bash
python -m scripts.bench_train_step --output-csv ""
```

常用对比字段：

```text
label
step_ms
forward_ms
backward_ms
optimizer_ms
tok_per_sec
flops_per_step
flops_per_sec
mfu_pct
```

判断实验是否变快时，优先看：

1. `step_ms` 是否下降
2. `tok_per_sec` 是否上升
3. `forward_ms/backward_ms/optimizer_ms` 哪一段变化最大
4. `flops_per_step` 是否变化，避免把“模型变小”误判成“实现变快”

## 注意事项

- 前几步会受 `torch.compile`、CUDA kernel 初始化、cache 等影响，所以要留 `--warmup-steps`。
- 测性能时建议 `--steps=30` 或更多；太少会有抖动。
- 脚本用随机 token，不代表模型质量，只代表训练 step 性能。
- 它不会产生 `val_bpb` 或 `CORE`，所以不能替代完整训练实验。
- 多卡时每段耗时会在 rank 间取 max，这样更接近真实 step 被最慢卡拖住的耗时。
- `MFU` 依赖 `nanochat.common.get_peak_flops()` 里的 GPU 峰值表，只适合作为相对参考。
- 如果只想看 eager 模式，可以加 `--no-compile`。

## 常见问题

### 为什么 MFU 可能超过 100%？

`flops_per_step` 是估算值，GPU 峰值也是按 `get_peak_flops()` 表估算。如果某个 GPU 的峰值表偏低，或 FLOPs 公式与实际 kernel 路径不完全匹配，MFU 可能超过 100%。这种情况下更适合做相同机器、相同配置下的相对比较。

### 为什么 benchmark 不读真实数据？

这个脚本要测的是模型 forward/backward/optimizer 的训练性能。真实 dataloader 会引入下载、parquet、tokenize、CPU prefetch 等额外变量，不利于快速判断某个模型或 optimizer 改动到底影响了哪里。

### 什么时候还需要完整训练？

当 benchmark 显示性能更快后，还需要跑较长训练检查质量指标，例如 `val_bpb` 和 `CORE metric`。一个改动可能让 step 更快，但训练质量变差；也可能 step 慢一点，但同等 FLOPs 下质量更好。
