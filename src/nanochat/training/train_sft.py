"""
Supervised fine-tuning (SFT) the model.
Run as:

python -m nanochat.scripts.chat_sft

Or torchrun for training:

torchrun --nproc_per_node=8 -m nanochat.scripts.chat_sft -- --device-batch-size=16
"""

import gc
import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import time
from dataclasses import asdict

import torch
import torch.distributed as dist

from nanochat.common import (
    autodetect_device_type,
    checkpoint_dir,
    compute_cleanup,
    compute_init,
    get_compute_dtype,
    get_compute_dtype_reason,
    get_device_sync,
    get_peak_flops,
    init_wandb,
    is_ddp_initialized,
    print0,
    print_banner,
)
from nanochat.common.flash_attention import HAS_FA3
from nanochat.config import Config
from nanochat.evaluation.chat_eval import run_chat_eval
from nanochat.evaluation.engine import Engine
from nanochat.evaluation.loss_eval import evaluate_bpb
from nanochat.report import get_report
from nanochat.tasks.base import TaskMixture
from nanochat.tasks.customjson import CustomJSON
from nanochat.tasks.gsm8k import GSM8K
from nanochat.tasks.mmlu import MMLU
from nanochat.tasks.smoltalk import SmolTalk
from nanochat.tasks.spellingbee import SimpleSpelling, SpellingBee
from nanochat.tokenizer import get_token_bytes
from nanochat.training.checkpoint import load_model_from_dir, load_optimizer_state, save_checkpoint
from nanochat.training.schedulers import create_muon_momentum_scheduler


def train_sft(config: Config):
    print_banner()
    user_config = asdict(config)
    base_dir = config.common.base_dir
    # -----------------------------------------------------------------------------

    # Init compute/precision
    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    ddp, ddp_rank, _, ddp_world_size, device = compute_init(device_type)
    print0(f"COMPUTE_DTYPE: {get_compute_dtype()} ({get_compute_dtype_reason()})")
    synchronize, get_max_memory = get_device_sync(device_type)
    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS

    # wandb logging init
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    wandb_run = init_wandb(config, master_process=master_process, project_suffix="sft", user_config=user_config)

    # Flash Attention status
    if not HAS_FA3:
        print0(
            "WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback. Training will be less efficient."
        )

    # Load the model and tokenizer
    model, tokenizer, meta = load_model_from_dir(
        base_dir=base_dir, device=device, phase="train", model_tag=config.sft.model_tag, step=config.sft.model_step
    )

    # Inherit training hyperparameters from pretrained checkpoint (None = inherit, explicit value = override)
    pretrain_user_config = meta.get("user_config", {})
    for name, fallback, source in [
        ("max_seq_len", 2048, meta),
        ("device_batch_size", 32, meta),
        ("total_batch_size", 524288, meta),
        ("embedding_lr", 0.3, pretrain_user_config),
        ("unembedding_lr", 0.004, pretrain_user_config),
        ("matrix_lr", 0.02, pretrain_user_config),
    ]:
        arg_val = getattr(config.sft, name)
        pretrain_val = source.get(name)
        if arg_val is None:
            resolved = pretrain_val if pretrain_val is not None else fallback
            setattr(config.sft, name, resolved)
            print0(f"Inherited {name}={resolved} from pretrained checkpoint")
        elif pretrain_val is not None and arg_val != pretrain_val:
            print0(f"NOTE: --{name.replace('_', '-')}={arg_val} overrides pretrained value of {pretrain_val}")
        else:
            print0(f"Using {name}={arg_val}")

    orig_model = model
    model = torch.compile(model, dynamic=False)
    depth = model.config.n_layer
    num_flops_per_token = model.estimate_flops()
    tokens_per_fwdbwd = config.sft.device_batch_size * config.sft.max_seq_len  # tokens per iteration for a single rank
    world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # total tokens per iteration for all ranks
    assert config.sft.total_batch_size % world_tokens_per_fwdbwd == 0
    grad_accum_steps = config.sft.total_batch_size // world_tokens_per_fwdbwd
    print0(
        f"Tokens / micro-batch / rank: {config.sft.device_batch_size} x {config.sft.max_seq_len} = {tokens_per_fwdbwd:,}"
    )
    print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
    print0(f"Total batch size {config.sft.total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")
    token_bytes = get_token_bytes(base_dir=config.common.base_dir, device=device)

    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    # Note that pretraining ramps weight_decay to zero by end of pretraining, so SFT continues with zero
    optimizer = model.setup_optimizer(
        unembedding_lr=config.sft.unembedding_lr,
        embedding_lr=config.sft.embedding_lr,
        matrix_lr=config.sft.matrix_lr,
        weight_decay=0.0,
    )

    # Optionally warm-start optimizer from pretrained checkpoint (momentum buffers etc.)
    # Note: load_state_dict overwrites param_group metadata (LRs, betas, etc.) with the
    # pretrained values. Since pretraining warmdown brings LRs to ~0, we must save and
    # restore our fresh SFT LRs after loading.

    if config.sft.load_optimizer:
        optimizer_data = load_optimizer_state(
            base_dir=config.common.base_dir,
            model_name="base",
            device=device,
            rank=ddp_rank,
            model_tag=config.sft.model_tag,
            step=config.sft.model_step,
        )
        if optimizer_data is not None:
            base_lrs = [group["lr"] for group in optimizer.param_groups]
            optimizer.load_state_dict(optimizer_data)
            del optimizer_data
            for group, base_lr in zip(optimizer.param_groups, base_lrs):
                group["lr"] = base_lr
            print0("Loaded optimizer state from pretrained checkpoint (momentum buffers only, LRs reset)")
        else:
            print0("WARNING: optimizer checkpoint not found, starting with fresh optimizer (slightly worse)")

    # GradScaler for fp16 training (bf16/fp32 don't need it)
    scaler = torch.amp.GradScaler() if get_compute_dtype() == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    # Override the initial learning rate as a fraction of the base learning rate
    for group in optimizer.param_groups:
        group["lr"] = group["lr"] * config.sft.init_lr_frac
        group["initial_lr"] = group["lr"]

    # SFT data mixture and DataLoader
    identity_conversations_filepath = os.path.join(base_dir, "identity.jsonl")
    train_tasks = [
        SmolTalk(split="train"),  # 460K rows of general conversations
        CustomJSON(filepath=identity_conversations_filepath),  # 1000 rows of synthetic identity conversations
        CustomJSON(filepath=identity_conversations_filepath),  # 2 epochs of these
        *[MMLU(subset="auxiliary_train", split="train") for _ in range(config.sft.mmlu_epochs)],  # 100K rows per epoch
        *[GSM8K(subset="main", split="train") for _ in range(config.sft.gsm8k_epochs)],  # 8K rows per epoch
        SimpleSpelling(size=200000, split="train"),  # 200K rows of Simple Spelling (e.g. spell the word 'apple')
        SpellingBee(size=80000, split="train"),  # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
    ]
    train_dataset = TaskMixture(train_tasks)
    print0(
        f"Training mixture: {len(train_dataset):,} rows (MMLU x{config.sft.mmlu_epochs}, GSM8K x{config.sft.gsm8k_epochs})"
    )
    val_dataset = TaskMixture(
        [
            SmolTalk(split="test"),  # 24K rows in test set
            MMLU(
                subset="all", split="test", stop=5200
            ),  # 14K rows in test set, use only 5.2K to match the train ratios
            GSM8K(
                subset="main", split="test", stop=420
            ),  # 1.32K rows in test set, use only 420 to match the train ratios
        ]
    )  # total: 24K + 14K + 1.32K ~= 39K rows
    # DataLoader is defined here, it emits inputs, targets : 2D tensors of shape (device_batch_size, max_seq_len)
    # A big problem is that we don't know the final num_iterations in advance. So we create
    # these two global variables and update them from within the data generator.
    last_step = False  # we will toggle this to True when we reach the end of the training dataset
    approx_progress = 0.0  # will go from 0 to 1 over the course of the epoch
    current_epoch = 1  # track epoch for logging

    def sft_data_generator_bos_bestfit(split: str, buffer_size: int = 100):
        """
        BOS-aligned dataloader for SFT with bestfit-pad packing.

        Each row in the batch starts with BOS (beginning of a conversation).
        Conversations are packed using best-fit algorithm. When no conversation fits,
        the row is padded (instead of cropping) to ensure no tokens are ever discarded.
        Padding positions have targets masked with -1 (ignore_index for cross-entropy).
        """
        nonlocal last_step, approx_progress, current_epoch
        assert split in {"train", "val"}, "split must be 'train' or 'val'"
        dataset = train_dataset if split == "train" else val_dataset
        dataset_size = len(dataset)
        assert dataset_size > 0
        row_capacity = config.sft.max_seq_len + 1  # +1 for target at last position
        bos_token = tokenizer.get_bos_token_id()

        # Conversation buffer: list of (token_ids, loss_mask) tuples
        conv_buffer = []
        cursor = ddp_rank  # Each rank processes different conversations (for fetching)
        consumed = ddp_rank  # Track actual consumption separately from buffering
        epoch = 1
        it = 0  # iteration counter

        def refill_buffer():
            nonlocal cursor, epoch
            while len(conv_buffer) < buffer_size:
                conversation = dataset[cursor]
                ids, mask = tokenizer.render_conversation(conversation)
                conv_buffer.append((ids, mask))
                cursor += ddp_world_size
                if cursor >= dataset_size:
                    cursor = cursor % dataset_size
                    epoch += 1
                    # Note: last_step is now triggered based on consumption, not fetching

        while True:
            rows = []
            mask_rows = []
            row_lengths = []  # Track actual content length (excluding padding) for each row
            for _ in range(config.sft.device_batch_size):
                row = []
                mask_row = []
                padded = False
                content_len = 0
                while len(row) < row_capacity:
                    # Ensure buffer has conversations
                    while len(conv_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - len(row)

                    # Find largest conversation that fits entirely
                    best_idx = -1
                    best_len = 0
                    for i, (conv, _) in enumerate(conv_buffer):
                        conv_len = len(conv)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        # Found a conversation that fits - use it entirely
                        conv, conv_mask = conv_buffer.pop(best_idx)
                        row.extend(conv)
                        mask_row.extend(conv_mask)
                        consumed += ddp_world_size  # Track actual consumption
                    else:
                        # No conversation fits - pad the remainder instead of cropping
                        # This ensures we never discard any tokens
                        content_len = len(row)
                        row.extend([bos_token] * remaining)  # Pad with BOS tokens
                        mask_row.extend([0] * remaining)
                        padded = True
                        break  # Row is now full (with padding)

                # Track content length: full row if no padding, otherwise the length before padding
                if padded:
                    row_lengths.append(content_len)
                else:
                    row_lengths.append(row_capacity)
                rows.append(row[:row_capacity])
                mask_rows.append(mask_row[:row_capacity])

            # Stopping condition to respect num_iterations, if given
            it += 1
            if 0 < config.sft.num_iterations <= it and split == "train":
                last_step = True

            # Update progress tracking (based on consumed, not cursor, to account for buffering)
            if split == "train":
                current_epoch = epoch
                if config.sft.num_iterations > 0:
                    approx_progress = it / config.sft.num_iterations
                else:
                    approx_progress = consumed / dataset_size
                # Trigger last_step when we've consumed enough (instead of when cursor wraps)
                if consumed >= dataset_size:
                    last_step = True

            # Build tensors
            use_cuda = device_type == "cuda"
            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            inputs = batch_tensor[:, :-1].to(device=device, dtype=torch.int32, non_blocking=use_cuda).contiguous()
            targets = batch_tensor[:, 1:].to(device=device, dtype=torch.int64, non_blocking=use_cuda).contiguous()

            # Apply the loss mask from render_conversation (mask=1 for assistant completions,
            # mask=0 for user prompts, BOS, special tokens, tool outputs). mask[1:] aligns
            # with targets (shifted by 1). Unmasked positions get -1 (ignore_index).
            mask_tensor = torch.tensor(mask_rows, dtype=torch.int8)
            mask_targets = mask_tensor[:, 1:].to(device=device)
            targets[mask_targets == 0] = -1

            # Mask out padding positions in targets (set to -1 = ignore_index)
            # For each row, positions >= (content_length - 1) in targets should be masked
            for i, content_len in enumerate(row_lengths):
                if content_len < row_capacity:
                    targets[i, content_len - 1 :] = -1

            yield inputs, targets

    train_loader = sft_data_generator_bos_bestfit("train")
    build_val_loader = lambda: sft_data_generator_bos_bestfit("val")
    progress = 0  # will go from 0 to 1 over the course of the epoch

    # Learning rate schedule (linear warmup, constant, linear warmdown)
    # Same shape as base_train but uses progress (0→1) instead of absolute step counts,
    # because SFT doesn't always know num_iterations in advance (dataset-driven stopping).
    def get_lr_multiplier(progress: float):
        if progress < config.sft.warmup_ratio:
            return (progress + 1e-8) / config.sft.warmup_ratio
        elif progress <= 1.0 - config.sft.warmdown_ratio:
            return 1.0
        else:
            decay = (progress - (1.0 - config.sft.warmdown_ratio)) / config.sft.warmdown_ratio
            return (1 - decay) * 1.0 + decay * config.sft.final_lr_frac

    get_muon_momentum = create_muon_momentum_scheduler()

    # -----------------------------------------------------------------------------
    # Training loop
    x, y = next(train_loader)  # prefetch the very first batch of data
    val_bpb: float | None = None
    min_val_bpb = float("inf")
    smooth_train_loss = 0  # EMA of training loss
    ema_beta = 0.9  # EMA decay factor
    total_training_time = 0  # total wall-clock time of training
    step = 0
    while True:
        flops_so_far = num_flops_per_token * config.sft.total_batch_size * step

        # Synchronize last_step across all ranks to avoid hangs in the distributed setting
        if ddp:
            last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
            dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
            last_step = bool(last_step_tensor.item())

        # once in a while: evaluate the val bpb (all ranks participate)
        if last_step or (config.sft.eval_every > 0 and step % config.sft.eval_every == 0):
            model.eval()
            val_loader = build_val_loader()
            eval_steps = config.sft.eval_tokens // (
                config.sft.device_batch_size * config.sft.max_seq_len * ddp_world_size
            )
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
            print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
            if val_bpb < min_val_bpb:
                min_val_bpb = val_bpb
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "val/bpb": val_bpb,
                }
            )
            model.train()

        # once in a while: estimate the ChatCORE metric (all ranks participate)
        # use the original uncompiled model because the inputs keep changing shape
        if config.sft.chatcore_every > 0 and (last_step or (step > 0 and step % config.sft.chatcore_every == 0)):
            model.eval()
            engine = Engine(orig_model, tokenizer)
            all_tasks = ["ARC-Easy", "ARC-Challenge", "MMLU", "GSM8K", "HumanEval", "SpellingBee"]
            categorical_tasks = {"ARC-Easy", "ARC-Challenge", "MMLU"}
            baseline_accuracies = {
                "ARC-Easy": 0.25,
                "ARC-Challenge": 0.25,
                "MMLU": 0.25,
                "GSM8K": 0.0,
                "HumanEval": 0.0,
                "SpellingBee": 0.0,
            }
            task_results = {}
            for task_name in all_tasks:
                limit = (
                    config.sft.chatcore_max_cat if task_name in categorical_tasks else config.sft.chatcore_max_sample
                )
                max_problems = None if limit < 0 else limit  # -1 means no limit
                acc = run_chat_eval(
                    task_name,
                    orig_model,
                    tokenizer,
                    engine,
                    batch_size=config.sft.device_batch_size,
                    max_problems=max_problems,
                )
                task_results[task_name] = acc
                print0(f"  {task_name}: {100 * acc:.2f}%")

            # Compute ChatCORE metrics (mean centered accuracy, ranges from 0=random to 1=perfect)
            def centered_mean(tasks: set[str]):
                return sum(
                    (task_results[t] - baseline_accuracies[t]) / (1.0 - baseline_accuracies[t]) for t in tasks
                ) / len(tasks)

            chatcore = centered_mean(all_tasks)
            chatcore_cat = centered_mean(categorical_tasks)
            print0(f"Step {step:05d} | ChatCORE: {chatcore:.4f} | ChatCORE_cat: {chatcore_cat:.4f}")
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "chatcore_metric": chatcore,
                    "chatcore_cat": chatcore_cat,
                    **{f"chatcore/{task_name}": acc for task_name, acc in task_results.items()},
                }
            )
            model.train()

        # save checkpoint at the end of the run (all ranks participate so each saves its optimizer shard)
        if last_step:
            output_dirname = config.sft.model_tag if config.sft.model_tag else f"d{depth}"  # e.g. d12
            ckpt_dir = checkpoint_dir(base_dir, "sft", output_dirname)
            save_checkpoint(
                ckpt_dir,
                step,
                orig_model.state_dict(),
                optimizer.state_dict(),
                {
                    "step": step,
                    "val_bpb": val_bpb,  # loss at last step
                    "model_config": {
                        "sequence_len": config.sft.max_seq_len,
                        "vocab_size": tokenizer.get_vocab_size(),
                        "n_layer": depth,
                        "n_head": model.config.n_head,
                        "n_kv_head": model.config.n_kv_head,
                        "n_embd": model.config.n_embd,
                        "window_pattern": model.config.window_pattern,
                    },
                    "user_config": user_config,  # inputs to the training script
                },
                rank=ddp_rank,
            )

        if last_step:
            break

        # Free eval/sample memory before training step
        if device_type == "mps":
            torch.mps.empty_cache()

        # -------------------------------------------------------------------------
        # single training step
        # evaluate the gradient
        synchronize()
        t0 = time.time()
        train_loss = torch.zeros(1, device=device)
        for _ in range(grad_accum_steps):
            loss = model(x, y)
            train_loss = loss.detach()  # for logging
            loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            x, y = next(train_loader)  # prefetch the next batch while the GPU is busy with forward/backward
            progress = max(progress, approx_progress)  # only increase progress monotonically
        # step the optimizer
        lrm = get_lr_multiplier(progress)
        muon_momentum = get_muon_momentum(step)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
        if scaler is not None:
            scaler.unscale_(optimizer)
            if is_ddp_initialized():
                for v in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        model.zero_grad(set_to_none=True)
        synchronize()
        t1 = time.time()
        dt = t1 - t0
        # -------------------------------------------------------------------------

        # State
        step += 1

        # logging
        smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()  # EMA the training loss
        debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
        pct_done = 100 * progress
        tok_per_sec = int(config.sft.total_batch_size / dt)
        flops_per_sec = num_flops_per_token * config.sft.total_batch_size / dt
        mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
        if step > 10:
            total_training_time += dt  # only count the time after the first 10 steps
        print0(
            f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | epoch: {current_epoch} | total time: {total_training_time / 60:.2f}m"
        )
        if step % 10 == 0:
            wandb_run.log(
                {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": current_epoch,
                }
            )

        # The garbage collector spends ~500ms scanning for cycles quite frequently.
        # We manually manage it to avoid these pauses during training.
        if step == 1:
            gc.collect()  # manually collect a lot of garbage from setup
            gc.freeze()  # freeze all currently surviving objects and exclude them from GC
            gc.disable()  # disable GC entirely except:
        elif step % 5000 == 0:  # every 5000 steps...
            gc.collect()  # manually collect, just to be safe for very long runs

    # print a few more stats
    print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
    print0(f"Total training time: {total_training_time / 60:.2f}m")

    # Log to report

    get_report(base_dir=config.common.base_dir).log(
        section="SFT",
        data=[
            user_config,  # CLI args
            {  # stats about the training setup
                "Number of iterations": step,
                "DDP world size": ddp_world_size,
            },
            {  # stats about training outcomes
                "Minimum validation bpb": min_val_bpb,
            },
        ],
    )

    # cleanup
    wandb_run.finish()  # wandb run finish
    compute_cleanup()
