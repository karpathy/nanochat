"""
Train model. From root directory of the project, run as:

python -m nanochat.scripts.base_train

or distributed as:

torchrun --nproc_per_node=8 -m nanochat.scripts.base_train

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m nanochat.scripts.base_train --depth=4 --max-seq-len=512 --device-batch-size=1 --eval-tokens=512 --core-metric-every=-1 --total-batch-size=512 --num-iterations=20
"""

import os

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
import gc
import json
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist

from nanochat.common import (
    autodetect_device_type,
    compute_cleanup,
    compute_init,
    init_wandb,
    get_compute_dtype,
    get_compute_dtype_reason,
    get_device_sync,
    get_peak_flops,
    is_ddp_initialized,
    print0,
    print_banner,
    checkpoint_dir,
)
from nanochat.config import Config
from nanochat.training.compression_metrics import CompressionMetrics
from nanochat.tokenizer import get_token_bytes, get_tokenizer
from nanochat.evaluation import Engine, evaluate_bpb, evaluate_core
from nanochat.common.flash_attention import HAS_FA3, _use_fa3
from nanochat.models.gpt import GPT, GPTConfig, Linear
from nanochat.training.checkpoint import load_checkpoint, save_checkpoint
from nanochat.training.dataloader import (
    tokenizing_distributed_data_loader_bos_bestfit,
    tokenizing_distributed_data_loader_with_state_bos_bestfit,
)
from nanochat.report import get_report
from nanochat.training.scaling import B_REF, compute_training_hyperparams, get_scaling_params
from nanochat.training.schedulers import create_lr_scheduler, create_muon_momentum_scheduler, create_weight_decay_scheduler


def build_model_meta(
    depth: int,
    aspect_ratio: int,
    head_dim: int,
    max_seq_len: int,
    window_pattern: str,
    vocab_size: int,
) -> GPT:
    """Build a GPT model on meta device for a given depth (shapes/dtypes only, no data)."""
    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    gpt_config = GPTConfig(
        sequence_len=max_seq_len,
        vocab_size=vocab_size,
        n_layer=depth,
        n_head=num_heads,
        n_kv_head=num_heads,
        n_embd=model_dim,
        window_pattern=window_pattern,
    )
    with torch.device("meta"):
        return GPT(gpt_config)


def train_base(config: Config):
    print_banner()

    # -----------------------------------------------------------------------------
    # Compute init and wandb logging

    device_type = autodetect_device_type() if config.common.device_type == "" else config.common.device_type
    _, ddp_rank, _, ddp_world_size, device = compute_init(device_type)


    synchronize, get_max_memory = get_device_sync(device_type)
    if device_type == "cuda":
        gpu_device_name = torch.cuda.get_device_name(0)
        gpu_peak_flops = get_peak_flops(gpu_device_name)
        print0(f"GPU: {gpu_device_name} | Peak FLOPS (BF16): {gpu_peak_flops:.2e}")
    else:
        gpu_peak_flops = float("inf")  # MFU not meaningful for CPU/MPS
    print0(f"COMPUTE_DTYPE: {get_compute_dtype()} ({get_compute_dtype_reason()})")

    # wandb logging init
    user_config = asdict(config)  # for logging
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    wandb_run = init_wandb(config,master_process=master_process, user_config=user_config)


    # Flash Attention status
    if _use_fa3():
        print0("✓ Using Flash Attention 3 (Hopper GPU detected), efficient, new and awesome.")
    else:
        print0("!" * 80)
        if HAS_FA3 and get_compute_dtype() != torch.bfloat16:
            print0(
                f"WARNING: Flash Attention 3 only supports bf16, but COMPUTE_DTYPE={get_compute_dtype()}. Using PyTorch SDPA fallback"
            )
        else:
            print0("WARNING: Flash Attention 3 not available, using PyTorch SDPA fallback")
        print0("WARNING: Training will be less efficient without FA3")
        if config.training.window_pattern != "L":
            print0(
                f"WARNING: SDPA has no support for sliding window attention (window_pattern='{config.training.window_pattern}'). Your GPU utilization will be terrible."
            )
            print0(
                "WARNING: Recommend using --window-pattern L for full context attention without alternating sliding window patterns."
            )
        print0("!" * 80)

    # -----------------------------------------------------------------------------
    # Tokenizer will be useful for evaluation and also we need the vocab size to init the model
    tokenizer = get_tokenizer(base_dir=config.common.base_dir)
    token_bytes = get_token_bytes(device=device, base_dir=config.common.base_dir)
    vocab_size = tokenizer.get_vocab_size()
    print0(f"Vocab size: {vocab_size:,}")

    # -----------------------------------------------------------------------------
    # Initialize the Model


    # Build the model, move to device, init the weights
    model = build_model_meta(config.training.depth, config.training.aspect_ratio, config.training.head_dim, config.training.max_seq_len, config.training.window_pattern, vocab_size)  # 1) Build on meta device (only shapes/dtypes, no data)
    model_config_kwargs = asdict(model.config)
    print0(f"Model config:\n{json.dumps(model_config_kwargs, indent=2)}")
    model.to_empty(device=device)  # 2) All tensors get storage on target device but with uninitialized (garbage) data
    model.init_weights()  # 3) All tensors get initialized

    # If we are resuming, overwrite the model parameters with those of the checkpoint
    base_dir = config.common.base_dir
    output_dirname = config.training.model_tag if config.training.model_tag else f"d{config.training.depth}"
    ckpt_dir = checkpoint_dir(base_dir, "base", output_dirname)
    os.makedirs(ckpt_dir, exist_ok=True)
    config.save(Path(ckpt_dir) / "config.toml")
    optimizer_data: dict[str, object] | None = None
    meta_data: dict[str, object] | None = None
    resuming = config.training.resume_from_step != -1
    if resuming:
        print0(f"Resuming optimization from step {config.training.resume_from_step}")
        model_data, optimizer_data, meta_data = load_checkpoint(
            ckpt_dir, config.training.resume_from_step, device, load_optimizer=True, rank=ddp_rank
        )
        model.load_state_dict(model_data, strict=True, assign=True)
        del model_data  # free up this memory after the copy

    # -----------------------------------------------------------------------------
    # FP8 training initialization and management (this has to be done before torch.compile)

    # Convert Linear layers to Float8Linear if --fp8 is set
    if config.training.fp8:
        if device_type != "cuda":
            print0("Warning: FP8 training requires CUDA, ignoring --fp8 flag")
        else:
            # our custom fp8 is simpler than torchao, written for exact API compatibility
            # from torchao.float8 import Float8LinearConfig, convert_to_float8_training
            import torch.nn as nn

            from nanochat.models.fp8 import Float8LinearConfig, convert_to_float8_training

            # Filter: dims must be divisible by 16 (FP8 hardware requirement) large enough
            def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
                if not isinstance(mod, nn.Linear):
                    return False
                if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
                    return False
                if min(mod.in_features, mod.out_features) < 128:
                    return False
                return True

            fp8_config = Float8LinearConfig.from_recipe_name(config.training.fp8_recipe)
            num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
            convert_to_float8_training(model, config=fp8_config, module_filter_fn=fp8_module_filter)
            num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
            num_skipped = num_linear - num_fp8
            print0(
                f"✓ FP8 training enabled ({config.training.fp8_recipe} scaling) - converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} (too small)"
            )


    # Context manager to temporarily disable FP8 so that model evaluation remains in BF16
    @contextmanager
    def disable_fp8(model: torch.nn.Module):
        """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

        CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
        we swap out Float8Linear modules entirely and restore them after.
        """

        # Find all Float8Linear modules and their locations
        fp8_locations = []  # list of (parent_module, attr_name, fp8_module)
        for name, module in model.named_modules():
            if "Float8" in type(module).__name__:
                if "." in name:
                    parent_name, attr_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name
                fp8_locations.append((parent, attr_name, module))

        if not fp8_locations:
            yield  # No FP8 modules, nothing to do
            return

        # Swap Float8Linear -> Linear (our custom class that casts weights to match input dtype)
        for parent, attr_name, fp8_module in fp8_locations:
            linear = Linear(
                fp8_module.in_features,
                fp8_module.out_features,
                bias=fp8_module.bias is not None,
                device=fp8_module.weight.device,
                dtype=fp8_module.weight.dtype,
            )
            linear.weight = fp8_module.weight  # share, don't copy
            if fp8_module.bias is not None:
                linear.bias = fp8_module.bias
            setattr(parent, attr_name, linear)

        try:
            yield
        finally:
            # Restore Float8Linear modules
            for parent, attr_name, fp8_module in fp8_locations:
                setattr(parent, attr_name, fp8_module)


    # -----------------------------------------------------------------------------
    # Compile the model

    orig_model = model  # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
    model = torch.compile(model, dynamic=False)  # the inputs to model will never change shape so dynamic=False is safe

    # -----------------------------------------------------------------------------
    # Scaling laws and muP extrapolations to determine the optimal training horizon, batch size, learning rates, weight decay.

    # Get the parameter counts of our model
    param_counts = model.num_scaling_params()
    print0("Parameter counts:")
    for key, value in param_counts.items():
        print0(f"{key:24s}: {value:,}")
    num_params = param_counts["total"]
    num_flops_per_token = model.estimate_flops()
    print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")


    # Scaling law computations: optimal tokens, batch size, LR scale, weight decay
    # ref: Power Lines paper https://arxiv.org/abs/2505.13738, T_epoch framework https://arxiv.org/abs/2405.13698
    d12_ref = build_model_meta(12, config.training.aspect_ratio, config.training.head_dim, config.training.max_seq_len, config.training.window_pattern, vocab_size)
    hp = compute_training_hyperparams(
        target_param_data_ratio=config.training.target_param_data_ratio,
        total_batch_size_override=config.training.total_batch_size,
        weight_decay=config.training.weight_decay,
        num_scaling_params=get_scaling_params(model),
        d12_scaling_params=get_scaling_params(d12_ref),
    )
    num_scaling_params = hp.num_scaling_params
    target_tokens = hp.target_tokens
    total_batch_size = hp.total_batch_size
    batch_lr_scale = hp.batch_lr_scale
    weight_decay_scaled = hp.weight_decay_scaled
    if config.training.total_batch_size == -1:
        print0(f"Auto-computed optimal batch size: {total_batch_size:,} tokens")
    if batch_lr_scale != 1.0:
        print0(f"Scaling LRs by {batch_lr_scale:.4f} for batch size {total_batch_size:,} (reference: {B_REF:,})")
    if weight_decay_scaled != config.training.weight_decay:
        print0(f"Scaling weight decay from {config.training.weight_decay:.6f} to {weight_decay_scaled:.6f} for depth {config.training.depth}")

    # -----------------------------------------------------------------------------
    # Initialize the Optimizer (combined MuonAdamW: Muon for matrix params, AdamW for rest)
    optimizer = model.setup_optimizer(
        # AdamW hyperparameters
        unembedding_lr=config.training.unembedding_lr * batch_lr_scale,
        embedding_lr=config.training.embedding_lr * batch_lr_scale,
        scalar_lr=config.training.scalar_lr * batch_lr_scale,
        # Muon hyperparameters
        matrix_lr=config.training.matrix_lr * batch_lr_scale,
        weight_decay=weight_decay_scaled,
    )

    if resuming:
        optimizer.load_state_dict(optimizer_data)
        del optimizer_data

    # -----------------------------------------------------------------------------
    # GradScaler for fp16 training (bf16/fp32 don't need it — bf16 has the same exponent range as fp32)
    scaler = torch.amp.GradScaler() if get_compute_dtype() == torch.float16 else None
    if scaler is not None:
        print0("GradScaler enabled for fp16 training")

    # -----------------------------------------------------------------------------
    # Initialize the DataLoaders for train/val
    dataloader_resume_state_dict = None
    if resuming:
        assert meta_data is not None
        dataloader_resume_state_dict = meta_data["dataloader_state_dict"]
    train_loader = tokenizing_distributed_data_loader_with_state_bos_bestfit(
        tokenizer,
        config.training.device_batch_size,
        config.training.max_seq_len,
        split="train",
        device=device,
        resume_state_dict=dataloader_resume_state_dict,
    )
    build_val_loader = lambda: tokenizing_distributed_data_loader_bos_bestfit(
        tokenizer, config.training.device_batch_size, config.training.max_seq_len, split="val", device=device
    )

    # -----------------------------------------------------------------------------
    # Calculate the number of iterations we will train for and set up the various schedulers

    # num_iterations: either it is given, or from target flops, or from target data:param ratio (in that order)
    assert config.training.num_iterations > 0 or config.training.target_param_data_ratio > 0 or config.training.target_flops > 0
    if config.training.num_iterations > 0:
        # Override num_iterations to a specific value if given
        num_iterations = config.training.num_iterations
        print0(f"Using user-provided number of iterations: {num_iterations:,}")
    elif config.training.target_flops > 0:
        # Calculate the number of iterations from the target flops (used in scaling laws analysis, e.g. runs/scaling_laws.sh)
        num_iterations = round(config.training.target_flops / (num_flops_per_token * total_batch_size))
        print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
    elif config.training.target_param_data_ratio > 0:
        # Calculate the number of iterations from the target param data ratio (the most common use case)
        num_iterations = target_tokens // total_batch_size
        print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
    else:
        raise ValueError("No training horizon specified")
    total_tokens = total_batch_size * num_iterations  # the actual number of tokens we will train for
    print0(f"Total number of training tokens: {total_tokens:,}")
    print0(
        f"Tokens : Scaling params ratio: {total_batch_size * num_iterations / num_scaling_params:.2f}"
    )  # e.g. Chinchilla was ~20
    print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")


    # Learning rate and optimizer schedulers
    get_lr_multiplier = create_lr_scheduler(num_iterations, config.training.warmup_steps, config.training.warmdown_ratio, config.training.final_lr_frac)
    get_muon_momentum = create_muon_momentum_scheduler()
    get_weight_decay = create_weight_decay_scheduler(weight_decay_scaled, num_iterations)


    # -----------------------------------------------------------------------------
    # Training loop


    def train_base_model():
        """Train base model. Captures all state from main() via closure."""
        # Initialize compression tracker if enabled
        compression_tracker = None
        if  config.training.track_compression:
            compression_tracker = CompressionMetrics(vocab_size=vocab_size)
            print0("✓ Compression metrics tracking enabled")
        # Loop state (variables updated by the training loop)
        if not resuming:
            step = 0
            val_bpb = None  # will be set if eval_every > 0
            min_val_bpb = float("inf")
            smooth_train_loss = 0  # EMA of training loss
            total_training_time = 0  # total wall-clock time of training
            dataloader_state_dict = dataloader_resume_state_dict
        else:
            assert meta_data is not None
            step = meta_data["step"]
            loop_state = cast(dict[str, object], meta_data["loop_state"])
            val_bpb = meta_data["val_bpb"]
            min_val_bpb = loop_state["min_val_bpb"]
            smooth_train_loss = loop_state["smooth_train_loss"]
            total_training_time = loop_state["total_training_time"]
            dataloader_state_dict = dataloader_resume_state_dict

        # Figure out the needed gradient accumulation micro-steps to reach the desired total batch size per step
        tokens_per_fwdbwd = config.training.device_batch_size * config.training.max_seq_len  # tokens per iteration for a single rank
        world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size  # total tokens per iteration for all ranks
        assert total_batch_size % world_tokens_per_fwdbwd == 0
        grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
        print0(f"Tokens / micro-batch / rank: {config.training.device_batch_size} x {config.training.max_seq_len} = {tokens_per_fwdbwd:,}")
        print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
        print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

        # Get first batch
        x, y, dataloader_state_dict = next(train_loader)

        # Go!
        mfu = 0.0
        while True:
            last_step = step == num_iterations  # loop runs num_iterations+1 times so that we can eval/save at the end
            flops_so_far = num_flops_per_token * total_batch_size * step

            # once in a while: evaluate the val bpb (all ranks participate)
            if config.training.eval_every > 0 and (last_step or step % config.training.eval_every == 0):
                model.eval()
                val_loader = build_val_loader()
                eval_steps = config.training.eval_tokens // (config.training.device_batch_size * config.training.max_seq_len * ddp_world_size)
                with disable_fp8(model):
                    val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
                print0(f"Step {step:05d} | Validation bpb: {val_bpb:.6f}")
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

            # once in a while: estimate the CORE metric (all ranks participate)
            # use the original uncompiled model because the inputs keep changing shape
            # disable FP8 for evaluation to use BF16 for more consistent/accurate results
            results = {}
            if  config.training.core_metric_every > 0 and (last_step or (step > 0 and step % config.training.core_metric_every == 0)):
                model.eval()
                with disable_fp8(orig_model):
                    results = evaluate_core(base_dir=base_dir, model=orig_model, tokenizer=tokenizer, device=device, max_per_task=config.training.core_metric_max_per_task)
                print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
                wandb_run.log(
                    {
                        "step": step,
                        "total_training_flops": flops_so_far,
                        "core_metric": results["core_metric"],
                        "centered_results": results["centered_results"],
                    }
                )
                model.train()

            # once in a while: sample from the model (only on master process)
            # use the original uncompiled model because the inputs keep changing shape
            if  config.training.sample_every > 0 and master_process and (last_step or (step > 0 and step % config.training.sample_every == 0)):
                model.eval()
                prompts = [
                    "The capital of France is",
                    "The chemical symbol of gold is",
                    "If yesterday was Friday, then tomorrow will be",
                    "The opposite of hot is",
                    "The planets of the solar system are:",
                    "My favorite color is",
                    "If 5*x + 3 = 13, then x is",
                ]
                engine = Engine(orig_model, tokenizer)  # use orig_model to avoid recompilation
                for prompt in prompts:
                    tokens = tokenizer(prompt, prepend="<|bos|>")
                    with disable_fp8(orig_model):
                        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
                    print0(tokenizer.decode(sample[0]))
                model.train()

            # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
            if last_step or (
                step > 0 and step != config.training.resume_from_step and config.training.save_every > 0 and step % config.training.save_every == 0
            ):
                save_checkpoint(
                    ckpt_dir,
                    step,
                    orig_model.state_dict(),  # model parameters
                    optimizer.state_dict(),  # optimizer state
                    {  # metadata saved as json
                        "step": step,
                        "val_bpb": val_bpb,  # loss at last step
                        "model_config": model_config_kwargs,
                        "user_config": user_config,  # inputs to the training script
                        "device_batch_size": config.training.device_batch_size,
                        "max_seq_len": config.training.max_seq_len,
                        "total_batch_size": total_batch_size,
                        "dataloader_state_dict": dataloader_state_dict,
                        "loop_state": {  # all loop state (other than step) so that we can resume training
                            "min_val_bpb": min_val_bpb,
                            "smooth_train_loss": smooth_train_loss,
                            "total_training_time": total_training_time,
                        },
                    },
                    rank=ddp_rank,
                )

            # termination conditions (TODO: possibly also add loss explosions etc.)
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
            logits_for_compression = None  # store logits for compression tracking
            train_loss = torch.zeros(1, device=device)
            for micro_step in range(grad_accum_steps):
                # Forward pass - capture logits if tracking compression
                if compression_tracker and step % config.training.compression_log_every == 0 and micro_step == 0:
                    with torch.amp.autocast(device_type=device_type, dtype=get_compute_dtype()):
                        logits_for_compression = model(x)
                    loss = torch.nn.functional.cross_entropy(
                        logits_for_compression.view(-1, logits_for_compression.size(-1)),
                        y.view(-1)
                    )
                else:
                    loss = model(x, y)
                train_loss = loss.detach()  # for logging
                loss = loss / grad_accum_steps  # each .backward() is a grad sum => normalize loss here
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                x, y, dataloader_state_dict = next(
                    train_loader
                )  # prefetch the next batch while the GPU is busy with forward/backward
            # step the optimizer
            lrm = get_lr_multiplier(step)
            muon_momentum = get_muon_momentum(step)
            muon_weight_decay = get_weight_decay(step)
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * lrm
                if group["kind"] == "muon":
                    group["momentum"] = muon_momentum
                    group["weight_decay"] = muon_weight_decay
            if scaler is not None:
                scaler.unscale_(optimizer)
                # In distributed training, all ranks must agree on whether to skip the step.
                # Each rank may independently encounter inf/nan gradients, so we all-reduce
                # the found_inf flag (MAX = if any rank found inf, all ranks skip).
                if is_ddp_initialized():
                    for v in scaler._found_inf_per_device(optimizer).values():
                        dist.all_reduce(v, op=dist.ReduceOp.MAX)
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            model.zero_grad(set_to_none=True)
            train_loss_f = train_loss.item()  # .item() is a CPU-GPU sync point
            synchronize()
            t1 = time.time()
            dt = t1 - t0
            # -------------------------------------------------------------------------

            # logging (CPU action only)
            ema_beta = 0.9  # EMA decay factor for some smoothing just for nicer logging
            smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f  # EMA the training loss
            debiased_smooth_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))  # debias the EMA
            pct_done = 100 * step / num_iterations
            tok_per_sec = int(total_batch_size / dt)
            flops_per_sec = num_flops_per_token * total_batch_size / dt
            mfu = 100 * flops_per_sec / (gpu_peak_flops * ddp_world_size)
            if step > 10:
                total_training_time += dt  # only count the time after the first 10 steps
            # Calculate ETA based on average time per step (excluding first 10 steps)
            steps_done = step - 10
            if steps_done > 0:
                avg_time_per_step = total_training_time / steps_done
                remaining_steps = num_iterations - step
                eta_seconds = remaining_steps * avg_time_per_step
                eta_str = f" | eta: {eta_seconds / 60:.1f}m"
            else:
                eta_str = ""
            epoch = f"{dataloader_state_dict['epoch']} pq: {dataloader_state_dict['pq_idx']} rg: {dataloader_state_dict['rg_idx']}"
            print0(
                f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_smooth_loss:.6f} | lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | bf16_mfu: {mfu:.2f} | epoch: {epoch} | total time: {total_training_time / 60:.2f}m{eta_str}"
            )
            # Track compression metrics if enabled
            if compression_tracker and step % config.training.compression_log_every == 0 and logits_for_compression is not None:
                with torch.no_grad():
                    compression_metrics = compression_tracker.log_metrics(
                        step=step,
                        tokens=y,  # target tokens
                        logits=logits_for_compression,
                        loss=train_loss_f,
                    )

                    # Check for overfitting if early stopping enabled
                    if config.training.compression_early_stop and compression_tracker.detect_overfitting():
                        print0(f"[Step {step}] Compression plateau detected - possible overfitting")

                    # Print compression metrics to console
                    print0(
                        f"[compression] step {step:05d} | entropy: {compression_metrics['entropy']:.4f} | "
                        f"ratio: {compression_metrics['compression_ratio']:.4f} | "
                        f"gzip: {compression_metrics['gzip_compression']:.4f} | "
                        f"efficiency: {compression_metrics['compression_efficiency']:.4f}"
                    )

                    # Log compression metrics to wandb
                    if master_process:
                        compression_log = {
                            f"compression/{k}": v
                            for k, v in compression_metrics.items()
                            if k != 'step'
                        }
                        wandb_run.log(compression_log)

            if step % 100 == 0:
                log_data = {
                    "step": step,
                    "total_training_flops": flops_so_far,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lrm,
                    "train/dt": dt,
                    "train/tok_per_sec": tok_per_sec,
                    "train/mfu": mfu,
                    "train/epoch": epoch,
                }
                wandb_run.log(log_data)

            # state update
            first_step_of_run = (step == 0) or (resuming and step == config.training.resume_from_step)
            step += 1

            # The garbage collector is sadly a little bit overactive and for some poorly understood reason,
            # it spends ~500ms scanning for cycles quite frequently, just to end up cleaning up very few tiny objects each time.
            # So we manually manage and help it out here
            if first_step_of_run:
                gc.collect()  # manually collect a lot of garbage from setup
                gc.freeze()  # immediately freeze all currently surviving objects and exclude them from GC
                gc.disable()  # nuclear intervention here: disable GC entirely except:
            elif step % 5000 == 0:  # every 5000 steps...
                gc.collect()  # manually collect, just to be safe for very, very long runs

        # print a few more stats
        print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
        print0(f"Total training time: {total_training_time / 60:.2f}m")
        if val_bpb is not None:
            print0(f"Minimum validation bpb: {min_val_bpb:.6f}")


        get_report(config.common.base_dir).log(
            section="Base model training",
            data=[
                user_config,  # CLI args
                {  # stats about the training setup
                    "Number of parameters": num_params,
                    "Number of FLOPs per token": f"{num_flops_per_token:e}",
                    "Calculated number of iterations": num_iterations,
                    "Number of training tokens": total_tokens,
                    "Tokens : Scaling params ratio": total_batch_size * num_iterations / num_scaling_params,
                    "DDP world size": ddp_world_size,
                    "warmup_steps": config.training.warmup_steps,
                    "warmdown_ratio": config.training.warmdown_ratio,
                    "final_lr_frac": config.training.final_lr_frac,
                },
                {  # stats about training outcomes
                    "Minimum validation bpb": min_val_bpb if val_bpb is not None else None,
                    "Final validation bpb": val_bpb,
                    "CORE metric estimate": results.get("core_metric", None),
                    "MFU %": f"{mfu:.2f}%",
                    "Total training flops": f"{flops_so_far:e}",
                    "Total training time": f"{total_training_time / 60:.2f}m",
                    "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
                },
            ],
        )

        return orig_model, {
            "val_bpb": val_bpb,
            "min_val_bpb": min_val_bpb,
            "core_metric": results.get("core_metric", None),
            "mfu": mfu,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "peak_memory": get_max_memory() / 1024 / 1024,
        }


    # -------------------------------------------------------------------------
    # Run training and cleanup
    train_base_model()
    wandb_run.finish()
    compute_cleanup()

