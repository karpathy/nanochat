//! Small, single-device Rust language-model trainer.

use std::{
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device, Tensor, backprop::GradStore};
use candle_nn::{VarMap, ops};

use crate::{
    data::{BatchSource, JsonlBatchSource, ParquetBatchSource, TokenBatch, TokenCorpus},
    model::GptModel,
    model_config::GptConfig,
    optim::{MuonAdamW, MuonAdamWConfig},
    tokenizer::BpeTokenizer,
};

pub struct TrainOptions {
    pub compute_dtype: DType,
    pub fp8: bool,
    pub steps: usize,
    pub steps_explicit: bool,
    pub batch_size: usize,
    pub sequence_len: usize,
    pub grad_accum_steps: usize,
    pub embedding_lr: f64,
    pub unembedding_lr: f64,
    pub matrix_lr: f64,
    pub scalar_lr: f64,
    pub weight_decay: f64,
    pub warmup_steps: usize,
    pub warmdown_ratio: f64,
    pub final_lr_frac: f64,
    pub sft_schedule: bool,
    pub initial_lr_frac: f64,
    pub warmup_ratio: f64,
    pub sft_warmdown_ratio: f64,
    pub full_epoch: bool,
    pub load_optimizer: bool,
    pub resume_data: bool,
    pub eval_every: usize,
    pub eval_tokens: usize,
    pub core_metric_every: usize,
    pub core_metric_max_per_task: usize,
    pub chatcore_every: usize,
    pub chatcore_max_cat: i64,
    pub chatcore_max_sample: usize,
    pub sample_every: usize,
    pub save_every: usize,
    pub output: std::path::PathBuf,
    pub run: Option<String>,
    pub wandb_config: serde_json::Value,
}

#[derive(Default)]
struct ResumeLoopState {
    val_bpb: Option<f64>,
    min_val_bpb: Option<f64>,
    smooth_train_loss: f64,
    total_training_time: f64,
}

pub fn train(
    config: GptConfig,
    corpus: &TokenCorpus,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
) -> Result<()> {
    validate_options(&config, options)?;
    let (model, variables) =
        GptModel::new_trainable_with_compute_dtype(config, device, options.compute_dtype)?;
    let model = model.with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let mut batches = CorpusBatchSource::new(
        corpus,
        options.batch_size,
        options.sequence_len,
        rank,
        world_size,
    )?;
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        None,
        None,
        None,
        Some(tokenizer),
        None,
    )
}

pub fn train_parquet(
    config: GptConfig,
    files: Vec<PathBuf>,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
) -> Result<()> {
    validate_options(&config, options)?;
    let (model, variables) =
        GptModel::new_trainable_with_compute_dtype(config, device, options.compute_dtype)?;
    let model = model.with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let mut batches = ParquetBatchSource::new_distributed(
        files.clone(),
        tokenizer,
        options.batch_size,
        options.sequence_len,
        false,
        None,
        rank,
        world_size,
    )?;
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        None,
        None,
        Some((tokenizer, files)),
        Some(tokenizer),
        None,
    )
}

pub fn train_jsonl(
    config: GptConfig,
    dataset: impl AsRef<Path>,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
    parse_jsonl: bool,
    validation_dataset: Option<PathBuf>,
) -> Result<()> {
    validate_options(&config, options)?;
    let (model, variables) =
        GptModel::new_trainable_with_compute_dtype(config, device, options.compute_dtype)?;
    let model = model.with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let mut batches = JsonlBatchSource::new(
        dataset,
        tokenizer,
        options.batch_size,
        options.sequence_len,
        None,
        parse_jsonl,
        rank,
        world_size,
    )?;
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        None,
        None,
        None,
        Some(tokenizer),
        validation_dataset.map(|path| (tokenizer, path, parse_jsonl)),
    )
}

pub fn fine_tune(
    config: GptConfig,
    checkpoint: impl AsRef<Path>,
    corpus: &TokenCorpus,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
) -> Result<()> {
    validate_options(&config, options)?;
    let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
    let (model, variables) = if checkpoint
        .extension()
        .and_then(|extension| extension.to_str())
        == Some("safetensors")
    {
        GptModel::load_trainable_safetensors(config, &checkpoint, device, DType::F32)?
    } else {
        GptModel::load_trainable_pth(config, &checkpoint, device, DType::F32)?
    };
    let model = model
        .with_compute_dtype(options.compute_dtype)
        .with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let mut batches = CorpusBatchSource::new(
        corpus,
        options.batch_size,
        options.sequence_len,
        rank,
        world_size,
    )?;
    let state_directory = options
        .load_optimizer
        .then(|| checkpoint.parent())
        .flatten();
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        state_directory,
        options
            .load_optimizer
            .then(|| numbered_checkpoint_step(&checkpoint))
            .flatten(),
        None,
        Some(tokenizer),
        None,
    )
}

pub fn fine_tune_parquet(
    config: GptConfig,
    checkpoint: impl AsRef<Path>,
    files: Vec<PathBuf>,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
) -> Result<()> {
    validate_options(&config, options)?;
    let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
    let (model, variables) = if checkpoint
        .extension()
        .and_then(|extension| extension.to_str())
        == Some("safetensors")
    {
        GptModel::load_trainable_safetensors(config, &checkpoint, device, DType::F32)?
    } else {
        GptModel::load_trainable_pth(config, &checkpoint, device, DType::F32)?
    };
    let model = model
        .with_compute_dtype(options.compute_dtype)
        .with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let resume_data_loader = if options.resume_data {
        saved_data_loader_state(&checkpoint, rank, numbered_checkpoint_step(&checkpoint))?
    } else {
        None
    };
    let mut batches = ParquetBatchSource::new_distributed(
        files.clone(),
        tokenizer,
        options.batch_size,
        options.sequence_len,
        false,
        resume_data_loader,
        rank,
        world_size,
    )?;
    let state_directory = options
        .load_optimizer
        .then(|| checkpoint.parent())
        .flatten();
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        state_directory,
        options
            .load_optimizer
            .then(|| numbered_checkpoint_step(&checkpoint))
            .flatten(),
        Some((tokenizer, files)),
        Some(tokenizer),
        None,
    )
}

pub fn fine_tune_jsonl(
    config: GptConfig,
    checkpoint: impl AsRef<Path>,
    dataset: impl AsRef<Path>,
    tokenizer: &BpeTokenizer,
    options: &TrainOptions,
    device: &Device,
    parse_jsonl: bool,
    validation_dataset: Option<PathBuf>,
) -> Result<()> {
    validate_options(&config, options)?;
    let checkpoint = crate::checkpoint::resolve_model_file(checkpoint)?;
    let (model, variables) = if checkpoint
        .extension()
        .and_then(|extension| extension.to_str())
        == Some("safetensors")
    {
        GptModel::load_trainable_safetensors(config, &checkpoint, device, DType::F32)?
    } else {
        GptModel::load_trainable_pth(config, &checkpoint, device, DType::F32)?
    };
    let model = model
        .with_compute_dtype(options.compute_dtype)
        .with_fp8_training(options.fp8);
    let (rank, world_size) = crate::distributed::DistributedContext::env_group()?;
    let resume = if options.resume_data {
        saved_jsonl_state(&checkpoint, rank, numbered_checkpoint_step(&checkpoint))?
    } else {
        None
    };
    let mut batches = JsonlBatchSource::new(
        dataset,
        tokenizer,
        options.batch_size,
        options.sequence_len,
        resume,
        parse_jsonl,
        rank,
        world_size,
    )?;
    let state_directory = options
        .load_optimizer
        .then(|| checkpoint.parent())
        .flatten();
    train_model(
        model,
        variables,
        &mut batches,
        options,
        device,
        state_directory,
        options
            .load_optimizer
            .then(|| numbered_checkpoint_step(&checkpoint))
            .flatten(),
        None,
        Some(tokenizer),
        validation_dataset.map(|path| (tokenizer, path, parse_jsonl)),
    )
}

fn validate_options(config: &GptConfig, options: &TrainOptions) -> Result<()> {
    ensure!(
        options.sequence_len > 1 && options.sequence_len <= config.sequence_len,
        "training sequence length is outside model context"
    );
    ensure!(
        options.batch_size > 0 && options.steps > 0,
        "batch size and step count must be positive"
    );
    ensure!(
        options.grad_accum_steps > 0,
        "gradient accumulation steps must be positive"
    );
    ensure!(
        options.eval_every == 0 || options.eval_tokens >= options.batch_size * options.sequence_len,
        "evaluation token budget must cover at least one batch"
    );
    Ok(())
}

/// Return `(start_step, total_horizon, steps_to_run)` for the selected stage.
/// Base training resumes an absolute global horizon; SFT starts a fresh stage
/// while retaining any warm-started optimizer moments.
fn resolve_loop_horizon(
    requested_steps: usize,
    loaded_optimizer_step: Option<usize>,
    sft_schedule: bool,
) -> (usize, usize, usize) {
    let start_step = if sft_schedule {
        0
    } else {
        loaded_optimizer_step.unwrap_or(0)
    };
    let total_steps = if start_step > 0 {
        requested_steps.max(start_step)
    } else {
        requested_steps
    };
    (
        start_step,
        total_steps,
        total_steps.saturating_sub(start_step),
    )
}

fn reference_wandb_log_step(step: usize, sft_schedule: bool) -> Option<usize> {
    if sft_schedule {
        let sft_step = step + 1;
        (sft_step % 10 == 0).then_some(sft_step)
    } else {
        (step % 100 == 0).then_some(step)
    }
}

fn reference_log_step(step: usize, sft_schedule: bool) -> usize {
    step + usize::from(sft_schedule)
}

fn sft_updates_for_batch_limit(limit: usize, grad_accum_steps: usize) -> usize {
    limit.saturating_sub(1).div_ceil(grad_accum_steps)
}

fn run_with_prefetched_batch<T>(
    batches: &mut impl BatchSource,
    compute: impl FnOnce() -> Result<T>,
) -> Result<(T, TokenBatch, f64)> {
    std::thread::scope(|scope| -> Result<_> {
        let prefetch = scope.spawn(|| batches.next_batch());
        let compute = compute();
        let wait_started = Instant::now();
        let next_batch = prefetch
            .join()
            .map_err(|_| anyhow::anyhow!("batch prefetch thread panicked"))??;
        Ok((compute?, next_batch, wait_started.elapsed().as_secs_f64()))
    })
}

fn resolve_sft_progress(
    step: usize,
    total_steps: usize,
    grad_accum_steps: usize,
    steps_explicit: bool,
    data_progress: Option<f64>,
) -> f64 {
    if steps_explicit {
        let fetched_batches = step
            .saturating_add(1)
            .saturating_mul(grad_accum_steps)
            .saturating_add(1);
        fetched_batches as f64 / total_steps.max(1) as f64
    } else {
        data_progress.unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        reference_log_step, reference_wandb_log_step, resolve_loop_horizon, resolve_sft_progress,
        run_with_prefetched_batch, sft_updates_for_batch_limit, training_logits_for_positions,
    };

    #[test]
    fn next_training_batch_is_prefetched_during_microbatch_compute() {
        use anyhow::Result;
        use std::sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        };
        use std::time::{Duration, Instant};

        use crate::data::{BatchSource, TokenBatch};

        struct Source(Arc<AtomicBool>);

        impl BatchSource for Source {
            fn next_batch(&mut self) -> Result<TokenBatch> {
                self.0.store(true, Ordering::Release);
                Ok(TokenBatch {
                    inputs: vec![1, 2],
                    valid_positions: vec![0],
                    valid_targets: vec![2],
                })
            }
        }

        let started = Arc::new(AtomicBool::new(false));
        let mut source = Source(Arc::clone(&started));
        let (value, batch, _) = run_with_prefetched_batch(&mut source, || {
            let deadline = Instant::now() + Duration::from_secs(1);
            while !started.load(Ordering::Acquire) {
                assert!(
                    Instant::now() < deadline,
                    "prefetch did not overlap compute"
                );
                std::thread::yield_now();
            }
            Ok(75)
        })
        .unwrap();
        assert_eq!(value, 75);
        assert_eq!(batch.valid_targets, vec![2]);
    }

    #[test]
    fn selected_training_logits_match_full_projection_and_gradients() {
        use crate::{model::GptModel, model_config::GptConfig};
        use candle_core::{DType, Device, Tensor};
        let config = GptConfig {
            sequence_len: 8,
            vocab_size: 64,
            n_layer: 2,
            n_head: 2,
            n_kv_head: 2,
            n_embd: 32,
            window_pattern: "L".into(),
        };
        let (model, variables) =
            GptModel::new_trainable_with_compute_dtype(config, &Device::Cpu, DType::F32).unwrap();
        let inputs: Vec<u32> = (0..8).collect();
        for positions in [vec![0, 1, 2, 3, 4, 5, 6, 7], vec![1, 4, 7]] {
            let actual = training_logits_for_positions(&model, &inputs, 1, 8, &positions).unwrap();
            let full = model
                .forward_training_logits(&inputs, 1, 8)
                .unwrap()
                .reshape((8, 64))
                .unwrap();
            let reference = full
                .index_select(
                    &Tensor::from_vec(positions.clone(), positions.len(), &Device::Cpu).unwrap(),
                    0,
                )
                .unwrap();
            let max_error = actual
                .sub(&reference)
                .unwrap()
                .abs()
                .unwrap()
                .max_all()
                .unwrap()
                .to_scalar::<f32>()
                .unwrap();
            assert!(max_error < 1e-5, "logits differ: {max_error}");
            let actual_grad = actual.sum_all().unwrap().backward().unwrap();
            let reference_grad = reference.sum_all().unwrap().backward().unwrap();
            for variable in variables.all_vars() {
                match (actual_grad.get(&variable), reference_grad.get(&variable)) {
                    (Some(a), Some(b)) => {
                        let error = a
                            .sub(b)
                            .unwrap()
                            .abs()
                            .unwrap()
                            .max_all()
                            .unwrap()
                            .to_scalar::<f32>()
                            .unwrap();
                        assert!(error < 1e-4, "gradient differs: {error}");
                    }
                    (None, None) => (),
                    _ => panic!("gradient missing from one path"),
                }
            }
        }
    }

    #[test]
    fn base_resume_uses_absolute_global_horizon() {
        assert_eq!(resolve_loop_horizon(2, Some(1), false), (1, 2, 1));
    }

    #[test]
    fn sft_restarts_stage_counter_but_keeps_optimizer_load_separate() {
        assert_eq!(resolve_loop_horizon(2, Some(100), true), (0, 2, 2));
    }

    #[test]
    fn base_resume_already_past_horizon_runs_no_updates() {
        assert_eq!(resolve_loop_horizon(4, Some(5), false), (5, 5, 0));
    }

    #[test]
    fn reference_wandb_training_log_cadence_matches_base_and_sft() {
        assert_eq!(reference_wandb_log_step(0, false), Some(0));
        assert_eq!(reference_wandb_log_step(99, false), None);
        assert_eq!(reference_wandb_log_step(100, false), Some(100));
        assert_eq!(reference_wandb_log_step(8, true), None);
        assert_eq!(reference_wandb_log_step(9, true), Some(10));
        assert_eq!(reference_wandb_log_step(19, true), Some(20));
    }

    #[test]
    fn reference_log_steps_match_base_and_sft_loss_debiasing() {
        assert_eq!(reference_log_step(0, false), 0);
        assert_eq!(reference_log_step(0, true), 1);
        assert_eq!(reference_log_step(10, true), 11);
    }

    #[test]
    fn sft_progress_follows_reference_generator_count() {
        assert_eq!(resolve_sft_progress(0, 10, 1, true, Some(0.9)), 0.2);
        assert_eq!(resolve_sft_progress(1, 10, 2, true, Some(0.9)), 0.5);
        assert_eq!(resolve_sft_progress(0, 10, 1, false, Some(0.1)), 0.1);
        assert_eq!(resolve_sft_progress(0, 10, 1, false, None), 0.0);
    }

    #[test]
    fn sft_batch_limit_counts_initial_prefetch_and_accumulation() {
        assert_eq!(sft_updates_for_batch_limit(1, 1), 0);
        assert_eq!(sft_updates_for_batch_limit(10, 1), 9);
        assert_eq!(sft_updates_for_batch_limit(10, 2), 5);
        assert_eq!(sft_updates_for_batch_limit(11, 2), 5);
    }
}

fn saved_data_loader_state(
    checkpoint: &Path,
    rank: usize,
    step: Option<usize>,
) -> Result<Option<crate::datasets::DataLoaderState>> {
    let Some(metadata_path) = checkpoint.parent().map(|parent| parent.join("meta.json")) else {
        return Ok(None);
    };
    if !metadata_path.is_file() {
        return Ok(None);
    }
    let rank_state_path = metadata_path.with_file_name(match step {
        Some(step) => format!("rank_{rank}_{step:06}.json"),
        None => format!("rank_{rank}.json"),
    });
    let state_path = if rank_state_path.is_file() {
        rank_state_path
    } else if let Some(step) = step {
        let numbered_metadata = metadata_path.with_file_name(format!("meta_{step:06}.json"));
        if numbered_metadata.is_file() {
            numbered_metadata
        } else {
            return Ok(None);
        }
    } else {
        metadata_path
    };
    let metadata = crate::json_compat::parse_value(
        &std::fs::read(&state_path)
            .with_context(|| format!("read checkpoint metadata {}", state_path.display()))?,
    )?;
    metadata
        .get("dataloader_state_dict")
        .filter(|state| !state.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(Into::into)
}

fn saved_jsonl_state(
    checkpoint: &Path,
    rank: usize,
    step: Option<usize>,
) -> Result<Option<crate::data::JsonlResume>> {
    let Some(metadata_path) = checkpoint.parent().map(|parent| parent.join("meta.json")) else {
        return Ok(None);
    };
    if !metadata_path.is_file() {
        return Ok(None);
    }
    let rank_state_path = metadata_path.with_file_name(match step {
        Some(step) => format!("rank_{rank}_{step:06}.json"),
        None => format!("rank_{rank}.json"),
    });
    let state_path = if rank_state_path.is_file() {
        rank_state_path
    } else if let Some(step) = step {
        let numbered_metadata = metadata_path.with_file_name(format!("meta_{step:06}.json"));
        if numbered_metadata.is_file() {
            numbered_metadata
        } else {
            return Ok(None);
        }
    } else {
        metadata_path
    };
    let metadata = crate::json_compat::parse_value(
        &std::fs::read(&state_path)
            .with_context(|| format!("read checkpoint metadata {}", state_path.display()))?,
    )?;
    metadata
        .get("jsonl_state_dict")
        .filter(|state| !state.is_null())
        .cloned()
        .map(serde_json::from_value)
        .transpose()
        .map_err(Into::into)
}

struct CorpusBatchSource<'a> {
    corpus: &'a TokenCorpus,
    batch_size: usize,
    sequence_len: usize,
    next_start: usize,
    rank_start: usize,
    stride_per_rank: usize,
    epoch_span: usize,
}

impl<'a> CorpusBatchSource<'a> {
    fn new(
        corpus: &'a TokenCorpus,
        batch_size: usize,
        sequence_len: usize,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        let stride = batch_size * sequence_len;
        let distributed_stride = stride * world_size;
        let available = corpus.ids.len().saturating_sub(stride);
        let blocks = available / distributed_stride;
        ensure!(
            rank < world_size && blocks > 0,
            "dataset is too small for the selected batch/sequence lengths across {world_size} ranks"
        );
        Ok(Self {
            corpus,
            batch_size,
            sequence_len,
            next_start: rank * stride,
            rank_start: rank * stride,
            stride_per_rank: distributed_stride,
            epoch_span: blocks * distributed_stride,
        })
    }
}

impl BatchSource for CorpusBatchSource<'_> {
    fn next_batch(&mut self) -> Result<TokenBatch> {
        let stride = self.batch_size * self.sequence_len;
        let start = self.next_start;
        self.next_start += self.stride_per_rank;
        if self.next_start >= self.epoch_span {
            self.next_start = self.rank_start;
        }
        let mut inputs = Vec::with_capacity(stride);
        let mut valid_positions = Vec::new();
        let mut valid_targets = Vec::new();
        for row in 0..self.batch_size {
            let row_start = start + row * self.sequence_len;
            inputs.extend_from_slice(&self.corpus.ids[row_start..row_start + self.sequence_len]);
            for position in 0..self.sequence_len {
                let target_index = row_start + position + 1;
                if self.corpus.loss_mask[target_index] != 0 {
                    valid_positions.push((row * self.sequence_len + position) as u32);
                    valid_targets.push(self.corpus.ids[target_index]);
                }
            }
        }
        ensure!(
            !valid_positions.is_empty(),
            "batch contains no supervised target tokens"
        );
        Ok(TokenBatch {
            inputs,
            valid_positions,
            valid_targets,
        })
    }
}

fn train_model(
    model: GptModel,
    variables: VarMap,
    batches: &mut impl BatchSource,
    options: &TrainOptions,
    device: &Device,
    optimizer_state_directory: Option<&Path>,
    optimizer_checkpoint_step: Option<usize>,
    mut validation: Option<(&BpeTokenizer, Vec<PathBuf>)>,
    evaluation_tokenizer: Option<&BpeTokenizer>,
    validation_jsonl: Option<(&BpeTokenizer, PathBuf, bool)>,
) -> Result<()> {
    let mut parameters: Vec<(String, candle_core::Var)> = variables
        .data()
        .lock()
        .map_err(|_| anyhow::anyhow!("model variable lock is poisoned"))?
        .iter()
        .map(|(name, parameter)| (name.clone(), parameter.clone()))
        .collect();
    parameters.sort_by(|left, right| left.0.cmp(&right.0));
    let mut optimizer = MuonAdamW::new(
        parameters.clone(),
        model.config.n_embd,
        MuonAdamWConfig {
            embedding_lr: options.embedding_lr,
            unembedding_lr: options.unembedding_lr,
            matrix_lr: options.matrix_lr,
            scalar_lr: options.scalar_lr,
            weight_decay: if options.sft_schedule {
                0.0
            } else {
                options.weight_decay
            },
        },
    )?;
    let mut distributed = crate::distributed::DistributedContext::from_env()?;
    ensure!(
        !options.full_epoch || batches.training_progress().is_some(),
        "--full-epoch requires a dataset source with known conversation progress (JSONL)"
    );
    distributed.broadcast_parameters(&parameters)?;
    optimizer.shard_state(distributed.rank(), distributed.world_size())?;
    let loaded_optimizer_step = optimizer_state_directory
        .map(|directory| {
            optimizer.load_state(directory, optimizer_checkpoint_step, &mut distributed)
        })
        .transpose()?
        .flatten();
    // Base pretraining resumes the absolute loop horizon from checkpoint
    // metadata. SFT starts a fresh stage at step zero while retaining any
    // warm-started optimizer moments from its input checkpoint.
    let resumed_loop_step = loaded_optimizer_step.filter(|_| !options.sft_schedule);
    let (start_step, total_steps, mut steps_to_run) =
        resolve_loop_horizon(options.steps, loaded_optimizer_step, options.sft_schedule);
    if options.sft_schedule && options.steps_explicit {
        steps_to_run = sft_updates_for_batch_limit(total_steps, options.grad_accum_steps);
    }
    let resume_loop_state = match (resumed_loop_step, optimizer_state_directory) {
        (Some(step), Some(directory)) => load_resume_loop_state(directory, Some(step))?,
        _ => ResumeLoopState::default(),
    };
    let training_started = Instant::now();
    let fp16_scaling = options.compute_dtype == DType::F16 && device.is_cuda();
    let mut loss_scale = if fp16_scaling { 65_536.0 } else { 1.0 };
    let mut loss_scale_growth_tracker = 0usize;
    let flops_per_token = GptModel::training_flops_per_token(&model.config) as f64;
    let gpu_peak_flops = device_peak_flops(device);
    let global_tokens_per_step = options.batch_size
        * options.sequence_len
        * options.grad_accum_steps
        * distributed.world_size();
    let mut val_bpb = resume_loop_state.val_bpb;
    let mut min_val_bpb = resume_loop_state.min_val_bpb;
    let mut smooth_train_loss = resume_loop_state.smooth_train_loss;
    let mut total_training_time = resume_loop_state.total_training_time;
    let eval_engine = evaluation_tokenizer.map(|tokenizer| {
        crate::engine::Engine::new(model.clone().with_fp8_training(false), tokenizer.clone())
    });
    let eval_model = model.clone().with_fp8_training(false);
    if options.eval_every > 0 && validation.is_none() && validation_jsonl.is_none() {
        anyhow::bail!("periodic BPB evaluation requires a validation dataset");
    }
    if (options.core_metric_every > 0 || options.chatcore_every > 0 || options.sample_every > 0)
        && eval_engine.is_none()
    {
        anyhow::bail!("CORE, ChatCORE, and sample evaluation require a tokenizer");
    }
    ensure!(
        options.chatcore_max_cat >= -1,
        "--chatcore-max-cat must be -1 or non-negative"
    );
    if options.chatcore_every > 0 {
        ensure!(
            options.chatcore_max_cat != 0,
            "--chatcore-max-cat must be -1 or positive when ChatCORE is enabled"
        );
        ensure!(
            options.chatcore_max_sample > 0,
            "--chatcore-max-sample must be positive when ChatCORE is enabled"
        );
    }
    if distributed.rank() == 0 {
        std::fs::create_dir_all(&options.output)?;
    }
    let mut wandb_run = if distributed.rank() == 0 {
        options
            .run
            .as_deref()
            .map(|name| {
                let core_path = crate::wandb::resolve_core_binary()?;
                let project = if options.sft_schedule {
                    "nanochat-sft"
                } else {
                    "nanochat"
                };
                crate::wandb::Run::start(
                    core_path,
                    project,
                    name,
                    &options.wandb_config,
                    Duration::from_secs(30),
                )
            })
            .transpose()?
    } else {
        None
    };
    if options.sft_schedule {
        batches.prefetch_batch()?;
    }
    let mut local_step = 0usize;
    let profile_steps = std::env::var_os("RUST_AGENT_PROFILE_STEPS").is_some();
    while local_step < steps_to_run {
        let stop_at_epoch =
            options.full_epoch || (options.sft_schedule && batches.training_progress().is_some());
        let local_epoch_done = stop_at_epoch
            && batches
                .training_progress()
                .is_some_and(|progress| progress.fraction >= 1.0);
        let epoch_done = if stop_at_epoch {
            let (finished_ranks, _) =
                distributed.all_reduce_counts(u64::from(local_epoch_done), 0)?;
            finished_ranks > 0
        } else {
            false
        };
        if epoch_done {
            break;
        }
        let step = start_step + local_step;
        if options.eval_every > 0 && step % options.eval_every == 0 {
            let current_val_bpb = if let Some((tokenizer, files)) = validation.as_mut() {
                crate::evaluation::evaluate_parquet_bpb_with_context(
                    &eval_model,
                    tokenizer,
                    files.clone(),
                    true,
                    options.sequence_len,
                    options.batch_size,
                    options.eval_tokens,
                    device,
                    &mut distributed,
                )?
            } else if let Some((tokenizer, path, parse_jsonl)) = validation_jsonl.as_ref() {
                evaluate_jsonl_bpb_with_context(
                    &eval_model,
                    tokenizer,
                    path,
                    *parse_jsonl,
                    options.sequence_len,
                    options.batch_size,
                    options.eval_tokens,
                    device,
                    &mut distributed,
                )?
            } else {
                anyhow::bail!("periodic BPB evaluation requires a validation dataset");
            };
            val_bpb = Some(current_val_bpb);
            min_val_bpb =
                Some(min_val_bpb.map_or(current_val_bpb, |best| best.min(current_val_bpb)));
            if distributed.rank() == 0 {
                println!("step {step}/{total_steps} validation_bpb {current_val_bpb:.6}");
                append_training_metric(
                    &options.output,
                    serde_json::json!({
                        "step": step,
                        "val/bpb": current_val_bpb,
                    }),
                    wandb_run.as_mut(),
                )?;
            }
        }
        if options.core_metric_every > 0 && step > 0 && step % options.core_metric_every == 0 {
            let results = crate::core_eval::evaluate_core_with_context(
                eval_engine.as_ref().expect("evaluation tokenizer checked"),
                device,
                Some(options.core_metric_max_per_task),
                &mut distributed,
            )?;
            if distributed.rank() == 0 {
                append_training_metric(
                    &options.output,
                    serde_json::json!({
                        "step": step,
                        "core_metric": results["core_metric"],
                        "centered_results": results["centered_results"],
                    }),
                    wandb_run.as_mut(),
                )?;
            }
        }
        if options.chatcore_every > 0 && step > 0 && step % options.chatcore_every == 0 {
            let results = evaluate_training_chatcore(
                eval_engine.as_ref().expect("evaluation tokenizer checked"),
                options,
            )?;
            if distributed.rank() == 0 {
                println!(
                    "step {step}/{total_steps} ChatCORE {:.4} categorical {:.4}",
                    results["ChatCORE"].as_f64().unwrap_or_default(),
                    results["ChatCORE_cat"].as_f64().unwrap_or_default(),
                );
                append_training_metric(
                    &options.output,
                    serde_json::json!({ "step": step, "chatcore": results }),
                    wandb_run.as_mut(),
                )?;
            }
        }
        if options.sample_every > 0
            && distributed.rank() == 0
            && step > 0
            && step % options.sample_every == 0
        {
            let samples = sample_training_prompts(
                eval_engine.as_ref().expect("evaluation tokenizer checked"),
            )?;
            append_training_metric(
                &options.output,
                serde_json::json!({ "step": step, "samples": samples }),
                wandb_run.as_mut(),
            )?;
        }
        let step_started = Instant::now();
        let mut accumulated = GradStore::default();
        let mut supervised_tokens = 0usize;
        let mut weighted_loss = 0.0f64;
        let mut last_batch_loss = 0.0f64;
        let mut data_seconds = 0.0f64;
        let mut forward_loss_seconds = 0.0f64;
        let mut backward_seconds = 0.0f64;
        let mut gradient_transfer_seconds = 0.0f64;
        // Long-context CUDA training can use most device memory for one
        // microbatch. Keep gradients from earlier microbatches on the host so
        // they do not compete with the next forward/backward pass.
        let host_accumulation = device.is_cuda() && options.grad_accum_steps > 1;
        let mut prefetched_batch = None;
        for microbatch_index in 0..options.grad_accum_steps {
            let data_started = Instant::now();
            let mut batch = match prefetched_batch.take() {
                Some(batch) => batch,
                None => batches.next_batch()?,
            };
            data_seconds += data_started.elapsed().as_secs_f64();
            let batch_tokens = batch.valid_targets.len();
            let forward_started = Instant::now();
            let mut run_microbatch = || -> Result<(GradStore, f64, f64, f64)> {
                let selected = training_logits_for_positions(
                    &model,
                    &batch.inputs,
                    options.batch_size,
                    options.sequence_len,
                    &batch.valid_positions,
                )?;
                let targets = Tensor::from_vec(
                    std::mem::take(&mut batch.valid_targets),
                    (selected.dim(0)?,),
                    device,
                )?;
                let loss = crate::model::training_softcap_cross_entropy(&selected, &targets)?;
                let loss_value = loss.to_scalar::<f32>()? as f64;
                let forward_seconds = forward_started.elapsed().as_secs_f64();
                // Python's train loops average each microbatch's mean loss
                // equally, even when SFT batches contain different numbers
                // of supervised assistant tokens.
                let scaled_loss = loss.affine(loss_scale / options.grad_accum_steps as f64, 0.0)?;
                let backward_started = Instant::now();
                let gradients = scaled_loss.backward()?;
                let backward_seconds = backward_started.elapsed().as_secs_f64();
                Ok((gradients, loss_value, forward_seconds, backward_seconds))
            };
            let (
                mut batch_gradients,
                batch_loss,
                forward_seconds,
                backward_elapsed,
                next_batch,
                wait,
            ) = if microbatch_index + 1 < options.grad_accum_steps {
                let ((gradients, loss, forward, backward), next_batch, wait) =
                    run_with_prefetched_batch(batches, run_microbatch)?;
                (gradients, loss, forward, backward, Some(next_batch), wait)
            } else {
                let (gradients, loss, forward, backward) = run_microbatch()?;
                (gradients, loss, forward, backward, None, 0.0)
            };
            prefetched_batch = next_batch;
            data_seconds += wait;
            last_batch_loss = batch_loss;
            forward_loss_seconds += forward_seconds;
            weighted_loss += last_batch_loss * batch_tokens as f64;
            backward_seconds += backward_elapsed;
            let gradient_transfer_started = Instant::now();
            if host_accumulation {
                for (_, parameter) in &parameters {
                    if let Some(gradient) = batch_gradients.remove(parameter.as_tensor()) {
                        let gradient = gradient.to_device(&Device::Cpu)?;
                        let combined = match accumulated.remove(parameter.as_tensor()) {
                            Some(previous) => previous.add(&gradient)?.detach(),
                            None => gradient,
                        };
                        accumulated.insert(parameter.as_tensor(), combined);
                    }
                }
            } else {
                accumulated.extend(batch_gradients)?;
            }
            gradient_transfer_seconds += gradient_transfer_started.elapsed().as_secs_f64();
            supervised_tokens += batch_tokens;
        }
        if options.sft_schedule {
            let data_started = Instant::now();
            batches.prefetch_batch()?;
            data_seconds += data_started.elapsed().as_secs_f64();
        }
        ensure!(
            supervised_tokens > 0,
            "accumulated batches contain no supervised targets"
        );
        if host_accumulation {
            let gradient_transfer_started = Instant::now();
            for (_, parameter) in &parameters {
                if let Some(gradient) = accumulated.remove(parameter.as_tensor()) {
                    accumulated.insert(parameter.as_tensor(), gradient.to_device(device)?);
                }
            }
            gradient_transfer_seconds += gradient_transfer_started.elapsed().as_secs_f64();
        }
        // Gradients already contain the per-microbatch mean divided by the
        // accumulation count. Average ranks equally, as PyTorch DDP does;
        // supervised-token weighting is used only for the aggregate metric.
        let scaled_gradients = distributed.average_gradients(&parameters, &accumulated)?;
        let mut squared_gradient_norm = None;
        for (_, parameter) in &parameters {
            if let Some(gradient) = scaled_gradients.get(parameter.as_tensor()) {
                let squared_norm = gradient.to_dtype(DType::F32)?.sqr()?.sum_all()?;
                squared_gradient_norm = Some(match squared_gradient_norm {
                    Some(total) => (total + squared_norm)?,
                    None => squared_norm,
                });
            }
        }
        let overflow = squared_gradient_norm
            .map(|norm| norm.to_scalar::<f32>().map(|value| !value.is_finite()))
            .transpose()?
            .unwrap_or(false);
        let mut gradients = GradStore::default();
        let optimizer_started = Instant::now();
        if !overflow {
            for (_, parameter) in &parameters {
                if let Some(gradient) = scaled_gradients.get(parameter.as_tensor()) {
                    gradients.insert(
                        parameter.as_tensor(),
                        gradient
                            .to_dtype(DType::F32)?
                            .affine(1.0 / loss_scale, 0.0)?,
                    );
                }
            }
        }
        if fp16_scaling {
            if overflow {
                loss_scale = (loss_scale * 0.5).max(1.0);
                loss_scale_growth_tracker = 0;
            } else {
                loss_scale_growth_tracker += 1;
                if loss_scale_growth_tracker == 2_000 {
                    loss_scale *= 2.0;
                    loss_scale_growth_tracker = 0;
                }
            }
        }
        let dataset_progress = batches.training_progress();
        let schedule_progress = if options.sft_schedule {
            resolve_sft_progress(
                step,
                total_steps,
                options.grad_accum_steps,
                options.steps_explicit,
                dataset_progress.map(|progress| progress.fraction),
            )
        } else {
            (step as f64 / total_steps.max(1) as f64).clamp(0.0, 1.0)
        };
        let schedule_warmdown_ratio = if options.sft_schedule {
            options.sft_warmdown_ratio
        } else {
            options.warmdown_ratio
        };
        let warmdown = (schedule_warmdown_ratio * total_steps as f64).round() as usize;
        let lr_multiplier = if options.sft_schedule {
            if schedule_progress < options.warmup_ratio {
                if options.warmup_ratio > 0.0 {
                    (schedule_progress + 1e-8) / options.warmup_ratio
                } else {
                    1.0
                }
            } else if schedule_progress <= 1.0 - schedule_warmdown_ratio {
                1.0
            } else if schedule_warmdown_ratio > 0.0 {
                let decay =
                    (schedule_progress - (1.0 - schedule_warmdown_ratio)) / schedule_warmdown_ratio;
                (1.0 - decay) + decay * options.final_lr_frac
            } else {
                options.final_lr_frac
            }
        } else {
            let warmup = options.warmup_steps.max(1);
            if step < options.warmup_steps {
                (step + 1) as f64 / warmup as f64
            } else if step <= total_steps.saturating_sub(warmdown) {
                1.0
            } else if warmdown > 0 {
                let progress = (total_steps - step) as f64 / warmdown as f64;
                progress + (1.0 - progress) * options.final_lr_frac
            } else {
                options.final_lr_frac
            }
        };
        let warmdown_start = total_steps.saturating_sub(warmdown);
        let momentum = if options.sft_schedule {
            let progress = (step as f64 / 300.0).min(1.0);
            0.85 + (0.95 - 0.85) * progress
        } else if step < 400 {
            0.85 + (0.97 - 0.85) * (step as f64 / 400.0)
        } else if step >= warmdown_start && warmdown > 0 {
            let progress = (step - warmdown_start) as f64 / warmdown as f64;
            0.97 * (1.0 - progress) + 0.90 * progress
        } else {
            0.97
        };
        let decay_multiplier =
            0.5 * (1.0 + (std::f64::consts::PI * step as f64 / total_steps as f64).cos());
        let effective_lr_multiplier = if options.sft_schedule {
            lr_multiplier * options.initial_lr_frac
        } else {
            lr_multiplier
        };
        if !overflow {
            optimizer.step_distributed(
                &gradients,
                effective_lr_multiplier,
                momentum,
                decay_multiplier,
                options.compute_dtype,
                &mut distributed,
            )?;
        }
        let optimizer_seconds = optimizer_started.elapsed().as_secs_f64();
        let mean_loss = weighted_loss / supervised_tokens as f64;
        if distributed.rank() == 0 {
            let step_elapsed = step_started.elapsed().as_secs_f64();
            smooth_train_loss = 0.9 * smooth_train_loss + 0.1 * last_batch_loss;
            let logged_step = reference_log_step(step, options.sft_schedule);
            if logged_step > 10 {
                total_training_time += step_elapsed;
            }
            let debiased_smooth_loss =
                smooth_train_loss / (1.0 - 0.9f64.powf((logged_step + 1) as f64));
            let elapsed = training_started.elapsed().as_secs_f64();
            let tokens_per_second =
                (global_tokens_per_step as f64 / step_elapsed.max(1e-9)).floor();
            let estimated_flops =
                (step + 1) as f64 * global_tokens_per_step as f64 * flops_per_token;
            let step_flops_per_second =
                flops_per_token * global_tokens_per_step as f64 / step_elapsed.max(1e-9);
            let estimated_flops_per_second = step_flops_per_second;
            let mfu_percent = gpu_peak_flops
                .map(|peak| {
                    step_flops_per_second / (peak * distributed.world_size() as f64) * 100.0
                })
                .unwrap_or(0.0);
            let steps_done = step.saturating_sub(10);
            let eta_seconds = if steps_done > 0 {
                total_training_time / steps_done as f64
                    * total_steps.saturating_sub(step + 1) as f64
            } else {
                0.0
            };
            let epoch_metric = if options.sft_schedule {
                dataset_progress.map_or(serde_json::Value::Null, |progress| {
                    serde_json::Value::from(progress.epoch)
                })
            } else if let Some(state) = batches.dataloader_state() {
                serde_json::Value::from(format!(
                    "{} pq: {} rg: {}",
                    state.epoch, state.pq_idx, state.rg_idx
                ))
            } else {
                dataset_progress.map_or(serde_json::Value::Null, |progress| {
                    serde_json::Value::from(progress.epoch)
                })
            };
            let epoch_display = if epoch_metric.is_null() {
                "-".to_owned()
            } else {
                epoch_metric
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| epoch_metric.to_string())
            };
            let progress_percent = if options.sft_schedule {
                schedule_progress * 100.0
            } else {
                step as f64 / total_steps.max(1) as f64 * 100.0
            };
            let display_step = if options.sft_schedule { step + 1 } else { step };
            let reference_log_step = reference_wandb_log_step(step, options.sft_schedule);
            let reference_wandb_record = reference_log_step.map(|reference_step| {
                serde_json::json!({
                    "step": reference_step,
                    "total_training_flops": step as f64 * global_tokens_per_step as f64 * flops_per_token,
                    "total_training_time": total_training_time,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lr_multiplier,
                    "train/dt": step_elapsed,
                    "train/tok_per_sec": tokens_per_second as u64,
                    "train/mfu": mfu_percent,
                    "train/epoch": epoch_metric,
                })
            });
            println!(
                "step {display_step:05}/{} ({progress_percent:.2}%) loss {:.6} lrm {:.2} dt {:.2}ms tok/sec {:.0} MFU {:.2}% epoch {} total time {:.2}m ETA {:.0}s",
                total_steps,
                debiased_smooth_loss,
                lr_multiplier,
                step_elapsed * 1000.0,
                tokens_per_second,
                mfu_percent,
                epoch_display,
                total_training_time / 60.0,
                eta_seconds
            );
            if profile_steps {
                let other_seconds = (step_elapsed
                    - data_seconds
                    - forward_loss_seconds
                    - backward_seconds
                    - gradient_transfer_seconds
                    - optimizer_seconds)
                    .max(0.0);
                println!(
                    "profile step {display_step:05}: data={data_seconds:.3}s forward_loss={forward_loss_seconds:.3}s backward={backward_seconds:.3}s grad_transfer={gradient_transfer_seconds:.3}s optimizer={optimizer_seconds:.3}s other={other_seconds:.3}s"
                );
            }
            std::io::stdout().flush()?;
            append_training_metric_with_payload(
                &options.output,
                serde_json::json!({
                    "step": step + 1,
                    "loss": mean_loss,
                    "train/loss": debiased_smooth_loss,
                    "train/lrm": lr_multiplier,
                    "train/dt": step_elapsed,
                    "train/tok_per_sec": tokens_per_second,
                    "train/mfu": mfu_percent,
                    "train/epoch": epoch_metric,
                    "fp16_loss_scale": fp16_scaling.then_some(loss_scale),
                    "fp16_overflow": fp16_scaling.then_some(overflow),
                    "lr_multiplier": lr_multiplier,
                    "elapsed_seconds": elapsed,
                    "total_training_time": total_training_time,
                    "tokens_per_second": tokens_per_second,
                    "estimated_training_flops": estimated_flops,
                    "estimated_flops_per_second": estimated_flops_per_second,
                    "step_flops_per_second": step_flops_per_second,
                    "eta_seconds": eta_seconds,
                    "dataset_progress": dataset_progress.map(|progress| progress.fraction),
                    "dataset_epoch": dataset_progress.map(|progress| progress.epoch),
                }),
                reference_wandb_record.as_ref(),
                wandb_run.as_mut(),
            )?;
        }
        let checkpoint_step = step + 1;
        if options.save_every > 0
            && checkpoint_step < total_steps
            && checkpoint_step % options.save_every == 0
        {
            save_rank_training_state(batches, options, &distributed, checkpoint_step)?;
            optimizer.save_state(
                &options.output,
                checkpoint_step,
                distributed.rank(),
                distributed.world_size(),
            )?;
            if distributed.rank() == 0 {
                save_numbered_checkpoint(
                    &variables,
                    &model,
                    batches,
                    options,
                    checkpoint_step,
                    distributed.world_size(),
                    val_bpb,
                    min_val_bpb,
                    smooth_train_loss,
                    total_training_time,
                )?;
            }
        }
        local_step += 1;
    }
    let final_step = start_step + local_step;
    let run_final_validation = options.eval_every > 0
        || (options.sft_schedule && (validation.is_some() || validation_jsonl.is_some()));
    if run_final_validation {
        let final_val_bpb = if let Some((tokenizer, files)) = validation.as_mut() {
            crate::evaluation::evaluate_parquet_bpb_with_context(
                &eval_model,
                tokenizer,
                files.clone(),
                true,
                options.sequence_len,
                options.batch_size,
                options.eval_tokens,
                device,
                &mut distributed,
            )?
        } else if let Some((tokenizer, path, parse_jsonl)) = validation_jsonl.as_ref() {
            evaluate_jsonl_bpb_with_context(
                &eval_model,
                tokenizer,
                path,
                *parse_jsonl,
                options.sequence_len,
                options.batch_size,
                options.eval_tokens,
                device,
                &mut distributed,
            )?
        } else {
            anyhow::bail!("final BPB evaluation requires a validation dataset");
        };
        if distributed.rank() == 0 {
            println!("step {final_step}/{total_steps} validation_bpb {final_val_bpb:.6}");
            append_training_metric(
                &options.output,
                serde_json::json!({ "step": final_step, "val/bpb": final_val_bpb }),
                wandb_run.as_mut(),
            )?;
        }
        val_bpb = Some(final_val_bpb);
        min_val_bpb = Some(min_val_bpb.map_or(final_val_bpb, |best| best.min(final_val_bpb)));
    }
    if options.core_metric_every > 0 {
        let results = crate::core_eval::evaluate_core_with_context(
            eval_engine.as_ref().expect("evaluation tokenizer checked"),
            device,
            Some(options.core_metric_max_per_task),
            &mut distributed,
        )?;
        if distributed.rank() == 0 {
            append_training_metric(
                &options.output,
                serde_json::json!({
                    "step": final_step,
                    "core_metric": results["core_metric"],
                    "centered_results": results["centered_results"],
                }),
                wandb_run.as_mut(),
            )?;
        }
    }
    if options.chatcore_every > 0 {
        let results = evaluate_training_chatcore(
            eval_engine.as_ref().expect("evaluation tokenizer checked"),
            options,
        )?;
        if distributed.rank() == 0 {
            println!(
                "step {final_step}/{total_steps} ChatCORE {:.4} categorical {:.4}",
                results["ChatCORE"].as_f64().unwrap_or_default(),
                results["ChatCORE_cat"].as_f64().unwrap_or_default(),
            );
            append_training_metric(
                &options.output,
                serde_json::json!({ "step": final_step, "chatcore": results }),
                wandb_run.as_mut(),
            )?;
        }
    }
    if options.sample_every > 0 && distributed.rank() == 0 {
        let samples =
            sample_training_prompts(eval_engine.as_ref().expect("evaluation tokenizer checked"))?;
        append_training_metric(
            &options.output,
            serde_json::json!({ "step": final_step, "samples": samples }),
            wandb_run.as_mut(),
        )?;
    }
    if distributed.rank() == 0 {
        std::fs::create_dir_all(&options.output)?;
    }
    save_rank_training_state(batches, options, &distributed, final_step)?;
    optimizer.save_state(
        &options.output,
        final_step,
        distributed.rank(),
        distributed.world_size(),
    )?;
    if distributed.rank() == 0 {
        save_numbered_checkpoint(
            &variables,
            &model,
            batches,
            options,
            final_step,
            distributed.world_size(),
            val_bpb,
            min_val_bpb,
            smooth_train_loss,
            total_training_time,
        )?;
        variables
            .save(options.output.join("model.safetensors"))
            .context("save Rust model checkpoint")?;
        crate::torch_save::save_state_dict(&variables, options.output.join("model.pt"))
            .context("save PyTorch-compatible model checkpoint")?;
        println!("Total training time: {:.2}m", total_training_time / 60.0);
        if let Some(minimum) = min_val_bpb {
            println!("Minimum validation bpb: {minimum:.6}");
        }
    }
    if let Some(run) = wandb_run.as_mut() {
        run.finish()?;
    }
    Ok(())
}

/// Avoid copying a complete logits matrix for dense batches, and avoid
/// projecting unsupervised positions through the vocabulary head for SFT.
fn training_logits_for_positions(
    model: &GptModel,
    inputs: &[u32],
    batch_size: usize,
    sequence_len: usize,
    positions: &[u32],
) -> Result<Tensor> {
    let total = batch_size
        .checked_mul(sequence_len)
        .context("batch size overflow")?;
    let dense = positions.len() == total
        && positions
            .iter()
            .enumerate()
            .all(|(index, &position)| position as usize == index);
    if dense {
        Ok(model
            .forward_training_logits(inputs, batch_size, sequence_len)?
            .reshape((total, model.config.vocab_size))?)
    } else {
        model.forward_positions_training_logits(inputs, batch_size, sequence_len, positions)
    }
}

fn device_peak_flops(device: &Device) -> Option<f64> {
    if !device.is_cuda() {
        return None;
    }
    let gpu_index = std::env::var("LOCAL_RANK").unwrap_or_else(|_| "0".to_owned());
    let output = Command::new("nvidia-smi")
        .args([
            "--id",
            gpu_index.as_str(),
            "--query-gpu=name",
            "--format=csv,noheader",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let name = String::from_utf8(output.stdout)
        .ok()?
        .trim()
        .to_ascii_lowercase();
    let models: &[(&[&str], f64)] = &[
        (&["gb200"], 2.5e15),
        (&["grace blackwell"], 2.5e15),
        (&["b200"], 2.25e15),
        (&["b100"], 1.8e15),
        (&["h200", "nvl"], 836e12),
        (&["h200", "pcie"], 836e12),
        (&["h200"], 989e12),
        (&["h100", "nvl"], 835e12),
        (&["h100", "pcie"], 756e12),
        (&["h100"], 989e12),
        (&["h800", "nvl"], 989e12),
        (&["h800"], 756e12),
        (&["a100"], 312e12),
        (&["a800"], 312e12),
        (&["a40"], 149.7e12),
        (&["a30"], 165e12),
        (&["l40s"], 362e12),
        (&["l40-s"], 362e12),
        (&["l40 s"], 362e12),
        (&["l4"], 121e12),
        (&["mi355"], 2.5e15),
        (&["mi325"], 1.3074e15),
        (&["mi300x"], 1.3074e15),
        (&["mi300a"], 980.6e12),
        (&["mi250x"], 383e12),
        (&["mi250"], 362.1e12),
        (&["5090"], 209.5e12),
        (&["4090"], 165.2e12),
        (&["3090"], 71e12),
    ];
    models
        .iter()
        .find(|(patterns, _)| patterns.iter().all(|pattern| name.contains(pattern)))
        .map(|(_, peak)| *peak)
}

fn evaluate_training_chatcore(
    engine: &crate::engine::Engine,
    options: &TrainOptions,
) -> Result<serde_json::Value> {
    use crate::task_eval::{ChatTask, evaluate_chat_tasks};

    let categorical_limit =
        (options.chatcore_max_cat >= 0).then_some(options.chatcore_max_cat as usize);
    let categorical = [ChatTask::ArcEasy, ChatTask::ArcChallenge, ChatTask::Mmlu];
    let generative = [ChatTask::Gsm8k, ChatTask::HumanEval];
    let categorical_results = evaluate_chat_tasks(
        engine,
        &categorical,
        options.batch_size,
        1,
        512,
        0.0,
        Some(50),
        42,
        categorical_limit,
        None,
    )?;
    let generative_results = evaluate_chat_tasks(
        engine,
        &generative,
        options.batch_size,
        1,
        512,
        0.0,
        Some(50),
        42,
        Some(options.chatcore_max_sample),
        None,
    )?;
    let mut results = categorical_results.as_object().cloned().unwrap_or_default();
    results.extend(generative_results.as_object().cloned().unwrap_or_default());
    let centered = |task: ChatTask| {
        let accuracy = results[task.name()].as_f64().unwrap_or_default();
        let baseline = if matches!(
            task,
            ChatTask::ArcEasy | ChatTask::ArcChallenge | ChatTask::Mmlu
        ) {
            0.25
        } else {
            0.0
        };
        (accuracy - baseline) / (1.0 - baseline)
    };
    let all_score = (categorical
        .iter()
        .chain(generative.iter())
        .map(|task| centered(*task))
        .sum::<f64>())
        / 5.0;
    let categorical_score = categorical.iter().map(|task| centered(*task)).sum::<f64>() / 3.0;
    results.insert("ChatCORE".into(), serde_json::json!(all_score));
    results.insert("ChatCORE_cat".into(), serde_json::json!(categorical_score));
    Ok(serde_json::Value::Object(results))
}

fn evaluate_jsonl_bpb_with_context(
    model: &GptModel,
    tokenizer: &BpeTokenizer,
    dataset: &Path,
    parse_jsonl: bool,
    sequence_len: usize,
    batch_size: usize,
    token_budget: usize,
    device: &Device,
    distributed: &mut crate::distributed::DistributedContext,
) -> Result<f64> {
    ensure!(
        sequence_len > 1,
        "evaluation sequence length must exceed one"
    );
    ensure!(batch_size > 0, "evaluation batch size must be positive");
    let global_tokens_per_batch = batch_size * sequence_len * distributed.world_size();
    let steps = token_budget / global_tokens_per_batch;
    ensure!(
        steps > 0,
        "evaluation token budget must cover at least one full batch"
    );
    let mut batches = JsonlBatchSource::new(
        dataset,
        tokenizer,
        batch_size,
        sequence_len,
        None,
        parse_jsonl,
        distributed.rank(),
        distributed.world_size(),
    )?;
    let mut total_nats = 0.0f32;
    let mut total_bytes = 0u64;
    for _ in 0..steps {
        let batch = batches.next_batch()?;
        let token_count = batch.valid_targets.len();
        let logits = model
            .forward(&batch.inputs, batch_size, sequence_len)?
            .reshape((batch_size * sequence_len, model.config.vocab_size))?;
        let selected = logits.index_select(
            &Tensor::from_vec(batch.valid_positions, (token_count,), device)?,
            0,
        )?;
        let log_probs = ops::log_softmax(&selected, 1)?;
        let targets = Tensor::from_vec(batch.valid_targets.clone(), (token_count,), device)?;
        let losses = log_probs
            .gather(&targets.unsqueeze(1)?, 1)?
            .to_vec2::<f32>()?;
        let mut batch_nats = 0.0f32;
        for (target, loss) in batch.valid_targets.iter().zip(losses.iter()) {
            let bytes = tokenizer.token_byte_len(*target)?;
            if bytes > 0 {
                batch_nats -= loss[0];
                total_bytes += bytes as u64;
            }
        }
        total_nats += batch_nats;
    }
    let total_nats = distributed.all_reduce_f32_sum(&[total_nats])?[0];
    let (_, total_bytes) = distributed.all_reduce_counts(0, total_bytes)?;
    ensure!(
        total_bytes > 0,
        "validation data contains no ordinary text tokens"
    );
    Ok(total_nats as f64 / (std::f64::consts::LN_2 * total_bytes as f64))
}

fn numbered_checkpoint_step(checkpoint: &Path) -> Option<usize> {
    let stem = checkpoint.file_stem()?.to_str()?;
    stem.strip_prefix("model_")?.parse().ok()
}

fn load_resume_loop_state(directory: &Path, step: Option<usize>) -> Result<ResumeLoopState> {
    let numbered_metadata = step.map(|step| directory.join(format!("meta_{step:06}.json")));
    let metadata_path = numbered_metadata
        .filter(|path| path.is_file())
        .unwrap_or_else(|| directory.join("meta.json"));
    if !metadata_path.is_file() {
        return Ok(ResumeLoopState::default());
    }
    let metadata = crate::json_compat::parse_value(
        &std::fs::read(&metadata_path)
            .with_context(|| format!("read checkpoint metadata {}", metadata_path.display()))?,
    )?;
    let loop_state = metadata.get("loop_state");
    Ok(ResumeLoopState {
        val_bpb: metadata.get("val_bpb").and_then(serde_json::Value::as_f64),
        min_val_bpb: loop_state
            .and_then(|state| state.get("min_val_bpb"))
            .and_then(serde_json::Value::as_f64),
        smooth_train_loss: loop_state
            .and_then(|state| state.get("smooth_train_loss"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_default(),
        total_training_time: loop_state
            .and_then(|state| state.get("total_training_time"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or_default(),
    })
}

fn save_numbered_checkpoint(
    variables: &VarMap,
    model: &GptModel,
    batches: &impl BatchSource,
    options: &TrainOptions,
    step: usize,
    world_size: usize,
    val_bpb: Option<f64>,
    min_val_bpb: Option<f64>,
    smooth_train_loss: f64,
    total_training_time: f64,
) -> Result<()> {
    std::fs::create_dir_all(&options.output)?;
    let model_path = options.output.join(format!("model_{step:06}.safetensors"));
    variables.save(&model_path)?;
    crate::torch_save::save_state_dict(
        variables,
        options.output.join(format!("model_{step:06}.pt")),
    )?;
    let total_batch_size = options
        .batch_size
        .checked_mul(options.sequence_len)
        .and_then(|tokens| tokens.checked_mul(options.grad_accum_steps))
        .and_then(|tokens| tokens.checked_mul(world_size))
        .context("checkpoint total batch size overflow")?;
    let metadata = serde_json::json!({
        "model_config": model.config,
        "step": step,
        "steps": step,
        "batch_size": options.batch_size,
        "device_batch_size": options.batch_size,
        "sequence_len": options.sequence_len,
        "max_seq_len": options.sequence_len,
        "grad_accum_steps": options.grad_accum_steps,
        "global_batch_tokens": total_batch_size,
        "total_batch_size": total_batch_size,
        "optimizer": "MuonAdamW",
        "user_config": &options.wandb_config,
        "val_bpb": val_bpb,
        "loop_state": {
            "min_val_bpb": min_val_bpb,
            "smooth_train_loss": smooth_train_loss,
            "total_training_time": total_training_time,
        },
        "dataloader_state_dict": batches.dataloader_state(),
        "jsonl_state_dict": batches.jsonl_state(),
        "optimizer_config": {
            "embedding_lr": options.embedding_lr,
            "unembedding_lr": options.unembedding_lr,
            "matrix_lr": options.matrix_lr,
            "scalar_lr": options.scalar_lr,
            "weight_decay": options.weight_decay,
            "warmup_steps": options.warmup_steps,
            "warmdown_ratio": options.warmdown_ratio,
            "final_lr_frac": options.final_lr_frac,
            "sft_schedule": options.sft_schedule,
            "initial_lr_frac": options.initial_lr_frac,
            "warmup_ratio": options.warmup_ratio,
            "sft_warmdown_ratio": options.sft_warmdown_ratio,
            "full_epoch": options.full_epoch,
        }
    });
    let metadata = serde_json::to_vec_pretty(&metadata)?;
    std::fs::write(
        options.output.join(format!("meta_{step:06}.json")),
        &metadata,
    )?;
    std::fs::write(options.output.join("meta.json"), metadata)?;
    Ok(())
}

fn save_rank_training_state(
    batches: &impl BatchSource,
    options: &TrainOptions,
    distributed: &crate::distributed::DistributedContext,
    step: usize,
) -> Result<()> {
    if distributed.world_size() <= 1 {
        return Ok(());
    }
    std::fs::create_dir_all(&options.output)?;
    let rank_state = serde_json::json!({
        "rank": distributed.rank(),
        "dataloader_state_dict": batches.dataloader_state(),
        "jsonl_state_dict": batches.jsonl_state(),
    });
    std::fs::write(
        options
            .output
            .join(format!("rank_{}.json", distributed.rank())),
        serde_json::to_vec_pretty(&rank_state)?,
    )?;
    std::fs::write(
        options
            .output
            .join(format!("rank_{}_{step:06}.json", distributed.rank())),
        serde_json::to_vec_pretty(&rank_state)?,
    )?;
    Ok(())
}

fn append_training_metric(
    output: &Path,
    record: serde_json::Value,
    wandb_run: Option<&mut crate::wandb::Run>,
) -> Result<()> {
    let wandb_payload = record.clone();
    append_training_metric_with_payload(output, record, Some(&wandb_payload), wandb_run)
}

fn append_training_metric_with_payload(
    output: &Path,
    record: serde_json::Value,
    wandb_payload: Option<&serde_json::Value>,
    wandb_run: Option<&mut crate::wandb::Run>,
) -> Result<()> {
    std::fs::create_dir_all(output)?;
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output.join("metrics.jsonl"))?;
    serde_json::to_writer(&mut file, &record)?;
    file.write_all(b"\n")?;
    if let (Some(wandb_run), Some(wandb_payload)) = (wandb_run, wandb_payload) {
        let object = wandb_payload
            .as_object()
            .context("training metric must be a JSON object")?;
        wandb_run.log(object)?;
    }
    Ok(())
}

fn sample_training_prompts(engine: &crate::engine::Engine) -> Result<Vec<String>> {
    let prompts = [
        "The capital of France is",
        "The chemical symbol of gold is",
        "If yesterday was Friday, then tomorrow will be",
        "The opposite of hot is",
        "The planets of the solar system are:",
        "My favorite color is",
        "If 5*x + 3 = 13, then x is",
    ];
    let mut samples = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let mut tokens = vec![engine.tokenizer.bos_id()];
        tokens.extend(engine.tokenizer.encode(prompt)?);
        let (sequences, _) = engine.generate_batch(&tokens, 1, 16, 0.0, None, 42)?;
        let sample = engine.tokenizer.decode(&sequences[0])?;
        println!("{sample}");
        samples.push(sample);
    }
    Ok(samples)
}
