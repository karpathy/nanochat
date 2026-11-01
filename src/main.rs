#![recursion_limit = "256"]

mod api;
mod base_eval;
mod checkpoint;
mod core_eval;
mod data;
mod datasets;
mod distributed;
mod engine;
mod evaluation;
mod execution;
pub mod fp8;
mod inference_bench;
mod json_compat;
mod model;
mod model_config;
mod numpy_rng;
mod optim;
mod paths;
mod python_runtime;
mod rl;
mod sft_data;
mod task_eval;
mod tokenizer;
mod tokenizer_eval;
mod torch_save;
mod training;
pub mod wandb;

use std::{
    ffi::OsString,
    io::{BufRead, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::{Child, Command as ProcessCommand, ExitCode},
    time::Duration,
};

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};

#[derive(Debug, Parser)]
#[command(
    name = "rust-agent",
    version,
    about = "Rust implementation of nanoAgent"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Launch a local multi-process Rust command without a Python torchrun install.
    Launch {
        #[arg(long, default_value_t = 1)]
        nproc_per_node: usize,
        /// Subcommand and its arguments, for example: train --depth 12.
        #[arg(required = true, num_args = 1.., trailing_var_arg = true)]
        command: Vec<OsString>,
    },
    /// Run a Python snippet in a short-lived, isolated working directory.
    Exec {
        code: String,
        #[arg(long, default_value_t = 5)]
        timeout_seconds: u64,
        /// Maximum address space in MiB. Use 0 to disable the limit.
        #[arg(long, default_value_t = execution::DEFAULT_WORKER_MEMORY_MIB)]
        memory_mib: u64,
    },
    #[command(name = "__python-worker", hide = true)]
    PythonWorker {
        #[arg(long)]
        memory_limit_bytes: Option<u64>,
    },
    /// Load and validate a model configuration JSON file.
    ValidateConfig { path: std::path::PathBuf },
    /// Resolve a nanochat checkpoint by source, tag, and optional step.
    CheckpointInfo {
        #[arg(long, value_parser = ["base", "sft", "rl"])]
        source: String,
        #[arg(long)]
        model_tag: Option<String>,
        #[arg(long)]
        step: Option<u64>,
        /// Override NANOCHAT_BASE_DIR or the default ~/.cache/nanochat root.
        #[arg(long)]
        base_dir: Option<PathBuf>,
    },
    /// Run one forward pass over token IDs using a PyTorch model checkpoint.
    Forward {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        /// Comma-separated token IDs.
        #[arg(long)]
        tokens: String,
        /// Write the last-position vocabulary logits as a JSON array.
        #[arg(long)]
        logits_output: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float32)]
        dtype: ComputeDtype,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Generate completion text from a local model checkpoint.
    Generate {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        prompt: String,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Number of independent completions to generate in one batch.
        #[arg(long, default_value_t = 1)]
        num_samples: usize,
        /// Print generated token IDs as JSON instead of decoded text.
        #[arg(long)]
        token_ids: bool,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long, allow_hyphen_values = true)]
        top_k: Option<i64>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float32)]
        dtype: ComputeDtype,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Start a line-oriented interactive chat session.
    Chat {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short = 'i', long, value_parser = ["base", "sft", "rl"], default_value = "sft")]
        source: String,
        #[arg(short, long)]
        model_tag: Option<String>,
        #[arg(short, long)]
        step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        #[arg(short, long, default_value = "")]
        prompt: String,
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        #[arg(long, default_value_t = 0.6)]
        temperature: f64,
        #[arg(short = 'k', long, allow_hyphen_values = true, default_value = "50")]
        top_k: Option<i64>,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Encode plain text with a rustAgent tokenizer.json file.
    Tokenize {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        text: String,
    },
    /// Convert nanochat's tokenizer.pkl to the native tokenizer.json format.
    ConvertTokenizer {
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
    /// Train the nanochat BPE tokenizer from a named corpus or local documents.
    TrainTokenizer {
        /// Local file/directory or one of climbmix, wiki, and crowdmind datasets.
        #[arg(long, default_value = "climbmix")]
        dataset: PathBuf,
        /// Prepend this dataset when training from ClimbMix, matching tok_train.py.
        #[arg(
            long,
            default_value = "crowdmind_2048",
            value_parser = ["none", "crowdmind_2048"]
        )]
        first_dataset: String,
        /// Defaults to $NANOCHAT_BASE_DIR/tokenizer/tokenizer.json or ~/.cache/nanochat/tokenizer/tokenizer.json.
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 32768)]
        vocab_size: u32,
        #[arg(long, default_value_t = 10_000)]
        doc_cap: usize,
        #[arg(long, default_value_t = 2_000_000_000)]
        max_chars: usize,
    },
    /// Download a Hugging Face training corpus to the nanochat-compatible cache.
    DownloadDataset {
        #[arg(long, value_name = "NAME")]
        dataset: String,
        #[arg(long, default_value_t = -1)]
        num_files: isize,
        #[arg(long, default_value_t = 4)]
        workers: usize,
    },
    /// Download an SFT dataset and write train/validation JSONL files.
    PrepareSft {
        #[arg(long, value_parser = ["standard", "tools", "post-training", "traces"])]
        mode: String,
        #[arg(long)]
        train_output: PathBuf,
        #[arg(long)]
        validation_output: PathBuf,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long, default_value_t = 3)]
        mmlu_epochs: usize,
        #[arg(long, default_value_t = 4)]
        gsm8k_epochs: usize,
    },
    /// Decode token IDs with a rustAgent tokenizer.json file.
    Detokenize {
        #[arg(long)]
        tokenizer: PathBuf,
        /// Comma-separated token IDs.
        #[arg(long)]
        tokens: String,
    },
    /// Render a chat conversation JSON file into model tokens and SFT mask.
    Render {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        conversation: PathBuf,
        #[arg(long, default_value_t = 2048)]
        max_tokens: usize,
    },
    /// Render a conversation with its final assistant answer removed.
    RenderForCompletion {
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        conversation: PathBuf,
    },
    /// Train or fine-tune locally from a text/JSONL dataset.
    Train {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        /// Model depth when starting fresh base pretraining.
        #[arg(long, default_value_t = 20)]
        depth: usize,
        /// Width multiplier used to derive model_dim = depth * aspect_ratio.
        #[arg(long, default_value_t = 64)]
        aspect_ratio: usize,
        /// Target attention head width when deriving a fresh model config.
        #[arg(long, default_value_t = 128)]
        head_dim: usize,
        /// Context length for fresh base pretraining.
        #[arg(long, alias = "sequence-len")]
        max_seq_len: Option<usize>,
        /// Sliding window pattern for a fresh model (L=full, S=quarter context).
        #[arg(long, default_value = "SSSL")]
        window_pattern: String,
        /// Local text, JSONL, or Parquet input. JSONL records may contain raw OpenAI-style SFT messages.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Prepare or reuse a nanochat SFT dataset mixture before training.
        #[arg(long, value_parser = ["standard", "tools", "post-training", "traces"])]
        data_mode: Option<String>,
        #[arg(long, default_value_t = 3)]
        mmlu_epochs: usize,
        #[arg(long, default_value_t = 4)]
        gsm8k_epochs: usize,
        #[arg(long, default_value_t = 42)]
        data_seed: u64,
        /// Held-out dataset for BPB evaluation (otherwise inferred from a sibling train/validation name).
        #[arg(long)]
        validation_dataset: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        /// W&B run name; use "dummy" to disable tracking like nanochat.
        #[arg(long, default_value = "dummy")]
        run: String,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        /// Resolve the input checkpoint from nanochat's base, SFT, or RL tag store.
        #[arg(short = 'i', long, value_parser = ["base", "sft", "rl"])]
        source: Option<String>,
        #[arg(short = 'g', long)]
        model_tag: Option<String>,
        #[arg(short = 's', long, conflicts_with = "resume_from_step")]
        model_step: Option<u64>,
        /// Resume a base run at this step; -1 disables resume like base_train.py.
        #[arg(long, allow_hyphen_values = true, conflicts_with = "model_step")]
        resume_from_step: Option<i64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        /// Save under ~/.cache/nanochat/chatsft_checkpoints/<tag>.
        #[arg(long)]
        output_model_tag: Option<String>,
        /// Load optimizer state from the input checkpoint (1=yes, 0=no).
        #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u8).range(0..=1))]
        load_optimizer: u8,
        /// Resume the checkpoint's dataset cursor; tag-based SFT defaults to a fresh cursor.
        #[arg(long, action = clap::ArgAction::Set)]
        resume_data: Option<bool>,
        #[arg(long, alias = "num-iterations", allow_hyphen_values = true)]
        steps: Option<i64>,
        #[arg(long, allow_hyphen_values = true)]
        target_flops: Option<f64>,
        #[arg(long, allow_hyphen_values = true)]
        target_param_data_ratio: Option<f64>,
        #[arg(long, alias = "device-batch-size")]
        batch_size: Option<usize>,
        #[arg(long)]
        grad_accum_steps: Option<usize>,
        #[arg(long, allow_hyphen_values = true)]
        total_batch_size: Option<i64>,
        /// Use tensorwise FP8 GEMMs for eligible CUDA linear layers.
        #[arg(long, default_value_t = false)]
        fp8: bool,
        #[arg(long, value_parser = ["rowwise", "tensorwise"], default_value = "tensorwise")]
        fp8_recipe: String,
        #[arg(long)]
        embedding_lr: Option<f64>,
        #[arg(long)]
        unembedding_lr: Option<f64>,
        #[arg(long)]
        matrix_lr: Option<f64>,
        #[arg(long, default_value_t = 0.5)]
        scalar_lr: f64,
        #[arg(long, default_value_t = 0.28)]
        weight_decay: f64,
        #[arg(long, default_value_t = 40)]
        warmup_steps: usize,
        /// Warmdown fraction; defaults to 0.5 for SFT and 0.65 for base pretraining.
        #[arg(long, alias = "sft-warmdown-ratio")]
        warmdown_ratio: Option<f64>,
        #[arg(long)]
        final_lr_frac: Option<f64>,
        /// Use the chat_sft progress-based warmup/constant/warmdown schedule on JSONL input.
        #[arg(long, default_value_t = false)]
        sft_schedule: bool,
        /// Initial learning-rate fraction used by the chat_sft schedule.
        #[arg(long, alias = "init-lr-frac", default_value_t = 0.8)]
        initial_lr_frac: f64,
        /// Progress fraction used for linear warmup by the chat_sft schedule.
        #[arg(long, default_value_t = 0.0)]
        warmup_ratio: f64,
        /// Stop after the JSONL training conversations have been consumed once, up to the step cap.
        #[arg(long, default_value_t = false)]
        full_epoch: bool,
        /// Validation interval; defaults to 200 steps for SFT and disabled otherwise.
        #[arg(long, allow_hyphen_values = true)]
        eval_every: Option<i64>,
        #[arg(long)]
        /// Validation token budget; defaults to 40*524288 for SFT, 80*524288 otherwise.
        eval_tokens: Option<usize>,
        #[arg(long, allow_hyphen_values = true)]
        core_metric_every: Option<i64>,
        #[arg(long, allow_hyphen_values = true, default_value_t = 500)]
        core_metric_max_per_task: i64,
        /// Evaluate ARC, MMLU, GSM8K, and HumanEval periodically and at the end.
        #[arg(long, allow_hyphen_values = true)]
        chatcore_every: Option<i64>,
        /// Maximum examples per categorical ChatCORE task; -1 evaluates all.
        #[arg(long, default_value_t = -1)]
        chatcore_max_cat: i64,
        /// Maximum examples per generative ChatCORE task.
        #[arg(long, default_value_t = 24)]
        chatcore_max_sample: usize,
        #[arg(long, allow_hyphen_values = true)]
        sample_every: Option<i64>,
        #[arg(long, allow_hyphen_values = true)]
        save_every: Option<i64>,
        /// Compute dtype (defaults to BF16 on supported CUDA devices, otherwise FP32).
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Serve a checkpoint through the streaming OpenAI-compatible chat API.
    Serve {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short = 'i', long, value_parser = ["base", "sft", "rl"], default_value = "sft")]
        source: String,
        #[arg(short = 'g', long)]
        model_tag: Option<String>,
        #[arg(short = 's', long)]
        step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        #[arg(long = "model-name", alias = "model-id", default_value = "nanochat")]
        model_name: String,
        #[arg(long)]
        host: Option<String>,
        #[arg(long)]
        port: Option<u16>,
        /// Legacy combined bind address (host:port).
        #[arg(long)]
        address: Option<String>,
        #[arg(long, default_value_t = 0.6)]
        temperature: f64,
        #[arg(long, allow_hyphen_values = true, default_value = "50")]
        top_k: Option<i64>,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Evaluate tokenizer-invariant bits per byte on a local text/JSONL corpus.
    EvalBpb {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value_t = 1024)]
        sequence_len: usize,
        #[arg(long, default_value_t = 1)]
        batch_size: usize,
        #[arg(long, default_value_t = 1_048_576)]
        eval_tokens: usize,
        #[arg(long, value_enum, default_value_t = ComputeDtype::Float32)]
        dtype: ComputeDtype,
        #[arg(long, default_value = "cpu")]
        device: String,
    },
    /// Evaluate chat benchmarks from Hugging Face: ARC-Easy, ARC-Challenge, MMLU, GSM8K, HumanEval.
    EvalChat {
        #[arg(long)]
        config: Option<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(short = 'i', long, value_parser = ["base", "sft", "rl"], default_value = "sft")]
        source: String,
        #[arg(short, long)]
        model_tag: Option<String>,
        #[arg(short, long)]
        step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        /// One task, or pipe-separated task names; defaults to all five.
        #[arg(long, default_value = "all")]
        task_name: String,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 1)]
        num_samples: usize,
        #[arg(long, default_value_t = 512)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, allow_hyphen_values = true, default_value_t = 50)]
        top_k: i64,
        #[arg(long, default_value_t = 42)]
        seed: u64,
        #[arg(long)]
        max_problems: Option<usize>,
        /// Write per-example expected/predicted answers or generated outcomes as JSONL.
        #[arg(long)]
        predictions_output: Option<PathBuf>,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Evaluate the base model on the CORE in-context learning suite.
    EvalCore {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long, allow_hyphen_values = true)]
        max_per_task: Option<i64>,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Run the unified base-model sample, BPB, and CORE evaluations.
    BaseEval {
        #[arg(long, default_value = "core,bpb,sample")]
        eval: String,
        #[arg(long)]
        model_tag: Option<String>,
        #[arg(long)]
        step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, allow_hyphen_values = true)]
        /// Maximum CORE examples per task; non-positive values evaluate all examples.
        max_per_task: Option<i64>,
        #[arg(long, default_value_t = 32)]
        device_batch_size: usize,
        #[arg(
            long,
            default_value = "climbmix",
            value_parser = ["climbmix", "wiki", "crowdmind_2048", "crowdmind_6144", "crowdmind_16384"]
        )]
        dataset: String,
        #[arg(long, default_value_t = 40 * 524_288)]
        split_tokens: usize,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Compare tokenizer compression on fixed samples and a dataset row group.
    EvalTokenizer {
        #[arg(
            long,
            default_value = "climbmix",
            value_parser = ["climbmix", "wiki", "crowdmind_2048", "crowdmind_6144", "crowdmind_16384"]
        )]
        dataset: String,
        /// Defaults to the cached tokenizer.json, falling back to tokenizer.pkl.
        #[arg(long)]
        tokenizer: Option<PathBuf>,
    },
    /// Fine-tune an SFT checkpoint with GSM8K reward optimization.
    ChatRl {
        #[arg(long, default_value = "dummy")]
        run: String,
        #[arg(long)]
        model_tag: Option<String>,
        #[arg(long)]
        model_step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 1)]
        num_epochs: usize,
        #[arg(long, value_parser = ["exact", "codex"], default_value = "exact")]
        reward_mode: String,
        #[arg(long)]
        judge_model: Option<String>,
        #[arg(long, default_value = "codex")]
        codex_command: String,
        #[arg(long, default_value_t = 120)]
        judge_timeout: u64,
        #[arg(long, default_value_t = 8)]
        device_batch_size: usize,
        #[arg(long, default_value_t = 16)]
        examples_per_step: usize,
        #[arg(long, default_value_t = 16)]
        num_samples: usize,
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,
        #[arg(long, default_value_t = 1.0)]
        temperature: f64,
        #[arg(long, allow_hyphen_values = true, default_value_t = 50)]
        top_k: i64,
        #[arg(long, default_value_t = 0.2)]
        embedding_lr: f64,
        #[arg(long, default_value_t = 0.004)]
        unembedding_lr: f64,
        #[arg(long, default_value_t = 0.02)]
        matrix_lr: f64,
        #[arg(long, default_value_t = 0.0)]
        weight_decay: f64,
        #[arg(long, default_value_t = 0.05)]
        init_lr_frac: f64,
        #[arg(long, default_value_t = 60)]
        eval_every: usize,
        #[arg(long, default_value_t = 400)]
        eval_examples: usize,
        #[arg(long, default_value_t = 60)]
        save_every: usize,
        /// Compute dtype (defaults to BF16 on supported CUDA devices, otherwise FP32).
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, visible_alias = "device-type", default_value = "auto")]
        device: String,
    },
    /// Sweep inference prefill/decode latency and architecture cost on one GPU.
    EvalBench {
        #[arg(short = 'i', long, value_parser = ["base", "sft", "rl"], default_value = "base")]
        source: String,
        #[arg(short = 'g', long)]
        model_tag: Option<String>,
        #[arg(short = 's', long)]
        step: Option<u64>,
        #[arg(long)]
        base_dir: Option<PathBuf>,
        #[arg(long)]
        tokenizer: Option<PathBuf>,
        #[arg(long, default_value_t = 2048)]
        prompt_tokens: usize,
        #[arg(long, default_value_t = 256)]
        decode_tokens: usize,
        #[arg(long, default_value = "1,8,32,128")]
        batch_sizes: String,
        #[arg(short = 't', long, default_value_t = 0.0)]
        temperature: f64,
        #[arg(long, value_enum)]
        dtype: Option<ComputeDtype>,
        #[arg(long, default_value = "cuda")]
        device: String,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ComputeDtype {
    Float32,
    Float16,
    Bfloat16,
}

impl From<ComputeDtype> for DType {
    fn from(value: ComputeDtype) -> Self {
        match value {
            ComputeDtype::Float32 => DType::F32,
            ComputeDtype::Float16 => DType::F16,
            ComputeDtype::Bfloat16 => DType::BF16,
        }
    }
}

fn main() -> Result<ExitCode> {
    std::thread::Builder::new()
        .name("rust-agent-command".to_owned())
        .stack_size(16 * 1024 * 1024)
        .spawn(run)?
        .join()
        .map_err(|_| anyhow::anyhow!("rustAgent command thread panicked"))?
}

fn run() -> Result<ExitCode> {
    let args = Args::parse();
    match args.command {
        Command::Launch {
            nproc_per_node,
            command,
        } => return launch_local(nproc_per_node, command),
        Command::Exec {
            code,
            timeout_seconds,
            memory_mib,
        } => {
            let memory_limit = (memory_mib != 0).then_some(memory_mib.saturating_mul(1024 * 1024));
            let result =
                execution::execute_code(&code, Duration::from_secs(timeout_seconds), memory_limit)?;
            print!("{}", result.stdout);
            eprint!("{}", result.stderr);
            if let Some(error) = result.error.as_deref() {
                if result.stderr.is_empty() {
                    eprintln!("{error}");
                }
            }
            if !result.success {
                return Ok(ExitCode::FAILURE);
            }
        }
        Command::PythonWorker { memory_limit_bytes } => {
            return Ok(ExitCode::from(python_runtime::run_worker(
                memory_limit_bytes,
            )?));
        }
        Command::ValidateConfig { path } => {
            let source = std::fs::read_to_string(&path)?;
            let config = model_config::GptConfig::from_value(serde_json::from_str(&source)?)?;
            config.validate()?;
            let long_layers = (0..config.n_layer)
                .filter(|&layer| config.is_long_window_layer(layer))
                .count();
            let short_window = (0..config.n_layer)
                .find(|&layer| !config.is_long_window_layer(layer))
                .map(|layer| config.attention_window(layer).0)
                .unwrap_or(0);
            println!(
                "valid GPT config: {} layers ({} full context, short window {}), width {}, {} query heads / {} KV heads, vocab {}",
                config.n_layer,
                long_layers,
                short_window,
                config.n_embd,
                config.n_head,
                config.n_kv_head,
                config.vocab_size
            );
        }
        Command::CheckpointInfo {
            source,
            model_tag,
            step,
            base_dir,
        } => {
            let selected = checkpoint::select_checkpoint(
                &source,
                base_dir.as_deref(),
                model_tag.as_deref(),
                step,
            )?;
            println!("{}", serde_json::to_string_pretty(&selected)?);
        }
        Command::Forward {
            config,
            checkpoint,
            tokens,
            logits_output,
            dtype,
            device,
        } => {
            let config_value = json_compat::parse_value(&std::fs::read(&config)?)?;
            let model_config = config_value
                .get("model_config")
                .cloned()
                .unwrap_or(config_value);
            let model_config = model_config::GptConfig::from_value(model_config)?;
            model_config.validate()?;
            let device = parse_device(&device)?;
            let tokens = tokens
                .split(',')
                .map(|part| part.trim().parse::<u32>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            anyhow::ensure!(!tokens.is_empty(), "provide at least one token ID");
            let model =
                model::GptModel::load_checkpoint(model_config, checkpoint, &device, dtype.into())?
                    .with_inference_attention(true);
            let logits = model.forward(&tokens, 1, tokens.len())?.to_vec3::<f32>()?;
            let last_logits = logits
                .last()
                .expect("non-empty sequence")
                .last()
                .expect("batch size is one");
            if let Some(path) = logits_output {
                std::fs::write(path, serde_json::to_vec(last_logits)?)?;
            }
            let (token, value) = last_logits
                .iter()
                .copied()
                .enumerate()
                .fold(None, |best: Option<(usize, f32)>, candidate| match best {
                    Some((_, best_value)) if !candidate.1.total_cmp(&best_value).is_gt() => best,
                    _ => Some(candidate),
                })
                .expect("non-empty vocabulary");
            println!("next_token={token} logit={value:.9}");
        }
        Command::Generate {
            config,
            checkpoint,
            tokenizer,
            prompt,
            max_tokens,
            num_samples,
            token_ids,
            temperature,
            top_k,
            seed,
            dtype,
            device,
        } => {
            let top_k = normalize_top_k(top_k);
            let config_value = json_compat::parse_value(&std::fs::read(&config)?)?;
            let config_value = config_value
                .get("model_config")
                .cloned()
                .unwrap_or(config_value);
            let config = model_config::GptConfig::from_value(config_value)?;
            config.validate()?;
            let device = parse_device(&device)?;
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            anyhow::ensure!(num_samples > 0, "number of samples must be positive");
            anyhow::ensure!(
                tokenizer.vocab_size() as usize == config.vocab_size,
                "tokenizer vocabulary does not match model configuration"
            );
            let tokens = tokenizer.render_user_prompt(&prompt)?;
            anyhow::ensure!(
                tokens.len() <= config.sequence_len,
                "prompt exceeds model context"
            );
            let model =
                model::GptModel::load_checkpoint(config, checkpoint, &device, dtype.into())?;
            let engine = engine::Engine::new(model, tokenizer);
            if num_samples == 1 {
                let generated = engine.generate(&tokens, max_tokens, temperature, top_k, seed)?;
                if token_ids {
                    println!("{}", serde_json::to_string(&generated)?);
                } else {
                    print!("{}", engine.tokenizer.decode(&generated)?);
                }
            } else {
                let (generated, _) = engine.generate_batch(
                    &tokens,
                    num_samples,
                    max_tokens,
                    temperature,
                    top_k,
                    seed,
                )?;
                if token_ids {
                    let completions = generated
                        .iter()
                        .map(|sequence| &sequence[tokens.len()..])
                        .collect::<Vec<_>>();
                    println!("{}", serde_json::to_string(&completions)?);
                } else {
                    let completions = generated
                        .iter()
                        .map(|sequence| engine.tokenizer.decode(&sequence[tokens.len()..]))
                        .collect::<Result<Vec<_>>>()?;
                    println!("{}", serde_json::to_string(&completions)?);
                }
            }
        }
        Command::Chat {
            config,
            checkpoint,
            tokenizer,
            source,
            model_tag,
            step,
            base_dir,
            prompt,
            max_tokens,
            temperature,
            top_k,
            seed,
            dtype,
            device,
        } => {
            let top_k = normalize_top_k(top_k);
            run_chat(
                config.as_deref(),
                checkpoint.as_deref(),
                tokenizer.as_deref(),
                &source,
                model_tag.as_deref(),
                step,
                base_dir.as_deref(),
                &prompt,
                max_tokens,
                temperature,
                top_k,
                seed,
                dtype,
                &device,
            )?;
        }
        Command::Tokenize { tokenizer, text } => {
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            let ids = tokenizer.encode(&text)?;
            println!(
                "{}",
                serde_json::json!({
                    "vocab_size": tokenizer.vocab_size(),
                    "tokens": ids,
                    "decoded": tokenizer.decode(&ids)?
                })
            );
        }
        Command::ConvertTokenizer { input, output } => {
            let count = tokenizer::BpeTokenizer::convert_tiktoken_pickle(input, output)?;
            println!("converted {count} mergeable tokens");
        }
        Command::TrainTokenizer {
            dataset,
            first_dataset,
            output,
            vocab_size,
            doc_cap,
            max_chars,
        } => {
            let output = output.unwrap_or(resolve_base_dir(None)?.join("tokenizer/tokenizer.json"));
            let (paths, dataset_order) = if dataset.exists() {
                let label = dataset.display().to_string();
                (vec![dataset], vec![label])
            } else {
                let selected = datasets::Dataset::parse(&dataset.to_string_lossy())?;
                let mut selected_datasets = Vec::new();
                if selected == datasets::Dataset::Climbmix && first_dataset != "none" {
                    selected_datasets.push(datasets::Dataset::CrowdMind2048);
                }
                selected_datasets.push(selected);
                let names = selected_datasets
                    .iter()
                    .map(|dataset| dataset.name().to_owned())
                    .collect::<Vec<_>>();
                let paths = selected_datasets
                    .into_iter()
                    .map(datasets::Dataset::local_cache_dir)
                    .collect::<Result<Vec<_>>>()?;
                (paths, names)
            };
            println!("max_chars: {max_chars}");
            println!("doc_cap: {doc_cap}");
            println!("vocab_size: {vocab_size}");
            println!("dataset order: {}", dataset_order.join(" -> "));
            let texts = data::text_documents_from_paths(paths, doc_cap, max_chars)?;
            let token_bytes_path = output
                .parent()
                .filter(|parent| !parent.as_os_str().is_empty())
                .unwrap_or_else(|| Path::new("."))
                .join("token_bytes.pt");
            let training_time = tokenizer::BpeTokenizer::train_to_file(texts, vocab_size, &output)?;
            println!("Training time: {:.2}s", training_time.as_secs_f64());
            println!("Saved token_bytes to {}", token_bytes_path.display());
        }
        Command::DownloadDataset {
            dataset,
            num_files,
            workers,
        } => {
            let dataset = datasets::Dataset::parse(&dataset)?;
            let directory = datasets::download(dataset, num_files, workers)?;
            println!("dataset_cache={}", directory.display());
        }
        Command::PrepareSft {
            mode,
            train_output,
            validation_output,
            seed,
            mmlu_epochs,
            gsm8k_epochs,
        } => {
            let mode = sft_data::SftMode::parse(&mode)?;
            let (train_rows, validation_rows) = sft_data::prepare(
                mode,
                &train_output,
                &validation_output,
                seed,
                mmlu_epochs,
                gsm8k_epochs,
            )?;
            println!(
                "train_rows={train_rows} validation_rows={validation_rows} train={} validation={}",
                train_output.display(),
                validation_output.display()
            );
        }
        Command::Detokenize { tokenizer, tokens } => {
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            let ids = tokens
                .split(',')
                .map(|part| part.trim().parse::<u32>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            println!("{}", tokenizer.decode(&ids)?);
        }
        Command::Render {
            tokenizer,
            conversation,
            max_tokens,
        } => {
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            let conversation: serde_json::Value =
                serde_json::from_slice(&std::fs::read(conversation)?)?;
            let (ids, mask) = tokenizer.render_conversation(&conversation, max_tokens)?;
            println!("{}", serde_json::json!({"tokens": ids, "mask": mask}));
        }
        Command::RenderForCompletion {
            tokenizer,
            conversation,
        } => {
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            let conversation: serde_json::Value =
                serde_json::from_slice(&std::fs::read(conversation)?)?;
            let ids = tokenizer.render_for_completion(&conversation)?;
            println!("{}", serde_json::to_string(&ids)?);
        }
        Command::Train {
            config,
            tokenizer,
            depth,
            aspect_ratio,
            head_dim,
            max_seq_len,
            window_pattern,
            dataset,
            data_mode,
            mmlu_epochs,
            gsm8k_epochs,
            data_seed,
            mut validation_dataset,
            output,
            run,
            checkpoint,
            source,
            model_tag,
            model_step,
            resume_from_step,
            base_dir,
            output_model_tag,
            load_optimizer,
            resume_data,
            steps,
            target_flops,
            target_param_data_ratio,
            batch_size,
            grad_accum_steps,
            total_batch_size,
            fp8,
            fp8_recipe,
            embedding_lr,
            unembedding_lr,
            matrix_lr,
            scalar_lr,
            weight_decay,
            warmup_steps,
            warmdown_ratio,
            final_lr_frac,
            sft_schedule,
            initial_lr_frac,
            warmup_ratio,
            full_epoch,
            eval_every,
            eval_tokens,
            core_metric_every,
            core_metric_max_per_task,
            chatcore_every,
            chatcore_max_cat,
            chatcore_max_sample,
            sample_every,
            save_every,
            dtype,
            device,
        } => {
            // Keep the parsed values for W&B's user_config. The Python entry
            // points log argparse values before resolving inherited defaults
            // or normalizing sentinel values for execution.
            let user_device_type = if device == "auto" {
                String::new()
            } else {
                device.clone()
            };
            let user_dataset = dataset.clone();
            let user_steps = steps.unwrap_or(-1);
            let user_total_batch_size = total_batch_size;
            let user_batch_size = batch_size;
            let user_embedding_lr = embedding_lr;
            let user_unembedding_lr = unembedding_lr;
            let user_matrix_lr = matrix_lr;
            let user_eval_every = eval_every;
            let user_eval_tokens = eval_tokens;
            let user_core_metric_every = core_metric_every;
            let user_chatcore_every = chatcore_every;
            let user_sample_every = sample_every;
            let user_save_every = save_every.unwrap_or(-1);
            let user_fp8 = fp8;
            let model_step = match resume_from_step {
                Some(-1) | None => model_step,
                Some(step) if step >= 0 => {
                    anyhow::ensure!(
                        model_step.is_none(),
                        "specify only one of --model-step and --resume-from-step"
                    );
                    Some(step as u64)
                }
                Some(step) => {
                    anyhow::bail!("--resume-from-step must be -1 or non-negative, got {step}")
                }
            };
            let steps = steps
                .filter(|steps| *steps > 0)
                .map(usize::try_from)
                .transpose()
                .context("step count exceeds this platform's supported range")?;
            let eval_every = eval_every.map(|value| value.max(0) as usize);
            let core_metric_every = core_metric_every.map(|value| value.max(0) as usize);
            let core_metric_max_per_task = core_metric_max_per_task.max(0) as usize;
            let chatcore_every = chatcore_every.map(|value| value.max(0) as usize);
            let sample_every = sample_every.map(|value| value.max(0) as usize);
            let save_every = save_every.unwrap_or(0).max(0) as usize;
            let fresh_pretraining_requested = source.is_none()
                && model_step.is_none()
                && checkpoint.is_none()
                && data_mode.is_none()
                && !sft_schedule;
            let resolve_from_store = source.is_some()
                || model_step.is_some()
                || (model_tag.is_some() && !fresh_pretraining_requested);
            let fresh_pretraining =
                !resolve_from_store && checkpoint.is_none() && data_mode.is_none() && !sft_schedule;
            let sft_requested = sft_schedule
                || data_mode.is_some()
                || source.as_deref().is_some_and(|source| source != "base");
            let data_mode = data_mode
                .or_else(|| (dataset.is_none() && sft_requested).then(|| "standard".to_owned()));
            let data_mode_selected = data_mode.is_some();
            let dataset = match (data_mode.as_deref(), dataset) {
                (Some(mode), None) => {
                    let cache_base = resolve_base_dir(base_dir.as_deref())?;
                    let cache_name =
                        format!("{mode}-mmlu{mmlu_epochs}-gsm{gsm8k_epochs}-seed{data_seed}");
                    let cache_dir = cache_base.join("prepared_sft").join(cache_name);
                    let train_path = cache_dir.join("train.jsonl");
                    let validation_path = cache_dir.join("validation.jsonl");
                    let mode = sft_data::SftMode::parse(mode)?;
                    let (train_rows, validation_rows, prepared) = sft_data::prepare_cached(
                        mode,
                        &cache_dir,
                        data_seed,
                        mmlu_epochs,
                        gsm8k_epochs,
                    )?;
                    if prepared {
                        println!(
                            "prepared_sft_rows train={train_rows} validation={validation_rows}"
                        );
                    } else {
                        println!(
                            "reused_prepared_sft_rows train={train_rows} validation={validation_rows}"
                        );
                    }
                    validation_dataset.get_or_insert(validation_path);
                    train_path
                }
                (Some(_), Some(_)) => anyhow::bail!(
                    "--data-mode prepares its own corpus and cannot be combined with --dataset"
                ),
                (None, Some(dataset)) => dataset,
                (None, None) if resolve_from_store && sft_requested => {
                    anyhow::bail!("provide --dataset or select an SFT --data-mode")
                }
                (None, None) => PathBuf::from("climbmix"),
            };
            let dataset = dataset
                .to_str()
                .and_then(|name| datasets::Dataset::parse(name).ok())
                .map(|dataset| dataset.local_cache_dir())
                .transpose()?
                .unwrap_or(dataset);
            let sft_training = sft_requested || data_mode_selected;
            let sft_schedule = sft_schedule || data_mode_selected || sft_training;
            let (base_warmdown_ratio, sft_warmdown_ratio) = resolve_warmdown_ratios(warmdown_ratio);
            let final_lr_frac = final_lr_frac.unwrap_or(if sft_schedule { 0.0 } else { 0.05 });
            let full_epoch = full_epoch
                || (sft_schedule
                    && steps.is_none()
                    && target_flops.is_none()
                    && target_param_data_ratio.is_none());
            let (config, checkpoint, tokenizer_path) = if resolve_from_store {
                anyhow::ensure!(
                    config.is_none() && checkpoint.is_none(),
                    "tag-based training cannot be combined with --config or --checkpoint"
                );
                let (config, checkpoint, tokenizer_path) = resolve_model_inputs(
                    None,
                    None,
                    tokenizer.as_deref(),
                    source
                        .as_deref()
                        .unwrap_or(if sft_requested { "sft" } else { "base" }),
                    model_tag.as_deref(),
                    model_step,
                    base_dir.as_deref(),
                )?;
                (config, Some(checkpoint), tokenizer_path)
            } else {
                let base = resolve_base_dir(base_dir.as_deref())?;
                let tokenizer_path =
                    tokenizer.unwrap_or_else(|| paths::cached_tokenizer_path(&base));
                let config = if let Some(config_path) = config {
                    let config_value = json_compat::parse_value(&std::fs::read(config_path)?)?;
                    let config_value = config_value
                        .get("model_config")
                        .cloned()
                        .unwrap_or(config_value);
                    model_config::GptConfig::from_value(config_value)?
                } else {
                    anyhow::ensure!(
                        fresh_pretraining,
                        "provide --config for a local checkpoint or select an input with --source/--model-tag"
                    );
                    anyhow::ensure!(head_dim > 0, "head dimension must be positive");
                    let tokenizer_data = tokenizer::BpeTokenizer::load(&tokenizer_path)?;
                    let base_dim = depth
                        .checked_mul(aspect_ratio)
                        .context("model width overflow")?;
                    let n_embd = base_dim.div_ceil(head_dim) * head_dim;
                    let n_heads = n_embd / head_dim;
                    model_config::GptConfig {
                        sequence_len: max_seq_len.unwrap_or(2048),
                        vocab_size: tokenizer_data.vocab_size() as usize,
                        n_layer: depth,
                        n_head: n_heads,
                        n_kv_head: n_heads,
                        n_embd,
                        window_pattern: window_pattern.clone(),
                    }
                };
                (config, checkpoint, tokenizer_path)
            };
            config.validate()?;
            let chatcore_every = chatcore_every.unwrap_or_else(|| {
                if sft_training && data_mode.as_deref().unwrap_or("standard") == "standard" {
                    200
                } else {
                    0
                }
            });
            let checkpoint_metadata = checkpoint
                .as_deref()
                .map(read_checkpoint_metadata)
                .transpose()?
                .unwrap_or_else(|| serde_json::json!({}));
            let checkpoint_user_config = checkpoint_metadata.get("user_config");
            let batch_size = batch_size.unwrap_or_else(|| {
                if sft_training {
                    checkpoint_metadata
                        .get("device_batch_size")
                        .and_then(serde_json::Value::as_u64)
                        .or_else(|| {
                            checkpoint_user_config
                                .and_then(|value| value.get("device_batch_size"))
                                .and_then(serde_json::Value::as_u64)
                        })
                        .unwrap_or(32) as usize
                } else if !sft_training {
                    32
                } else {
                    1
                }
            });
            let sequence_len = resolve_training_sequence_len(
                max_seq_len,
                sft_training,
                &checkpoint_metadata,
                config.sequence_len,
            );
            let total_batch_size = match total_batch_size {
                None | Some(-1) => None,
                Some(value) => {
                    anyhow::ensure!(
                        value > 0,
                        "total batch size must be positive or -1 for auto"
                    );
                    Some(value as usize)
                }
            }
            .or_else(|| {
                if sft_training && grad_accum_steps.is_none() {
                    checkpoint_metadata
                        .get("total_batch_size")
                        .and_then(serde_json::Value::as_u64)
                        .or_else(|| {
                            checkpoint_user_config
                                .and_then(|value| value.get("total_batch_size"))
                                .and_then(serde_json::Value::as_u64)
                        })
                        .or(Some(524_288))
                        .map(|value| value as usize)
                } else {
                    None
                }
            });
            let embedding_lr = embedding_lr
                .or_else(|| {
                    sft_training
                        .then(|| {
                            checkpoint_user_config
                                .and_then(|value| value.get("embedding_lr"))
                                .and_then(serde_json::Value::as_f64)
                        })
                        .flatten()
                })
                .unwrap_or(0.3);
            let unembedding_lr = unembedding_lr
                .or_else(|| {
                    sft_training
                        .then(|| {
                            checkpoint_user_config
                                .and_then(|value| value.get("unembedding_lr"))
                                .and_then(serde_json::Value::as_f64)
                        })
                        .flatten()
                })
                .unwrap_or(if sft_training { 0.004 } else { 0.008 });
            let matrix_lr = matrix_lr
                .or_else(|| {
                    sft_training
                        .then(|| {
                            checkpoint_user_config
                                .and_then(|value| value.get("matrix_lr"))
                                .and_then(serde_json::Value::as_f64)
                        })
                        .flatten()
                })
                .unwrap_or(0.02);
            let mut plan = resolve_training_plan(
                &config,
                steps,
                target_flops,
                target_param_data_ratio.or_else(|| (!sft_training).then_some(12.0)),
                batch_size,
                sequence_len,
                grad_accum_steps,
                total_batch_size,
                weight_decay,
            )?;
            if sft_training {
                // chat_sft inherits optimizer rates directly and does not use
                // the base trainer's global-batch square-root LR correction.
                plan.lr_scale = 1.0;
            }
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer_path)?;
            anyhow::ensure!(
                tokenizer.vocab_size() as usize == config.vocab_size,
                "tokenizer vocabulary does not match model configuration"
            );
            let parquet_input = dataset.is_dir()
                || dataset
                    .extension()
                    .is_some_and(|extension| extension == "parquet");
            let jsonl_input = dataset
                .extension()
                .is_some_and(|extension| extension == "jsonl" || extension == "ndjson");
            let text_file_input = dataset
                .extension()
                .is_some_and(|extension| extension == "txt" || extension == "text");
            let line_input = jsonl_input || text_file_input;
            let validation_dataset = validation_dataset.or_else(|| {
                line_input
                    .then(|| infer_validation_dataset(&dataset))
                    .flatten()
            });
            anyhow::ensure!(
                !full_epoch || line_input,
                "--full-epoch requires local text or JSONL input"
            );
            if full_epoch
                && steps.is_none()
                && target_flops.is_none()
                && target_param_data_ratio.is_none()
            {
                let line_count = std::io::BufReader::new(std::fs::File::open(&dataset)?)
                    .lines()
                    .try_fold(0usize, |count, line| -> Result<usize> {
                        let line = line?;
                        Ok(count + usize::from(!line.trim().is_empty()))
                    })?;
                anyhow::ensure!(line_count > 0, "training dataset contains no records");
                plan.steps = line_count;
            }
            anyhow::ensure!(
                validation_dataset.is_none() || line_input,
                "--validation-dataset is only supported with local text or JSONL input"
            );
            let parquet_files = if parquet_input {
                Some(if dataset.is_dir() {
                    datasets::local_files(&dataset)?
                } else {
                    vec![dataset.clone()]
                })
            } else {
                None
            };
            let corpus = if parquet_input || line_input {
                None
            } else {
                Some(data::TokenCorpus::from_path_split(
                    &dataset, &tokenizer, false,
                )?)
            };
            let eval_every = eval_every.unwrap_or(if sft_training { 200 } else { 250 });
            let core_metric_every =
                core_metric_every.unwrap_or(if sft_training { 0 } else { 2000 });
            let sample_every = sample_every.unwrap_or(if sft_training { 0 } else { 2000 });
            let eval_tokens = eval_tokens.unwrap_or(if sft_training {
                40 * 524_288
            } else {
                80 * 524_288
            });
            let device = resolve_requested_device(&device)?;
            let compute_dtype = resolve_compute_dtype(dtype, &device)?;
            anyhow::ensure!(
                !fp8 || !sft_training,
                "--fp8 is supported only for base training"
            );
            anyhow::ensure!(
                !fp8 || fp8_recipe == "tensorwise",
                "only the tensorwise FP8 recipe is supported"
            );
            let fp8 = if fp8 && !device.is_cuda() {
                eprintln!("Warning: FP8 training requires CUDA; ignoring --fp8");
                false
            } else {
                fp8
            };
            #[cfg(not(feature = "cuda-base"))]
            anyhow::ensure!(!fp8, "--fp8 requires a build with --features cuda-base");
            let wandb_config = if sft_training {
                serde_json::json!({
                    "run": &run,
                    "device_type": &user_device_type,
                    "model_tag": &model_tag,
                    "source": source.as_deref().unwrap_or("base"),
                    "output_model_tag": &output_model_tag,
                    "model_step": model_step,
                    "load_optimizer": load_optimizer as i64,
                    "num_iterations": user_steps,
                    "max_seq_len": max_seq_len,
                    "device_batch_size": user_batch_size,
                    "total_batch_size": user_total_batch_size,
                    "embedding_lr": user_embedding_lr,
                    "unembedding_lr": user_unembedding_lr,
                    "matrix_lr": user_matrix_lr,
                    "init_lr_frac": initial_lr_frac,
                    "warmup_ratio": warmup_ratio,
                    "warmdown_ratio": sft_warmdown_ratio,
                    "final_lr_frac": final_lr_frac,
                    "eval_every": user_eval_every.unwrap_or(200),
                    "eval_tokens": user_eval_tokens.unwrap_or(40 * 524_288),
                    "chatcore_every": user_chatcore_every.unwrap_or(200),
                    "chatcore_max_cat": chatcore_max_cat,
                    "chatcore_max_sample": chatcore_max_sample,
                    "data_mode": data_mode.as_deref().unwrap_or("standard"),
                    "mmlu_epochs": mmlu_epochs,
                    "gsm8k_epochs": gsm8k_epochs,
                })
            } else {
                serde_json::json!({
                    "run": &run,
                    "device_type": &user_device_type,
                    "fp8": user_fp8,
                    "fp8_recipe": &fp8_recipe,
                    "depth": depth,
                    "aspect_ratio": aspect_ratio,
                    "head_dim": head_dim,
                    "max_seq_len": max_seq_len.unwrap_or(2048),
                    "dataset": user_dataset.as_ref().map_or_else(|| "climbmix".to_owned(), |path| path.display().to_string()),
                    "window_pattern": &window_pattern,
                    "num_iterations": user_steps,
                    "target_flops": target_flops.unwrap_or(-1.0),
                    "target_param_data_ratio": target_param_data_ratio.unwrap_or(12.0),
                    "device_batch_size": user_batch_size.unwrap_or(32),
                    "total_batch_size": user_total_batch_size.unwrap_or(-1),
                    "embedding_lr": user_embedding_lr.unwrap_or(0.3),
                    "unembedding_lr": user_unembedding_lr.unwrap_or(0.008),
                    "weight_decay": weight_decay,
                    "matrix_lr": user_matrix_lr.unwrap_or(0.02),
                    "scalar_lr": scalar_lr,
                    "warmup_steps": warmup_steps,
                    "warmdown_ratio": base_warmdown_ratio,
                    "final_lr_frac": final_lr_frac,
                    "resume_from_step": resume_from_step.unwrap_or(-1),
                    "eval_every": user_eval_every.unwrap_or(250),
                    "eval_tokens": user_eval_tokens.unwrap_or(80 * 524_288),
                    "core_metric_every": user_core_metric_every.unwrap_or(2000),
                    "core_metric_max_per_task": core_metric_max_per_task,
                    "sample_every": user_sample_every.unwrap_or(2000),
                    "save_every": user_save_every,
                    "model_tag": &model_tag,
                })
            };
            let options = training::TrainOptions {
                compute_dtype: compute_dtype.into(),
                fp8,
                steps: plan.steps,
                steps_explicit: user_steps > 0,
                batch_size,
                sequence_len,
                grad_accum_steps: plan.grad_accum_steps,
                embedding_lr: embedding_lr * plan.lr_scale,
                unembedding_lr: unembedding_lr * plan.lr_scale,
                matrix_lr: matrix_lr * plan.lr_scale,
                scalar_lr: scalar_lr * plan.lr_scale,
                weight_decay: plan.weight_decay,
                warmup_steps,
                warmdown_ratio: base_warmdown_ratio,
                final_lr_frac,
                sft_schedule,
                initial_lr_frac,
                warmup_ratio,
                sft_warmdown_ratio,
                full_epoch,
                load_optimizer: load_optimizer == 1,
                resume_data: resume_data.unwrap_or(!(resolve_from_store && sft_training)),
                eval_every,
                eval_tokens,
                core_metric_every,
                core_metric_max_per_task,
                chatcore_every,
                chatcore_max_cat,
                chatcore_max_sample,
                sample_every,
                save_every,
                output: match output {
                    Some(output) => {
                        anyhow::ensure!(
                            output_model_tag.is_none(),
                            "--output and --output-model-tag cannot be used together"
                        );
                        output
                    }
                    None if !sft_training => {
                        let base = resolve_base_dir(base_dir.as_deref())?;
                        base.join("base_checkpoints")
                            .join(model_tag.clone().unwrap_or_else(|| format!("d{depth}")))
                    }
                    None => {
                        let default_tag = model_tag
                            .as_deref()
                            .map(str::to_owned)
                            .unwrap_or_else(|| format!("d{}", config.n_layer));
                        let tag = output_model_tag
                            .as_deref()
                            .or_else(|| resolve_from_store.then_some(default_tag.as_str()))
                            .with_context(|| {
                                "provide --output, --output-model-tag, or select an input model tag"
                            })?;
                        resolve_base_dir(base_dir.as_deref())?
                            .join("chatsft_checkpoints")
                            .join(tag)
                    }
                },
                run: (run != "dummy").then_some(run),
                wandb_config,
            };
            match (checkpoint, parquet_files, line_input, corpus.as_ref()) {
                (Some(checkpoint), Some(files), _, _) => training::fine_tune_parquet(
                    config, checkpoint, files, &tokenizer, &options, &device,
                )?,
                (None, Some(files), _, _) => {
                    training::train_parquet(config, files, &tokenizer, &options, &device)?
                }
                (Some(checkpoint), None, true, _) => training::fine_tune_jsonl(
                    config,
                    checkpoint,
                    dataset,
                    &tokenizer,
                    &options,
                    &device,
                    jsonl_input,
                    validation_dataset,
                )?,
                (None, None, true, _) => training::train_jsonl(
                    config,
                    dataset,
                    &tokenizer,
                    &options,
                    &device,
                    jsonl_input,
                    validation_dataset,
                )?,
                (Some(checkpoint), None, false, Some(corpus)) => {
                    training::fine_tune(config, checkpoint, corpus, &tokenizer, &options, &device)?
                }
                (None, None, false, Some(corpus)) => {
                    training::train(config, corpus, &tokenizer, &options, &device)?
                }
                _ => unreachable!("every dataset input is Parquet, JSONL, or a text corpus"),
            }
        }
        Command::Serve {
            config,
            checkpoint,
            tokenizer,
            source,
            model_tag,
            step,
            base_dir,
            model_name,
            host,
            port,
            address,
            temperature,
            top_k,
            dtype,
            device,
        } => {
            let top_k = normalize_top_k(top_k);
            let (config, checkpoint_path, tokenizer_path) = resolve_model_inputs(
                config.as_deref(),
                checkpoint.as_deref(),
                tokenizer.as_deref(),
                &source,
                model_tag.as_deref(),
                step,
                base_dir.as_deref(),
            )?;
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer_path)?;
            anyhow::ensure!(
                tokenizer.vocab_size() as usize == config.vocab_size,
                "tokenizer vocabulary does not match model configuration"
            );
            let device = resolve_requested_device(&device)?;
            let dtype = resolve_compute_dtype(dtype, &device)?;
            let model =
                model::GptModel::load_checkpoint(config, checkpoint_path, &device, dtype.into())?;
            let engine = engine::Engine::new(model, tokenizer);
            let address = address.unwrap_or_else(|| {
                let host = host.as_deref().unwrap_or("127.0.0.1");
                let host = if host.contains(':') {
                    format!("[{host}]")
                } else {
                    host.to_owned()
                };
                format!("{host}:{}", port.unwrap_or(8000))
            });
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?;
            runtime.block_on(api::serve(
                engine,
                model_name,
                address,
                std::env::var("NANOCHAT_API_KEY")
                    .ok()
                    .filter(|key| !key.is_empty()),
                temperature,
                top_k,
            ))?;
        }
        Command::EvalBpb {
            config,
            checkpoint,
            tokenizer,
            dataset,
            sequence_len,
            batch_size,
            eval_tokens,
            dtype,
            device,
        } => {
            let config_value = json_compat::parse_value(&std::fs::read(config)?)?;
            let config_value = config_value
                .get("model_config")
                .cloned()
                .unwrap_or(config_value);
            let config = model_config::GptConfig::from_value(config_value)?;
            config.validate()?;
            let device = parse_device(&device)?;
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            let model =
                model::GptModel::load_checkpoint(config, checkpoint, &device, dtype.into())?;
            let parquet_input = dataset.is_dir()
                || dataset
                    .extension()
                    .is_some_and(|extension| extension == "parquet");
            let bpb = if parquet_input {
                let files = if dataset.is_dir() {
                    datasets::local_files(&dataset)?
                } else {
                    vec![dataset]
                };
                evaluation::evaluate_parquet_bpb(
                    &model,
                    &tokenizer,
                    files,
                    true,
                    sequence_len,
                    batch_size,
                    eval_tokens,
                    &device,
                )?
            } else {
                let corpus = data::TokenCorpus::from_path_split(&dataset, &tokenizer, true)?;
                evaluation::evaluate_bpb(&model, &tokenizer, &corpus, sequence_len, &device)?
            };
            println!("val_bpb={bpb:.6}");
        }
        Command::EvalChat {
            config,
            checkpoint,
            tokenizer,
            source,
            model_tag,
            step,
            base_dir,
            task_name,
            batch_size,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            seed,
            max_problems,
            predictions_output,
            dtype,
            device,
        } => {
            let (config, checkpoint, tokenizer) = resolve_model_inputs(
                config.as_deref(),
                checkpoint.as_deref(),
                tokenizer.as_deref(),
                &source,
                model_tag.as_deref(),
                step,
                base_dir.as_deref(),
            )?;
            let device = resolve_requested_device(&device)?;
            let dtype = resolve_compute_dtype(dtype, &device)?;
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            anyhow::ensure!(
                tokenizer.vocab_size() as usize == config.vocab_size,
                "tokenizer vocabulary does not match model configuration"
            );
            let model =
                model::GptModel::load_checkpoint(config, checkpoint, &device, dtype.into())?;
            let engine = engine::Engine::new(model, tokenizer);
            let tasks = if task_name == "all" {
                vec![
                    task_eval::ChatTask::ArcEasy,
                    task_eval::ChatTask::ArcChallenge,
                    task_eval::ChatTask::Mmlu,
                    task_eval::ChatTask::Gsm8k,
                    task_eval::ChatTask::HumanEval,
                ]
            } else {
                task_name
                    .split('|')
                    .map(task_eval::ChatTask::parse)
                    .collect::<Result<Vec<_>>>()?
            };
            let results = task_eval::evaluate_chat_tasks(
                &engine,
                &tasks,
                batch_size,
                num_samples,
                max_new_tokens,
                temperature,
                normalize_top_k(Some(top_k)),
                seed,
                max_problems,
                predictions_output.as_deref(),
            )?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::EvalCore {
            config,
            checkpoint,
            tokenizer,
            max_per_task,
            dtype,
            device,
        } => {
            let config_value = json_compat::parse_value(&std::fs::read(config)?)?;
            let config_value = config_value
                .get("model_config")
                .cloned()
                .unwrap_or(config_value);
            let config = model_config::GptConfig::from_value(config_value)?;
            config.validate()?;
            let device = resolve_requested_device(&device)?;
            let dtype = resolve_compute_dtype(dtype, &device)?;
            let tokenizer = tokenizer::BpeTokenizer::load(tokenizer)?;
            anyhow::ensure!(
                tokenizer.vocab_size() as usize == config.vocab_size,
                "tokenizer vocabulary does not match model configuration"
            );
            let model =
                model::GptModel::load_checkpoint(config, checkpoint, &device, dtype.into())?;
            let engine = engine::Engine::new(model, tokenizer);
            let max_per_task = max_per_task
                .filter(|limit| *limit > 0)
                .map(|limit| limit as usize);
            let results = core_eval::evaluate_core(&engine, &device, max_per_task)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::EvalTokenizer { dataset, tokenizer } => {
            let results = tokenizer_eval::evaluate(&dataset, tokenizer)?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::BaseEval {
            eval,
            model_tag,
            step,
            base_dir,
            tokenizer,
            max_per_task,
            device_batch_size,
            dataset,
            split_tokens,
            dtype,
            device,
        } => {
            let device = resolve_requested_device(&device)?;
            let dtype = resolve_compute_dtype(dtype, &device)?;
            let results = base_eval::run(
                &base_eval::BaseEvalOptions {
                    eval,
                    model_tag,
                    step,
                    base_dir,
                    tokenizer,
                    max_per_task: max_per_task
                        .and_then(|limit| usize::try_from(limit).ok())
                        .filter(|limit| *limit > 0),
                    device_batch_size,
                    dataset,
                    split_tokens,
                },
                &device,
                dtype.into(),
            )?;
            println!("{}", serde_json::to_string_pretty(&results)?);
        }
        Command::EvalBench {
            source,
            model_tag,
            step,
            base_dir,
            tokenizer,
            prompt_tokens,
            decode_tokens,
            batch_sizes,
            temperature,
            dtype,
            device,
        } => {
            let device = parse_device(&device)?;
            let dtype = resolve_compute_dtype(dtype, &device)?;
            inference_bench::run(
                &source,
                model_tag.as_deref(),
                step,
                base_dir.as_deref(),
                tokenizer,
                prompt_tokens,
                decode_tokens,
                &batch_sizes,
                temperature,
                dtype.into(),
                &device,
            )?;
        }
        Command::ChatRl {
            run,
            model_tag,
            model_step,
            base_dir,
            tokenizer,
            num_epochs,
            reward_mode,
            judge_model,
            codex_command,
            judge_timeout,
            device_batch_size,
            examples_per_step,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            embedding_lr,
            unembedding_lr,
            matrix_lr,
            weight_decay,
            init_lr_frac,
            eval_every,
            eval_examples,
            save_every,
            dtype,
            device,
        } => {
            let wandb_config = serde_json::json!({
                "run": &run,
                "device_type": if device == "auto" { "" } else { device.as_str() },
                "model_tag": &model_tag,
                "model_step": model_step,
                "num_epochs": num_epochs,
                "reward_mode": &reward_mode,
                "judge_model": &judge_model,
                "codex_command": &codex_command,
                "judge_timeout": judge_timeout,
                "device_batch_size": device_batch_size,
                "examples_per_step": examples_per_step,
                "num_samples": num_samples,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "embedding_lr": embedding_lr,
                "unembedding_lr": unembedding_lr,
                "matrix_lr": matrix_lr,
                "weight_decay": weight_decay,
                "init_lr_frac": init_lr_frac,
                "eval_every": eval_every,
                "eval_examples": eval_examples,
                "save_every": save_every,
            });
            let top_k = normalize_top_k(Some(top_k)).unwrap_or(0);
            let device = resolve_requested_device(&device)?;
            let compute_dtype = resolve_compute_dtype(dtype, &device)?;
            rl::train(
                &rl::RlOptions {
                    run,
                    wandb_config,
                    model_tag,
                    model_step,
                    base_dir,
                    tokenizer,
                    num_epochs,
                    reward_mode,
                    judge_model,
                    codex_command,
                    judge_timeout,
                    device_batch_size,
                    examples_per_step,
                    num_samples,
                    max_new_tokens,
                    temperature,
                    top_k,
                    embedding_lr,
                    unembedding_lr,
                    matrix_lr,
                    weight_decay,
                    init_lr_frac,
                    eval_every,
                    eval_examples,
                    save_every,
                    compute_dtype: compute_dtype.into(),
                },
                &device,
            )?;
        }
    }
    Ok(ExitCode::SUCCESS)
}

fn launch_local(nproc_per_node: usize, command: Vec<OsString>) -> Result<ExitCode> {
    anyhow::ensure!(nproc_per_node > 0, "--nproc-per-node must be positive");
    anyhow::ensure!(!command.is_empty(), "provide a Rust subcommand to launch");
    let master_addr = std::env::var("MASTER_ADDR").unwrap_or_else(|_| "127.0.0.1".to_owned());
    let master_port = match std::env::var("MASTER_PORT") {
        Ok(port) => port,
        Err(_) => TcpListener::bind((master_addr.as_str(), 0))?
            .local_addr()?
            .port()
            .to_string(),
    };
    let executable = std::env::current_exe()?;
    let world_size = nproc_per_node.to_string();
    let local_world_size = world_size.clone();
    let mut children: Vec<(usize, Child)> = Vec::with_capacity(nproc_per_node);

    for rank in 0..nproc_per_node {
        let child = ProcessCommand::new(&executable)
            .args(&command)
            .env("RANK", rank.to_string())
            .env("LOCAL_RANK", rank.to_string())
            .env("WORLD_SIZE", &world_size)
            .env("LOCAL_WORLD_SIZE", &local_world_size)
            .env("MASTER_ADDR", &master_addr)
            .env("MASTER_PORT", &master_port)
            .spawn();
        match child {
            Ok(child) => children.push((rank, child)),
            Err(error) => {
                for (_, child) in &mut children {
                    let _ = child.kill();
                    let _ = child.wait();
                }
                return Err(error).context("start Rust worker process");
            }
        }
    }

    let mut failed = None;
    while !children.is_empty() {
        let mut index = 0;
        while index < children.len() {
            match children[index].1.try_wait()? {
                Some(status) if status.success() => {
                    children.swap_remove(index);
                }
                Some(status) => {
                    failed = Some((children[index].0, status));
                    break;
                }
                None => index += 1,
            }
        }
        if failed.is_some() {
            break;
        }
        if !children.is_empty() {
            std::thread::sleep(Duration::from_millis(25));
        }
    }

    if let Some((rank, status)) = failed {
        for (_, child) in &mut children {
            let _ = child.kill();
        }
        for (_, child) in &mut children {
            let _ = child.wait();
        }
        eprintln!("rust-agent: worker rank {rank} exited with {status}");
        return Ok(ExitCode::FAILURE);
    }
    Ok(ExitCode::SUCCESS)
}

#[allow(clippy::too_many_arguments)]
fn resolve_model_inputs(
    config_path: Option<&Path>,
    checkpoint_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    source: &str,
    model_tag: Option<&str>,
    step: Option<u64>,
    base_dir: Option<&Path>,
) -> Result<(model_config::GptConfig, PathBuf, PathBuf)> {
    anyhow::ensure!(
        config_path.is_some() == checkpoint_path.is_some(),
        "provide both --config and --checkpoint for a local model, or neither to resolve --source/--model-tag/--step"
    );
    let base_dir = if config_path.is_none() || tokenizer_path.is_none() || base_dir.is_some() {
        Some(resolve_base_dir(base_dir)?)
    } else {
        None
    };
    let (config_path, checkpoint_path) = if let (Some(config), Some(checkpoint)) =
        (config_path, checkpoint_path)
    {
        (config.to_path_buf(), checkpoint.to_path_buf())
    } else {
        let selected = checkpoint::select_checkpoint(source, base_dir.as_deref(), model_tag, step)?;
        let metadata_path = selected
            .metadata_file
            .context("selected checkpoint has no metadata file")?;
        (metadata_path, selected.model_file)
    };
    let checkpoint_path = checkpoint::resolve_model_file(checkpoint_path)?;
    let config_value = json_compat::parse_value(&std::fs::read(&config_path)?)?;
    let config_value = config_value
        .get("model_config")
        .cloned()
        .unwrap_or(config_value);
    let config = model_config::GptConfig::from_value(config_value)?;
    config.validate()?;
    let tokenizer_path = tokenizer_path
        .map(Path::to_path_buf)
        .or_else(|| {
            base_dir
                .as_ref()
                .map(|base_dir| paths::cached_tokenizer_path(base_dir))
        })
        .context("a tokenizer path or nanochat base directory is required")?;
    Ok((config, checkpoint_path, tokenizer_path))
}

fn infer_validation_dataset(training_path: &Path) -> Option<PathBuf> {
    let extension = training_path.extension()?.to_str()?;
    let stem = training_path.file_stem()?.to_str()?;
    let validation_stem = if stem == "train" {
        "validation".to_owned()
    } else if let Some(prefix) = stem.strip_suffix("-train") {
        format!("{prefix}-validation")
    } else if let Some(prefix) = stem.strip_suffix("_train") {
        format!("{prefix}_validation")
    } else {
        return None;
    };
    let candidate = training_path.with_file_name(format!("{validation_stem}.{extension}"));
    candidate.is_file().then_some(candidate)
}

fn resolve_requested_device(value: &str) -> Result<Device> {
    if value == "auto" {
        let local_rank = std::env::var("LOCAL_RANK")
            .ok()
            .and_then(|rank| rank.parse::<usize>().ok())
            .unwrap_or(0);
        let cuda = Device::cuda_if_available(local_rank)?;
        if cuda.is_cuda() {
            Ok(cuda)
        } else {
            Device::metal_if_available(local_rank).map_err(Into::into)
        }
    } else if value == "mps" {
        parse_device("metal")
    } else {
        parse_device(value)
    }
}

fn resolve_compute_dtype(requested: Option<ComputeDtype>, device: &Device) -> Result<ComputeDtype> {
    if let Some(dtype) = requested {
        return Ok(dtype);
    }
    if let Some(dtype) = std::env::var_os("NANOCHAT_DTYPE") {
        return match dtype.to_string_lossy().as_ref() {
            "bfloat16" => Ok(ComputeDtype::Bfloat16),
            "float16" => Ok(ComputeDtype::Float16),
            "float32" => Ok(ComputeDtype::Float32),
            other => anyhow::bail!("unsupported NANOCHAT_DTYPE `{other}`"),
        };
    }
    Ok(if device.is_cuda() && device.supports_bf16() {
        ComputeDtype::Bfloat16
    } else {
        ComputeDtype::Float32
    })
}

/// Python's sampler treats every non-positive top-k value as disabled.
fn normalize_top_k(top_k: Option<i64>) -> Option<usize> {
    top_k
        .and_then(|value| usize::try_from(value).ok())
        .filter(|&value| value > 0)
}

#[allow(clippy::too_many_arguments)]
fn run_chat(
    config_path: Option<&Path>,
    checkpoint_path: Option<&Path>,
    tokenizer_path: Option<&Path>,
    source: &str,
    model_tag: Option<&str>,
    step: Option<u64>,
    base_dir: Option<&Path>,
    initial_prompt: &str,
    max_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    seed: u64,
    requested_dtype: Option<ComputeDtype>,
    requested_device: &str,
) -> Result<()> {
    let (config, checkpoint_path, tokenizer_path) = resolve_model_inputs(
        config_path,
        checkpoint_path,
        tokenizer_path,
        source,
        model_tag,
        step,
        base_dir,
    )?;
    let device = resolve_requested_device(requested_device)?;
    let dtype = resolve_compute_dtype(requested_dtype, &device)?;
    let tokenizer = tokenizer::BpeTokenizer::load(&tokenizer_path)?;
    anyhow::ensure!(
        tokenizer.vocab_size() as usize == config.vocab_size,
        "tokenizer vocabulary does not match model configuration"
    );
    let model = model::GptModel::load_checkpoint(config, checkpoint_path, &device, dtype.into())?;
    let engine = engine::Engine::new(model, tokenizer);
    let assistant_end = engine.tokenizer.special_id("<|assistant_end|>")?;
    let user_start = engine.tokenizer.special_id("<|user_start|>")?;
    let user_end = engine.tokenizer.special_id("<|user_end|>")?;
    let assistant_start = engine.tokenizer.special_id("<|assistant_start|>")?;
    let mut conversation = vec![engine.tokenizer.bos_id()];

    println!("\nNanoChat Interactive Mode");
    println!("{}", "-".repeat(50));
    println!("Type 'quit' or 'exit' to end the conversation");
    println!("Type 'clear' to start a new conversation");
    println!("{}", "-".repeat(50));
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        let input = if initial_prompt.is_empty() {
            print!("\nUser: ");
            std::io::stdout().flush()?;
            match lines.next() {
                Some(line) => line?,
                None => {
                    println!("\nGoodbye!");
                    break;
                }
            }
            .trim()
            .to_owned()
        } else {
            initial_prompt.to_owned()
        };
        match input.to_lowercase().as_str() {
            "quit" | "exit" => {
                println!("Goodbye!");
                break;
            }
            "clear" => {
                conversation.clear();
                conversation.push(engine.tokenizer.bos_id());
                println!("Conversation cleared.");
                if !initial_prompt.is_empty() {
                    break;
                }
                continue;
            }
            _ if input.is_empty() => {
                if !initial_prompt.is_empty() {
                    break;
                }
                continue;
            }
            _ => {}
        }
        conversation.push(user_start);
        conversation.extend(engine.tokenizer.encode(&input)?);
        conversation.push(user_end);
        conversation.push(assistant_start);
        print!("\nAssistant: ");
        std::io::stdout().flush()?;
        let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
        let mut response = Vec::new();
        let mut stdout = std::io::stdout().lock();
        engine.generate_with_terminal(
            &conversation,
            max_tokens,
            temperature,
            top_k,
            seed,
            |token, terminal| {
                response.push(token);
                if let Ok(bytes) = engine.tokenizer.token_bytes(token) {
                    let mut decoded = String::new();
                    if terminal {
                        let _ = decoder.decode_to_string(&[], &mut decoded, true);
                    } else {
                        let _ = decoder.decode_to_string(&bytes, &mut decoded, false);
                    }
                    if stdout.write_all(decoded.as_bytes()).is_err() {
                        return false;
                    }
                    if terminal && stdout.write_all(&bytes).is_err() {
                        return false;
                    }
                    let _ = stdout.flush();
                }
                true
            },
        )?;
        let mut decoded = String::new();
        let _ = decoder.decode_to_string(&[], &mut decoded, true);
        stdout.write_all(decoded.as_bytes())?;
        stdout.write_all(b"\n")?;
        stdout.flush()?;
        drop(stdout);
        if response.last() != Some(&assistant_end) {
            response.push(assistant_end);
        }
        conversation.extend(response);
        if !initial_prompt.is_empty() {
            break;
        }
    }
    Ok(())
}

fn resolve_base_dir(base_dir: Option<&Path>) -> Result<PathBuf> {
    if let Some(base_dir) = base_dir {
        return Ok(base_dir.to_path_buf());
    }
    if let Some(base_dir) = paths::base_dir_override() {
        return Ok(PathBuf::from(base_dir));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE must be set to locate nanochat checkpoints")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}

fn read_checkpoint_metadata(checkpoint: &Path) -> Result<serde_json::Value> {
    let Some(directory) = checkpoint.parent() else {
        return Ok(serde_json::json!({}));
    };
    let numbered_metadata = checkpoint
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.strip_prefix("model_"))
        .and_then(|step| step.parse::<u64>().ok())
        .map(|step| directory.join(format!("meta_{step:06}.json")));
    let metadata_path = numbered_metadata
        .filter(|path| path.is_file())
        .unwrap_or_else(|| directory.join("meta.json"));
    if !metadata_path.is_file() {
        return Ok(serde_json::json!({}));
    }
    json_compat::parse_value(&std::fs::read(&metadata_path).map_err(|error| {
        anyhow::anyhow!(
            "read checkpoint metadata {}: {error}",
            metadata_path.display()
        )
    })?)
}

fn parse_device(value: &str) -> Result<Device> {
    match value {
        "cpu" => Ok(Device::Cpu),
        "cuda" => {
            let local_rank = std::env::var("LOCAL_RANK")
                .ok()
                .map(|rank| rank.parse::<usize>())
                .transpose()?
                .unwrap_or(0);
            Ok(Device::new_cuda(local_rank)?)
        }
        "metal" | "mps" => {
            let local_rank = std::env::var("LOCAL_RANK")
                .ok()
                .map(|rank| rank.parse::<usize>())
                .transpose()?
                .unwrap_or(0);
            Ok(Device::new_metal(local_rank)?)
        }
        other => anyhow::bail!("unknown device '{other}', choose cpu, cuda, or metal"),
    }
}

struct TrainingPlan {
    steps: usize,
    grad_accum_steps: usize,
    lr_scale: f64,
    weight_decay: f64,
}

#[allow(clippy::too_many_arguments)]
fn resolve_training_plan(
    config: &model_config::GptConfig,
    requested_steps: Option<usize>,
    requested_flops: Option<f64>,
    requested_ratio: Option<f64>,
    batch_size: usize,
    sequence_len: usize,
    requested_accumulation: Option<usize>,
    requested_total_batch: Option<usize>,
    base_weight_decay: f64,
) -> Result<TrainingPlan> {
    anyhow::ensure!(
        batch_size > 0 && sequence_len > 0,
        "batch size and sequence length must be positive"
    );
    if let Some(accumulation) = requested_accumulation {
        anyhow::ensure!(
            accumulation > 0,
            "gradient accumulation steps must be positive"
        );
    }
    if let Some(batch) = requested_total_batch {
        anyhow::ensure!(batch > 0, "total batch size must be positive");
    }
    let (_, world_size) = distributed::DistributedContext::env_group()?;
    let microbatch_tokens = batch_size
        .checked_mul(sequence_len)
        .and_then(|tokens| tokens.checked_mul(world_size))
        .context("global microbatch token count overflow")?;
    let ratio = requested_ratio.filter(|ratio| ratio.is_finite() && *ratio > 0.0);
    let target_flops = requested_flops.filter(|flops| flops.is_finite() && *flops > 0.0);
    let scaling_enabled =
        ratio.is_some() || target_flops.is_some() || requested_total_batch.is_some();
    let ratio_for_scaling = ratio.unwrap_or(12.0);
    let scaling_tokens =
        ratio_for_scaling * model::GptModel::scaling_parameter_count(config) as f64;

    let auto_total_batch = if requested_total_batch.is_none()
        && requested_accumulation.is_none()
        && (ratio.is_some() || target_flops.is_some())
    {
        let reference = reference_config(config)?;
        let reference_tokens =
            ratio_for_scaling * model::GptModel::scaling_parameter_count(&reference) as f64;
        let predicted = 524_288.0
            * 2.0f64.powf(
                (scaling_tokens / reference_tokens)
                    .powf(0.383)
                    .log2()
                    .round(),
            );
        anyhow::ensure!(
            predicted.is_finite() && predicted >= 1.0,
            "automatic total batch size is invalid"
        );
        Some(predicted as usize)
    } else {
        None
    };
    let total_batch = requested_total_batch.or(auto_total_batch);
    let grad_accum_steps = if let Some(total_batch) = total_batch {
        anyhow::ensure!(
            total_batch % microbatch_tokens == 0,
            "total-batch-size ({total_batch}) must be divisible by batch-size * sequence-len * world-size ({microbatch_tokens})"
        );
        let derived = total_batch / microbatch_tokens;
        if let Some(requested) = requested_accumulation {
            anyhow::ensure!(
                requested == derived,
                "grad-accum-steps conflicts with total-batch-size"
            );
        }
        anyhow::ensure!(
            derived > 0,
            "total-batch-size is smaller than one global microbatch"
        );
        derived
    } else {
        requested_accumulation.unwrap_or(1)
    };
    let global_batch_tokens = microbatch_tokens
        .checked_mul(grad_accum_steps)
        .context("global batch token count overflow")?;
    let lr_scale = if scaling_enabled {
        (global_batch_tokens as f64 / 524_288.0).sqrt()
    } else {
        1.0
    };
    let d_ref = ratio_for_scaling
        * model::GptModel::scaling_parameter_count(&reference_config(config)?) as f64;
    let weight_decay = if scaling_enabled {
        base_weight_decay * lr_scale * d_ref / scaling_tokens
    } else {
        base_weight_decay
    };
    let steps = if let Some(steps) = requested_steps.filter(|steps| *steps > 0) {
        steps
    } else if let Some(target_flops) = target_flops {
        (target_flops
            / (model::GptModel::training_flops_per_token(config) as f64
                * global_batch_tokens as f64))
            .round() as usize
    } else if let Some(ratio) = ratio {
        (ratio * model::GptModel::scaling_parameter_count(config) as f64) as usize
            / global_batch_tokens
    } else {
        100
    };
    anyhow::ensure!(steps > 0, "training horizon produces zero optimizer steps");
    Ok(TrainingPlan {
        steps,
        grad_accum_steps,
        lr_scale,
        weight_decay,
    })
}

fn reference_config(config: &model_config::GptConfig) -> Result<model_config::GptConfig> {
    let head_dim = config.n_embd / config.n_head;
    anyhow::ensure!(head_dim > 0, "model head dimension must be positive");
    let aspect_ratio = config.n_embd / config.n_layer;
    let width = (12 * aspect_ratio).div_ceil(head_dim) * head_dim;
    let heads = width / head_dim;
    let kv_heads = if config.n_kv_head == config.n_head {
        heads
    } else {
        (heads * config.n_kv_head / config.n_head).max(1)
    };
    Ok(model_config::GptConfig {
        sequence_len: config.sequence_len,
        vocab_size: config.vocab_size,
        n_layer: 12,
        n_head: heads,
        n_kv_head: kv_heads,
        n_embd: width,
        window_pattern: config.window_pattern.clone(),
    })
}

fn resolve_warmdown_ratios(requested: Option<f64>) -> (f64, f64) {
    (requested.unwrap_or(0.65), requested.unwrap_or(0.5))
}

fn resolve_training_sequence_len(
    requested: Option<usize>,
    sft_training: bool,
    checkpoint_metadata: &serde_json::Value,
    config_sequence_len: usize,
) -> usize {
    if let Some(requested) = requested {
        return requested;
    }
    if !sft_training {
        return config_sequence_len;
    }
    checkpoint_metadata
        .get("max_seq_len")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| {
            checkpoint_metadata
                .get("user_config")
                .and_then(|value| value.get("max_seq_len"))
                .and_then(serde_json::Value::as_u64)
        })
        .or_else(|| {
            checkpoint_metadata
                .get("model_config")
                .and_then(|value| value.get("sequence_len"))
                .and_then(serde_json::Value::as_u64)
        })
        .and_then(|value| usize::try_from(value).ok())
        .unwrap_or(config_sequence_len)
}

#[cfg(test)]
mod train_cli_parity_tests {
    use super::{Args, Command, resolve_training_sequence_len, resolve_warmdown_ratios};
    use clap::Parser;

    #[test]
    fn sft_training_options_accept_python_spellings_and_keep_legacy_aliases() {
        for args in [
            [
                "rust-agent",
                "train",
                "--max-seq-len",
                "512",
                "--init-lr-frac",
                "0.7",
                "--warmdown-ratio",
                "0.4",
            ],
            [
                "rust-agent",
                "train",
                "--sequence-len",
                "512",
                "--initial-lr-frac",
                "0.7",
                "--sft-warmdown-ratio",
                "0.4",
            ],
        ] {
            let Args {
                command:
                    Command::Train {
                        max_seq_len,
                        initial_lr_frac,
                        warmdown_ratio,
                        ..
                    },
            } = Args::try_parse_from(args).unwrap()
            else {
                panic!("expected train command")
            };
            assert_eq!(max_seq_len, Some(512));
            assert_eq!(initial_lr_frac, 0.7);
            assert_eq!(warmdown_ratio, Some(0.4));
        }
    }

    #[test]
    fn sequence_override_and_warmdown_defaults_follow_training_mode() {
        let metadata = serde_json::json!({
            "max_seq_len": 1024,
            "model_config": { "sequence_len": 2048 }
        });
        assert_eq!(
            resolve_training_sequence_len(Some(512), true, &metadata, 2048),
            512
        );
        assert_eq!(
            resolve_training_sequence_len(None, true, &metadata, 2048),
            1024
        );
        assert_eq!(
            resolve_training_sequence_len(None, false, &metadata, 2048),
            2048
        );
        assert_eq!(resolve_warmdown_ratios(None), (0.65, 0.5));
        assert_eq!(resolve_warmdown_ratios(Some(0.4)), (0.4, 0.4));
    }
}
