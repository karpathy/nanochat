//! Inference latency, throughput, and architecture-cost benchmark.

use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
    time::Instant,
};

use anyhow::{Context, Result, ensure};
use candle_core::{DType, Device};
use serde_json::{Value, json};

use crate::{
    checkpoint, engine::Engine, model::GptModel, model_config::GptConfig, tokenizer::BpeTokenizer,
};

#[derive(Debug)]
struct GpuInfo {
    name: String,
    total_memory_bytes: u64,
}

#[derive(Debug)]
struct Measurement {
    ttft_sec: f64,
    step_times: Vec<f64>,
    peak_vram_bytes: Option<u64>,
}

pub fn run(
    source: &str,
    model_tag: Option<&str>,
    step: Option<u64>,
    base_dir: Option<&Path>,
    tokenizer_path: Option<PathBuf>,
    prompt_tokens: usize,
    decode_tokens: usize,
    batch_sizes: &str,
    temperature: f64,
    dtype: DType,
    device: &Device,
) -> Result<Value> {
    ensure!(
        matches!(device, Device::Cuda(_)),
        "eval-bench currently requires a CUDA device"
    );
    ensure!(
        std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            == 1,
        "eval-bench requires a single process; do not launch it with torchrun"
    );
    ensure!(decode_tokens > 0, "decode token count must be positive");
    ensure!(
        temperature >= 0.0 && temperature.is_finite(),
        "temperature must be finite and non-negative"
    );
    let selection = checkpoint::select_checkpoint(source, base_dir, model_tag, step)?;
    let metadata_path = selection
        .metadata_file
        .context("selected checkpoint has no metadata JSON")?;
    let metadata = crate::json_compat::parse_value(
        &std::fs::read(&metadata_path)
            .with_context(|| format!("read checkpoint metadata {}", metadata_path.display()))?,
    )?;
    let model_config_value = metadata
        .get("model_config")
        .cloned()
        .context("checkpoint metadata has no model_config")?;
    let config = GptConfig::from_value(model_config_value.clone())?;
    config.validate()?;
    let tokenizer_path = tokenizer_path.unwrap_or(default_tokenizer_path(base_dir)?);
    let tokenizer = BpeTokenizer::load(&tokenizer_path)
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
    ensure!(
        tokenizer.vocab_size() as usize == config.vocab_size,
        "tokenizer vocabulary does not match model configuration"
    );
    let model = GptModel::load_checkpoint(config.clone(), &selection.model_file, device, dtype)?;
    let engine = Engine::new(model, tokenizer);

    let gpu_index = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let gpu = query_gpu(gpu_selector(gpu_index))?;
    let peak_bandwidth = peak_bandwidth(&gpu.name);
    let peak_flops = peak_flops(&gpu.name);
    let prompt_max = config
        .sequence_len
        .checked_sub(decode_tokens)
        .context("decode tokens exceed the model context")?;
    let prompt_len = prompt_tokens.min(prompt_max);
    ensure!(prompt_len > 0, "prompt length is zero after context clamp");
    let prompt = build_prompt(&engine.tokenizer, prompt_len)?;
    let context_mid = prompt_len + decode_tokens / 2;
    let weight_bytes = engine_model(&engine).weight_bytes() as u64;
    let num_params = engine_model(&engine).parameter_count() as u64;
    let kv_bytes_per_token = engine_model(&engine).kv_bytes_per_token() as u64;
    let kv_read_bytes = engine_model(&engine).kv_read_bytes(context_mid) as u64;
    let decode_flops = engine_model(&engine).estimate_decode_flops(context_mid);
    let prefill_flops = engine_model(&engine).estimate_prefill_flops(prompt_len);
    let ceiling_bs1 =
        peak_bandwidth.map(|bandwidth| bandwidth / (weight_bytes + kv_read_bytes) as f64);
    let max_rows = if kv_bytes_per_token > 0 {
        ((gpu.total_memory_bytes.saturating_sub(weight_bytes))
            / (kv_bytes_per_token * config.sequence_len as u64)) as u64
    } else {
        0
    };

    let mut payload = json!({
        "source": source,
        "model_tag": selection.model_tag,
        "step": selection.step,
        "model_config": model_config_value,
        "gpu": gpu.name,
        "peak_bandwidth_bytes_per_sec": peak_bandwidth,
        "total_vram_bytes": gpu.total_memory_bytes,
        "num_params": num_params,
        "param_dtypes": { format_dtype(dtype): num_params },
        "weight_bytes": weight_bytes,
        "kv_bytes_per_token": kv_bytes_per_token,
        "kv_read_bytes_per_step": kv_read_bytes,
        "context_mid": context_mid,
        "peak_flops_per_sec": peak_flops,
        "decode_flops_per_token": decode_flops,
        "peak_vram_source": "nvidia-smi per-process memory sampled every 500ms",
        "ceiling_bs1_tok_per_sec": ceiling_bs1.map(|value| round(value, 1)),
        "max_full_context_rows": max_rows,
        "prompt_tokens": prompt_len,
        "decode_tokens": decode_tokens,
        "temperature": temperature,
        "sweep": [],
    });

    print!("{}", "=".repeat(100));
    println!(
        "\nModel: {source} {} (step {}) | depth {}, dim {}, heads {}, kv heads {} (GQA)",
        selection.model_tag,
        selection.step.unwrap_or_default(),
        config.n_layer,
        config.n_embd,
        config.n_head,
        config.n_kv_head
    );
    println!(
        "GPU: {} | peak bandwidth {} | peak compute {} | VRAM {:.0} GiB",
        gpu.name,
        format_peak(peak_bandwidth, "TB/s"),
        format_peak(peak_flops, "TFLOPS"),
        gpu.total_memory_bytes as f64 / 2f64.powi(30)
    );
    println!(
        "Parameters: {num_params} ({}) | weight bytes as stored: {:.0} MiB",
        format_dtype(dtype),
        weight_bytes as f64 / 2f64.powi(20)
    );
    println!(
        "KV cache: {kv_bytes_per_token} bytes/token stored | {kv_read_bytes} bytes read/step at context {context_mid} (window pattern {})",
        config.window_pattern
    );
    println!(
        "Theoretical decode ceiling at batch 1: {} tok/s | max ~{max_rows} full-context rows in VRAM",
        ceiling_bs1
            .map(|value| format!("{value:.0}"))
            .unwrap_or_else(|| "unknown".to_owned())
    );
    println!("{}", "=".repeat(100));

    let prefill_warmup = benchmark_generation(&engine, &prompt, 1, 2, temperature, None)?;
    let prefill = benchmark_generation(&engine, &prompt, 1, 2, temperature, None)?;
    let prefill_time = prefill.ttft_sec;
    let prefill_tok_per_sec = prompt_len as f64 / prefill_time;
    let prefill_mfu = peak_flops.map(|peak| prefill_flops as f64 / prefill_time / peak * 100.0);
    println!(
        "Prefill (batch 1, {prompt_len} tokens): {prefill_tok_per_sec:.0} tok/s | MFU {}",
        prefill_mfu
            .map(|value| format!("{value:.1}%"))
            .unwrap_or_else(|| "unknown".to_owned())
    );
    payload["prefill"] = json!({
        "tok_per_sec": round(prefill_tok_per_sec, 1),
        "mfu_percent": prefill_mfu.map(|value| round(value, 2)),
        "time_sec": round(prefill_time, 6),
    });
    drop(prefill_warmup);

    let batches = parse_batch_sizes(batch_sizes)?;
    let header = format!(
        "{:>6} {:>9} {:>9} {:>10} {:>7} {:>7} {:>9} {:>6}",
        "batch", "TTFT ms", "TPOT ms", "tok/s", "MBU %", "MFU %", "VRAM GiB", "steps"
    );
    println!("{header}\n{}", "-".repeat(header.len()));
    let mut sweep = Vec::new();
    for batch_size in batches {
        let _warmup = benchmark_generation(&engine, &prompt, batch_size, 8, temperature, None)?;
        let result = benchmark_generation(
            &engine,
            &prompt,
            batch_size,
            decode_tokens,
            temperature,
            None,
        )?;
        if result.step_times.is_empty() {
            println!("{batch_size:>6} all rows terminated during prefill?! skipping");
            continue;
        }
        let mut sorted_times = result.step_times.clone();
        sorted_times.sort_by(f64::total_cmp);
        let tpot = sorted_times[sorted_times.len() / 2];
        let tok_per_sec =
            (batch_size * result.step_times.len()) as f64 / result.step_times.iter().sum::<f64>();
        let bytes_per_step = weight_bytes as f64 + batch_size as f64 * kv_read_bytes as f64;
        let mbu = peak_bandwidth.map(|peak| 100.0 * bytes_per_step / tpot / peak);
        let mfu =
            peak_flops.map(|peak| 100.0 * batch_size as f64 * decode_flops as f64 / tpot / peak);
        let vram_gib = result
            .peak_vram_bytes
            .map(|bytes| bytes as f64 / 2f64.powi(30));
        println!(
            "{batch_size:>6} {:>9.1} {:>9.2} {:>10.0} {:>7} {:>7} {:>9} {:>6}{}",
            result.ttft_sec * 1e3,
            tpot * 1e3,
            tok_per_sec,
            fmt_opt(mbu, 1),
            fmt_opt(mfu, 2),
            vram_gib
                .map(|value| format!("{value:.2}"))
                .unwrap_or_else(|| "n/a".to_owned()),
            result.step_times.len(),
            if result.step_times.len() + 1 == decode_tokens {
                ""
            } else {
                " (early stop)"
            }
        );
        sweep.push(json!({
            "batch_size": batch_size,
            "ttft_sec": round(result.ttft_sec, 6),
            "tpot_sec": round(tpot, 6),
            "tok_per_sec": round(tok_per_sec, 1),
            "mbu_percent": mbu.map(|value| round(value, 2)),
            "mfu_percent": mfu.map(|value| round(value, 4)),
            "peak_vram_bytes": result.peak_vram_bytes,
            "peak_vram_source": "nvidia-smi per-process memory sampled every 500ms",
            "decode_steps": result.step_times.len(),
        }));
    }
    payload["sweep"] = json!(sweep);
    println!(
        "{}\n{}",
        "-".repeat(header.len()),
        serde_json::to_string(&payload)?
    );
    Ok(payload)
}

// Engine owns the model to keep generation APIs compact; expose benchmark cost
// accounting through the model stored behind a narrow accessor.
fn engine_model(engine: &Engine) -> &GptModel {
    engine.model_ref()
}

fn benchmark_generation(
    engine: &Engine,
    prompt: &[u32],
    batch: usize,
    tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
) -> Result<Measurement> {
    let monitor = GpuMemoryMonitor::start();
    let started = Instant::now();
    let mut previous = started;
    let mut ttft = None;
    let mut step_times = Vec::new();
    engine.generate_batch_with(prompt, batch, tokens, temperature, top_k, 42, |step, _| {
        let now = Instant::now();
        if step == 0 {
            ttft = Some(now.duration_since(started).as_secs_f64());
        } else {
            step_times.push(now.duration_since(previous).as_secs_f64());
        }
        previous = now;
        true
    })?;
    Ok(Measurement {
        ttft_sec: ttft.unwrap_or(0.0),
        step_times,
        peak_vram_bytes: monitor.finish(),
    })
}

struct GpuMemoryMonitor {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    worker: Option<thread::JoinHandle<()>>,
}

impl GpuMemoryMonitor {
    fn start() -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let worker_stop = Arc::clone(&stop);
        let worker_peak = Arc::clone(&peak);
        let process_id = std::process::id();
        let worker = thread::Builder::new()
            .name("rust-agent-gpu-memory-monitor".to_owned())
            .spawn(move || {
                while !worker_stop.load(Ordering::Relaxed) {
                    if let Some(bytes) = process_gpu_memory(process_id) {
                        worker_peak.fetch_max(bytes, Ordering::Relaxed);
                    }
                    thread::sleep(Duration::from_millis(500));
                }
            })
            .ok();
        Self { stop, peak, worker }
    }

    fn finish(mut self) -> Option<u64> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        match self.peak.load(Ordering::Relaxed) {
            0 => None,
            bytes => Some(bytes),
        }
    }
}

impl Drop for GpuMemoryMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

fn process_gpu_memory(process_id: u32) -> Option<u64> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()?
        .lines()
        .filter_map(|line| {
            let (pid, memory) = line.split_once(',')?;
            if pid.trim().parse::<u32>().ok()? != process_id {
                return None;
            }
            memory
                .trim()
                .parse::<u64>()
                .ok()
                .map(|mib| mib * 1024 * 1024)
        })
        .max()
}

fn build_prompt(tokenizer: &BpeTokenizer, num_tokens: usize) -> Result<Vec<u32>> {
    const PARAGRAPH: &str = "The history of science is the study of the development of science, including both the natural and social sciences. Science is a body of empirical, theoretical, and practical knowledge about the natural world. ";
    let text = PARAGRAPH.repeat(num_tokens / 10 + 2);
    let mut tokens = vec![tokenizer.bos_id()];
    tokens.extend(tokenizer.encode(&text)?);
    ensure!(
        tokens.len() >= num_tokens,
        "prompt text too short; increase repetition"
    );
    tokens.truncate(num_tokens);
    Ok(tokens)
}

fn parse_batch_sizes(value: &str) -> Result<Vec<usize>> {
    let values = value
        .split(',')
        .map(|part| part.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        !values.is_empty() && values.iter().all(|size| *size > 0),
        "batch sizes must be positive integers"
    );
    Ok(values)
}

fn gpu_selector(local_rank: usize) -> String {
    std::env::var("CUDA_VISIBLE_DEVICES")
        .ok()
        .and_then(|devices| {
            devices
                .split(',')
                .nth(local_rank)
                .map(str::trim)
                .map(str::to_owned)
        })
        .filter(|device| !device.is_empty())
        .unwrap_or_else(|| local_rank.to_string())
}

fn query_gpu(selector: String) -> Result<GpuInfo> {
    let output = Command::new("nvidia-smi")
        .args([
            "-i",
            &selector,
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .context("run nvidia-smi to read GPU details")?;
    ensure!(
        output.status.success(),
        "nvidia-smi did not return GPU details"
    );
    let line = String::from_utf8(output.stdout)?;
    let (name, memory) = line
        .trim()
        .split_once(',')
        .context("unexpected nvidia-smi GPU response")?;
    let total_memory_bytes = memory.trim().parse::<u64>()? * 1024 * 1024;
    Ok(GpuInfo {
        name: name.trim().to_owned(),
        total_memory_bytes,
    })
}

fn default_tokenizer_path(base_dir: Option<&Path>) -> Result<PathBuf> {
    let base_dir = if let Some(path) = base_dir {
        path.to_owned()
    } else if let Some(path) = crate::paths::base_dir_override() {
        PathBuf::from(path)
    } else {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .context("HOME or USERPROFILE must be set to locate tokenizer")?;
        PathBuf::from(home).join(".cache").join("nanochat")
    };
    Ok(crate::paths::cached_tokenizer_path(&base_dir))
}

fn peak_flops(name: &str) -> Option<f64> {
    let name = name.to_ascii_lowercase();
    let patterns: &[(&[&str], f64)] = &[
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
    patterns
        .iter()
        .find(|(all, _)| all.iter().all(|pattern| name.contains(pattern)))
        .map(|(_, peak)| *peak)
}

fn peak_bandwidth(name: &str) -> Option<f64> {
    let name = name.to_ascii_lowercase();
    let patterns: &[(&[&str], f64)] = &[
        (&["gb200"], 8.0e12),
        (&["grace blackwell"], 8.0e12),
        (&["b200"], 8.0e12),
        (&["b100"], 8.0e12),
        (&["h200"], 4.8e12),
        (&["h100", "nvl"], 3.9e12),
        (&["h100", "pcie"], 2.0e12),
        (&["h100"], 3.35e12),
        (&["h800", "pcie"], 2.0e12),
        (&["h800"], 3.35e12),
        (&["a100"], 2.0e12),
        (&["a800"], 2.0e12),
        (&["a40"], 696e9),
        (&["a30"], 933e9),
        (&["l40s"], 864e9),
        (&["l40-s"], 864e9),
        (&["l40 s"], 864e9),
        (&["l4"], 300e9),
        (&["mi355"], 8.0e12),
        (&["mi325"], 6.0e12),
        (&["mi300x"], 5.3e12),
        (&["mi300a"], 5.3e12),
        (&["mi250x"], 3.28e12),
        (&["mi250"], 3.28e12),
        (&["5090"], 1.79e12),
        (&["4090"], 1.01e12),
        (&["3090"], 936e9),
    ];
    patterns
        .iter()
        .find(|(all, _)| all.iter().all(|pattern| name.contains(pattern)))
        .map(|(_, peak)| *peak)
}

fn format_dtype(dtype: DType) -> String {
    match dtype {
        DType::F32 => "float32",
        DType::F16 => "float16",
        DType::BF16 => "bfloat16",
        _ => "unknown",
    }
    .to_owned()
}

fn round(value: f64, digits: i32) -> f64 {
    let factor = 10f64.powi(digits);
    (value * factor).round() / factor
}

fn format_peak(value: Option<f64>, unit: &str) -> String {
    value
        .map(|value| format!("{:.2} {unit}", value / 1e12))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn fmt_opt(value: Option<f64>, digits: usize) -> String {
    value
        .map(|value| format!("{value:.digits$}"))
        .unwrap_or_else(|| "n/a".to_owned())
}
