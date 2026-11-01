//! Unified base-model evaluation migrated from `scripts/base_eval.py`.

use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Device};
use serde_json::{Value, json};

use crate::{
    checkpoint, core_eval,
    datasets::{self, Dataset},
    engine::Engine,
    evaluation,
    model::GptModel,
    model_config::GptConfig,
    tokenizer::BpeTokenizer,
};

pub struct BaseEvalOptions {
    pub eval: String,
    pub model_tag: Option<String>,
    pub step: Option<u64>,
    pub base_dir: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub max_per_task: Option<usize>,
    pub device_batch_size: usize,
    pub dataset: String,
    pub split_tokens: usize,
}

pub fn run(options: &BaseEvalOptions, device: &Device, dtype: DType) -> Result<Value> {
    let master = std::env::var("RANK")
        .ok()
        .and_then(|rank| rank.parse::<usize>().ok())
        .unwrap_or(0)
        == 0;
    let modes = options
        .eval
        .split(',')
        .map(str::trim)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    ensure!(!modes.is_empty(), "select at least one evaluation mode");
    for mode in &modes {
        if !matches!(*mode, "core" | "bpb" | "sample") {
            bail!("unknown base evaluation mode {mode:?}; choose core,bpb,sample");
        }
    }
    ensure!(
        options.device_batch_size > 0,
        "device batch size must be positive"
    );

    let base_dir = resolve_base_dir(options.base_dir.as_deref())?;
    let selection = checkpoint::select_checkpoint(
        "base",
        Some(&base_dir),
        options.model_tag.as_deref(),
        options.step,
    )?;
    let metadata_path = selection
        .metadata_file
        .as_ref()
        .context("selected checkpoint has no metadata JSON")?;
    let metadata = crate::json_compat::parse_value(&fs::read(metadata_path)?)?;
    let config_value = metadata
        .get("model_config")
        .cloned()
        .context("checkpoint metadata has no model_config")?;
    let config = GptConfig::from_value(config_value)?;
    config.validate()?;
    let tokenizer_path = options
        .tokenizer
        .clone()
        .unwrap_or_else(|| crate::paths::cached_tokenizer_path(&base_dir));
    let tokenizer = BpeTokenizer::load(&tokenizer_path)
        .with_context(|| format!("load tokenizer {}", tokenizer_path.display()))?;
    ensure!(
        tokenizer.vocab_size() as usize == config.vocab_size,
        "tokenizer vocabulary does not match checkpoint"
    );
    let model = GptModel::load_checkpoint(config.clone(), &selection.model_file, device, dtype)?;
    let engine = Engine::new(model, tokenizer);
    let mut result = json!({
        "model_tag": selection.model_tag,
        "step": selection.step,
        "eval": modes,
    });

    if modes.contains(&"sample") && master {
        let prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ];
        let mut conditioned = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            let mut tokens = vec![engine.tokenizer.bos_id()];
            tokens.extend(engine.tokenizer.encode(prompt)?);
            let (sequences, _) = engine.generate_batch(&tokens, 1, 16, 0.0, None, 42)?;
            let sample = engine.tokenizer.decode(&sequences[0])?;
            println!("{sample}");
            conditioned.push(sample);
        }
        let prompt = [engine.tokenizer.bos_id()];
        let (sequences, _) = engine.generate_batch(&prompt, 8, 128, 1.0, None, 42)?;
        let unconditioned = sequences
            .iter()
            .map(|sequence| engine.tokenizer.decode(sequence))
            .collect::<Result<Vec<_>>>()?;
        for sample in &unconditioned {
            println!("{sample}");
        }
        result["samples"] = json!({"conditioned": conditioned, "unconditioned": unconditioned});
    }

    if modes.contains(&"bpb") {
        let dataset = Dataset::parse(&options.dataset)?;
        let directory = dataset.local_cache_dir_in(&base_dir);
        let files = datasets::local_files(&directory).with_context(|| format!("load cached dataset {}; download its shards first with `rust-agent download-dataset --dataset {}`", directory.display(), options.dataset))?;
        let world_size = std::env::var("WORLD_SIZE")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(1)
            .max(1);
        let tokens_per_step = options
            .device_batch_size
            .checked_mul(config.sequence_len)
            .and_then(|tokens| tokens.checked_mul(world_size))
            .context("evaluation batch size overflow")?;
        let split_tokens = options.split_tokens / tokens_per_step * tokens_per_step;
        ensure!(
            split_tokens >= tokens_per_step,
            "split token budget must cover one evaluation batch"
        );
        let mut bpb_results = serde_json::Map::new();
        for (name, validation) in [("train", false), ("val", true)] {
            let score = evaluation::evaluate_parquet_bpb(
                &engine.model_ref(),
                &engine.tokenizer,
                files.clone(),
                validation,
                config.sequence_len,
                options.device_batch_size,
                split_tokens,
                device,
            )?;
            println!("{name} bpb: {score:.6}");
            bpb_results.insert(name.to_owned(), json!(score));
        }
        result["bpb"] = Value::Object(bpb_results);
    }

    if modes.contains(&"core") {
        result["core"] = core_eval::evaluate_core(&engine, device, options.max_per_task)?;
    }
    if let Some(step) = selection.step.filter(|_| master) {
        let output_dir = base_dir.join("base_eval");
        fs::create_dir_all(&output_dir)?;
        let output_path = output_dir.join(format!("base_model_{step:06}.json"));
        fs::write(&output_path, serde_json::to_vec_pretty(&result)?)?;
        println!("Wrote base evaluation results to {}", output_path.display());
        if let Some(core) = result.get("core") {
            let mut csv = String::from("Task, Accuracy, Centered\n");
            if let Some(results) = core.get("results").and_then(Value::as_object) {
                for (label, accuracy) in results {
                    let centered = core["centered_results"][label].as_f64().unwrap_or_default();
                    csv.push_str(&format!(
                        "{label}, {:.6}, {centered:.6}\n",
                        accuracy.as_f64().unwrap_or_default()
                    ));
                }
                csv.push_str(&format!(
                    "CORE, , {:.6}\n",
                    core["core_metric"].as_f64().unwrap_or_default()
                ));
                fs::write(output_dir.join(format!("base_model_{step:06}.csv")), csv)?;
            }
        }
    }
    Ok(result)
}

fn resolve_base_dir(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(path) = explicit {
        return Ok(path.to_owned());
    }
    if let Some(path) = crate::paths::base_dir_override() {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE is required to locate checkpoints")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}
