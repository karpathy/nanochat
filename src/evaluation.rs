//! Tokenizer-invariant bits-per-byte evaluation.

use anyhow::{Result, ensure};
use candle_core::{Device, Tensor};
use candle_nn::ops;

use crate::{
    data::{BatchSource, ParquetBatchSource, TokenCorpus},
    model::GptModel,
    tokenizer::BpeTokenizer,
};

pub fn evaluate_bpb(
    model: &GptModel,
    tokenizer: &BpeTokenizer,
    corpus: &TokenCorpus,
    sequence_len: usize,
    device: &Device,
) -> Result<f64> {
    ensure!(
        sequence_len > 1 && sequence_len <= model.config.sequence_len,
        "evaluation sequence length is outside model context"
    );
    let mut total_nats = 0.0f64;
    let mut total_bytes = 0u64;
    let stride = sequence_len - 1;
    for start in (0..corpus.ids.len().saturating_sub(1)).step_by(stride) {
        let available = (corpus.ids.len() - start - 1).min(stride);
        if available == 0 {
            break;
        }
        let input_len = available;
        let input = &corpus.ids[start..start + input_len];
        let targets = &corpus.ids[start + 1..start + input_len + 1];
        let logits = model
            .forward(input, 1, input_len)?
            .narrow(1, 0, input_len)?
            .reshape((input_len, model.config.vocab_size))?;
        let log_probs = ops::log_softmax(&logits, 1)?;
        let target_tensor = Tensor::from_vec(targets.to_vec(), (input_len,), device)?;
        let losses = log_probs
            .gather(&target_tensor.unsqueeze(1)?, 1)?
            .to_vec2::<f32>()?;
        for (index, (&target, loss)) in targets.iter().zip(losses.iter()).enumerate() {
            if corpus.loss_mask[start + index + 1] == 0 {
                continue;
            }
            let bytes = tokenizer.token_byte_len(target)?;
            if bytes > 0 {
                total_nats -= loss[0] as f64;
                total_bytes += bytes as u64;
            }
        }
    }
    ensure!(
        total_bytes > 0,
        "evaluation dataset contains no ordinary text tokens"
    );
    Ok(total_nats / (std::f64::consts::LN_2 * total_bytes as f64))
}

/// Bounded BPB evaluation over a Parquet split using the training best-fit rows.
pub fn evaluate_parquet_bpb(
    model: &GptModel,
    tokenizer: &BpeTokenizer,
    files: Vec<std::path::PathBuf>,
    validation: bool,
    sequence_len: usize,
    batch_size: usize,
    token_budget: usize,
    device: &Device,
) -> Result<f64> {
    evaluate_parquet_bpb_distributed(
        model,
        tokenizer,
        files,
        validation,
        sequence_len,
        batch_size,
        token_budget,
        device,
    )
}

pub fn evaluate_parquet_bpb_distributed(
    model: &GptModel,
    tokenizer: &BpeTokenizer,
    files: Vec<std::path::PathBuf>,
    validation: bool,
    sequence_len: usize,
    batch_size: usize,
    token_budget: usize,
    device: &Device,
) -> Result<f64> {
    let mut distributed = crate::distributed::DistributedContext::from_env()?;
    evaluate_parquet_bpb_with_context(
        model,
        tokenizer,
        files,
        validation,
        sequence_len,
        batch_size,
        token_budget,
        device,
        &mut distributed,
    )
}

pub fn evaluate_parquet_bpb_with_context(
    model: &GptModel,
    tokenizer: &BpeTokenizer,
    files: Vec<std::path::PathBuf>,
    validation: bool,
    sequence_len: usize,
    batch_size: usize,
    token_budget: usize,
    device: &Device,
    distributed: &mut crate::distributed::DistributedContext,
) -> Result<f64> {
    ensure!(
        sequence_len > 1 && sequence_len <= model.config.sequence_len,
        "evaluation sequence length is outside model context"
    );
    ensure!(batch_size > 0, "evaluation batch size must be positive");
    let tokens_per_batch = batch_size * sequence_len;
    let tokens_per_global_step = tokens_per_batch * distributed.world_size();
    let steps = token_budget / tokens_per_global_step;
    ensure!(
        steps > 0,
        "evaluation token budget must cover at least one full batch"
    );
    let mut batches = ParquetBatchSource::new_distributed(
        files,
        tokenizer,
        batch_size,
        sequence_len,
        validation,
        None,
        distributed.rank(),
        distributed.world_size(),
    )?;
    let mut total_nats = 0.0f32;
    let mut total_bytes = 0u64;
    for _ in 0..steps {
        let batch = batches.next_batch()?;
        let logits = model
            .forward(&batch.inputs, batch_size, sequence_len)?
            .reshape((tokens_per_batch, model.config.vocab_size))?;
        let log_probs = ops::log_softmax(&logits, 1)?;
        let targets = Tensor::from_vec(batch.valid_targets.clone(), (tokens_per_batch,), device)?;
        let losses = log_probs
            .gather(&targets.unsqueeze(1)?, 1)?
            .to_vec2::<f32>()?;
        let mut batch_nats = 0.0f32;
        for (target, loss) in batch.valid_targets.iter().zip(losses.iter()) {
            let byte_len = tokenizer.token_byte_len(*target)?;
            if byte_len > 0 {
                batch_nats -= loss[0];
                total_bytes += byte_len as u64;
            }
        }
        total_nats += batch_nats;
    }
    ensure!(
        total_bytes > 0,
        "evaluation split contains no ordinary text tokens"
    );
    let total_nats = distributed.all_reduce_f32_sum(&[total_nats])?[0];
    let (_, total_bytes) = distributed.all_reduce_counts(0, total_bytes)?;
    Ok(total_nats as f64 / (std::f64::consts::LN_2 * total_bytes as f64))
}
