//! Autoregressive sampling and generated-Python tool handling.

use std::collections::VecDeque;

use anyhow::{Result, ensure};
use candle_core::Tensor;
use rand::{Rng, SeedableRng, rngs::StdRng};

use crate::{
    model::{GptModel, KvCache},
    tokenizer::BpeTokenizer,
};

pub struct Engine {
    model: GptModel,
    pub tokenizer: BpeTokenizer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

impl Engine {
    pub fn new(model: GptModel, tokenizer: BpeTokenizer) -> Self {
        Self {
            model: model.with_inference_attention(true),
            tokenizer,
        }
    }

    pub fn model_context(&self) -> usize {
        self.model.config.sequence_len
    }

    pub fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }

    pub fn model_ref(&self) -> &GptModel {
        &self.model
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        batch_size: usize,
        sequence_len: usize,
    ) -> Result<Tensor> {
        self.model.forward(tokens, batch_size, sequence_len)
    }

    /// Run a policy-loss forward pass with attention operations that support
    /// backward propagation. Generation and evaluation use inference kernels.
    pub(crate) fn forward_train(
        &self,
        tokens: &[u32],
        batch_size: usize,
        sequence_len: usize,
    ) -> Result<Tensor> {
        self.model
            .clone()
            .with_inference_attention(false)
            .forward(tokens, batch_size, sequence_len)
    }

    /// Greedy or temperature/top-k sampling with a per-layer incremental KV cache.
    pub fn generate(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::with_capacity(max_tokens);
        self.generate_with(prompt, max_tokens, temperature, top_k, seed, |token| {
            generated.push(token);
            true
        })?;
        Ok(generated)
    }

    pub fn generate_with<F>(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
        mut on_token: F,
    ) -> Result<FinishReason>
    where
        F: FnMut(u32) -> bool,
    {
        self.generate_with_terminal(
            prompt,
            max_tokens,
            temperature,
            top_k,
            seed,
            |token, terminal| {
                if terminal { true } else { on_token(token) }
            },
        )
    }

    /// Like `generate_with`, but also reports the sampled stop token before
    /// returning. The interactive reference CLI decodes these special tokens;
    /// API callers use `generate_with` and keep them out of completion text.
    pub fn generate_with_terminal<F>(
        &self,
        prompt: &[u32],
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
        mut on_token: F,
    ) -> Result<FinishReason>
    where
        F: FnMut(u32, bool) -> bool,
    {
        ensure!(!prompt.is_empty(), "prompt must contain at least one token");
        ensure!(temperature >= 0.0, "temperature must be non-negative");
        let mut rng = StdRng::seed_from_u64(seed);
        let mut torch_cpu_rng = TorchCpuRng::new(seed);
        let mut torch_cuda_rng = TorchCudaRng::new(seed, self.model.device().is_cuda());
        let mut torch_mps_rng = TorchMpsRng::new(seed);
        let use_torch_cpu_sampling = self.model.device().is_cpu();
        let use_torch_cuda_sampling = self.model.device().is_cuda();
        let use_torch_mps_sampling = self.model.device().is_metal();
        let end_token = self.tokenizer.special_id("<|assistant_end|>")?;
        let bos_token = self.tokenizer.bos_id();
        let mut cache = KvCache::new(&self.model.config);
        let mut logits =
            self.model
                .forward_with_cache(prompt, 1, prompt.len(), Some(&mut cache))?;
        let python_start = self.tokenizer.special_id("<|python_start|>")?;
        let python_end = self.tokenizer.special_id("<|python_end|>")?;
        let output_start = self.tokenizer.special_id("<|output_start|>")?;
        let output_end = self.tokenizer.special_id("<|output_end|>")?;
        let mut forced_tokens = VecDeque::new();
        let mut python_code = Vec::new();
        let mut in_python_block = false;
        let device_greedy = temperature == 0.0 && !use_torch_cpu_sampling;
        let greedy_indices = if device_greedy {
            Some(
                candle_core::Tensor::arange(
                    0u32,
                    self.model.config.vocab_size as u32,
                    self.model.device(),
                )?
                .reshape((1, 1, self.model.config.vocab_size))?,
            )
        } else {
            None
        };
        let greedy_sentinel = if device_greedy {
            Some(candle_core::Tensor::full(
                u32::MAX,
                (1, 1, self.model.config.vocab_size),
                self.model.device(),
            )?)
        } else {
            None
        };
        for _ in 0..max_tokens {
            // nanochat samples before substituting a queued calculator result,
            // so forced tool tokens still advance the sampling RNG.
            let sampled = if device_greedy {
                let last_position = logits.dim(1)? - 1;
                let last_logits = logits.narrow(1, last_position, 1)?;
                let max_logit = last_logits.max_keepdim(2)?;
                let maxima = last_logits.eq(&max_logit.broadcast_as(last_logits.shape())?)?;
                let indices = greedy_indices
                    .as_ref()
                    .expect("temperature-zero indices initialized");
                let sentinel = greedy_sentinel
                    .as_ref()
                    .expect("temperature-zero sentinel initialized");
                maxima
                    .where_cond(&indices.broadcast_as(last_logits.shape())?, sentinel)?
                    .min(2)?
                    .flatten_all()?
                    .to_vec1::<u32>()?[0]
            } else if use_torch_cpu_sampling {
                let rows = logits.to_vec3::<f32>()?;
                sample_torch_cpu(
                    rows[0].last().expect("non-empty context"),
                    temperature,
                    top_k,
                    &mut torch_cpu_rng,
                )
            } else {
                let rows = logits.to_vec3::<f32>()?;
                let last = rows[0].last().expect("non-empty context");
                if use_torch_cuda_sampling {
                    sample_torch_cuda(last, temperature, top_k, &mut torch_cuda_rng, 0, 1)
                } else if use_torch_mps_sampling {
                    sample_torch_mps(&[last], temperature, top_k, &mut torch_mps_rng)[0]
                } else {
                    sample(last, temperature, top_k, &mut rng)
                }
            };
            let token = forced_tokens.pop_front().unwrap_or(sampled);
            if is_terminal_token(token, end_token, bos_token) {
                let _ = on_token(token, true);
                return Ok(FinishReason::Stop);
            }
            if !on_token(token, false) {
                return Ok(FinishReason::Length);
            }
            if token == python_start {
                in_python_block = true;
                python_code.clear();
            } else if token == python_end && in_python_block {
                in_python_block = false;
                let code = self.tokenizer.decode(&python_code)?;
                if let Some(output) = evaluate_calculator(&code) {
                    forced_tokens.push_back(output_start);
                    forced_tokens.extend(self.tokenizer.encode(&output)?);
                    forced_tokens.push_back(output_end);
                }
            } else if in_python_block {
                python_code.push(token);
            }
            if cache.position() >= self.model.config.sequence_len {
                return Ok(FinishReason::Length);
            }
            logits = self
                .model
                .forward_with_cache(&[token], 1, 1, Some(&mut cache))?;
        }
        Ok(FinishReason::Length)
    }

    /// Generate raw completion tokens for several independent rows using a
    /// shared prompt prefill and a replicated KV cache. The callback receives
    /// each sampled token column, including stop tokens.
    pub fn generate_batch_with<F>(
        &self,
        prompt: &[u32],
        num_samples: usize,
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
        mut on_step: F,
    ) -> Result<usize>
    where
        F: FnMut(usize, &[u32]) -> bool,
    {
        self.generate_batch_with_masks(
            prompt,
            num_samples,
            max_tokens,
            temperature,
            top_k,
            seed,
            |step, tokens, _| on_step(step, tokens),
        )
    }

    pub fn generate_batch_with_masks<F>(
        &self,
        prompt: &[u32],
        num_samples: usize,
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
        mut on_step: F,
    ) -> Result<usize>
    where
        F: FnMut(usize, &[u32], &[u8]) -> bool,
    {
        ensure!(!prompt.is_empty(), "prompt must contain at least one token");
        ensure!(num_samples > 0, "num_samples must be positive");
        ensure!(
            prompt.len() <= self.model_context(),
            "prompt exceeds model context"
        );
        ensure!(temperature >= 0.0, "temperature must be non-negative");
        let end_token = self.tokenizer.special_id("<|assistant_end|>")?;
        let bos_token = self.tokenizer.bos_id();
        let python_start = self.tokenizer.special_id("<|python_start|>")?;
        let python_end = self.tokenizer.special_id("<|python_end|>")?;
        let output_start = self.tokenizer.special_id("<|output_start|>")?;
        let output_end = self.tokenizer.special_id("<|output_end|>")?;
        let mut cache = KvCache::new(&self.model.config);
        let logits = self
            .model
            .forward_with_cache(prompt, 1, prompt.len(), Some(&mut cache))?;
        cache.replicate_batch(num_samples)?;
        let last = logits
            .narrow(1, prompt.len() - 1, 1)?
            .reshape((1, self.model.config.vocab_size))?
            .to_vec2::<f32>()?
            .remove(0);
        let use_torch_cpu_sampling = self.model.device().is_cpu();
        let use_torch_cuda_sampling = self.model.device().is_cuda();
        let use_torch_mps_sampling = self.model.device().is_metal();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut torch_cpu_rng = TorchCpuRng::new(seed);
        let mut torch_cuda_rng = TorchCudaRng::new(seed, self.model.device().is_cuda());
        let mut torch_mps_rng = TorchMpsRng::new(seed);
        let mut states = (0..num_samples)
            .map(|_| BatchGenerationState::default())
            .collect::<Vec<_>>();
        let mut tokens = vec![0u32; num_samples];
        let mut masks = vec![0u8; num_samples];
        let mut rows = vec![last; num_samples];
        let mut steps = 0;
        for step in 0..max_tokens {
            steps = step + 1;
            let sampled_tokens = if use_torch_mps_sampling {
                let row_slices = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
                sample_torch_mps(&row_slices, temperature, top_k, &mut torch_mps_rng)
            } else {
                (0..num_samples)
                    .map(|row| {
                        if use_torch_cpu_sampling {
                            sample_torch_cpu(&rows[row], temperature, top_k, &mut torch_cpu_rng)
                        } else if use_torch_cuda_sampling {
                            sample_torch_cuda(
                                &rows[row],
                                temperature,
                                top_k,
                                &mut torch_cuda_rng,
                                row,
                                num_samples,
                            )
                        } else {
                            sample(&rows[row], temperature, top_k, &mut rng)
                        }
                    })
                    .collect::<Vec<_>>()
            };
            for row in 0..num_samples {
                let sampled = sampled_tokens[row];
                if let Some(forced) = states[row].forced_tokens.pop_front() {
                    tokens[row] = forced;
                    masks[row] = 0;
                } else {
                    tokens[row] = sampled;
                    masks[row] = 1;
                }
                let token = tokens[row];
                states[row].completed |= is_terminal_token(token, end_token, bos_token);
                if token == python_start {
                    states[row].in_python_block = true;
                    states[row].python_code.clear();
                } else if token == python_end && states[row].in_python_block {
                    states[row].in_python_block = false;
                    let expression = self.tokenizer.decode(&states[row].python_code)?;
                    if let Some(result) = evaluate_calculator(&expression) {
                        states[row].forced_tokens.push_back(output_start);
                        states[row]
                            .forced_tokens
                            .extend(self.tokenizer.encode(&result)?);
                        states[row].forced_tokens.push_back(output_end);
                    }
                } else if states[row].in_python_block {
                    states[row].python_code.push(token);
                }
            }
            if !on_step(step, &tokens, &masks) {
                return Ok(step + 1);
            }
            if states.iter().all(|state| state.completed)
                || step + 1 == max_tokens
                || cache.position() >= self.model_context()
            {
                break;
            }
            let logits =
                self.model
                    .forward_with_cache(&tokens, num_samples, 1, Some(&mut cache))?;
            rows = logits
                .reshape((num_samples, self.model.config.vocab_size))?
                .to_vec2::<f32>()?;
        }
        Ok(steps)
    }

    pub fn generate_batch(
        &self,
        prompt: &[u32],
        num_samples: usize,
        max_tokens: usize,
        temperature: f64,
        top_k: Option<usize>,
        seed: u64,
    ) -> Result<(Vec<Vec<u32>>, Vec<Vec<u8>>)> {
        let mut sequences = vec![prompt.to_vec(); num_samples];
        let mut masks = vec![vec![0u8; prompt.len()]; num_samples];
        let end_token = self.tokenizer.special_id("<|assistant_end|>")?;
        let bos_token = self.tokenizer.bos_id();
        let mut completed = vec![false; num_samples];
        self.generate_batch_with_masks(
            prompt,
            num_samples,
            max_tokens,
            temperature,
            top_k,
            seed,
            |_, tokens, token_masks| {
                for row in 0..num_samples {
                    if completed[row] {
                        continue;
                    }
                    if is_terminal_token(tokens[row], end_token, bos_token) {
                        completed[row] = true;
                        continue;
                    }
                    sequences[row].push(tokens[row]);
                    masks[row].push(token_masks[row]);
                }
                !completed.iter().all(|done| *done)
            },
        )?;
        Ok((sequences, masks))
    }
}

#[derive(Default)]
struct BatchGenerationState {
    completed: bool,
    forced_tokens: VecDeque<u32>,
    in_python_block: bool,
    python_code: Vec<u32>,
}

fn sample(logits: &[f32], temperature: f64, top_k: Option<usize>, rng: &mut StdRng) -> u32 {
    if temperature == 0.0 {
        return argmax(logits) as u32;
    }
    let mut candidates: Vec<(usize, f64)> = logits
        .iter()
        .enumerate()
        .map(|(index, value)| (index, *value as f64 / temperature))
        .collect();
    if let Some(k) = top_k.filter(|&k| k > 0) {
        // Match torch.topk's sorted candidate order before torch.multinomial.
        // The order does not change the distribution, but it does change how a
        // seeded categorical draw maps onto token IDs.
        let count = k.min(candidates.len());
        if count < candidates.len() {
            candidates.select_nth_unstable_by(count - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(count);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit)
        .fold(f64::NEG_INFINITY, f64::max);
    let mut total = 0.0;
    for (_, logit) in &mut candidates {
        *logit = (*logit - max_logit).exp();
        total += *logit;
    }
    let mut target = rng.random::<f64>() * total;
    for (index, weight) in candidates {
        target -= weight;
        if target <= 0.0 {
            return index as u32;
        }
    }
    logits.len().saturating_sub(1) as u32
}

fn sample_torch_cpu(
    logits: &[f32],
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut TorchCpuRng,
) -> u32 {
    if temperature == 0.0 {
        return argmax(logits) as u32;
    }
    let mut candidates = logits
        .iter()
        .enumerate()
        .map(|(index, value)| (index, *value))
        .collect::<Vec<_>>();
    if let Some(k) = top_k.filter(|&k| k > 0) {
        let count = k.min(candidates.len());
        if count < candidates.len() {
            candidates.select_nth_unstable_by(count - 1, |a, b| b.1.total_cmp(&a.1));
            candidates.truncate(count);
        }
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    }
    let temperature = temperature as f32;
    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = candidates
        .iter()
        .map(|(_, logit)| (*logit / temperature - max_logit).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().copied().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= total;
    }

    // CPU torch.multinomial(..., 1) draws one exponential variate per category
    // and returns argmax(probability / variate).
    let mut best = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (position, probability) in probabilities.into_iter().enumerate() {
        let uniform = rng.uniform_f64();
        let exponential = (-(-uniform).ln_1p()) as f32;
        let score = probability / exponential;
        if score > best_score {
            best = position;
            best_score = score;
        }
    }
    candidates[best].0 as u32
}

fn sample_torch_cuda(
    logits: &[f32],
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut TorchCudaRng,
    row: usize,
    rows: usize,
) -> u32 {
    if temperature == 0.0 {
        return argmax(logits) as u32;
    }
    let temperature = temperature as f32;
    let mut candidates = logits.iter().copied().enumerate().collect::<Vec<_>>();
    if let Some(k) = top_k.filter(|&k| k > 0) {
        let count = k.min(candidates.len());
        candidates = cuda_topk_candidates(logits, count);
    }
    let max_logit = candidates
        .iter()
        .map(|(_, logit)| *logit / temperature)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probabilities = candidates
        .iter()
        .map(|(_, logit)| (*logit / temperature - max_logit).exp())
        .collect::<Vec<_>>();
    let total = probabilities.iter().copied().sum::<f32>();
    for probability in &mut probabilities {
        *probability /= total;
    }

    // PyTorch's default torch.multinomial path is without replacement. For
    // one sample it uses the Gumbel trick: argmax(p / q), q ~ Exp(1).
    // CUDA draws one Philox value per candidate for exponential_(). When
    // top-k is active, PyTorch samples the compact top-k tensor.
    let candidate_count = candidates.len();
    let total_random_values = rows * candidate_count;
    let mut best_token = 0;
    let mut best_score = f32::NEG_INFINITY;
    for (token, probability) in probabilities.into_iter().enumerate() {
        let random_index = row * candidate_count + token;
        let uniform = rng.uniform(random_index, total_random_values);
        let log_uniform = if uniform >= 1.0 - f32::EPSILON / 2.0 {
            -f32::EPSILON / 2.0
        } else {
            uniform.ln()
        };
        let exponential = -log_uniform;
        let score = probability / exponential;
        if score > best_score {
            best_token = candidates[token].0;
            best_score = score;
        }
    }
    best_token as u32
}

// PyTorch gathers values above the cutoff in input order, then fills the
// remaining slots with cutoff ties. Its CUDA small-sort path uses a 32-entry
// bitonic network, whose equal-value swaps affect seeded multinomial draws.
fn cuda_topk_candidates(logits: &[f32], count: usize) -> Vec<(usize, f32)> {
    let mut sorted = logits.iter().copied().enumerate().collect::<Vec<_>>();
    sorted.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    if count == 0 {
        return Vec::new();
    }
    if count > 32 {
        sorted.truncate(count);
        return sorted;
    }
    let cutoff = sorted[count - 1].1;
    if logits.iter().any(|value| value.is_nan()) {
        sorted.truncate(count);
        return sorted;
    }
    let mut gathered = logits
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| *value > cutoff)
        .collect::<Vec<_>>();
    let remaining = count - gathered.len();
    gathered.extend(
        logits
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, value)| *value == cutoff)
            .take(remaining),
    );
    let mut slots = gathered.into_iter().map(Some).collect::<Vec<_>>();
    slots.resize(32, None);
    let mut size = 2;
    while size < 32 {
        let mut stride = size / 2;
        while stride > 0 {
            for thread in 0..16 {
                let pos = 2 * thread - (thread & (stride - 1));
                let swap = match (slots[pos], slots[pos + stride]) {
                    (Some(a), Some(b)) => a.1 > b.1,
                    (_, None) => true,
                    (None, Some(_)) => false,
                };
                if swap == (thread & (size / 2) != 0) {
                    slots.swap(pos, pos + stride);
                }
            }
            stride /= 2;
        }
        size *= 2;
    }
    let mut stride = 16;
    while stride > 0 {
        for thread in 0..16 {
            let pos = 2 * thread - (thread & (stride - 1));
            let swap = match (slots[pos], slots[pos + stride]) {
                (Some(a), Some(b)) => a.1 > b.1,
                (_, None) => true,
                (None, Some(_)) => false,
            };
            if !swap {
                slots.swap(pos, pos + stride);
            }
        }
        stride /= 2;
    }
    slots.into_iter().take(count).map(Option::unwrap).collect()
}

#[cfg(test)]
mod sampling_tests {
    use super::{TorchCudaRng, cuda_topk_candidates, is_terminal_token, sample_torch_cuda};

    #[test]
    fn cuda_topk_ties_match_pytorch_fixture() {
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/cuda-topk-ties.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let size = case["size"].as_u64().unwrap() as usize;
            let k = case["k"].as_u64().unwrap() as usize;
            let logits = (0..size)
                .map(|i| {
                    if case["pattern"] == "mixed" {
                        ((i * 37 + i / 7) % 11) as f32
                    } else {
                        ((size - i - 1) / 3) as f32
                    }
                })
                .collect::<Vec<_>>();
            let actual = cuda_topk_candidates(&logits, k)
                .into_iter()
                .map(|(i, _)| i)
                .collect::<Vec<_>>();
            let expected = case["indices"]
                .as_array()
                .unwrap()
                .iter()
                .map(|i| i.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(actual, expected, "size={size}, k={k}");
        }
    }

    #[test]
    fn assistant_end_and_bos_are_the_only_terminal_tokens() {
        assert!(is_terminal_token(260, 260, 261));
        assert!(is_terminal_token(261, 260, 261));
        assert!(!is_terminal_token(259, 260, 261));
        assert!(!is_terminal_token(262, 260, 261));
    }

    #[test]
    fn cuda_multinomial_top_k_matches_pytorch_reference() {
        if candle_core::Device::new_cuda(0).is_err() {
            return;
        }
        let logits = (0..128).map(|value| value as f32).collect::<Vec<_>>();
        let mut rng = TorchCudaRng::new(42, true);
        let actual = (0..32)
            .map(|_| sample_torch_cuda(&logits, 0.8, Some(50), &mut rng, 0, 1))
            .collect::<Vec<_>>();
        // PyTorch 2.9.1 returns these positions in the compact top-k tensor;
        // ascending logits map compact positions 0, 1, and 2 to tokens 127,
        // 126, and 125 respectively.
        let expected = [
            127, 127, 127, 127, 126, 127, 127, 127, 127, 127, 127, 127, 126, 127, 127, 127, 127,
            126, 127, 127, 125, 126, 127, 125, 127, 127, 126, 127, 127, 126, 127, 125,
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn cuda_multinomial_with_tied_top_k_matches_pytorch() {
        if candle_core::Device::new_cuda(0).is_err() {
            return;
        }
        let logits = (0..128)
            .map(|i| ((i * 37 + i / 7) % 11) as f32)
            .collect::<Vec<_>>();
        let mut rng = TorchCudaRng::new(42, true);
        let actual = (0..16)
            .map(|_| sample_torch_cuda(&logits, 0.8, Some(5), &mut rng, 0, 1))
            .collect::<Vec<_>>();
        // PyTorch 2.9.1 CUDA topk + multinomial, seed 42.
        assert_eq!(
            actual,
            [
                29, 29, 29, 29, 21, 37, 21, 45, 21, 45, 53, 21, 53, 37, 29, 21
            ]
        );
    }

    #[test]
    fn cuda_multinomial_matches_real_depth12_top_k_fixture() {
        if candle_core::Device::new_cuda(0).is_err() {
            return;
        }
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cuda-seeded-topk-step9.json"
        ))
        .unwrap();
        let logits = fixture["topk_logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let indices = fixture["topk_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let seed = fixture["seed"].as_u64().unwrap();
        let temperature = fixture["temperature"].as_f64().unwrap();
        let top_k = fixture["top_k"].as_u64().unwrap() as usize;
        let prior_samples = fixture["preceding_samples"].as_array().unwrap().len();
        let mut rng = TorchCudaRng::new(seed, true);
        rng.offset = (prior_samples * 4) as u64;
        let selected_position =
            sample_torch_cuda(&logits, temperature, Some(top_k), &mut rng, 0, 1) as usize;
        assert_eq!(
            selected_position,
            fixture["expected_position"].as_u64().unwrap() as usize
        );
        assert_eq!(
            indices[selected_position],
            fixture["expected_token"].as_u64().unwrap() as u32
        );
    }

    #[cfg(feature = "cuda-base")]
    #[test]
    #[ignore = "requires RUST_AGENT_SEEDED_CHECKPOINT to point to the local depth-12 checkpoint"]
    fn diagnose_cuda_depth12_seeded_decode_logits() {
        use candle_core::{DType, Device};

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/cuda-seeded-topk-step9.json"
        ))
        .unwrap();
        let checkpoint =
            std::env::var("RUST_AGENT_SEEDED_CHECKPOINT").expect("RUST_AGENT_SEEDED_CHECKPOINT");
        let config =
            crate::model_config::GptConfig::from_value(fixture["model_config"].clone()).unwrap();
        let device = Device::new_cuda(0).unwrap();
        let model = crate::model::GptModel::load_checkpoint(
            config.clone(),
            checkpoint,
            &device,
            DType::BF16,
        )
        .unwrap()
        .with_inference_attention(true);
        let mut cache = crate::model::KvCache::new(&config);
        let prompt = fixture["prompt_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as u32)
            .collect::<Vec<_>>();
        let mut logits = model
            .forward_with_cache(&prompt, 1, prompt.len(), Some(&mut cache))
            .unwrap();
        for token in fixture["preceding_samples"].as_array().unwrap() {
            logits = model
                .forward_with_cache(&[token.as_u64().unwrap() as u32], 1, 1, Some(&mut cache))
                .unwrap();
        }
        let logits = logits.to_vec3::<f32>().unwrap().remove(0).remove(0);
        let indices = fixture["topk_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_u64().unwrap() as usize)
            .collect::<Vec<_>>();
        let expected_logits = fixture["topk_logits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let max_error = indices
            .iter()
            .zip(expected_logits)
            .map(|(&index, expected)| (logits[index] - expected).abs())
            .fold(0.0f32, f32::max);
        let rust_topk = cuda_topk_candidates(&logits, indices.len())
            .into_iter()
            .map(|(index, _)| index)
            .collect::<Vec<_>>();
        let topk_mismatches = rust_topk
            .iter()
            .zip(&indices)
            .filter(|(actual, expected)| actual != expected)
            .count();
        let mut rng = TorchCudaRng::new(fixture["seed"].as_u64().unwrap(), true);
        rng.offset = (fixture["preceding_samples"].as_array().unwrap().len() * 4) as u64;
        let rust_token = sample_torch_cuda(
            &logits,
            fixture["temperature"].as_f64().unwrap(),
            Some(fixture["top_k"].as_u64().unwrap() as usize),
            &mut rng,
            0,
            1,
        );
        let python_token = fixture["expected_token"].as_u64().unwrap() as u32;
        eprintln!(
            "depth-12 step-9 cached logits max_error={max_error}; top-k index mismatches={topk_mismatches}; Rust token={rust_token}; Python token={python_token}"
        );
        assert_eq!(rust_token, python_token);
    }
}

/// MPS uses the generic one-sample multinomial path: an exponential random
/// value is generated for every probability and each row takes argmax(p / q).
/// The categorical math and flat batched layout match that path, but this
/// host-side conversion of Philox words to random values is only an
/// approximation of MPSGraph's random op.
fn sample_torch_mps(
    rows: &[&[f32]],
    temperature: f64,
    top_k: Option<usize>,
    rng: &mut TorchMpsRng,
) -> Vec<u32> {
    if temperature == 0.0 {
        return rows.iter().map(|row| argmax(row) as u32).collect();
    }
    if rows.is_empty() {
        return Vec::new();
    }

    let temperature = temperature as f32;
    let mut candidates = Vec::with_capacity(rows.len());
    let mut probabilities = Vec::with_capacity(rows.len());
    for row in rows {
        let mut row_candidates = row
            .iter()
            .enumerate()
            .map(|(index, value)| (index, *value))
            .collect::<Vec<_>>();
        if let Some(k) = top_k.filter(|&k| k > 0) {
            let count = k.min(row_candidates.len());
            if count < row_candidates.len() {
                row_candidates.select_nth_unstable_by(count - 1, |a, b| b.1.total_cmp(&a.1));
                row_candidates.truncate(count);
            }
            row_candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        }
        let max_logit = row_candidates
            .iter()
            .map(|(_, logit)| *logit / temperature)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut row_probabilities = row_candidates
            .iter()
            .map(|(_, logit)| (*logit / temperature - max_logit).exp())
            .collect::<Vec<_>>();
        let total = row_probabilities.iter().copied().sum::<f32>();
        for probability in &mut row_probabilities {
            *probability /= total;
        }
        candidates.push(row_candidates);
        probabilities.push(row_probabilities);
    }

    let categories = probabilities[0].len();
    let numel = rows.len() * categories;
    let base_offset = rng.offset;
    let mut current_thread = usize::MAX;
    let mut random_words = [0u32; 4];
    let mut sampled = Vec::with_capacity(rows.len());
    for row in 0..rows.len() {
        let mut best_position = 0;
        let mut best_score = f32::NEG_INFINITY;
        for position in 0..categories {
            let linear_index = row * categories + position;
            let thread = linear_index / 4;
            if thread != current_thread {
                random_words = philox4x32_mps(rng.seed, base_offset + thread as u64);
                current_thread = thread;
            }
            let uniform =
                (random_words[linear_index % 4] & 0x7fff_ffff) as f32 * 4.656_612_734_2e-10_f32;
            let uniform = uniform.clamp(f32::EPSILON / 2.0, 1.0 - f32::EPSILON);
            let exponential = -(1.0_f32 - uniform).ln();
            let score = probabilities[row][position] / exponential;
            if score > best_score {
                best_score = score;
                best_position = position;
            }
        }
        sampled.push(candidates[row][best_position].0 as u32);
    }
    rng.offset += numel.div_ceil(4) as u64;
    sampled
}

/// PyTorch-compatible Philox4_32 raw stream used by the host-side MPS
/// sampling and initialization paths. The value transform remains an
/// approximation: PyTorch feeds Philox state into MPSGraph, which generates
/// the random tensor on the backend.
pub(crate) struct TorchMpsRng {
    seed: u64,
    offset: u64,
}

impl TorchMpsRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self { seed, offset: 0 }
    }

    pub(crate) fn uniform_values(&mut self, count: usize, low: f32, high: f32) -> Vec<f32> {
        let threads = count.div_ceil(4);
        let mut values = Vec::with_capacity(count);
        for thread in 0..threads {
            let words = philox4x32_mps(self.seed, self.offset + thread as u64);
            for word in words {
                if values.len() == count {
                    break;
                }
                let sample = mps_uniform_f32(word);
                let value = low + (high - low) * sample;
                values.push(if value == high { low } else { value });
            }
        }
        self.offset += threads as u64;
        values
    }

    pub(crate) fn normal_values(&mut self, count: usize, mean: f32, std: f32) -> Vec<f32> {
        let threads = count.div_ceil(4);
        let mut values = Vec::with_capacity(count);
        for thread in 0..threads {
            let words = philox4x32_mps(self.seed, self.offset + thread as u64);
            for pair in [[words[0], words[1]], [words[2], words[3]]] {
                let u1 = mps_uniform_f32(pair[0]).max(f32::EPSILON);
                let u2 = mps_uniform_f32(pair[1]);
                let radius = (-2.0 * u1.ln()).sqrt();
                let angle = std::f32::consts::TAU * u2;
                for sample in [radius * angle.cos(), radius * angle.sin()] {
                    if values.len() == count {
                        break;
                    }
                    values.push(mean + std * sample);
                }
            }
        }
        self.offset += threads as u64;
        values
    }
}

fn mps_uniform_f32(value: u32) -> f32 {
    (value & 0x7fff_ffff) as f32 * 4.656_612_734_2e-10
}

fn philox4x32_mps(seed: u64, index: u64) -> [u32; 4] {
    const M0: u32 = 0xd251_1f53;
    const M1: u32 = 0xcd9e_8d57;
    const W0: u32 = 0x9e37_79b9;
    const W1: u32 = 0xbb67_ae85;
    // PyTorch's MPSGeneratorImpl uses Philox4_32(seed, subsequence=0,
    // offset=0), then advances this sequential offset one 128-bit block at a
    // time. PhiloxRNGEngine stores offset in counter words 0-1 and the seed
    // in low/high key order.
    let mut counter = [index as u32, (index >> 32) as u32, 0, 0];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for round in 0..10 {
        let product0 = M0 as u64 * counter[0] as u64;
        let product1 = M1 as u64 * counter[2] as u64;
        counter = [
            (product1 >> 32) as u32 ^ counter[1] ^ key[0],
            product1 as u32,
            (product0 >> 32) as u32 ^ counter[3] ^ key[1],
            product0 as u32,
        ];
        if round < 9 {
            key[0] = key[0].wrapping_add(W0);
            key[1] = key[1].wrapping_add(W1);
        }
    }
    counter
}

/// CPU generator used by PyTorch's `torch.Generator().manual_seed(seed)`.
pub(crate) struct TorchCpuRng {
    state: [u32; 624],
    index: usize,
}

impl TorchCpuRng {
    pub(crate) fn new(seed: u64) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed as u32;
        for index in 1..state.len() {
            state[index] = 1_812_433_253u32
                .wrapping_mul(state[index - 1] ^ (state[index - 1] >> 30))
                .wrapping_add(index as u32);
        }
        Self { state, index: 624 }
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= self.state.len() {
            for index in 0..self.state.len() {
                let joined = (self.state[index] & 0x8000_0000)
                    | (self.state[(index + 1) % 624] & 0x7fff_ffff);
                let twist = (joined >> 1) ^ if joined & 1 == 0 { 0 } else { 0x9908_b0df };
                self.state[index] = self.state[(index + 397) % 624] ^ twist;
            }
            self.index = 0;
        }
        let mut value = self.state[self.index];
        self.index += 1;
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c_5680;
        value ^= (value << 15) & 0xefc6_0000;
        value ^= value >> 18;
        value
    }

    pub(crate) fn uniform_f32(&mut self) -> f32 {
        let value = self.next_u32() & 0x00ff_ffff;
        value as f32 * (1.0 / 16_777_216.0)
    }

    fn uniform_f64(&mut self) -> f64 {
        let random = ((self.next_u32() as u64) << 32) | self.next_u32() as u64;
        ((random & ((1u64 << 53) - 1)) as f64) / ((1u64 << 53) as f64)
    }
}

/// CUDA Philox stream used by torch.Generator(device="cuda").
pub(crate) struct TorchCudaRng {
    seed: u64,
    offset: u64,
    grid_block_limit: usize,
}

impl TorchCudaRng {
    pub(crate) fn new(seed: u64, on_cuda: bool) -> Self {
        Self {
            seed,
            offset: 0,
            grid_block_limit: if on_cuda {
                torch_cuda_grid_block_limit().unwrap_or(usize::MAX)
            } else {
                usize::MAX
            },
        }
    }

    fn uniform(&mut self, row: usize, rows: usize) -> f32 {
        let block_size = 256usize;
        let grid = rows.div_ceil(block_size).min(self.grid_block_limit).max(1);
        let threads = block_size * grid;
        let unroll = 4usize;
        let iterations = rows.div_ceil(threads * unroll).max(1);
        let thread = row % threads;
        let lane = (row / threads) % unroll;
        let iteration = row / (threads * unroll);
        let counter = self.offset / 4 + iteration as u64;
        let bits = philox4x32_10(self.seed, thread as u64, counter)[lane];
        let value = bits as f32 * (1.0 / 4_294_967_296.0) + (1.0 / 8_589_934_592.0);
        let uniform = if value == 1.0 { 0.0 } else { value };
        if row + 1 == rows {
            self.offset += (iterations * 4) as u64;
        }
        uniform
    }

    pub(crate) fn uniform_f32(&mut self, row: usize, rows: usize) -> f32 {
        self.uniform(row, rows)
    }

    pub(crate) fn normal_f32(&mut self, row: usize, rows: usize) -> f32 {
        let block_size = 256usize;
        let grid = rows.div_ceil(block_size).min(self.grid_block_limit).max(1);
        let threads = block_size * grid;
        let iterations = rows.div_ceil(threads * 4).max(1);
        let thread = row % threads;
        let lane = (row / threads) % 4;
        let iteration = row / (threads * 4);
        let counter = self.offset / 4 + iteration as u64;
        let random = philox4x32_10(self.seed, thread as u64, counter);
        let pair = match lane {
            0 | 1 => box_muller(random[0], random[1]),
            _ => box_muller(random[2], random[3]),
        };
        let value = if lane % 2 == 0 { pair.0 } else { pair.1 };
        if row + 1 == rows {
            self.offset += (iterations * 4) as u64;
        }
        value
    }
}

fn box_muller(first: u32, second: u32) -> (f32, f32) {
    const TWO_POW32_INV: f32 = 1.0 / 4_294_967_296.0;
    let radius_input = first as f32 * TWO_POW32_INV + TWO_POW32_INV / 2.0;
    let angle = second as f32 * (TWO_POW32_INV * std::f32::consts::TAU)
        + (TWO_POW32_INV * std::f32::consts::TAU) / 2.0;
    let radius = (-2.0 * radius_input.ln()).sqrt();
    (radius * angle.sin(), radius * angle.cos())
}

#[cfg(feature = "cuda-base")]
fn torch_cuda_grid_block_limit() -> Option<usize> {
    use candle_core::cuda_backend::cudarc::driver::{CudaContext, sys};

    let ordinal = std::env::var("LOCAL_RANK")
        .ok()
        .and_then(|rank| rank.parse::<usize>().ok())
        .unwrap_or(0);
    let context = CudaContext::new(ordinal).ok()?;
    let multiprocessors = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()? as usize;
    let max_threads_per_sm = context
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .ok()? as usize;
    Some(multiprocessors * (max_threads_per_sm / 256))
}

#[cfg(not(feature = "cuda-base"))]
fn torch_cuda_grid_block_limit() -> Option<usize> {
    None
}

fn philox4x32_10(seed: u64, subsequence: u64, counter: u64) -> [u32; 4] {
    const M0: u32 = 0xd251_1f53;
    const M1: u32 = 0xcd9e_8d57;
    const W0: u32 = 0x9e37_79b9;
    const W1: u32 = 0xbb67_ae85;
    let mut ctr = [
        counter as u32,
        (counter >> 32) as u32,
        subsequence as u32,
        (subsequence >> 32) as u32,
    ];
    let mut key = [seed as u32, (seed >> 32) as u32];
    for _ in 0..10 {
        let product0 = (M0 as u64) * (ctr[0] as u64);
        let product1 = (M1 as u64) * (ctr[2] as u64);
        ctr = [
            (product1 >> 32) as u32 ^ ctr[1] ^ key[0],
            product1 as u32,
            (product0 >> 32) as u32 ^ ctr[3] ^ key[1],
            product0 as u32,
        ];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    ctr
}

fn is_terminal_token(token: u32, assistant_end: u32, bos: u32) -> bool {
    token == assistant_end || token == bos
}

fn argmax(values: &[f32]) -> usize {
    let mut best_index = 0;
    for index in 1..values.len() {
        if values[index].total_cmp(&values[best_index]).is_gt() {
            best_index = index;
        }
    }
    best_index
}

fn evaluate_calculator(expression: &str) -> Option<String> {
    let expression = expression.replace(',', "");
    if let Some((source, needle)) = parse_count_expression(&expression) {
        return Some(source.matches(needle).count().to_string());
    }
    let allowed = expression.chars().all(|character| {
        character.is_ascii_digit()
            || matches!(character, '*' | '+' | '-' | '/' | '.' | '(' | ')' | ' ')
    });
    if !allowed || expression.contains("**") {
        return None;
    }
    let mut parser = ArithmeticParser {
        characters: expression.chars().collect(),
        position: 0,
    };
    let value = parser.expression()?;
    parser.skip_whitespace();
    if parser.position != parser.characters.len() || !value.value.is_finite() {
        return None;
    }
    Some(if value.is_float {
        format!("{:?}", value.value)
    } else {
        format!("{:.0}", value.value)
    })
}

fn parse_count_expression(expression: &str) -> Option<(&str, &str)> {
    let expression = expression.trim();
    let bytes = expression.as_bytes();
    let quote = *bytes.first()?;
    if quote != b'\'' && quote != b'"' {
        return None;
    }
    let source_end = bytes[1..].iter().position(|byte| *byte == quote)? + 1;
    let source = &expression[1..source_end];
    let rest = expression[source_end + 1..].trim_start();
    let rest = rest.strip_prefix(".count(")?.trim_end();
    let rest = rest.strip_suffix(')')?.trim();
    let rest_bytes = rest.as_bytes();
    let needle_quote = *rest_bytes.first()?;
    if needle_quote != b'\'' && needle_quote != b'"' {
        return None;
    }
    if *rest_bytes.last()? != needle_quote {
        return None;
    }
    Some((source, &rest[1..rest.len() - 1]))
}

#[derive(Clone, Copy)]
struct ArithmeticValue {
    value: f64,
    is_float: bool,
}

struct ArithmeticParser {
    characters: Vec<char>,
    position: usize,
}

impl ArithmeticParser {
    fn expression(&mut self) -> Option<ArithmeticValue> {
        let mut left = self.term()?;
        loop {
            self.skip_whitespace();
            let Some(operator) = self.peek() else {
                return Some(left);
            };
            if operator != '+' && operator != '-' {
                return Some(left);
            }
            self.position += 1;
            let right = self.term()?;
            left = ArithmeticValue {
                value: if operator == '+' {
                    left.value + right.value
                } else {
                    left.value - right.value
                },
                is_float: left.is_float || right.is_float,
            };
        }
    }

    fn term(&mut self) -> Option<ArithmeticValue> {
        let mut left = self.factor()?;
        loop {
            self.skip_whitespace();
            let Some(operator) = self.peek() else {
                return Some(left);
            };
            if operator != '*' && operator != '/' {
                return Some(left);
            }
            self.position += 1;
            let right = self.factor()?;
            if operator == '/' && right.value == 0.0 {
                return None;
            }
            left = ArithmeticValue {
                value: if operator == '*' {
                    left.value * right.value
                } else {
                    left.value / right.value
                },
                is_float: left.is_float || right.is_float || operator == '/',
            };
        }
    }

    fn factor(&mut self) -> Option<ArithmeticValue> {
        self.skip_whitespace();
        if self.consume('+') {
            return self.factor();
        }
        if self.consume('-') {
            let mut value = self.factor()?;
            value.value = -value.value;
            return Some(value);
        }
        if self.consume('(') {
            let value = self.expression()?;
            self.skip_whitespace();
            return self.consume(')').then_some(value);
        }
        let start = self.position;
        let mut saw_digit = false;
        let mut saw_dot = false;
        while let Some(character) = self.peek() {
            if character.is_ascii_digit() {
                saw_digit = true;
                self.position += 1;
            } else if character == '.' && !saw_dot {
                saw_dot = true;
                self.position += 1;
            } else {
                break;
            }
        }
        if !saw_digit {
            return None;
        }
        let text = self.characters[start..self.position]
            .iter()
            .collect::<String>();
        Some(ArithmeticValue {
            value: text.parse().ok()?,
            is_float: saw_dot,
        })
    }

    fn skip_whitespace(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.position += 1;
        }
    }

    fn peek(&self) -> Option<char> {
        self.characters.get(self.position).copied()
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek() == Some(expected) {
            self.position += 1;
            true
        } else {
            false
        }
    }
}
