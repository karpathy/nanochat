//! In-context base-model CORE evaluation.

use std::{
    fs::{self, File, OpenOptions},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::{Device, Tensor};
use candle_nn::ops;
use fs2::FileExt;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::Value;
use zip::ZipArchive;

use crate::{distributed::DistributedContext, engine::Engine};

const EVAL_BUNDLE_URL: &str = "https://karpathy-public.s3.us-west-2.amazonaws.com/eval_bundle.zip";

#[derive(Debug, Deserialize)]
struct CoreConfig {
    icl_tasks: Vec<CoreTaskConfig>,
}

#[derive(Debug, Deserialize)]
struct CoreTaskConfig {
    label: String,
    icl_task_type: String,
    dataset_uri: String,
    num_fewshot: Vec<usize>,
    continuation_delimiter: Option<String>,
}

#[derive(Debug)]
enum TaskKind {
    MultipleChoice,
    Schema,
    LanguageModeling,
}

impl TaskKind {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "multiple_choice" => Ok(Self::MultipleChoice),
            "schema" => Ok(Self::Schema),
            "language_modeling" => Ok(Self::LanguageModeling),
            _ => bail!("unsupported CORE task type {value}"),
        }
    }
}

/// Evaluate the CORE benchmark bundle and return raw, centered, and aggregate
/// accuracies, matching `scripts/base_eval.py`.
pub fn evaluate_core(
    engine: &Engine,
    device: &Device,
    max_per_task: Option<usize>,
) -> Result<Value> {
    let mut distributed = DistributedContext::from_env()?;
    evaluate_core_with_context(engine, device, max_per_task, &mut distributed)
}

pub fn evaluate_core_with_context(
    engine: &Engine,
    device: &Device,
    max_per_task: Option<usize>,
    distributed: &mut DistributedContext,
) -> Result<Value> {
    let bundle = ensure_eval_bundle()?;
    let config: CoreConfig = serde_yaml::from_str(
        &fs::read_to_string(bundle.join("core.yaml")).context("read CORE task configuration")?,
    )?;
    let baselines = read_baselines(&bundle.join("eval_meta_data.csv"))?;
    ensure!(
        !config.icl_tasks.is_empty(),
        "CORE config contains no tasks"
    );

    let mut results = serde_json::Map::new();
    let mut centered = serde_json::Map::new();
    let mut centered_sum = 0.0;
    for task in config.icl_tasks {
        let kind = TaskKind::parse(&task.icl_task_type)?;
        let fewshot = *task
            .num_fewshot
            .first()
            .context("CORE task has empty num_fewshot list")?;
        let delimiter = task.continuation_delimiter.as_deref().unwrap_or(" ");
        let dataset_path = safe_bundle_path(&bundle.join("eval_data"), &task.dataset_uri)?;
        let mut data = read_jsonl(&dataset_path)?;
        shuffle(&mut data, 1337);
        if let Some(limit) = max_per_task.filter(|limit| *limit > 0) {
            data.truncate(limit);
        }
        ensure!(!data.is_empty(), "CORE task {} has no examples", task.label);

        let mut correct = 0u64;
        let mut total = 0u64;
        let local_indices = (distributed.rank()..data.len())
            .step_by(distributed.world_size())
            .collect::<Vec<_>>();
        for indices in local_indices.chunks(2) {
            let batch =
                evaluate_examples_batch(engine, device, &data, indices, fewshot, &kind, delimiter)
                    .with_context(|| {
                        format!(
                            "CORE task {} examples {}..{}",
                            task.label,
                            indices[0],
                            indices[indices.len() - 1]
                        )
                    })?;
            correct += batch.iter().filter(|is_correct| **is_correct).count() as u64;
            total += batch.len() as u64;
        }
        let (correct, total) = distributed.all_reduce_counts(correct, total)?;
        ensure!(total > 0, "CORE task {} has no scored examples", task.label);
        let accuracy = correct as f64 / total as f64;
        let random_baseline = baselines
            .get(&task.label)
            .with_context(|| format!("missing random baseline for CORE task {}", task.label))?;
        let centered_score = (accuracy - 0.01 * random_baseline) / (1.0 - 0.01 * random_baseline);
        println!(
            "{} accuracy: {:.4} | centered: {:.4}",
            task.label, accuracy, centered_score
        );
        results.insert(task.label.clone(), Value::from(accuracy));
        centered.insert(task.label, Value::from(centered_score));
        centered_sum += centered_score;
    }
    let core_metric = centered_sum / results.len() as f64;
    println!("CORE metric: {core_metric:.4}");
    Ok(serde_json::json!({
        "results": results,
        "centered_results": centered,
        "core_metric": core_metric
    }))
}

struct PreparedCoreExample {
    prompts: Vec<Vec<u32>>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    gold: usize,
    lm_target: bool,
}

fn prepare_core_example(
    engine: &Engine,
    data: &[Value],
    index: usize,
    fewshot_count: usize,
    kind: &TaskKind,
    delimiter: &str,
) -> Result<PreparedCoreExample> {
    let fewshot = choose_fewshot(data, index, fewshot_count)?;
    let item = &data[index];
    let (prompts, start, end, gold, lm_target) = match kind {
        TaskKind::MultipleChoice => {
            let choices = strings(item.get("choices"), "choices")?;
            let gold = usize_field(item, "gold")?;
            ensure!(gold < choices.len(), "CORE gold index is outside choices");
            let query = string_field(item, "query")?;
            let prompts = choices
                .iter()
                .map(|choice| render_mc(query, delimiter, &fewshot, choice))
                .collect::<Result<Vec<_>>>()?;
            let tokenized = encode_batch(&engine.tokenizer, &prompts)?;
            let start = common_prefix_length(&tokenized);
            let end = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
            (tokenized, vec![start; end.len()], end, gold, false)
        }
        TaskKind::Schema => {
            let contexts = strings(item.get("context_options"), "context_options")?;
            let gold = usize_field(item, "gold")?;
            ensure!(
                gold < contexts.len(),
                "CORE gold index is outside context options"
            );
            let continuation = string_field(item, "continuation")?;
            let prompts = contexts
                .iter()
                .map(|context| render_schema(context, delimiter, &fewshot, continuation))
                .collect::<Result<Vec<_>>>()?;
            let tokenized = encode_batch(&engine.tokenizer, &prompts)?;
            let suffix = common_suffix_length(&tokenized);
            let end = tokenized.iter().map(Vec::len).collect::<Vec<_>>();
            let start = end.iter().map(|end| end - suffix).collect();
            (tokenized, start, end, gold, false)
        }
        TaskKind::LanguageModeling => {
            let context = string_field(item, "context")?.trim();
            let continuation = string_field(item, "continuation")?;
            let prompt_without = render_lm(context, delimiter, &fewshot, "")
                .trim()
                .to_owned();
            let prompt_with = render_lm(context, delimiter, &fewshot, continuation);
            let without = encode(&engine.tokenizer, &prompt_without)?;
            let with = encode(&engine.tokenizer, &prompt_with)?;
            ensure!(
                with.starts_with(&without) && with.len() > without.len(),
                "CORE LM continuation is not a token suffix"
            );
            let start = without.len();
            let end = with.len();
            (vec![with], vec![start], vec![end], 0, true)
        }
    };
    Ok(PreparedCoreExample {
        prompts,
        starts: start,
        ends: end,
        gold,
        lm_target,
    })
}

fn evaluate_examples_batch(
    engine: &Engine,
    device: &Device,
    data: &[Value],
    indices: &[usize],
    fewshot_count: usize,
    kind: &TaskKind,
    delimiter: &str,
) -> Result<Vec<bool>> {
    let mut prompts = Vec::new();
    let mut starts = Vec::new();
    let mut ends = Vec::new();
    let mut groups = Vec::with_capacity(indices.len());
    for &index in indices {
        let prepared = prepare_core_example(engine, data, index, fewshot_count, kind, delimiter)?;
        let begin = prompts.len();
        prompts.extend(prepared.prompts);
        starts.extend(prepared.starts);
        ends.extend(prepared.ends);
        groups.push((begin, prompts.len(), prepared.gold, prepared.lm_target));
    }
    let scored = score_continuations(engine, device, &prompts, &starts, &ends)?;
    Ok(groups
        .into_iter()
        .map(|(begin, end, gold, lm_target)| {
            if lm_target {
                scored[begin].1
            } else {
                let mut prediction = begin;
                for index in begin + 1..end {
                    // Match Python's f32 tie behavior despite tiny GEMM
                    // rounding differences between selected and full logits.
                    let candidate = scored[index].0;
                    let current = scored[prediction].0;
                    let tolerance =
                        f32::EPSILON as f64 * candidate.abs().max(current.abs()).max(1.0);
                    if candidate < current - tolerance {
                        prediction = index;
                    }
                }
                prediction - begin == gold
            }
        })
        .collect())
}

fn score_continuations(
    engine: &Engine,
    device: &Device,
    sequences: &[Vec<u32>],
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<(f64, bool)>> {
    ensure!(
        sequences.len() == starts.len() && starts.len() == ends.len(),
        "CORE prompt boundaries do not match prompt count"
    );
    let context = engine.model_context();
    let mut sequences = sequences.to_vec();
    let mut starts = starts.to_vec();
    let mut ends = ends.to_vec();
    for index in 0..sequences.len() {
        ensure!(starts[index] > 0 && ends[index] >= starts[index]);
        if sequences[index].len() > context {
            let crop = sequences[index].len() - context;
            ensure!(
                starts[index] >= crop && ends[index] >= crop,
                "CORE continuation exceeds model context"
            );
            sequences[index].drain(..crop);
            starts[index] -= crop;
            ends[index] -= crop;
        }
    }
    let max_len = sequences
        .iter()
        .map(Vec::len)
        .max()
        .context("no CORE prompts")?;
    let bos = engine.tokenizer.bos_id();
    let mut input = Vec::with_capacity(sequences.len() * max_len);
    let mut flat_positions = Vec::new();
    let mut targets = Vec::new();
    let mut counts = Vec::with_capacity(sequences.len());
    for (row, sequence) in sequences.iter().enumerate() {
        input.extend_from_slice(sequence);
        input.resize(input.len() + max_len - sequence.len(), bos);
        counts.push(ends[row] - starts[row]);
        for target_position in starts[row]..ends[row] {
            flat_positions.push((row * max_len + target_position - 1) as u32);
            targets.push(sequence[target_position]);
        }
    }
    if counts.iter().all(|count| *count == 0) {
        return Ok(counts.into_iter().map(|_| (f64::NAN, true)).collect());
    }
    let selected =
        engine
            .model_ref()
            .forward_positions(&input, sequences.len(), max_len, &flat_positions)?;
    let log_probs = ops::log_softmax(&selected, 1)?;
    let target_tensor = Tensor::from_vec(targets.clone(), (targets.len(),), device)?;
    let losses = log_probs.gather(&target_tensor.unsqueeze(1)?, 1)?;
    let selected_logits = selected.to_vec2::<f32>()?;
    let mut results = Vec::with_capacity(sequences.len());
    let mut offset = 0usize;
    for count in counts {
        let end = offset + count;
        if count == 0 {
            results.push((f64::NAN, true));
            continue;
        }
        let loss = -(losses
            .narrow(0, offset, count)?
            .mean_all()?
            .to_scalar::<f32>()? as f64);
        let exact = selected_logits[offset..end]
            .iter()
            .zip(&targets[offset..end])
            .all(|(row, target)| {
                row.iter()
                    .enumerate()
                    .fold(None, |best: Option<(usize, &f32)>, candidate| match best {
                        Some((_, best_value)) if !candidate.1.total_cmp(best_value).is_gt() => best,
                        _ => Some(candidate),
                    })
                    .is_some_and(|(predicted, _)| predicted as u32 == *target)
            });
        results.push((loss, exact));
        offset = end;
    }
    Ok(results)
}

fn render_mc(query: &str, delimiter: &str, examples: &[Value], choice: &str) -> Result<String> {
    let mut output = String::new();
    for example in examples {
        let query = string_field(example, "query")?;
        let choices = strings(example.get("choices"), "few-shot choices")?;
        let gold = usize_field(example, "gold")?;
        let answer = choices
            .get(gold)
            .context("few-shot gold is outside choices")?;
        output.push_str(query);
        output.push_str(delimiter);
        output.push_str(answer);
        output.push_str("\n\n");
    }
    output.push_str(query);
    output.push_str(delimiter);
    output.push_str(choice);
    Ok(output)
}

fn render_schema(
    context: &str,
    delimiter: &str,
    examples: &[Value],
    continuation: &str,
) -> Result<String> {
    let mut output = String::new();
    for example in examples {
        let options = strings(example.get("context_options"), "few-shot context_options")?;
        let gold = usize_field(example, "gold")?;
        let answer = options
            .get(gold)
            .context("few-shot gold is outside context_options")?;
        let example_continuation = string_field(example, "continuation")?;
        output.push_str(answer);
        output.push_str(delimiter);
        output.push_str(example_continuation);
        output.push_str("\n\n");
    }
    output.push_str(context);
    output.push_str(delimiter);
    output.push_str(continuation);
    Ok(output)
}

fn render_lm(context: &str, delimiter: &str, examples: &[Value], continuation: &str) -> String {
    let mut output = String::new();
    for example in examples {
        let context = example
            .get("context")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        let continuation = example
            .get("continuation")
            .and_then(Value::as_str)
            .unwrap_or_default();
        output.push_str(context);
        output.push_str(delimiter);
        output.push_str(continuation);
        output.push_str("\n\n");
    }
    output.push_str(context.trim());
    output.push_str(delimiter);
    output.push_str(continuation);
    output
}

fn encode_batch(
    tokenizer: &crate::tokenizer::BpeTokenizer,
    prompts: &[String],
) -> Result<Vec<Vec<u32>>> {
    prompts
        .iter()
        .map(|prompt| encode(tokenizer, prompt))
        .collect()
}

fn encode(tokenizer: &crate::tokenizer::BpeTokenizer, prompt: &str) -> Result<Vec<u32>> {
    let mut tokens = Vec::with_capacity(128);
    tokens.push(tokenizer.bos_id());
    tokens.extend(tokenizer.encode(prompt)?);
    Ok(tokens)
}

fn common_prefix_length(sequences: &[Vec<u32>]) -> usize {
    let Some(first) = sequences.first() else {
        return 0;
    };
    (0..sequences.iter().map(Vec::len).min().unwrap_or(0))
        .take_while(|index| {
            sequences
                .iter()
                .all(|sequence| sequence[*index] == first[*index])
        })
        .count()
}

fn common_suffix_length(sequences: &[Vec<u32>]) -> usize {
    let Some(first) = sequences.first() else {
        return 0;
    };
    (0..sequences.iter().map(Vec::len).min().unwrap_or(0))
        .take_while(|offset| {
            let first_index = first.len() - 1 - offset;
            sequences
                .iter()
                .all(|sequence| sequence[sequence.len() - 1 - offset] == first[first_index])
        })
        .count()
}

fn choose_fewshot(data: &[Value], index: usize, count: usize) -> Result<Vec<Value>> {
    ensure!(
        count <= data.len().saturating_sub(1),
        "not enough CORE examples for few-shot selection"
    );
    if count == 0 {
        return Ok(Vec::new());
    }
    let mut rng = PythonRandom::seed(1234 + index as u64);
    let mut selected = Vec::with_capacity(count);
    let mut pool = (0..data.len())
        .filter(|candidate| *candidate != index)
        .collect::<Vec<_>>();
    if pool.len() <= sample_set_size(count) {
        for offset in 0..count {
            let selected_offset = rng.randbelow(pool.len() - offset);
            selected.push(data[pool[selected_offset]].clone());
            pool[selected_offset] = pool[pool.len() - offset - 1];
        }
    } else {
        let mut chosen = std::collections::HashSet::with_capacity(count);
        while selected.len() < count {
            let candidate = rng.randbelow(data.len());
            if candidate != index && chosen.insert(candidate) {
                selected.push(data[candidate].clone());
            }
        }
    }
    Ok(selected)
}

fn sample_set_size(count: usize) -> usize {
    let mut size = 21usize;
    if count > 5 {
        let target = count.saturating_mul(3);
        let mut power = 1usize;
        while power < target {
            power = power.saturating_mul(4);
        }
        size = size.saturating_add(power);
    }
    size
}

fn read_jsonl(path: &Path) -> Result<Vec<Value>> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("read CORE dataset {}", path.display()))?;
    contents
        .lines()
        .enumerate()
        .filter(|(_, line)| !line.trim().is_empty())
        .map(|(line, text)| {
            serde_json::from_str(text).with_context(|| {
                format!("parse CORE JSONL line {} in {}", line + 1, path.display())
            })
        })
        .collect()
}

fn strings<'a>(value: Option<&'a Value>, label: &str) -> Result<Vec<&'a str>> {
    value
        .and_then(Value::as_array)
        .with_context(|| format!("CORE field {label} must be an array"))?
        .iter()
        .map(|item| {
            item.as_str()
                .with_context(|| format!("CORE field {label} must contain strings"))
        })
        .collect()
}

fn string_field<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value
        .get(field)
        .and_then(Value::as_str)
        .with_context(|| format!("CORE item has no string field {field}"))
}

fn usize_field(value: &Value, field: &str) -> Result<usize> {
    value
        .get(field)
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .with_context(|| format!("CORE item has no integer field {field}"))
}

fn read_baselines(path: &Path) -> Result<std::collections::HashMap<String, f64>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("read CORE baseline file {}", path.display()))?;
    let mut rows = text.lines();
    let header = rows.next().context("CORE baseline CSV is empty")?;
    let columns = parse_csv_line(header);
    let name_index = columns
        .iter()
        .position(|column| column == "Eval Task")
        .context("CORE CSV has no Eval Task column")?;
    let baseline_index = columns
        .iter()
        .position(|column| column == "Random baseline")
        .context("CORE CSV has no Random baseline column")?;
    let mut result = std::collections::HashMap::new();
    for line in rows {
        let columns = parse_csv_line(line);
        if columns.len() <= name_index.max(baseline_index) {
            continue;
        }
        result.insert(
            columns[name_index].clone(),
            columns[baseline_index].trim().parse::<f64>()?,
        );
    }
    Ok(result)
}

fn parse_csv_line(line: &str) -> Vec<String> {
    let mut fields = Vec::new();
    let mut current = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(character) = chars.next() {
        match character {
            '"' if quoted && chars.peek() == Some(&'"') => {
                chars.next();
                current.push('"');
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(current.trim().to_owned());
                current.clear();
            }
            _ => current.push(character),
        }
    }
    fields.push(current.trim().to_owned());
    fields
}

pub(crate) fn shuffle<T>(items: &mut [T], seed: u64) {
    let mut rng = PythonRandom::seed(seed);
    for index in (1..items.len()).rev() {
        items.swap(index, rng.randbelow(index + 1));
    }
}

/// Small CPython-compatible MT19937 wrapper for random.Random integer seeds,
/// getrandbits, shuffle, and sample(population, k) used by base_eval.py.
struct PythonRandom {
    state: [u32; 624],
    index: usize,
}

impl PythonRandom {
    fn seed(seed: u64) -> Self {
        let mut rng = Self {
            state: [0; 624],
            index: 624,
        };
        rng.state[0] = 19_650_218;
        for i in 1..624 {
            rng.state[i] = 1_812_433_253u32
                .wrapping_mul(rng.state[i - 1] ^ (rng.state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        let mut key = vec![seed as u32];
        if seed >> 32 != 0 {
            key.push((seed >> 32) as u32);
        }
        let mut i = 1usize;
        let mut j = 0usize;
        for _ in 0..624.max(key.len()) {
            rng.state[i] = (rng.state[i]
                ^ (rng.state[i - 1] ^ (rng.state[i - 1] >> 30)).wrapping_mul(1_664_525))
            .wrapping_add(key[j])
            .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                rng.state[0] = rng.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            rng.state[i] = (rng.state[i]
                ^ (rng.state[i - 1] ^ (rng.state[i - 1] >> 30)).wrapping_mul(1_566_083_941))
            .wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                rng.state[0] = rng.state[623];
                i = 1;
            }
        }
        rng.state[0] = 0x8000_0000;
        rng
    }

    fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for i in 0..624 {
                let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
                self.state[i] = self.state[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 == 0 { 0 } else { 0x9908_b0df };
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    fn randbelow(&mut self, upper: usize) -> usize {
        assert!(upper > 0);
        let bits = usize::BITS - upper.leading_zeros();
        loop {
            let value = if bits <= 32 {
                (self.next_u32() >> (32 - bits)) as usize
            } else {
                let low = self.next_u32() as u64;
                let high = (self.next_u32() >> (64 - bits)) as u64;
                (low | (high << 32)) as usize
            };
            if value < upper {
                return value;
            }
        }
    }
}

fn safe_bundle_path(root: &Path, relative: &str) -> Result<PathBuf> {
    let path = Path::new(relative);
    ensure!(
        path.components()
            .all(|component| matches!(component, Component::Normal(_))),
        "unsafe path in CORE bundle: {relative}"
    );
    Ok(root.join(path))
}

fn ensure_eval_bundle() -> Result<PathBuf> {
    let base = base_dir()?;
    let bundle = base.join("eval_bundle");
    if bundle.is_dir() {
        return Ok(bundle);
    }
    fs::create_dir_all(&base)?;
    // Match nanochat.common.download_file_with_lock's cache lock name.
    let lock = base.join("eval_bundle.zip.lock");
    let _guard = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock)
        .with_context(|| format!("open CORE bundle lock {}", lock.display()))?;
    FileExt::lock_exclusive(&_guard)
        .with_context(|| format!("lock CORE bundle cache {}", lock.display()))?;
    if bundle.is_dir() {
        return Ok(bundle);
    }
    if bundle.is_dir() {
        return Ok(bundle);
    }
    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(3600))
        .build()?;
    let mut response = client.get(EVAL_BUNDLE_URL).send()?.error_for_status()?;
    let zip_path = base.join("eval_bundle.zip.part");
    let mut output = File::create(&zip_path)?;
    std::io::copy(&mut response, &mut output)?;
    output.sync_all()?;
    let extraction = tempfile::tempdir_in(&base)?;
    let mut archive = ZipArchive::new(File::open(&zip_path)?)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry
            .enclosed_name()
            .context("unsafe path in CORE ZIP archive")?
            .to_owned();
        let destination = extraction.path().join(&name);
        if entry.is_dir() {
            fs::create_dir_all(&destination)?;
        } else {
            if let Some(parent) = destination.parent() {
                fs::create_dir_all(parent)?;
            }
            let mut file = File::create(&destination)?;
            std::io::copy(&mut entry, &mut file)?;
        }
    }
    let extracted_bundle = extraction.path().join("eval_bundle");
    ensure!(
        extracted_bundle.is_dir(),
        "CORE archive has no eval_bundle directory"
    );
    fs::rename(&extracted_bundle, &bundle)?;
    let _ = fs::remove_file(zip_path);
    Ok(bundle)
}

fn base_dir() -> Result<PathBuf> {
    if let Some(path) = crate::paths::base_dir_override() {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE is required to locate eval data")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}

#[cfg(test)]
mod prompt_render_tests {
    use super::{render_lm, render_mc, render_schema};
    use serde_json::json;

    #[test]
    fn core_prompt_templates_match_jinja_whitespace_and_trimming() {
        let mc = render_mc(
            "Question?\n",
            " => ",
            &[json!({"query":"Prior?", "choices":["wrong", "right"], "gold":1})],
            "Alpha",
        )
        .unwrap();
        assert_eq!(mc, "Prior? => right\n\nQuestion?\n => Alpha");

        let schema = render_schema(
            "ctx A",
            ":",
            &[json!({
                "context_options":["old A", "old B"],
                "continuation":" old",
                "gold":0
            })],
            " answer",
        )
        .unwrap();
        assert_eq!(schema, "old A: old\n\nctx A: answer");

        let lm = render_lm(
            "  passage with space  ",
            "\n",
            &[json!({"context":" old context  ", "continuation":" continuation"})],
            " next",
        );
        assert_eq!(
            lm,
            "old context\n continuation\n\npassage with space\n next"
        );
    }
}
