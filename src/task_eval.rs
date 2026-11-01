//! Dataset-backed chat task evaluation migrated from `scripts/chat_eval.py`.

use std::{
    fs::{self, File, OpenOptions},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use candle_core::Tensor;
use fancy_regex::Regex;
use fs2::FileExt;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Field,
};
use reqwest::blocking::Client;
use serde_json::{Value, json};

use crate::{engine::Engine, execution::execute_code};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatTask {
    ArcEasy,
    ArcChallenge,
    Mmlu,
    Gsm8k,
    HumanEval,
}

impl ChatTask {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "ARC-Easy" | "arc-easy" => Ok(Self::ArcEasy),
            "ARC-Challenge" | "arc-challenge" => Ok(Self::ArcChallenge),
            "MMLU" | "mmlu" => Ok(Self::Mmlu),
            "GSM8K" | "gsm8k" => Ok(Self::Gsm8k),
            "HumanEval" | "humaneval" => Ok(Self::HumanEval),
            _ => bail!("unknown chat task '{value}'"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::ArcEasy => "ARC-Easy",
            Self::ArcChallenge => "ARC-Challenge",
            Self::Mmlu => "MMLU",
            Self::Gsm8k => "GSM8K",
            Self::HumanEval => "HumanEval",
        }
    }

    fn dataset(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::ArcEasy => ("allenai/ai2_arc", "ARC-Easy", "test"),
            Self::ArcChallenge => ("allenai/ai2_arc", "ARC-Challenge", "test"),
            Self::Mmlu => ("cais/mmlu", "all", "test"),
            Self::Gsm8k => ("openai/gsm8k", "main", "test"),
            Self::HumanEval => ("openai/openai_humaneval", "openai_humaneval", "test"),
        }
    }

    fn categorical(self) -> bool {
        matches!(self, Self::ArcEasy | Self::ArcChallenge | Self::Mmlu)
    }
}

pub struct ChatExample {
    pub conversation: Value,
    pub answer: Value,
}

/// Evaluate one or more Python `chat_eval` tasks. Returns per-task accuracy
/// and the centered ChatCORE score when all five tasks are requested.
pub fn evaluate_chat_tasks(
    engine: &Engine,
    tasks: &[ChatTask],
    batch_size: usize,
    num_samples: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    seed: u64,
    max_problems: Option<usize>,
    predictions_output: Option<&Path>,
) -> Result<Value> {
    ensure!(batch_size > 0, "evaluation batch size must be positive");
    ensure!(num_samples > 0, "number of samples must be positive");
    let mut distributed = crate::distributed::DistributedContext::from_env()?;
    let rank = distributed.rank();
    let world_size = distributed.world_size();
    ensure!(
        predictions_output.is_none() || world_size == 1,
        "per-example prediction export currently requires a single rank"
    );
    let mut predictions = predictions_output
        .map(|path| File::create(path).map(BufWriter::new))
        .transpose()?;
    let mut results = serde_json::Map::new();
    for task in tasks {
        let examples = load_chat_examples(*task, 42)?;
        let limit = max_problems.unwrap_or(examples.len()).min(examples.len());
        let local = if task.categorical() {
            evaluate_categorical(
                engine,
                *task,
                &examples[..limit],
                batch_size,
                rank,
                world_size,
                &mut predictions,
            )?
        } else {
            evaluate_generative(
                engine,
                *task,
                &examples[..limit],
                num_samples,
                max_new_tokens,
                temperature,
                top_k,
                seed,
                rank,
                world_size,
                &mut predictions,
            )?
        };
        let (passed, total) = distributed.all_reduce_counts(local.0, local.1)?;
        ensure!(total > 0, "evaluation selected no examples");
        let accuracy = passed as f64 / total as f64;
        results.insert(task.name().to_owned(), json!(accuracy));
        println!(
            "{} accuracy: {:.2}% ({}/{})",
            task.name(),
            100.0 * accuracy,
            (accuracy * limit as f64).round() as usize,
            limit
        );
    }
    let all = [
        ChatTask::ArcEasy,
        ChatTask::ArcChallenge,
        ChatTask::Mmlu,
        ChatTask::Gsm8k,
        ChatTask::HumanEval,
    ];
    if all.iter().all(|task| results.contains_key(task.name())) {
        let mut score = 0.0;
        for task in all {
            let accuracy = results[task.name()].as_f64().unwrap_or(0.0);
            let baseline = if task.categorical() { 0.25 } else { 0.0 };
            score += (accuracy - baseline) / (1.0 - baseline);
        }
        results.insert("ChatCORE".into(), json!(score / all.len() as f64));
    }
    if let Some(writer) = predictions.as_mut() {
        writer.flush()?;
    }
    Ok(Value::Object(results))
}

fn evaluate_categorical(
    engine: &Engine,
    task: ChatTask,
    examples: &[ChatExample],
    batch_size: usize,
    rank: usize,
    world_size: usize,
    predictions: &mut Option<BufWriter<File>>,
) -> Result<(u64, u64)> {
    let batches = examples.len().div_ceil(batch_size);
    let mut passed = 0u64;
    let mut total = 0u64;
    for batch_index in (rank..batches).step_by(world_size) {
        let start = batch_index * batch_size;
        let stop = (start + batch_size).min(examples.len());
        let chunk = &examples[start..stop];
        let prompts = chunk
            .iter()
            .map(|example| {
                engine
                    .tokenizer
                    .render_for_completion(&example.conversation)
            })
            .collect::<Result<Vec<_>>>()?;
        let max_len = prompts.iter().map(Vec::len).max().unwrap_or(0);
        let bos = engine.tokenizer.bos_id();
        let mut tokens = Vec::with_capacity(chunk.len() * max_len);
        for prompt in &prompts {
            tokens.extend_from_slice(prompt);
            tokens.resize(tokens.len() + max_len - prompt.len(), bos);
        }
        let answers = chunk
            .iter()
            .map(|example| categorical_answer(task, &example.answer))
            .collect::<Result<Vec<_>>>()?;
        let max_choices = answers
            .iter()
            .map(|(letters, _)| letters.len())
            .max()
            .context("categorical batch has no choices")?;
        let mut choice_ids = Vec::with_capacity(chunk.len() * max_choices);
        for (letters, _) in &answers {
            let ids = letters
                .iter()
                .map(|letter| {
                    let ids = engine.tokenizer.encode(letter)?;
                    ensure!(ids.len() == 1, "answer choice {letter:?} must be one token");
                    Ok(ids[0])
                })
                .collect::<Result<Vec<_>>>()?;
            choice_ids.extend_from_slice(&ids);
            choice_ids.resize(choice_ids.len() + max_choices - ids.len(), 0);
        }
        let selected_positions = prompts
            .iter()
            .enumerate()
            .map(|(index, prompt)| (index * max_len + prompt.len() - 1) as u32)
            .collect::<Vec<_>>();
        let logits = engine.forward(&tokens, chunk.len(), max_len)?;
        let vocab_size = logits.dim(2)?;
        let selected_logits = logits
            .reshape((chunk.len() * max_len, vocab_size))?
            .index_select(
                &Tensor::from_vec(selected_positions, (chunk.len(),), logits.device())?,
                0,
            )?;
        let choice_logits = selected_logits
            .gather(
                &Tensor::from_vec(choice_ids, (chunk.len(), max_choices), logits.device())?,
                1,
            )?
            .to_vec2::<f32>()?;
        for (index, (letters, expected)) in answers.iter().enumerate() {
            let (choice_index, _) = letters
                .iter()
                .enumerate()
                .map(|(choice_index, _)| (choice_index, choice_logits[index][choice_index]))
                .fold(None, |best: Option<(usize, f32)>, candidate| match best {
                    Some((_, best_value)) if !candidate.1.total_cmp(&best_value).is_gt() => best,
                    _ => Some(candidate),
                })
                .context("categorical task has no answer choices")?;
            passed += u64::from(letters[choice_index].as_str() == expected.as_str());
            write_prediction(
                predictions,
                json!({
                    "task": task.name(),
                    "index": start + index,
                    "expected": expected,
                    "predicted": letters[choice_index],
                    "choice_logits": choice_logits[index],
                    "passed": letters[choice_index].as_str() == expected.as_str(),
                }),
            )?;
            total += 1;
        }
    }
    Ok((passed, total))
}

fn evaluate_generative(
    engine: &Engine,
    task: ChatTask,
    examples: &[ChatExample],
    num_samples: usize,
    max_new_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    seed: u64,
    rank: usize,
    world_size: usize,
    predictions: &mut Option<BufWriter<File>>,
) -> Result<(u64, u64)> {
    let mut passed = 0u64;
    let mut total = 0u64;
    for index in (rank..examples.len()).step_by(world_size) {
        let example = &examples[index];
        let prompt = engine
            .tokenizer
            .render_for_completion(&example.conversation)?;
        let (generated, _) = engine.generate_batch(
            &prompt,
            num_samples,
            max_new_tokens,
            temperature,
            top_k,
            seed,
        )?;
        let mut any_passed = false;
        let mut completions = Vec::with_capacity(generated.len());
        let mut outcomes = Vec::with_capacity(generated.len());
        for sequence in generated {
            let completion = engine.tokenizer.decode(&sequence[prompt.len()..])?;
            let passed = evaluate_completion(task, example, &completion)?;
            if passed {
                any_passed = true;
            }
            outcomes.push(passed);
            completions.push(completion);
        }
        passed += u64::from(any_passed);
        write_prediction(
            predictions,
            json!({
                "task": task.name(),
                "index": index,
                "expected": example.answer.get("expected"),
                "completions": completions,
                "outcomes": outcomes,
                "passed": any_passed,
            }),
        )?;
        total += 1;
    }
    Ok((passed, total))
}

fn write_prediction(predictions: &mut Option<BufWriter<File>>, row: Value) -> Result<()> {
    if let Some(writer) = predictions {
        serde_json::to_writer(&mut *writer, &row)?;
        writer.write_all(b"\n")?;
    }
    Ok(())
}

fn evaluate_completion(task: ChatTask, example: &ChatExample, completion: &str) -> Result<bool> {
    match task {
        ChatTask::Gsm8k => {
            let expected = example.answer.get("expected").and_then(Value::as_str);
            Ok(expected.is_some() && extract_gsm_answer(completion).as_deref() == expected)
        }
        ChatTask::HumanEval => {
            let prompt = example
                .answer
                .get("prompt")
                .and_then(Value::as_str)
                .context("HumanEval prompt metadata missing")?;
            let imports = prompt
                .lines()
                .take_while(|line| {
                    let line = line.trim();
                    line.is_empty()
                        || line.starts_with('#')
                        || line.starts_with("import ")
                        || line.starts_with("from ")
                })
                .filter(|line| {
                    let line = line.trim();
                    line.starts_with("import ") || line.starts_with("from ")
                })
                .collect::<Vec<_>>()
                .join("\n");
            let code = extract_code_block(completion);
            let entry_point = example
                .answer
                .get("entry_point")
                .and_then(Value::as_str)
                .context("HumanEval entry_point missing")?;
            let tests = example
                .answer
                .get("test")
                .and_then(Value::as_str)
                .context("HumanEval tests missing")?;
            let program = format!("{imports}\n\n{code}\n\n{tests}\ncheck({entry_point})");
            // HumanEval uses the reference runner's five-second timeout and
            // 256 MiB process limit. Apply the limit after RustPython startup
            // so its interpreter bootstrap does not consume the user budget.
            let result = execute_code(&program, Duration::from_secs(5), Some(256 * 1024 * 1024))?;
            Ok(result.success)
        }
        _ => bail!("{} is not a generative task", task.name()),
    }
}

fn extract_code_block(completion: &str) -> &str {
    let Ok(pattern) = Regex::new(r"(?s)```(?:python)?\s*\n(.*?)\n```") else {
        return completion.trim();
    };
    pattern
        .captures(completion)
        .ok()
        .flatten()
        .and_then(|captures| captures.get(1))
        .map_or_else(|| completion.trim(), |code| code.as_str().trim())
}

fn categorical_answer(task: ChatTask, answer: &Value) -> Result<(Vec<String>, String)> {
    let answer = answer
        .as_object()
        .context("categorical answer must be an object")?;
    let letters = answer
        .get("letters")
        .and_then(Value::as_array)
        .context("categorical answer has no letters")?
        .iter()
        .map(|value| {
            value
                .as_str()
                .map(str::to_owned)
                .context("answer letter must be text")
        })
        .collect::<Result<Vec<_>>>()?;
    let expected = answer
        .get("expected")
        .and_then(Value::as_str)
        .context("categorical answer has no expected letter")?
        .to_owned();
    ensure!(
        task.categorical(),
        "task {} is not categorical",
        task.name()
    );
    Ok((letters, expected))
}

pub fn load_chat_examples(task: ChatTask, seed: u64) -> Result<Vec<ChatExample>> {
    load_chat_examples_from_split(task, None, seed)
}

pub fn load_gsm8k_examples(split: &str, seed: u64) -> Result<Vec<ChatExample>> {
    ensure!(
        matches!(split, "train" | "test"),
        "GSM8K split must be train or test"
    );
    load_chat_examples_from_split(ChatTask::Gsm8k, Some(split), seed)
}

fn load_chat_examples_from_split(
    task: ChatTask,
    split_override: Option<&str>,
    seed: u64,
) -> Result<Vec<ChatExample>> {
    let (repo, subset, default_split) = task.dataset();
    let split = split_override.unwrap_or(default_split);
    let rows = load_hub_rows(repo, subset, split)?;
    let mut examples = rows
        .into_iter()
        .map(|row| make_example(task, row))
        .collect::<Result<Vec<_>>>()?;
    crate::numpy_rng::shuffle(&mut examples, seed);
    Ok(examples)
}

fn make_example(task: ChatTask, row: Value) -> Result<ChatExample> {
    let example = match task {
        ChatTask::ArcEasy | ChatTask::ArcChallenge => {
            let question = text_field(&row, "question")?;
            let answer = text_field(&row, "answerKey")?;
            let choices = row.get("choices").context("ARC row has no choices")?;
            let labels = string_array(choices.get("label"), "choices.label")?;
            let texts = string_array(choices.get("text"), "choices.text")?;
            ensure!(
                labels.len() == texts.len(),
                "ARC choices have mismatched labels and text"
            );
            ensure!(
                labels.iter().any(|label| label == &answer),
                "ARC answer key is not a choice"
            );
            let prompt = render_multiple_choice(&question, &labels, &texts);
            ChatExample {
                conversation: conversation(&prompt, &answer),
                answer: json!({"letters": labels, "expected": answer}),
            }
        }
        ChatTask::Mmlu => {
            let question = text_field(&row, "question")?;
            let choices = string_array(row.get("choices"), "choices")?;
            ensure!(choices.len() == 4, "MMLU must have four choices");
            let letters: Vec<String> = vec!["A", "B", "C", "D"]
                .into_iter()
                .map(str::to_owned)
                .collect();
            let answer_index =
                row.get("answer")
                    .and_then(Value::as_u64)
                    .context("MMLU answer must be an integer index")? as usize;
            ensure!(
                answer_index < letters.len(),
                "MMLU answer index is outside choices"
            );
            let answer = letters[answer_index].clone();
            let prompt = render_multiple_choice(&question, &letters, &choices);
            ChatExample {
                conversation: conversation(&prompt, &answer),
                answer: json!({"letters": letters, "expected": answer}),
            }
        }
        ChatTask::Gsm8k => {
            let question = text_field(&row, "question")?;
            let answer = text_field(&row, "answer")?;
            ChatExample {
                conversation: json!({"messages": [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ]}),
                answer: json!({
                    "expected": extract_gsm_answer(&answer),
                    "reference_solution": format_gsm_reference(&answer),
                }),
            }
        }
        ChatTask::HumanEval => {
            let prompt = text_field(&row, "prompt")?;
            let entry_point = text_field(&row, "entry_point")?;
            let test = text_field(&row, "test")?;
            let solution = text_field(&row, "canonical_solution")?;
            let completed = format!("{prompt}\n{solution}");
            ChatExample {
                conversation: conversation(&prompt, &completed),
                answer: json!({"entry_point": entry_point, "test": test, "prompt": prompt}),
            }
        }
    };
    Ok(example)
}

fn format_gsm_reference(answer: &str) -> String {
    let mut formatted = String::new();
    let mut remaining = answer;
    while let Some(start) = remaining.find("<<") {
        formatted.push_str(&remaining[..start]);
        let Some(relative_end) = remaining[start + 2..].find(">>") else {
            formatted.push_str(&remaining[start..]);
            return formatted;
        };
        let end = start + 2 + relative_end;
        let tool = &remaining[start + 2..end];
        if let Some((expression, output)) = tool.rsplit_once('=') {
            formatted.push_str(&format!("[python: {expression}][python_output: {output}]"));
        } else {
            formatted.push_str(&format!("[python: {tool}][python_output: ]"));
        }
        remaining = &remaining[end + 2..];
    }
    formatted.push_str(remaining);
    formatted
}

fn conversation(prompt: &str, answer: &str) -> Value {
    json!({"messages": [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": answer}
    ]})
}

fn render_multiple_choice(question: &str, letters: &[String], choices: &[String]) -> String {
    let mut prompt = format!("Multiple Choice question: {question}\n");
    for (letter, choice) in letters.iter().zip(choices) {
        prompt.push_str(&format!("- {choice}={letter}\n"));
    }
    prompt.push_str("\nRespond only with the letter of the correct answer.");
    prompt
}

fn text_field(row: &Value, key: &str) -> Result<String> {
    row.get(key)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .with_context(|| format!("dataset row has no string field {key}"))
}

fn string_array(value: Option<&Value>, name: &str) -> Result<Vec<String>> {
    value
        .and_then(Value::as_array)
        .with_context(|| format!("dataset field {name} must be an array"))?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .map(str::to_owned)
                .with_context(|| format!("dataset field {name} must contain strings"))
        })
        .collect()
}

pub fn extract_gsm_answer(text: &str) -> Option<String> {
    static ANSWER_PATTERN: OnceLock<Regex> = OnceLock::new();
    let pattern = ANSWER_PATTERN
        .get_or_init(|| Regex::new(r"#### (-?[0-9.,]+)").expect("valid GSM8K answer pattern"));
    pattern
        .captures(text)
        .ok()
        .flatten()
        .and_then(|captures| captures.get(1))
        .map(|answer| answer.as_str().replace(',', ""))
}

pub fn gsm8k_reward(expected: Option<&Value>, completion: &str) -> f32 {
    let expected = expected.and_then(Value::as_str);
    if expected.is_some() && extract_gsm_answer(completion).as_deref() == expected {
        1.0
    } else {
        0.0
    }
}

fn load_hub_rows(repo: &str, subset: &str, split: &str) -> Result<Vec<Value>> {
    let base = base_dir()?;
    let slug = repo.replace('/', "--");
    let directory = base.join("task_data").join(slug).join(subset).join(split);
    fs::create_dir_all(&directory)?;
    let manifest = directory.join("manifest.json");
    // Always take the lock briefly: another process may have created the
    // manifest but still be writing it while holding the same Python-compatible
    // lock.
    download_task_dataset(repo, subset, split, &directory, &manifest)?;
    let filenames: Vec<String> = serde_json::from_slice(&fs::read(&manifest)?)?;
    ensure!(
        !filenames.is_empty(),
        "task dataset manifest contains no shards"
    );
    let mut rows = Vec::new();
    for filename in filenames {
        let path = directory.join(filename);
        let reader = SerializedFileReader::new(
            File::open(&path).with_context(|| format!("open task shard {}", path.display()))?,
        )?;
        for row in reader.get_row_iter(None)? {
            let row = row?;
            let object = row
                .get_column_iter()
                .map(|(name, field)| (name.to_owned(), field_to_json(field)))
                .collect::<serde_json::Map<_, _>>();
            rows.push(Value::Object(object));
        }
    }
    ensure!(!rows.is_empty(), "task dataset contains no rows");
    Ok(rows)
}

fn download_task_dataset(
    repo: &str,
    subset: &str,
    split: &str,
    directory: &Path,
    manifest: &Path,
) -> Result<()> {
    // Python's FileLock uses this exact suffix. Hold an OS lock rather than
    // claiming the path with create_new so crashes do not strand the cache.
    let lock = manifest.with_file_name("manifest.json.lock");
    let _guard = open_cache_lock(&lock)?;
    if manifest.is_file() {
        return Ok(());
    }
    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(3600))
        .build()?;
    let listing = format!("https://huggingface.co/api/datasets/{repo}/parquet/{subset}/{split}");
    let response = client.get(&listing).send()?.error_for_status()?;
    let listing: Value = response.json()?;
    let urls = listing
        .as_array()
        .context("Hugging Face task shard listing must be an array")?
        .iter()
        .map(|entry| {
            entry
                .as_str()
                .or_else(|| entry.get("url").and_then(Value::as_str))
                .map(str::to_owned)
                .context("Hugging Face task shard is missing its URL")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !urls.is_empty(),
        "Hugging Face returned no task data shards"
    );
    let mut filenames = Vec::with_capacity(urls.len());
    for (index, url) in urls.iter().enumerate() {
        let filename = format!("{index:05}.parquet");
        let destination = directory.join(&filename);
        if !destination.is_file() {
            let mut response = client.get(url).send()?.error_for_status()?;
            let temp = destination.with_extension("parquet.part");
            let mut file = File::create(&temp)?;
            io::copy(&mut response, &mut file)?;
            file.sync_all()?;
            fs::rename(temp, &destination)?;
        }
        filenames.push(filename);
    }
    let manifest_temp = manifest.with_extension("json.tmp");
    fs::write(&manifest_temp, serde_json::to_vec(&filenames)?)?;
    fs::rename(manifest_temp, manifest)?;
    Ok(())
}

fn open_cache_lock(path: &Path) -> Result<File> {
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open task data lock {}", path.display()))?;
    FileExt::lock_exclusive(&lock)
        .with_context(|| format!("lock task data cache {}", path.display()))?;
    Ok(lock)
}

fn base_dir() -> Result<PathBuf> {
    if let Some(path) = crate::paths::base_dir_override() {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE is required to find task datasets")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}

fn field_to_json(field: &Field) -> Value {
    match field {
        Field::Null => Value::Null,
        Field::Bool(value) => json!(value),
        Field::Byte(value) => json!(value),
        Field::Short(value) => json!(value),
        Field::Int(value) => json!(value),
        Field::Long(value) => json!(value),
        Field::UByte(value) => json!(value),
        Field::UShort(value) => json!(value),
        Field::UInt(value) => json!(value),
        Field::ULong(value) => json!(value),
        Field::Float16(value) => json!(f32::from(*value)),
        Field::Float(value) => json!(value),
        Field::Double(value) => json!(value),
        Field::Decimal(_) => Value::Null,
        Field::Str(value) => json!(value),
        Field::Bytes(value) => json!(value.data()),
        Field::Date(value) => json!(value),
        Field::TimeMillis(value) => json!(value),
        Field::TimeMicros(value) => json!(value),
        Field::TimestampMillis(value) => json!(value),
        Field::TimestampMicros(value) => json!(value),
        Field::Group(row) => Value::Object(
            row.get_column_iter()
                .map(|(name, field)| (name.to_owned(), field_to_json(field)))
                .collect(),
        ),
        Field::ListInternal(list) => {
            Value::Array(list.elements().iter().map(field_to_json).collect())
        }
        Field::MapInternal(map) => Value::Array(
            map.entries()
                .iter()
                .map(|(key, value)| json!([field_to_json(key), field_to_json(value)]))
                .collect(),
        ),
    }
}
