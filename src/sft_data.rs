//! Native download and preparation of nanochat SFT datasets.

use std::{
    fs::{self, File},
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::{Field, Row},
};
use reqwest::blocking::Client;
use serde_json::{Map, Value};

#[derive(Clone, Copy, Debug)]
pub enum SftMode {
    Standard,
    Tools,
    PostTraining,
    Traces,
}

impl SftMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "standard" => Ok(Self::Standard),
            "tools" => Ok(Self::Tools),
            "post-training" => Ok(Self::PostTraining),
            "traces" => Ok(Self::Traces),
            _ => bail!(
                "unknown SFT mode `{value}` (expected standard, tools, post-training, or traces)"
            ),
        }
    }

    fn dataset_id(self) -> Result<&'static str> {
        match self {
            Self::Standard => bail!("the standard mixture loads multiple datasets"),
            Self::Tools => Ok("CrowdMind/tools-sft"),
            Self::PostTraining => Ok("CrowdMind/post-training"),
            Self::Traces => bail!("the trace mixture loads multiple datasets"),
        }
    }
}

fn prepare_standard(
    train_output: &Path,
    validation_output: &Path,
    seed: u64,
    mmlu_epochs: usize,
    gsm8k_epochs: usize,
) -> Result<(usize, usize)> {
    let smol_train = shuffled(
        load_rows("HuggingFaceTB/smol-smoltalk", "default", "train")?,
        seed,
    );
    let smol_val = shuffled(
        load_rows("HuggingFaceTB/smol-smoltalk", "default", "test")?,
        seed,
    );
    let mmlu_train = shuffled(load_rows("cais/mmlu", "all", "auxiliary_train")?, seed)
        .iter()
        .map(transform_mmlu)
        .collect::<Result<Vec<_>>>()?;
    let mmlu_val = shuffled(load_rows("cais/mmlu", "all", "test")?, seed)
        .iter()
        .take(5200)
        .map(transform_mmlu)
        .collect::<Result<Vec<_>>>()?;
    let gsm8k_train = shuffled(load_rows("openai/gsm8k", "main", "train")?, seed)
        .iter()
        .map(transform_gsm8k)
        .collect::<Result<Vec<_>>>()?;
    let gsm8k_val = shuffled(load_rows("openai/gsm8k", "main", "test")?, seed)
        .iter()
        .take(420)
        .map(transform_gsm8k)
        .collect::<Result<Vec<_>>>()?;

    let train_tasks = std::iter::once(&smol_train)
        .chain(std::iter::repeat_n(&mmlu_train, mmlu_epochs))
        .chain(std::iter::repeat_n(&gsm8k_train, gsm8k_epochs))
        .collect::<Vec<_>>();
    let train_length = train_tasks.iter().map(|task| task.len()).sum();
    let mut train_index = Vec::with_capacity(train_length);
    for (task_index, task) in train_tasks.iter().enumerate() {
        train_index.extend((0..task.len()).map(|row_index| (task_index, row_index)));
    }
    crate::core_eval::shuffle(&mut train_index, 42);
    let train_rows = train_index
        .iter()
        .map(|(task, row)| &train_tasks[*task][*row]);

    let val_tasks = [&smol_val, &mmlu_val, &gsm8k_val];
    let validation_count = val_tasks.iter().map(|task| task.len()).sum::<usize>();
    let mut val_index = Vec::with_capacity(validation_count);
    for (task_index, task) in val_tasks.iter().enumerate() {
        val_index.extend((0..task.len()).map(|row_index| (task_index, row_index)));
    }
    crate::core_eval::shuffle(&mut val_index, 42);
    let validation_rows = val_index.iter().map(|(task, row)| &val_tasks[*task][*row]);
    write_jsonl_iter(train_output, train_rows)?;
    write_jsonl_iter(validation_output, validation_rows)?;
    Ok((train_length, validation_count))
}

const TRACE_DATASETS: [&str; 4] = [
    "armand0e/qwen3.7-max-pi-traces",
    "TeichAI/Ox-Alpha-Pi-Traces",
    "armand0e/claude-opus-4.8-pi-traces",
    "armand0e/minimax-m3-claude-code-traces",
];

fn prepare_traces(
    train_output: &Path,
    validation_output: &Path,
    seed: u64,
) -> Result<(usize, usize)> {
    let mut train_tasks = Vec::<Vec<Value>>::with_capacity(TRACE_DATASETS.len());
    let mut validation_tasks = Vec::<Vec<Value>>::with_capacity(TRACE_DATASETS.len());
    for dataset_id in TRACE_DATASETS {
        let records = if dataset_id == "TeichAI/Ox-Alpha-Pi-Traces" {
            load_teich_trace_rows()?
        } else {
            load_rows(dataset_id, "default", "train")?
        };
        let mut records = shuffled(records, seed);
        let validation_count = (records.len() / 20).max(1);
        let split = records.len().saturating_sub(validation_count);
        let mut validation = records.split_off(split);
        train_tasks.push(records);
        validation_tasks.push(std::mem::take(&mut validation));
    }
    let train_count = write_task_mixture(train_output, &train_tasks)?;
    let validation_count = write_task_mixture(validation_output, &validation_tasks)?;
    Ok((train_count, validation_count))
}

fn write_task_mixture(path: &Path, tasks: &[Vec<Value>]) -> Result<usize> {
    let length = tasks.iter().map(Vec::len).sum::<usize>();
    ensure!(
        length > 0,
        "trace mixture contains no trainable conversations"
    );
    let mut index_map = Vec::with_capacity(length);
    for (task_index, task) in tasks.iter().enumerate() {
        index_map.extend((0..task.len()).map(|row_index| (task_index, row_index)));
    }
    crate::core_eval::shuffle(&mut index_map, 42);
    let rows = index_map.iter().map(|(task, row)| &tasks[*task][*row]);
    write_jsonl_iter(path, rows)?;
    Ok(length)
}

fn load_teich_trace_rows() -> Result<Vec<Value>> {
    let base = cache_base_directory()?;
    let cache = base
        .join("task_data")
        .join("TeichAI--Ox-Alpha-Pi-Traces")
        .join("normalized-train");
    fs::create_dir_all(&cache)?;
    let normalized_jsonl = cache.join("traces.jsonl");
    let _lock = acquire_cache_lock(&cache.join("manifest.json.lock"))?;
    if normalized_jsonl.is_file() {
        return read_jsonl_rows(&normalized_jsonl);
    }
    let normalized_parquet = cache.join("traces.parquet");
    if normalized_parquet.is_file() {
        let reader = SerializedFileReader::new(File::open(&normalized_parquet)?)?;
        let mut rows = Vec::new();
        for row in reader.get_row_iter(None)? {
            rows.push(row_to_json(&row?)?);
        }
        write_jsonl_iter(&normalized_jsonl, rows.iter())?;
        return Ok(rows);
    }

    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(60))
        .build()?;
    let raw_dir = cache.join("raw");
    fs::create_dir_all(&raw_dir)?;
    let mut next_url = Some("https://huggingface.co/api/datasets/TeichAI/Ox-Alpha-Pi-Traces/tree/main?recursive=true&expand=false&limit=1000".to_owned());
    let mut files = Vec::<String>::new();
    while let Some(url) = next_url.take() {
        let response = client.get(&url).send()?.error_for_status()?;
        let next_link = response
            .headers()
            .get(reqwest::header::LINK)
            .and_then(|value| value.to_str().ok())
            .and_then(next_link);
        let entries: Vec<Value> = response.json()?;
        files.extend(entries.iter().filter_map(|entry| {
            let path = entry.get("path")?.as_str()?;
            (entry.get("type").and_then(Value::as_str) == Some("file") && path.ends_with(".jsonl"))
                .then(|| path.to_owned())
        }));
        next_url = next_link;
    }
    ensure!(
        !files.is_empty(),
        "Ox-Alpha Pi traces contain no JSONL session files"
    );

    let mut records = Vec::new();
    for file_path in files {
        let filename = file_path.rsplit('/').next().unwrap_or(&file_path);
        let local_path = raw_dir.join(filename);
        if !local_path.is_file() {
            let url = format!(
                "https://huggingface.co/datasets/TeichAI/Ox-Alpha-Pi-Traces/resolve/main/{}",
                encode_path(&file_path)
            );
            download_file(&client, &url, &local_path)?;
        }
        let mut events = Vec::new();
        for line in BufReader::new(File::open(&local_path)?).lines() {
            let line = line?;
            if let Ok(event) = serde_json::from_str::<Value>(&line) {
                events.push(event);
            }
        }
        let (messages, tools) = convert_trace_events(&events);
        if messages
            .first()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("user")
            && messages
                .last()
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str)
                == Some("assistant")
        {
            records.push(serde_json::json!({
                "messages": serde_json::to_string(&messages)?,
                "tools": serde_json::to_string(&tools)?,
            }));
        }
    }
    ensure!(
        !records.is_empty(),
        "Ox-Alpha Pi traces contain no trainable sessions"
    );
    write_jsonl_iter(&normalized_jsonl, records.iter())?;
    Ok(records)
}

fn read_jsonl_rows(path: &Path) -> Result<Vec<Value>> {
    BufReader::new(File::open(path)?)
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(
                serde_json::from_str(&line)
                    .with_context(|| format!("parse cached trace JSONL row {}", index + 1)),
            ),
            Err(error) => Some(Err(error.into())),
        })
        .collect()
}

fn convert_trace_events(events: &[Value]) -> (Vec<Value>, Value) {
    let mut messages = Vec::new();
    let mut tools = Value::Array(Vec::new());
    for event in events {
        if event.get("type").and_then(Value::as_str) == Some("custom")
            && event.get("customType").and_then(Value::as_str) == Some("teich-available-tools")
        {
            if let Some(value) = event.get("data").and_then(|data| data.get("tools")) {
                tools = value.clone();
            }
            continue;
        }
        if event.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let message = event.get("message").unwrap_or(&Value::Null);
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if role == "toolResult" {
            messages.push(serde_json::json!({
                "role":"tool",
                "tool_call_id":message.get("toolCallId").cloned().unwrap_or(Value::String(String::new())),
                "name":message.get("toolName").cloned().unwrap_or(Value::String("tool".to_owned())),
                "content":message.get("content").cloned().unwrap_or(Value::String(String::new())),
            }));
            continue;
        }
        if !matches!(role, "system" | "user" | "assistant") {
            continue;
        }
        let source_content = message
            .get("content")
            .cloned()
            .unwrap_or(Value::String(String::new()));
        let mut content = source_content.clone();
        let mut tool_calls = Vec::new();
        if role == "assistant" {
            if let Some(parts) = source_content.as_array() {
                let mut texts = Vec::new();
                for part in parts {
                    match part.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(text) = part.get("text").and_then(Value::as_str) {
                                texts.push(text.to_owned());
                            }
                        }
                        Some("toolCall") => tool_calls.push(serde_json::json!({
                            "id":part.get("id").cloned().unwrap_or(Value::String(String::new())),
                            "type":"function",
                            "function": {
                                "name":part.get("name").cloned().unwrap_or(Value::String(String::new())),
                                "arguments":part.get("arguments").cloned().unwrap_or(serde_json::json!({})),
                            }
                        })),
                        _ => {}
                    }
                }
                content = Value::String(texts.join("\n"));
            }
        } else if let Some(parts) = source_content.as_array() {
            content = Value::String(content_text(parts));
        }
        if tool_calls.is_empty() {
            if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                tool_calls = calls.clone();
            }
        }
        let mut converted = serde_json::json!({"role":role,"content":content});
        if !tool_calls.is_empty() {
            converted["tool_calls"] = Value::Array(tool_calls);
        }
        messages.push(converted);
    }
    (messages, tools)
}

fn content_text(parts: &[Value]) -> String {
    parts
        .iter()
        .filter_map(|part| {
            part.as_str()
                .or_else(|| part.get("text").and_then(Value::as_str))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|entry| {
        if !entry.contains("rel=\"next\"") {
            return None;
        }
        let start = entry.find('<')? + 1;
        let end = entry[start..].find('>')? + start;
        Some(entry[start..end].to_owned())
    })
}

fn encode_path(path: &str) -> String {
    let mut encoded = String::with_capacity(path.len());
    for byte in path.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'/' | b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn shuffled(mut rows: Vec<Value>, seed: u64) -> Vec<Value> {
    crate::numpy_rng::shuffle(&mut rows, seed);
    rows
}

fn transform_mmlu(row: &Value) -> Result<Value> {
    let question = row
        .get("question")
        .and_then(Value::as_str)
        .context("MMLU row is missing question text")?;
    let choices = row
        .get("choices")
        .and_then(Value::as_array)
        .context("MMLU row is missing choices")?;
    ensure!(choices.len() == 4, "MMLU row must have four choices");
    let answer = row
        .get("answer")
        .and_then(Value::as_u64)
        .context("MMLU answer must be an integer")? as usize;
    ensure!(answer < 4, "MMLU answer index is outside A-D");
    let letters = ["A", "B", "C", "D"];
    let mut prompt = format!("Multiple Choice question: {question}\n");
    for (choice, letter) in choices.iter().zip(letters) {
        let choice = choice.as_str().context("MMLU choice must be a string")?;
        prompt.push_str(&format!("- {choice}={letter}\n"));
    }
    prompt.push_str("\nRespond only with the letter of the correct answer.");
    Ok(serde_json::json!({
        "messages": [
            {"role":"user", "content":prompt},
            {"role":"assistant", "content":letters[answer]},
        ]
    }))
}

fn transform_gsm8k(row: &Value) -> Result<Value> {
    let question = row
        .get("question")
        .and_then(Value::as_str)
        .context("GSM8K row is missing question text")?;
    let answer = row
        .get("answer")
        .and_then(Value::as_str)
        .context("GSM8K row is missing answer text")?;
    let mut parts = Vec::new();
    let mut cursor = 0usize;
    let mut scan = cursor;
    while let Some(relative_start) = answer[scan..].find("<<") {
        let start = scan + relative_start;
        let Some(relative_end) = answer[start + 2..].find(">>") else {
            break;
        };
        let end = start + 2 + relative_end;
        let inner = &answer[start + 2..end];
        if inner.is_empty() {
            scan = end + 2;
            continue;
        }
        if start > cursor {
            parts.push(serde_json::json!({"type":"text", "text":&answer[cursor..start]}));
        }
        let (expression, result) = inner.rsplit_once('=').unwrap_or((inner, ""));
        parts.push(serde_json::json!({"type":"python", "text":expression}));
        parts.push(serde_json::json!({"type":"python_output", "text":result}));
        cursor = end + 2;
        scan = cursor;
    }
    if cursor < answer.len() || parts.is_empty() {
        parts.push(serde_json::json!({"type":"text", "text":&answer[cursor..]}));
    }
    Ok(serde_json::json!({
        "messages": [
            {"role":"user", "content":question},
            {"role":"assistant", "content":parts},
        ]
    }))
}

/// Download, shuffle and split a CrowdMind dataset, writing raw OpenAI-style
/// rows that the native Rust SFT renderer consumes directly.
pub fn prepare(
    mode: SftMode,
    train_output: &Path,
    validation_output: &Path,
    seed: u64,
    mmlu_epochs: usize,
    gsm8k_epochs: usize,
) -> Result<(usize, usize)> {
    ensure!(
        train_output != validation_output,
        "training and validation outputs must be different files"
    );
    if let SftMode::Standard = mode {
        return prepare_standard(
            train_output,
            validation_output,
            seed,
            mmlu_epochs,
            gsm8k_epochs,
        );
    }
    if let SftMode::Traces = mode {
        return prepare_traces(train_output, validation_output, seed);
    }
    let dataset_id = mode.dataset_id()?;
    let mut rows = load_rows(dataset_id, "default", "train")?;
    ensure!(!rows.is_empty(), "{} contains no records", dataset_id);
    crate::numpy_rng::shuffle(&mut rows, seed);
    let validation_count = (rows.len() / 20).max(1);
    let split = rows.len() - validation_count;
    write_jsonl_iter(train_output, rows[..split].iter())?;
    write_jsonl_iter(validation_output, rows[split..].iter())?;
    Ok((split, validation_count))
}

/// Prepare an SFT corpus once for all training ranks, using a cache lock and a
/// completion manifest so interrupted writes are regenerated on the next run.
pub fn prepare_cached(
    mode: SftMode,
    cache_dir: &Path,
    seed: u64,
    mmlu_epochs: usize,
    gsm8k_epochs: usize,
) -> Result<(usize, usize, bool)> {
    fs::create_dir_all(cache_dir)?;
    let _lock = acquire_cache_lock(&cache_dir.join("prepare.lock"))?;
    let train_output = cache_dir.join("train.jsonl");
    let validation_output = cache_dir.join("validation.jsonl");
    let manifest = cache_dir.join("complete.json");
    if train_output.is_file() && validation_output.is_file() && manifest.is_file() {
        let counts: Value = serde_json::from_slice(&fs::read(&manifest)?)?;
        if let (Some(train), Some(validation)) = (
            counts.get("train_rows").and_then(Value::as_u64),
            counts.get("validation_rows").and_then(Value::as_u64),
        ) {
            return Ok((train as usize, validation as usize, false));
        }
    }
    let (train_rows, validation_rows) = prepare(
        mode,
        &train_output,
        &validation_output,
        seed,
        mmlu_epochs,
        gsm8k_epochs,
    )?;
    let temporary_manifest = cache_dir.join("complete.json.tmp");
    fs::write(
        &temporary_manifest,
        serde_json::to_vec(&serde_json::json!({
            "train_rows": train_rows,
            "validation_rows": validation_rows,
        }))?,
    )?;
    if manifest.exists() {
        fs::remove_file(&manifest)?;
    }
    fs::rename(&temporary_manifest, &manifest)?;
    Ok((train_rows, validation_rows, true))
}

fn cache_directory(dataset_id: &str, subset: &str, split: &str) -> Result<PathBuf> {
    let base = cache_base_directory()?;
    let safe_id = dataset_id.replace('/', "--");
    Ok(base
        .join("task_data")
        .join(safe_id)
        .join(subset)
        .join(split))
}

fn cache_base_directory() -> Result<PathBuf> {
    if let Some(path) = crate::paths::base_dir_override() {
        return Ok(PathBuf::from(path));
    }
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .context("HOME or USERPROFILE is required for the SFT data cache")?;
    Ok(PathBuf::from(home).join(".cache").join("nanochat"))
}

fn download_parquet(dataset_id: &str, subset: &str, split: &str) -> Result<Vec<PathBuf>> {
    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(60))
        .build()?;
    let endpoint =
        format!("https://huggingface.co/api/datasets/{dataset_id}/parquet/{subset}/{split}");
    let response = client.get(endpoint).send()?.error_for_status()?;
    let entries: Vec<Value> = response.json()?;
    ensure!(
        !entries.is_empty(),
        "Hugging Face listed no SFT shards for {dataset_id}"
    );
    let cache = cache_directory(dataset_id, subset, split)?;
    fs::create_dir_all(&cache)?;
    let _lock = acquire_cache_lock(&cache.join("manifest.json.lock"))?;
    let mut paths = Vec::with_capacity(entries.len());
    let mut filenames = Vec::with_capacity(entries.len());
    for (index, entry) in entries.iter().enumerate() {
        let url = entry
            .as_str()
            .or_else(|| entry.get("url").and_then(Value::as_str))
            .context("unexpected Hugging Face Parquet shard entry")?;
        let path = cache.join(format!("{index:05}.parquet"));
        filenames.push(format!("{index:05}.parquet"));
        if !path.is_file() {
            let temp = path.with_extension("parquet.tmp");
            let result = (|| -> Result<()> {
                let mut response = client
                    .get(url)
                    .timeout(Duration::from_secs(60 * 60))
                    .send()?
                    .error_for_status()?;
                let mut output = File::create(&temp)?;
                io::copy(&mut response, &mut output)?;
                output.sync_all()?;
                fs::rename(&temp, &path)?;
                Ok(())
            })();
            if let Err(error) = result {
                let _ = fs::remove_file(&temp);
                return Err(error).with_context(|| format!("download SFT shard {url}"));
            }
        }
        paths.push(path);
    }
    let manifest = cache.join("manifest.json");
    if !manifest.is_file() {
        let temp = cache.join("manifest.json.tmp");
        fs::write(&temp, serde_json::to_vec(&filenames)?)?;
        fs::rename(&temp, &manifest)?;
    }
    Ok(paths)
}

/// Match the reference loader's manifest lock so concurrent preparation
/// processes cannot race while downloading shards or publishing a cache.
fn acquire_cache_lock(path: &Path) -> Result<File> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let lock = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .with_context(|| format!("open cache lock {}", path.display()))?;
    FileExt::lock_exclusive(&lock)
        .with_context(|| format!("lock data cache {}", path.display()))?;
    Ok(lock)
}

fn download_file(client: &Client, url: &str, destination: &Path) -> Result<()> {
    if let Some(parent) = destination.parent() {
        fs::create_dir_all(parent)?;
    }
    let temp = destination.with_extension("download.tmp");
    let result = (|| -> Result<()> {
        let mut response = client
            .get(url)
            .timeout(Duration::from_secs(60 * 60))
            .send()?
            .error_for_status()?;
        let mut output = File::create(&temp)?;
        io::copy(&mut response, &mut output)?;
        output.sync_all()?;
        fs::rename(&temp, destination)?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_file(&temp);
        return Err(error).with_context(|| format!("download {url}"));
    }
    Ok(())
}

fn load_rows(dataset_id: &str, subset: &str, split: &str) -> Result<Vec<Value>> {
    let files = download_parquet(dataset_id, subset, split)?;
    let mut rows = Vec::new();
    for path in files {
        let reader = SerializedFileReader::new(
            File::open(&path).with_context(|| format!("open SFT shard {}", path.display()))?,
        )?;
        for row in reader.get_row_iter(None)? {
            rows.push(row_to_json(&row?)?);
        }
    }
    ensure!(
        !rows.is_empty(),
        "{dataset_id}/{subset}/{split} contains no records"
    );
    Ok(rows)
}

fn write_jsonl_iter<'a>(path: &Path, rows: impl IntoIterator<Item = &'a Value>) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temp = path.with_extension("jsonl.tmp");
    let file = File::create(&temp).with_context(|| format!("create {}", temp.display()))?;
    let mut writer = BufWriter::new(file);
    for row in rows {
        serde_json::to_writer(&mut writer, row)?;
        writer.write_all(b"\n")?;
    }
    writer.flush()?;
    writer.get_ref().sync_all()?;
    fs::rename(&temp, path).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn row_to_json(row: &Row) -> Result<Value> {
    let mut object = Map::new();
    for (name, field) in row.get_column_iter() {
        object.insert(name.clone(), field_to_json(field)?);
    }
    Ok(Value::Object(object))
}

fn field_to_json(field: &Field) -> Result<Value> {
    Ok(match field {
        Field::Null => Value::Null,
        Field::Bool(value) => Value::Bool(*value),
        Field::Byte(value) => Value::from(*value),
        Field::Short(value) => Value::from(*value),
        Field::Int(value) => Value::from(*value),
        Field::Long(value) => Value::from(*value),
        Field::UByte(value) => Value::from(*value),
        Field::UShort(value) => Value::from(*value),
        Field::UInt(value) => Value::from(*value),
        Field::ULong(value) => Value::from(*value),
        Field::Float16(value) => Value::from(f32::from(*value) as f64),
        Field::Float(value) => Value::from(*value as f64),
        Field::Double(value) => Value::from(*value),
        Field::Decimal(value) => Value::String(format!("{value:?}")),
        Field::Str(value) => Value::String(value.clone()),
        Field::Bytes(value) => {
            Value::Array(value.data().iter().map(|byte| Value::from(*byte)).collect())
        }
        Field::Date(value) => Value::from(*value),
        Field::TimeMillis(value) => Value::from(*value),
        Field::TimeMicros(value) | Field::TimestampMicros(value) => Value::from(*value),
        Field::TimestampMillis(value) => Value::from(*value),
        Field::Group(row) => row_to_json(row)?,
        Field::ListInternal(list) => Value::Array(
            list.elements()
                .iter()
                .map(field_to_json)
                .collect::<Result<Vec<_>>>()?,
        ),
        Field::MapInternal(map) => {
            let mut object = Map::new();
            for (key, value) in map.entries() {
                let key = match key {
                    Field::Str(key) => key.clone(),
                    _ => key.to_string(),
                };
                object.insert(key, field_to_json(value)?);
            }
            Value::Object(object)
        }
    })
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{convert_trace_events, next_link};

    #[test]
    fn converts_ox_alpha_tool_calls_and_results() {
        let events = vec![
            json!({
                "type":"custom",
                "customType":"teich-available-tools",
                "data":{"tools":[{"name":"read_file"}]}
            }),
            json!({"type":"message","message":{"role":"user","content":"Inspect this file."}}),
            json!({
                "type":"message",
                "message":{"role":"assistant","content":[
                    {"type":"text","text":"I will read it."},
                    {"type":"toolCall","id":"call-1","name":"read_file","arguments":{"path":"a.txt"}}
                ]}
            }),
            json!({"type":"message","message":{"role":"toolResult","toolCallId":"call-1","toolName":"read_file","content":"contents"}}),
            json!({"type":"message","message":{"role":"assistant","content":"Done."}}),
        ];
        let (messages, tools) = convert_trace_events(&events);
        assert_eq!(tools, json!([{"name":"read_file"}]));
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1]["content"], "I will read it.");
        assert_eq!(
            messages[1]["tool_calls"][0]["function"]["name"],
            "read_file"
        );
        assert_eq!(messages[2]["role"], "tool");
        assert_eq!(messages[2]["tool_call_id"], "call-1");
        assert_eq!(messages[3]["content"], "Done.");
    }

    #[test]
    fn follows_hugging_face_next_page_link() {
        let header = "<https://huggingface.co/api?page=2>; rel=\"next\", <https://huggingface.co/api?page=9>; rel=\"last\"";
        assert_eq!(
            next_link(header).as_deref(),
            Some("https://huggingface.co/api?page=2")
        );
    }
}
