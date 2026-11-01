//! Tokenizer compression comparison matching `scripts/tok_eval.py`.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fs::{self, File},
    io,
    path::PathBuf,
    time::Duration,
};

use anyhow::{Context, Result, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fancy_regex::Regex;
use reqwest::blocking::Client;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::{datasets, tokenizer::BpeTokenizer};

const R50K_URL: &str = "https://openaipublic.blob.core.windows.net/encodings/r50k_base.tiktoken";
const R50K_SHA256: &str = "306cd27f03c1a714eca7108e03d66b7dc042abe8c258b44c199a7ed9838dd930";
const CL100K_URL: &str =
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken";
const CL100K_SHA256: &str = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";
const R50K_PATTERN: &str =
    r"'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s";
const CL100K_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s";

#[derive(Clone, Debug)]
struct ReferenceTokenizer {
    pattern: Regex,
    ranks: HashMap<Vec<u8>, u32>,
    bytes_by_id: Vec<Vec<u8>>,
    vocab_size: usize,
}

#[derive(Debug)]
struct Piece {
    bytes: Vec<u8>,
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    generation: u32,
    active: bool,
}

impl ReferenceTokenizer {
    fn from_tiktoken_file(path: &PathBuf, pattern: &str, vocab_size: usize) -> Result<Self> {
        let pattern = Regex::new(pattern).context("compile reference tokenizer pattern")?;
        let content = fs::read_to_string(path)?;
        let mut ranks = HashMap::new();
        let mut bytes_by_id: Vec<Vec<u8>> = Vec::new();
        for (line_number, line) in content.lines().enumerate() {
            let mut fields = line.split_ascii_whitespace();
            let encoded = fields.next().with_context(|| {
                format!("missing token bytes on rank file line {}", line_number + 1)
            })?;
            let rank: usize = fields
                .next()
                .with_context(|| format!("missing rank on rank file line {}", line_number + 1))?
                .parse()
                .with_context(|| format!("invalid rank on rank file line {}", line_number + 1))?;
            ensure!(
                fields.next().is_none(),
                "extra data on rank file line {}",
                line_number + 1
            );
            let bytes = STANDARD.decode(encoded).with_context(|| {
                format!("decode token bytes on rank file line {}", line_number + 1)
            })?;
            ensure!(
                rank < vocab_size,
                "rank {rank} exceeds tokenizer vocabulary"
            );
            if bytes_by_id.len() <= rank {
                bytes_by_id.resize(rank + 1, Vec::new());
            }
            ensure!(bytes_by_id[rank].is_empty(), "duplicate token rank {rank}");
            ensure!(
                ranks.insert(bytes.clone(), rank as u32).is_none(),
                "duplicate token bytes"
            );
            bytes_by_id[rank] = bytes;
        }
        for rank in 0..256 {
            ensure!(
                ranks.contains_key(&[rank as u8][..]),
                "missing base byte rank {rank}"
            );
        }
        Ok(Self {
            pattern,
            ranks,
            bytes_by_id,
            vocab_size,
        })
    }

    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for matched in self.pattern.find_iter(text) {
            let chunk = matched
                .context("reference tokenizer pattern match failed")?
                .as_str();
            ids.extend(self.encode_piece(chunk.as_bytes())?);
        }
        Ok(ids)
    }

    fn encode_piece(&self, bytes: &[u8]) -> Result<Vec<u32>> {
        if bytes.is_empty() {
            return Ok(Vec::new());
        }
        let mut pieces = Vec::with_capacity(bytes.len());
        for (index, byte) in bytes.iter().enumerate() {
            let token = self
                .ranks
                .get(&[*byte][..])
                .copied()
                .with_context(|| format!("reference tokenizer has no rank for byte {byte}"))?;
            pieces.push(Piece {
                bytes: vec![*byte],
                token,
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                generation: 0,
                active: true,
            });
        }

        type Candidate = Reverse<(u32, usize, usize, u32, u32)>;
        let mut candidates = BinaryHeap::<Candidate>::new();
        for index in 0..pieces.len().saturating_sub(1) {
            add_candidate(index, &pieces, &self.ranks, &mut candidates);
        }
        while let Some(Reverse((rank, left, right, left_generation, right_generation))) =
            candidates.pop()
        {
            if !pieces[left].active
                || !pieces[right].active
                || pieces[left].next != Some(right)
                || pieces[left].generation != left_generation
                || pieces[right].generation != right_generation
            {
                continue;
            }
            let previous = pieces[left].previous;
            let next = pieces[right].next;
            let right_bytes = pieces[right].bytes.clone();
            pieces[left].bytes.extend_from_slice(&right_bytes);
            pieces[left].token = rank;
            pieces[left].generation = pieces[left].generation.wrapping_add(1);
            pieces[left].next = next;
            if let Some(next) = next {
                pieces[next].previous = Some(left);
            }
            pieces[right].active = false;
            pieces[right].generation = pieces[right].generation.wrapping_add(1);
            if let Some(previous) = previous {
                add_candidate(previous, &pieces, &self.ranks, &mut candidates);
            }
            add_candidate(left, &pieces, &self.ranks, &mut candidates);
        }

        let mut ids = Vec::new();
        let mut current = pieces
            .iter()
            .position(|piece| piece.active && piece.previous.is_none());
        while let Some(index) = current {
            ids.push(pieces[index].token);
            current = pieces[index].next;
        }
        Ok(ids)
    }

    fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend_from_slice(
                self.bytes_by_id
                    .get(*id as usize)
                    .filter(|bytes| !bytes.is_empty())
                    .with_context(|| format!("token ID {id} is outside tokenizer ranks"))?,
            );
        }
        String::from_utf8(bytes).context("tokenizer output does not decode as UTF-8")
    }
}

fn add_candidate(
    left: usize,
    pieces: &[Piece],
    ranks: &HashMap<Vec<u8>, u32>,
    candidates: &mut BinaryHeap<Reverse<(u32, usize, usize, u32, u32)>>,
) {
    if !pieces[left].active {
        return;
    }
    let Some(right) = pieces[left].next else {
        return;
    };
    if !pieces[right].active {
        return;
    }
    let mut combined = Vec::with_capacity(pieces[left].bytes.len() + pieces[right].bytes.len());
    combined.extend_from_slice(&pieces[left].bytes);
    combined.extend_from_slice(&pieces[right].bytes);
    if let Some(&rank) = ranks.get(&combined) {
        candidates.push(Reverse((
            rank,
            left,
            right,
            pieces[left].generation,
            pieces[right].generation,
        )));
    }
}

#[derive(serde::Serialize)]
struct Compression {
    bytes: usize,
    tokens: usize,
    ratio: f64,
}

pub fn evaluate(dataset_name: &str, tokenizer_path: Option<PathBuf>) -> Result<Value> {
    let dataset = datasets::Dataset::parse(dataset_name)?;
    let files = datasets::local_files(dataset.local_cache_dir()?)?;
    ensure!(
        !files.is_empty(),
        "no local Parquet shards for {dataset_name}; run `download-dataset --dataset {dataset_name}` first"
    );
    let train = datasets::first_row_group_text(&files, false)?.join("\n");
    let val = datasets::first_row_group_text(&files, true)?.join("\n");

    let mut samples: Vec<(String, String)> =
        serde_json::from_str(include_str!("../assets/tokenizer_eval_samples.json"))?;
    samples.push((format!("{dataset_name}-train"), train));
    if !val.is_empty() {
        samples.push((format!("{dataset_name}-val"), val));
    }

    let cache = tokenizer_cache_dir()?;
    let gpt2 = ReferenceTokenizer::from_tiktoken_file(
        &ensure_rank_file(&cache, "r50k_base", R50K_URL, R50K_SHA256)?,
        R50K_PATTERN,
        50_257,
    )?;
    let gpt4 = ReferenceTokenizer::from_tiktoken_file(
        &ensure_rank_file(&cache, "cl100k_base", CL100K_URL, CL100K_SHA256)?,
        CL100K_PATTERN,
        100_277,
    )?;
    let ours_path = tokenizer_path
        .or_else(|| std::env::var_os("NANOCHAT_TOKENIZER").map(PathBuf::from))
        .unwrap_or(crate::paths::cached_tokenizer_path(&tokenizer_base_dir()?));
    let ours = BpeTokenizer::load(&ours_path).with_context(|| {
        format!(
            "load native tokenizer {} (set NANOCHAT_TOKENIZER to override)",
            ours_path.display()
        )
    })?;

    println!(
        "\nVocab sizes: GPT-2 50257 | GPT-4/cl100k 100277 | Ours {}",
        ours.vocab_size()
    );
    let mut all_results = serde_json::Map::new();
    for (name, tokenizer, vocab_size) in [
        ("gpt2", Some(&gpt2), gpt2.vocab_size),
        ("gpt4", Some(&gpt4), gpt4.vocab_size),
        ("ours", None, ours.vocab_size() as usize),
    ] {
        let mut results = serde_json::Map::new();
        for (sample_name, text) in &samples {
            let ids = if let Some(tokenizer) = tokenizer {
                tokenizer.encode(text)?
            } else {
                ours.encode(text)?
            };
            let decoded = if let Some(tokenizer) = tokenizer {
                tokenizer.decode(&ids)?
            } else {
                ours.decode(&ids)?
            };
            ensure!(
                decoded == *text,
                "{name} tokenizer did not round-trip sample {sample_name}"
            );
            let bytes = text.len();
            let ratio = bytes as f64 / ids.len().max(1) as f64;
            results.insert(
                sample_name.clone(),
                serde_json::to_value(Compression {
                    bytes,
                    tokens: ids.len(),
                    ratio,
                })?,
            );
        }
        all_results.insert(
            name.to_owned(),
            serde_json::json!({ "vocab_size": vocab_size, "samples": results }),
        );
    }
    for baseline in ["gpt2", "gpt4"] {
        let baseline_label = if baseline == "gpt2" { "GPT-2" } else { "GPT-4" };
        println!("\nComparison with {baseline_label}:");
        println!(
            "{:<24} {:>10} {:>10} {:>8} {:>10} {:>8} {:>10} {:>8}",
            "Text Type", "Bytes", "Tokens", "Ratio", "Ours tok", "Ours ratio", "Diff %", "Better"
        );
        for (sample_name, _) in &samples {
            let baseline_data = &all_results[baseline]["samples"][sample_name];
            let ours_data = &all_results["ours"]["samples"][sample_name];
            let bytes = baseline_data["bytes"].as_u64().unwrap_or(0);
            let baseline_tokens = all_results[baseline]["samples"][sample_name]["tokens"]
                .as_u64()
                .unwrap_or(0);
            let baseline_ratio = baseline_data["ratio"].as_f64().unwrap_or(0.0);
            let ours_tokens = ours_data["tokens"].as_u64().unwrap_or(0);
            let ours_ratio = ours_data["ratio"].as_f64().unwrap_or(0.0);
            let reduction = if baseline_tokens == 0 {
                0.0
            } else {
                (baseline_tokens as f64 - ours_tokens as f64) * 100.0 / baseline_tokens as f64
            };
            let better = if baseline_ratio > ours_ratio {
                baseline_label
            } else if ours_ratio > baseline_ratio {
                "Ours"
            } else {
                "Tie"
            };
            println!(
                "{sample_name:<24} {bytes:>10} {baseline_tokens:>10} {baseline_ratio:>8.2} {ours_tokens:>10} {ours_ratio:>10.2} {reduction:>+9.1}% {better:>8}"
            );
        }
    }
    Ok(serde_json::json!({ "dataset": dataset_name, "tokenizers": all_results }))
}

fn tokenizer_cache_dir() -> Result<PathBuf> {
    Ok(tokenizer_base_dir()?.join("tokenizer_eval"))
}

fn tokenizer_base_dir() -> Result<PathBuf> {
    let base = if let Some(path) = crate::paths::base_dir_override() {
        PathBuf::from(path)
    } else {
        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .context("HOME or USERPROFILE is required for tokenizer cache")?;
        PathBuf::from(home).join(".cache").join("nanochat")
    };
    Ok(base)
}

fn ensure_rank_file(
    cache: &PathBuf,
    name: &str,
    url: &str,
    expected_sha256: &str,
) -> Result<PathBuf> {
    fs::create_dir_all(cache)?;
    let destination = cache.join(format!("{name}.tiktoken"));
    if destination.is_file() {
        if sha256(&destination)? == expected_sha256 {
            return Ok(destination);
        }
        fs::remove_file(&destination).context("remove invalid cached tokenizer ranks")?;
    }
    let temp = cache.join(format!("{name}.{}.part", std::process::id()));
    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(600))
        .build()?;
    let download = (|| -> Result<()> {
        let mut response = client.get(url).send()?.error_for_status()?;
        let mut output = File::create(&temp)?;
        io::copy(&mut response, &mut output)?;
        output.sync_all()?;
        ensure!(
            sha256(&temp)? == expected_sha256,
            "SHA-256 mismatch while downloading {name} ranks"
        );
        match fs::rename(&temp, &destination) {
            Ok(()) => Ok(()),
            Err(_error)
                if destination.is_file()
                    && sha256(&destination).ok().as_deref() == Some(expected_sha256) =>
            {
                let _ = fs::remove_file(&temp);
                Ok(())
            }
            Err(error) => {
                if destination.is_file() {
                    fs::remove_file(&destination)?;
                    fs::rename(&temp, &destination)?;
                    Ok(())
                } else {
                    Err(error.into())
                }
            }
        }
    })();
    if download.is_err() {
        let _ = fs::remove_file(&temp);
    }
    download?;
    Ok(destination)
}

fn sha256(path: &PathBuf) -> Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
