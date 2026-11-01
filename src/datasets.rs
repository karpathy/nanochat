//! Hugging Face shard management and local Parquet readers.

use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver},
    },
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use parquet::{
    file::reader::{FileReader, SerializedFileReader},
    record::Field,
};
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dataset {
    Climbmix,
    Wiki,
    CrowdMind2048,
    CrowdMind6144,
    CrowdMind16384,
}

impl Dataset {
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "climbmix" => Ok(Self::Climbmix),
            "wiki" => Ok(Self::Wiki),
            "crowdmind_2048" => Ok(Self::CrowdMind2048),
            "crowdmind_6144" => Ok(Self::CrowdMind6144),
            "crowdmind_16384" => Ok(Self::CrowdMind16384),
            _ => bail!("unknown dataset `{value}`"),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Climbmix => "climbmix",
            Self::Wiki => "wiki",
            Self::CrowdMind2048 => "crowdmind_2048",
            Self::CrowdMind6144 => "crowdmind_6144",
            Self::CrowdMind16384 => "crowdmind_16384",
        }
    }

    pub fn cache_dir(self) -> Result<PathBuf> {
        let base = if let Some(path) = crate::paths::base_dir_override() {
            PathBuf::from(path)
        } else {
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .context("HOME or USERPROFILE is required to locate the dataset cache")?;
            PathBuf::from(home).join(".cache").join("nanochat")
        };
        Ok(self.cache_dir_in(base))
    }

    pub fn cache_dir_in(self, base: impl AsRef<Path>) -> PathBuf {
        base.as_ref().join(format!("base_data_{}", self.name()))
    }

    /// Resolve the existing local read cache, preserving Python's legacy
    /// ClimbMix fallback while leaving downloads on the current cache path.
    pub fn local_cache_dir(self) -> Result<PathBuf> {
        let base = if let Some(path) = crate::paths::base_dir_override() {
            PathBuf::from(path)
        } else {
            let home = std::env::var_os("HOME")
                .or_else(|| std::env::var_os("USERPROFILE"))
                .context("HOME or USERPROFILE is required to locate the dataset cache")?;
            PathBuf::from(home).join(".cache").join("nanochat")
        };
        Ok(self.local_cache_dir_in(base))
    }

    pub fn local_cache_dir_in(self, base: impl AsRef<Path>) -> PathBuf {
        let current = self.cache_dir_in(&base);
        if current.exists() || self != Self::Climbmix {
            return current;
        }
        let legacy = base.as_ref().join("base_data");
        if legacy.exists() { legacy } else { current }
    }

    fn hf_dataset_id(self) -> Option<&'static str> {
        match self {
            Self::Climbmix => None,
            Self::Wiki => Some("lucadiliello/wikipedia_512_pretraining"),
            Self::CrowdMind2048 => Some("CrowdMind/2048-pretraining"),
            Self::CrowdMind6144 => Some("CrowdMind/6144-pretraining"),
            Self::CrowdMind16384 => Some("CrowdMind/16384-pretraining"),
        }
    }
}

#[cfg(test)]
mod cache_tests {
    use super::{Dataset, select_download_jobs};
    use tempfile::tempdir;

    #[test]
    fn climbmix_reads_legacy_cache_only_when_new_cache_is_absent() {
        let base = tempdir().unwrap();
        let legacy = base.path().join("base_data");
        std::fs::create_dir(&legacy).unwrap();
        assert_eq!(
            Dataset::Climbmix.cache_dir_in(base.path()),
            base.path().join("base_data_climbmix")
        );
        assert_eq!(Dataset::Climbmix.local_cache_dir_in(base.path()), legacy);

        let current = base.path().join("base_data_climbmix");
        std::fs::create_dir(&current).unwrap();
        assert_eq!(Dataset::Climbmix.local_cache_dir_in(base.path()), current);
        assert_eq!(
            Dataset::Wiki.local_cache_dir_in(base.path()),
            base.path().join("base_data_wiki")
        );
    }

    #[test]
    fn missing_climbmix_cache_keeps_the_current_download_path() {
        let base = tempdir().unwrap();
        let current = base.path().join("base_data_climbmix");
        assert_eq!(Dataset::Climbmix.local_cache_dir_in(base.path()), current);
        assert_eq!(Dataset::Climbmix.cache_dir_in(base.path()), current);
    }

    #[test]
    fn zero_climbmix_train_shards_still_selects_validation_shard() {
        let remote = vec!["train-0".to_owned(), "validation".to_owned()];
        assert_eq!(
            select_download_jobs(Dataset::Climbmix, &remote, 0).unwrap(),
            vec![(6542, "validation".to_owned())]
        );
        assert!(select_download_jobs(Dataset::Wiki, &remote[..1], 0).is_err());
    }
}

#[derive(Deserialize)]
struct ParquetUrl {
    url: String,
}

fn list_remote(dataset: Dataset) -> Result<Vec<String>> {
    let client = Client::builder()
        .user_agent("rustAgent")
        .timeout(Duration::from_secs(60))
        .build()?;
    if dataset == Dataset::Climbmix {
        return Ok((0..6542)
            .map(|index| {
                format!(
                    "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_{index:05}.parquet"
                )
            })
            .chain(std::iter::once(
                "https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main/shard_06542.parquet".to_owned(),
            ))
            .collect());
    }
    let dataset_id = dataset.hf_dataset_id().expect("HF dataset");
    let endpoint =
        format!("https://huggingface.co/api/datasets/{dataset_id}/parquet/default/train");
    let response = client.get(endpoint).send()?.error_for_status()?;
    let value: serde_json::Value = response.json()?;
    if let Ok(urls) = serde_json::from_value::<Vec<String>>(value.clone()) {
        return Ok(urls);
    }
    if let Ok(entries) = serde_json::from_value::<Vec<ParquetUrl>>(value) {
        return Ok(entries.into_iter().map(|entry| entry.url).collect());
    }
    bail!("Hugging Face returned an unexpected Parquet shard listing for {dataset_id}")
}

pub fn download(dataset: Dataset, num_files: isize, workers: usize) -> Result<PathBuf> {
    ensure!(workers > 0, "worker count must be positive");
    let cache = dataset.cache_dir()?;
    fs::create_dir_all(&cache)?;
    let remote = list_remote(dataset)?;
    let jobs = select_download_jobs(dataset, &remote, num_files)?;
    let results: Vec<Result<()>> = jobs
        .par_iter()
        .map(|(index, url)| download_one(url, &cache.join(format!("shard_{index:05}.parquet"))))
        .collect();
    for result in results {
        result?;
    }
    Ok(cache)
}

fn select_download_jobs(
    dataset: Dataset,
    remote: &[String],
    num_files: isize,
) -> Result<Vec<(usize, String)>> {
    ensure!(
        !remote.is_empty(),
        "Hugging Face returned no shards for {}",
        dataset.name()
    );
    let available_train = if matches!(dataset, Dataset::Climbmix | Dataset::Wiki) {
        remote.len() - 1
    } else {
        remote.len()
    };
    let train_count = if num_files < 0 {
        available_train
    } else {
        (num_files as usize).min(available_train)
    };
    ensure!(
        train_count > 0 || dataset == Dataset::Climbmix,
        "at least one training shard must be selected"
    );
    let mut jobs: Vec<(usize, String)> = remote
        .iter()
        .take(train_count)
        .cloned()
        .enumerate()
        .collect();
    if dataset == Dataset::Wiki {
        let (index, url) = (remote.len() - 1, remote.last().unwrap().clone());
        jobs.retain(|(job_index, _)| *job_index != index);
        jobs.push((index, url));
    }
    if dataset == Dataset::Climbmix {
        // ClimbMix's last shard is the fixed validation shard, even when fewer
        // training shards were requested. Python also permits zero train shards.
        let val = remote.last().unwrap().clone();
        jobs.push((6542, val));
    }
    Ok(jobs)
}

fn download_one(url: &str, destination: &Path) -> Result<()> {
    if destination.is_file() {
        return Ok(());
    }
    let client = Client::builder()
        .user_agent("nanochat")
        .timeout(Duration::from_secs(60 * 60))
        .build()?;
    let temp = destination.with_extension("parquet.tmp");
    let mut last_error = None;
    for attempt in 1..=5 {
        let result = (|| -> Result<()> {
            let mut response = client.get(url).send()?.error_for_status()?;
            let mut output = File::create(&temp)?;
            io::copy(&mut response, &mut output)?;
            output.sync_all()?;
            fs::rename(&temp, destination)?;
            Ok(())
        })();
        match result {
            Ok(()) => return Ok(()),
            Err(error) => {
                last_error = Some(error);
                let _ = fs::remove_file(&temp);
                if attempt < 5 {
                    thread::sleep(Duration::from_secs(1 << attempt));
                }
            }
        }
    }
    Err(last_error.unwrap()).with_context(|| format!("download shard {url}"))
}

pub fn local_files(directory: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let mut files = fs::read_dir(directory.as_ref())?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "parquet")
        })
        .collect::<Vec<_>>();
    files.sort();
    Ok(files)
}

/// Read the first Parquet row group from the selected train/validation split,
/// matching `parquets_iter_batched(...).__next__()` in the Python evaluator.
pub fn first_row_group_text(files: &[PathBuf], validation: bool) -> Result<Vec<String>> {
    let segments = split_segments(files, validation)?;
    let (_, path, start, stop) = segments
        .first()
        .context("dataset split has no Parquet files")?;
    ensure!(start < stop, "dataset split contains no row groups");
    let reader = SerializedFileReader::new(
        File::open(path).with_context(|| format!("open Parquet file {}", path.display()))?,
    )?;
    let row_group = reader.get_row_group(*start)?;
    row_group
        .get_row_iter(None)?
        .map(|row| {
            let row = row?;
            row.get_column_iter()
                .find_map(|(name, field)| match (name.as_str(), field) {
                    ("text", Field::Str(text)) => Some(Ok(text.clone())),
                    ("text", Field::Null) => {
                        Some(Err(anyhow::anyhow!("Parquet `text` field is null")))
                    }
                    ("text", _) => Some(Err(anyhow::anyhow!("Parquet `text` column is not UTF-8"))),
                    _ => None,
                })
                .context("Parquet row has no `text` column")?
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct DataLoaderState {
    pub pq_idx: usize,
    pub rg_idx: usize,
    pub epoch: usize,
}

fn split_segments(
    files: &[PathBuf],
    validation: bool,
) -> Result<Vec<(usize, PathBuf, usize, usize)>> {
    ensure!(!files.is_empty(), "no local Parquet shards were found");
    let selected: Vec<(usize, &PathBuf)> = if files.len() > 1 {
        if validation {
            // `get_parquet_splits(...)["val"]` is a one-file list in Python,
            // so its row-group cursor uses split-local index 0.
            vec![(0, &files[files.len() - 1])]
        } else {
            files[..files.len() - 1].iter().enumerate().collect()
        }
    } else {
        vec![(0, &files[0])]
    };
    let mut segments = Vec::with_capacity(selected.len());
    for (pq_idx, path) in selected {
        let reader = SerializedFileReader::new(File::open(path)?)?;
        let row_groups = reader.metadata().num_row_groups();
        if files.len() == 1 {
            ensure!(
                row_groups >= 2,
                "cannot split a single-shard dataset with fewer than two row groups"
            );
            let val_groups = ((row_groups as f64 * 0.05).ceil() as usize).max(1);
            let split = row_groups - val_groups;
            let (start, stop) = if validation {
                (split, row_groups)
            } else {
                (0, split)
            };
            segments.push((pq_idx, path.clone(), start, stop));
        } else {
            segments.push((pq_idx, path.clone(), 0, row_groups));
        }
    }
    Ok(segments)
}

/// Read one pass over the selected split, preserving file and row-group order.
pub fn read_split<F>(files: &[PathBuf], validation: bool, mut consume: F) -> Result<()>
where
    F: FnMut(String) -> Result<bool>,
{
    read_split_from(files, validation, None, 1, |text, _| consume(text)).map(|_| ())
}

pub fn read_split_from<F>(
    files: &[PathBuf],
    validation: bool,
    resume: Option<DataLoaderState>,
    epoch: usize,
    consume: F,
) -> Result<usize>
where
    F: FnMut(String, DataLoaderState) -> Result<bool>,
{
    read_split_sharded_from(files, validation, resume, epoch, 0, 1, consume)
}

pub fn read_split_sharded_from<F>(
    files: &[PathBuf],
    validation: bool,
    resume: Option<DataLoaderState>,
    epoch: usize,
    rank: usize,
    world_size: usize,
    mut consume: F,
) -> Result<usize>
where
    F: FnMut(String, DataLoaderState) -> Result<bool>,
{
    ensure!(
        world_size > 0 && rank < world_size,
        "invalid data shard rank/world size"
    );
    let segments = split_segments(files, validation)?;
    if let Some(state) = resume {
        ensure!(
            state.pq_idx < segments.len(),
            "data-loader shard index is outside the selected split"
        );
    }
    let shardable_groups = segments
        .iter()
        .map(|(_, _, start, stop)| stop - start)
        .sum::<usize>();
    ensure!(
        shardable_groups >= world_size,
        "Parquet split has {shardable_groups} row groups for {world_size} distributed ranks"
    );
    let epoch = resume.map_or(epoch.max(1), |state| state.epoch.max(1));
    let mut documents_read = 0usize;
    for (pq_idx, path, rg_start, rg_stop) in segments {
        if resume.is_some_and(|state| pq_idx < state.pq_idx) {
            continue;
        }
        let start = if let Some(state) = resume.filter(|state| pq_idx == state.pq_idx) {
            // nanochat stores the last row group it consumed, then advances
            // one full rank stride before adding this rank's offset. The
            // saved cursor can be rank-local or shared from rank 0; this
            // formula handles both layouts.
            let rank_stride = state
                .rg_idx
                .saturating_sub(rg_start)
                .checked_div(world_size)
                .context("invalid Parquet resume world size")?
                .saturating_add(1);
            rg_start
                .saturating_add(rank_stride.saturating_mul(world_size))
                .saturating_add(rank)
        } else {
            // Python restarts rank sharding at the beginning of each Parquet
            // file rather than carrying a cumulative offset across files.
            rg_start.saturating_add(rank)
        };
        let count = read_file_rows(
            &path,
            pq_idx,
            start,
            rg_start,
            rg_stop,
            epoch,
            rank,
            world_size,
            &mut consume,
        )?;
        documents_read += count;
    }
    Ok(documents_read)
}

/// Bounded document stream that reopens the selected split for each epoch.
pub struct ParquetTextStream {
    receiver: Receiver<std::result::Result<Vec<(String, DataLoaderState)>, String>>,
    state: Arc<Mutex<Option<DataLoaderState>>>,
}

impl ParquetTextStream {
    pub fn new_distributed(
        files: Vec<PathBuf>,
        validation: bool,
        resume: Option<DataLoaderState>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        ensure!(!files.is_empty(), "no local Parquet shards were found");
        ensure!(
            world_size > 0 && rank < world_size,
            "invalid data shard rank/world size"
        );
        let (sender, receiver) = mpsc::sync_channel(128);
        let state = Arc::new(Mutex::new(None));
        std::thread::Builder::new()
            .name("rust-agent-parquet-reader".to_owned())
            .spawn(move || {
                let mut resume_state = resume;
                let mut epoch = resume_state.map_or(1, |state| state.epoch.max(1));
                loop {
                    let mut batch = Vec::with_capacity(128);
                    let mut batch_row_group = None;
                    let result = read_split_sharded_from(
                        &files,
                        validation,
                        resume_state,
                        epoch,
                        rank,
                        world_size,
                        |text, cursor| {
                            let row_group = (cursor.pq_idx, cursor.rg_idx, cursor.epoch);
                            if !batch.is_empty() && batch_row_group != Some(row_group) {
                                sender.send(Ok(std::mem::take(&mut batch))).map_err(|_| {
                                    anyhow::anyhow!("Parquet document consumer stopped")
                                })?;
                            }
                            batch_row_group = Some(row_group);
                            batch.push((text, cursor));
                            if batch.len() == 128 {
                                sender.send(Ok(std::mem::take(&mut batch))).map_err(|_| {
                                    anyhow::anyhow!("Parquet document consumer stopped")
                                })?;
                            }
                            Ok(true)
                        },
                    );
                    match result {
                        Ok(0) if resume_state.is_some() => {
                            // A checkpoint at the final row group resumes at the next epoch.
                            resume_state = None;
                            epoch += 1;
                        }
                        Ok(0) => {
                            let _ = sender.send(Err(
                                "selected Parquet split contains no documents".to_owned(),
                            ));
                            break;
                        }
                        Ok(_) => {
                            if !batch.is_empty()
                                && sender.send(Ok(std::mem::take(&mut batch))).is_err()
                            {
                                break;
                            }
                            resume_state = None;
                            epoch += 1;
                        }
                        Err(error) => {
                            let _ =
                                sender.send(Err(format!("read Parquet training split: {error:#}")));
                            break;
                        }
                    }
                }
            })?;
        Ok(Self { receiver, state })
    }

    pub fn next_text_batch(&self) -> Result<Vec<String>> {
        match self.receiver.recv() {
            Ok(Ok(batch)) => {
                if let Some((_, state)) = batch.last() {
                    *self
                        .state
                        .lock()
                        .map_err(|_| anyhow::anyhow!("data-loader state lock is poisoned"))? =
                        Some(*state);
                }
                Ok(batch.into_iter().map(|(text, _)| text).collect())
            }
            Ok(Err(error)) => bail!(error),
            Err(_) => bail!("Parquet document stream ended unexpectedly"),
        }
    }

    pub fn state(&self) -> Option<DataLoaderState> {
        self.state.lock().ok().and_then(|state| *state)
    }
}

fn read_file_rows<F>(
    path: &Path,
    pq_idx: usize,
    start: usize,
    segment_start: usize,
    stop: usize,
    epoch: usize,
    rank: usize,
    world_size: usize,
    consume: &mut F,
) -> Result<usize>
where
    F: FnMut(String, DataLoaderState) -> Result<bool>,
{
    let reader = SerializedFileReader::new(
        File::open(path).with_context(|| format!("open Parquet file {}", path.display()))?,
    )?;
    let end = stop.min(reader.metadata().num_row_groups());
    let mut documents_read = 0usize;
    for index in start..end {
        if (index - segment_start) % world_size != rank {
            continue;
        }
        let row_group = reader.get_row_group(index)?;
        let rows = row_group.get_row_iter(None)?;
        for row in rows {
            let row = row?;
            let text = row
                .get_column_iter()
                .find_map(|(name, field)| match (name.as_str(), field) {
                    ("text", Field::Str(text)) => Some(Ok(text.clone())),
                    ("text", Field::Null) => {
                        Some(Err(anyhow::anyhow!("Parquet `text` field is null")))
                    }
                    ("text", _) => Some(Err(anyhow::anyhow!("Parquet `text` column is not UTF-8"))),
                    _ => None,
                })
                .context("Parquet row has no `text` column")??;
            documents_read += 1;
            if !consume(
                text,
                DataLoaderState {
                    pq_idx,
                    rg_idx: index,
                    epoch,
                },
            )? {
                return Ok(documents_read);
            }
        }
    }
    Ok(documents_read)
}
