//! Local text/JSONL corpus reader used by the first Rust training path.

use std::{
    collections::VecDeque,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        mpsc::{self, Receiver, SyncSender},
    },
};

use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::tokenizer::BpeTokenizer;

#[derive(Debug)]
pub struct TokenCorpus {
    pub ids: Vec<u32>,
    /// Whether each token contributes to the next-token loss.
    pub loss_mask: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenBatch {
    pub inputs: Vec<u32>,
    pub valid_positions: Vec<u32>,
    pub valid_targets: Vec<u32>,
}

pub trait BatchSource: Send {
    fn next_batch(&mut self) -> Result<TokenBatch>;

    fn prefetch_batch(&mut self) -> Result<()> {
        Ok(())
    }

    fn training_progress(&self) -> Option<TrainingProgress> {
        None
    }

    fn dataloader_state(&self) -> Option<crate::datasets::DataLoaderState> {
        None
    }

    fn jsonl_state(&self) -> Option<JsonlResume> {
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrainingProgress {
    pub fraction: f64,
    pub epoch: usize,
}

/// Bounded BOS best-fit packer used by nanochat's base training and BPB eval.
pub struct ParquetBatchSource<'a> {
    stream: crate::datasets::ParquetTextStream,
    tokenizer: &'a BpeTokenizer,
    documents: VecDeque<Vec<u32>>,
    batch_size: usize,
    sequence_len: usize,
    bos_id: u32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct JsonlState {
    pub line_idx: usize,
    pub epoch: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub next_epoch: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonlDocument {
    ids: Vec<u32>,
    mask: Vec<u8>,
    #[serde(default = "default_epoch")]
    epoch: usize,
}

fn default_epoch() -> usize {
    1
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct JsonlResume {
    cursor: JsonlState,
    buffered: Vec<JsonlDocument>,
    #[serde(default)]
    consumed_documents: usize,
    #[serde(default = "default_epoch")]
    consumed_epoch: usize,
    #[serde(default)]
    pending_batch: Option<TokenBatch>,
    #[serde(default)]
    batch_size: Option<usize>,
    #[serde(default)]
    sequence_len: Option<usize>,
    #[serde(default)]
    rank: Option<usize>,
    #[serde(default)]
    world_size: Option<usize>,
    #[serde(default)]
    parse_jsonl: Option<bool>,
}

impl JsonlResume {
    fn validate(
        &self,
        batch_size: usize,
        sequence_len: usize,
        rank: usize,
        world_size: usize,
        parse_jsonl: bool,
        vocab_size: u32,
    ) -> Result<()> {
        ensure!(
            self.batch_size.is_none_or(|value| value == batch_size),
            "JSONL resume batch size differs from checkpoint"
        );
        ensure!(
            self.sequence_len.is_none_or(|value| value == sequence_len),
            "JSONL resume sequence length differs from checkpoint"
        );
        ensure!(
            self.rank.is_none_or(|value| value == rank)
                && self.world_size.is_none_or(|value| value == world_size),
            "JSONL resume rank or world size differs from checkpoint"
        );
        ensure!(
            self.parse_jsonl.is_none_or(|value| value == parse_jsonl),
            "JSONL resume data format differs from checkpoint"
        );
        if let Some(batch) = &self.pending_batch {
            let expected = batch_size
                .checked_mul(sequence_len)
                .context("JSONL resume batch size overflow")?;
            ensure!(
                batch.inputs.len() == expected,
                "JSONL resume pending batch dimensions differ from current settings"
            );
            ensure!(
                batch.valid_positions.len() == batch.valid_targets.len()
                    && !batch.valid_targets.is_empty(),
                "JSONL resume pending batch has invalid targets"
            );
            ensure!(
                batch
                    .valid_positions
                    .iter()
                    .all(|&position| (position as usize) < expected),
                "JSONL resume pending batch has an invalid target position"
            );
            ensure!(
                batch
                    .inputs
                    .iter()
                    .chain(batch.valid_targets.iter())
                    .all(|&token| token < vocab_size),
                "JSONL resume pending batch has a token outside the vocabulary"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod resume_tests {
    use super::{JsonlResume, advance_jsonl_cursor, logical_record_index, spawn_jsonl_reader};

    #[test]
    fn legacy_jsonl_resume_without_pending_batch_still_loads() {
        let state: JsonlResume =
            serde_json::from_str(r#"{"cursor":{"line_idx":3,"epoch":1},"buffered":[]}"#).unwrap();
        assert!(state.pending_batch.is_none());
        assert_eq!(state.cursor.line_idx, 3);
        state.validate(1, 8, 0, 1, true, 100).unwrap();
    }

    #[test]
    fn resume_rejects_changed_layout_and_invalid_pending_batch() {
        let state: JsonlResume = serde_json::from_str(r#"{"cursor":{"line_idx":1,"epoch":1},"buffered":[],"batch_size":1,"sequence_len":2,"rank":0,"world_size":2,"parse_jsonl":true,"pending_batch":{"inputs":[1,2],"valid_positions":[0],"valid_targets":[3]}}"#).unwrap();
        state.validate(1, 2, 0, 2, true, 4).unwrap();
        assert!(state.validate(1, 2, 0, 1, true, 4).is_err());
        assert!(state.validate(1, 2, 0, 2, true, 3).is_err());
        assert!(state.validate(2, 2, 0, 2, true, 4).is_err());
    }

    #[test]
    fn sharded_cursor_wraps_across_logical_records_and_blank_lines() {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        use std::io::Write;
        write!(file, "a\n\nb\nc\n\nd\ne\n").unwrap();
        assert_eq!(logical_record_index(file.path(), 4).unwrap(), 2);
        assert!(logical_record_index(file.path(), 5).is_err());
        let mut cursor = 0;
        let mut seen = Vec::new();
        for _ in 0..6 {
            seen.push(cursor);
            cursor = advance_jsonl_cursor(cursor, 2, 5).0;
        }
        assert_eq!(seen, [0, 2, 4, 1, 3, 0]);
        let reader = spawn_jsonl_reader(file.path().to_owned(), 5, 0, 1, 2).unwrap();
        let first: Vec<_> = (0..3).map(|_| reader.recv().unwrap().unwrap()).collect();
        assert_eq!(
            first
                .iter()
                .map(|(line, _)| line.as_str())
                .collect::<Vec<_>>(),
            ["a", "c", "e"]
        );
        let saved = first.last().unwrap().1;
        assert_eq!(saved.next_epoch, Some(2));
        drop(reader);
        let saved_index = logical_record_index(file.path(), saved.line_idx).unwrap();
        let (next, wrapped) = advance_jsonl_cursor(saved_index, 2, 5);
        let resumed = spawn_jsonl_reader(
            file.path().to_owned(),
            5,
            next,
            saved.epoch + usize::from(wrapped),
            2,
        )
        .unwrap();
        let next_rows: Vec<_> = (0..3).map(|_| resumed.recv().unwrap().unwrap()).collect();
        assert_eq!(
            next_rows
                .iter()
                .map(|(line, _)| line.as_str())
                .collect::<Vec<_>>(),
            ["b", "d", "a"]
        );
        assert_eq!(
            next_rows
                .iter()
                .map(|(_, state)| state.epoch)
                .collect::<Vec<_>>(),
            [2, 2, 3]
        );
    }
}

pub struct JsonlBatchSource<'a> {
    receiver: Receiver<std::result::Result<(String, JsonlState), String>>,
    state: Arc<Mutex<Option<JsonlState>>>,
    tokenizer: &'a BpeTokenizer,
    conversations: VecDeque<JsonlDocument>,
    batch_size: usize,
    sequence_len: usize,
    bos_id: u32,
    parse_jsonl: bool,
    consumed_documents: usize,
    consumed_epoch: usize,
    dataset_size: usize,
    initial_progress: usize,
    world_size: usize,
    pending_batch: Option<TokenBatch>,
}

impl<'a> JsonlBatchSource<'a> {
    pub fn new(
        path: impl AsRef<Path>,
        tokenizer: &'a BpeTokenizer,
        batch_size: usize,
        sequence_len: usize,
        resume: Option<JsonlResume>,
        parse_jsonl: bool,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        ensure!(
            batch_size > 0 && sequence_len > 1,
            "batch dimensions must be positive"
        );
        let path = path.as_ref().to_owned();
        ensure!(
            path.is_file(),
            "JSONL dataset is not a file: {}",
            path.display()
        );
        ensure!(
            world_size > 0 && rank < world_size,
            "invalid data shard rank/world size"
        );
        let dataset_size = BufReader::new(File::open(&path)?).lines().try_fold(
            0usize,
            |count, line| -> Result<usize> {
                let line = line?;
                Ok(count + usize::from(!line.trim().is_empty()))
            },
        )?;
        ensure!(dataset_size > 0, "JSONL dataset contains no usable records");
        ensure!(
            rank < dataset_size,
            "JSONL dataset has fewer records than distributed ranks"
        );
        if let Some(saved) = &resume {
            saved.validate(
                batch_size,
                sequence_len,
                rank,
                world_size,
                parse_jsonl,
                tokenizer.vocab_size(),
            )?;
        }
        let (resume_cursor, buffered, consumed_documents, consumed_epoch, pending_batch) = resume
            .map(|saved| {
                (
                    Some(saved.cursor),
                    saved.buffered,
                    saved.consumed_documents,
                    saved.consumed_epoch,
                    saved.pending_batch,
                )
            })
            .unwrap_or_else(|| (None, Vec::new(), 0, 1, None));
        let (start_index, start_epoch) = if let Some(saved) = resume_cursor {
            let saved_index = logical_record_index(&path, saved.line_idx)?;
            let (next, wrapped) = advance_jsonl_cursor(saved_index, world_size, dataset_size);
            (next, saved.epoch.max(1) + usize::from(wrapped))
        } else {
            (rank, 1)
        };
        let receiver =
            spawn_jsonl_reader(path, dataset_size, start_index, start_epoch, world_size)?;
        let state = Arc::new(Mutex::new(resume_cursor));
        Ok(Self {
            receiver,
            state,
            tokenizer,
            conversations: if buffered.is_empty() {
                VecDeque::with_capacity(128)
            } else {
                buffered.into()
            },
            batch_size,
            sequence_len,
            bos_id: tokenizer.bos_id(),
            parse_jsonl,
            consumed_documents,
            consumed_epoch,
            dataset_size,
            initial_progress: rank,
            world_size,
            pending_batch,
        })
    }

    fn refill(&mut self) -> Result<()> {
        while self.conversations.len() < 100 {
            match self.receiver.recv() {
                Ok(Ok((line, state))) => {
                    *self
                        .state
                        .lock()
                        .map_err(|_| anyhow::anyhow!("JSONL cursor lock is poisoned"))? =
                        Some(state);
                    let (ids, mask) = if self.parse_jsonl {
                        encode_training_record(&line, self.tokenizer, self.sequence_len)?
                    } else {
                        let mut ids = vec![self.bos_id];
                        ids.extend(self.tokenizer.encode(&line)?);
                        ids.truncate(self.sequence_len);
                        let mask = vec![1; ids.len()];
                        (ids, mask)
                    };
                    self.conversations.push_back(JsonlDocument {
                        ids,
                        mask,
                        epoch: state.epoch,
                    });
                }
                Ok(Err(error)) => anyhow::bail!(error),
                Err(_) => anyhow::bail!("JSONL document stream ended unexpectedly"),
            }
        }
        Ok(())
    }
}

fn spawn_jsonl_reader(
    path: PathBuf,
    dataset_size: usize,
    start_index: usize,
    start_epoch: usize,
    world_size: usize,
) -> Result<Receiver<std::result::Result<(String, JsonlState), String>>> {
    let (sender, receiver) = mpsc::sync_channel(1);
    std::thread::Builder::new()
            .name("rust-agent-jsonl-reader".to_owned())
            .spawn(move || {
                let mut next_index = start_index;
                let mut epoch = start_epoch;
                loop {
                    let result = (|| -> Result<()> {
                        let file = File::open(&path)
                            .with_context(|| format!("open JSONL dataset {}", path.display()))?;
                        let mut record_index = 0usize;
                        for (line_idx, line) in BufReader::new(file).lines().enumerate() {
                            let line_idx = line_idx + 1;
                            let line =
                                line.with_context(|| format!("read JSONL line {line_idx}"))?;
                            if line.trim().is_empty() {
                                continue;
                            }
                            let current = record_index;
                            record_index += 1;
                            if current != next_index { continue; }
                            let (next, wrapped) = advance_jsonl_cursor(current, world_size, dataset_size);
                            let cursor = JsonlState { line_idx, epoch, next_epoch: Some(epoch + usize::from(wrapped)) };
                            sender
                                .send(Ok((line, cursor)))
                                .map_err(|_| anyhow::anyhow!("JSONL batch consumer stopped"))?;
                            next_index = next;
                            if wrapped {
                                epoch += 1;
                                return Ok(());
                            }
                        }
                        anyhow::bail!("JSONL dataset changed while reading or no longer contains the expected record")
                    })();
                    match result {
                        Ok(()) => {}
                        Err(error) => {
                            let _ =
                                sender.send(Err(format!("read JSONL training data: {error:#}")));
                            break;
                        }
                    }
                }
            })?;
    Ok(receiver)
}

fn advance_jsonl_cursor(index: usize, world_size: usize, dataset_size: usize) -> (usize, bool) {
    let next = index + world_size;
    (next % dataset_size, next >= dataset_size)
}

fn logical_record_index(path: &Path, line_number: usize) -> Result<usize> {
    let mut index = 0usize;
    for (line_idx, line) in BufReader::new(File::open(path)?).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if line_idx + 1 == line_number {
            return Ok(index);
        }
        index += 1;
    }
    anyhow::bail!(
        "JSONL resume cursor does not identify a record in {}",
        path.display()
    )
}

impl BatchSource for JsonlBatchSource<'_> {
    fn next_batch(&mut self) -> Result<TokenBatch> {
        if let Some(batch) = self.pending_batch.take() {
            return Ok(batch);
        }
        let time = self.sequence_len;
        let row_capacity = time + 1;
        let mut inputs = Vec::with_capacity(self.batch_size * time);
        let mut valid_positions = Vec::new();
        let mut valid_targets = Vec::new();
        for row_index in 0..self.batch_size {
            let mut row_ids = Vec::with_capacity(row_capacity);
            let mut row_mask = Vec::with_capacity(row_capacity);
            while row_ids.len() < row_capacity {
                self.refill()?;
                let remaining = row_capacity - row_ids.len();
                let mut best = None;
                let mut best_len = 0usize;
                for (index, doc) in self.conversations.iter().enumerate() {
                    if doc.ids.len() <= remaining && doc.ids.len() > best_len {
                        best = Some((index, doc.ids.len()));
                        best_len = doc.ids.len();
                    }
                }
                if let Some((index, len)) = best {
                    let document = self
                        .conversations
                        .remove(index)
                        .expect("selected conversation");
                    self.consumed_documents += 1;
                    self.consumed_epoch = document.epoch;
                    row_ids.extend(document.ids.into_iter().take(len));
                    row_mask.extend(document.mask.into_iter().take(len));
                } else {
                    row_ids.extend(std::iter::repeat_n(self.bos_id, remaining));
                    row_mask.extend(std::iter::repeat_n(0, remaining));
                }
            }
            ensure!(
                row_ids.len() == row_capacity && row_mask.len() == row_capacity,
                "SFT packer produced an incomplete row"
            );
            inputs.extend_from_slice(&row_ids[..time]);
            for position in 0..time {
                if row_mask[position + 1] != 0 {
                    valid_positions.push((row_index * time + position) as u32);
                    valid_targets.push(row_ids[position + 1]);
                }
            }
        }
        ensure!(
            !valid_positions.is_empty(),
            "batch contains no supervised assistant tokens"
        );
        Ok(TokenBatch {
            inputs,
            valid_positions,
            valid_targets,
        })
    }

    fn jsonl_state(&self) -> Option<JsonlResume> {
        Some(JsonlResume {
            cursor: self.state.lock().ok().and_then(|state| *state)?,
            buffered: self.conversations.iter().cloned().collect(),
            consumed_documents: self.consumed_documents,
            consumed_epoch: self.consumed_epoch,
            pending_batch: self.pending_batch.clone(),
            batch_size: Some(self.batch_size),
            sequence_len: Some(self.sequence_len),
            rank: Some(self.initial_progress),
            world_size: Some(self.world_size),
            parse_jsonl: Some(self.parse_jsonl),
        })
    }

    fn prefetch_batch(&mut self) -> Result<()> {
        if self.pending_batch.is_none() {
            self.pending_batch = Some(self.next_batch()?);
        }
        Ok(())
    }

    fn training_progress(&self) -> Option<TrainingProgress> {
        let numerator = self.initial_progress + self.consumed_documents * self.world_size;
        let reader_epoch = self
            .state
            .lock()
            .ok()
            .and_then(|state| (*state).map(|state| state.next_epoch.unwrap_or(state.epoch)))
            .unwrap_or(self.consumed_epoch);
        Some(TrainingProgress {
            fraction: numerator as f64 / self.dataset_size as f64,
            epoch: reader_epoch,
        })
    }
}

fn encode_training_record(
    line: &str,
    tokenizer: &BpeTokenizer,
    sequence_len: usize,
) -> Result<(Vec<u32>, Vec<u8>)> {
    if let Ok(record) = serde_json::from_str::<Value>(line) {
        if record.get("messages").is_some() {
            return tokenizer.render_conversation(&record, sequence_len);
        }
        if let Some(text) = record.get("text").and_then(Value::as_str) {
            let mut ids = vec![tokenizer.bos_id()];
            ids.extend(tokenizer.encode(text)?);
            ids.truncate(sequence_len);
            let mask = vec![1; ids.len()];
            return Ok((ids, mask));
        }
    }
    let mut ids = vec![tokenizer.bos_id()];
    ids.extend(tokenizer.encode(line)?);
    ids.truncate(sequence_len);
    let mask = vec![1; ids.len()];
    Ok((ids, mask))
}

impl<'a> ParquetBatchSource<'a> {
    pub fn new_distributed(
        files: Vec<PathBuf>,
        tokenizer: &'a BpeTokenizer,
        batch_size: usize,
        sequence_len: usize,
        validation: bool,
        resume: Option<crate::datasets::DataLoaderState>,
        rank: usize,
        world_size: usize,
    ) -> Result<Self> {
        ensure!(
            batch_size > 0 && sequence_len > 0,
            "batch dimensions must be positive"
        );
        Ok(Self {
            stream: crate::datasets::ParquetTextStream::new_distributed(
                files, validation, resume, rank, world_size,
            )?,
            tokenizer,
            documents: VecDeque::with_capacity(1024),
            batch_size,
            sequence_len,
            bos_id: tokenizer.bos_id(),
        })
    }

    fn refill(&mut self) -> Result<()> {
        while self.documents.len() < 1000 {
            for text in self.stream.next_text_batch()? {
                let mut tokens = Vec::with_capacity(text.len().min(self.sequence_len * 2) + 1);
                tokens.push(self.bos_id);
                tokens.extend(self.tokenizer.encode(&text)?);
                self.documents.push_back(tokens);
            }
        }
        Ok(())
    }
}

impl BatchSource for ParquetBatchSource<'_> {
    fn next_batch(&mut self) -> Result<TokenBatch> {
        let time = self.sequence_len;
        let row_capacity = time + 1;
        let mut inputs = Vec::with_capacity(self.batch_size * time);
        let mut valid_positions = Vec::with_capacity(self.batch_size * time);
        let mut valid_targets = Vec::with_capacity(self.batch_size * time);
        for row_index in 0..self.batch_size {
            let mut row = Vec::with_capacity(row_capacity);
            while row.len() < row_capacity {
                self.refill()?;
                let remaining = row_capacity - row.len();
                let mut best = None;
                let mut best_len = 0usize;
                for (index, document) in self.documents.iter().enumerate() {
                    if document.len() <= remaining && document.len() > best_len {
                        best = Some((index, document.len()));
                        best_len = document.len();
                    }
                }
                if let Some((index, length)) = best {
                    row.extend(
                        self.documents
                            .remove(index)
                            .expect("selected buffered document")
                            .into_iter()
                            .take(length),
                    );
                } else {
                    let (index, length) = self
                        .documents
                        .iter()
                        .enumerate()
                        .min_by_key(|(_, document)| document.len())
                        .map(|(index, document)| (index, document.len()))
                        .context("no documents available to fill training row")?;
                    let document = self
                        .documents
                        .remove(index)
                        .expect("selected shortest document");
                    row.extend(document.into_iter().take(remaining.min(length)));
                }
            }
            ensure!(
                row.len() == row_capacity,
                "best-fit packer produced an incomplete training row"
            );
            inputs.extend_from_slice(&row[..time]);
            for position in 0..time {
                valid_positions.push((row_index * time + position) as u32);
                valid_targets.push(row[position + 1]);
            }
        }
        Ok(TokenBatch {
            inputs,
            valid_positions,
            valid_targets,
        })
    }

    fn dataloader_state(&self) -> Option<crate::datasets::DataLoaderState> {
        self.stream.state()
    }
}

/// Bounded, single-pass document stream for native tokenizer training.
pub struct TextDocumentStream {
    receiver: Receiver<std::result::Result<String, String>>,
}

impl Iterator for TextDocumentStream {
    type Item = Result<String>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.receiver.recv() {
            Ok(Ok(text)) => Some(Ok(text)),
            Ok(Err(error)) => Some(Err(anyhow::anyhow!(error))),
            Err(_) => None,
        }
    }
}

/// Stream tokenizer documents in the same order as nanochat's training CLI.
/// JSONL `text` records are unwrapped; conversation records contribute their
/// visible text and tool-code parts in message order.
pub fn text_documents_from_paths(
    paths: Vec<PathBuf>,
    doc_cap: usize,
    max_chars: usize,
) -> Result<TextDocumentStream> {
    ensure!(
        !paths.is_empty(),
        "at least one tokenizer dataset is required"
    );
    let (sender, receiver) = mpsc::sync_channel(128);
    std::thread::Builder::new()
        .name("rust-agent-tokenizer-reader".to_owned())
        .spawn(move || {
            if let Err(error) = stream_text_documents(&paths, doc_cap, max_chars, &sender) {
                let _ = sender.send(Err(format!("read tokenizer training data: {error:#}")));
            }
        })?;
    Ok(TextDocumentStream { receiver })
}

fn stream_text_documents(
    paths: &[PathBuf],
    doc_cap: usize,
    max_chars: usize,
    sender: &SyncSender<std::result::Result<String, String>>,
) -> Result<()> {
    let mut total_chars = 0usize;
    let mut document_count = 0usize;
    let mut emit = |text: String| -> Result<bool> {
        let document = text.chars().take(doc_cap).collect::<String>();
        total_chars += document.chars().count();
        document_count += 1;
        sender
            .send(Ok(document))
            .map_err(|_| anyhow::anyhow!("tokenizer document consumer stopped"))?;
        Ok(total_chars <= max_chars)
    };

    let mut exhausted = false;
    for path in paths {
        if path.is_dir()
            || path
                .extension()
                .is_some_and(|extension| extension == "parquet")
        {
            let files = if path.is_dir() {
                crate::datasets::local_files(path)?
            } else {
                vec![path.to_owned()]
            };
            crate::datasets::read_split(&files, false, |text| {
                let keep_reading = emit(text)?;
                if !keep_reading {
                    exhausted = true;
                }
                Ok(keep_reading)
            })?;
        } else {
            let file =
                File::open(path).with_context(|| format!("open dataset {}", path.display()))?;
            for (line_number, line) in BufReader::new(file).lines().enumerate() {
                let line = line.with_context(|| format!("read line {}", line_number + 1))?;
                if line.trim().is_empty() {
                    continue;
                }
                let document = match serde_json::from_str::<Value>(&line) {
                    Ok(record) => {
                        if let Some(text) = record.get("text").and_then(Value::as_str) {
                            text.to_owned()
                        } else if let Some(messages) =
                            record.get("messages").and_then(Value::as_array)
                        {
                            messages
                                .iter()
                                .filter_map(|message| message.get("content"))
                                .map(content_text)
                                .collect::<Vec<_>>()
                                .join("\n\n")
                        } else {
                            line
                        }
                    }
                    Err(_) => line,
                };
                if !emit(document)? {
                    exhausted = true;
                    break;
                }
            }
        }
        if exhausted {
            break;
        }
    }

    ensure!(document_count > 0, "dataset contains no documents");
    Ok(())
}

fn content_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join(""),
        _ => String::new(),
    }
}

impl TokenCorpus {
    pub fn from_path_split(
        path: impl AsRef<Path>,
        tokenizer: &BpeTokenizer,
        validation: bool,
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir()
            || path
                .extension()
                .is_some_and(|extension| extension == "parquet")
        {
            let files = if path.is_dir() {
                crate::datasets::local_files(path)?
            } else {
                vec![path.to_owned()]
            };
            let mut ids = Vec::new();
            let mut loss_mask = Vec::new();
            crate::datasets::read_split(&files, validation, |text| {
                append_document(&mut ids, &mut loss_mask, tokenizer, &text)?;
                Ok(true)
            })?;
            ensure!(ids.len() >= 2, "dataset contains fewer than two tokens");
            return Ok(Self { ids, loss_mask });
        }
        let file = File::open(path).with_context(|| format!("open dataset {}", path.display()))?;
        let mut ids = Vec::new();
        let mut loss_mask = Vec::new();
        for (line_number, line) in BufReader::new(file).lines().enumerate() {
            let line = line.with_context(|| format!("read line {}", line_number + 1))?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(record) = serde_json::from_str::<Value>(&line) {
                if record.get("messages").is_some() {
                    let (tokens, mask) = tokenizer
                        .render_conversation(&record, usize::MAX)
                        .with_context(|| {
                            format!("render conversation on line {}", line_number + 1)
                        })?;
                    ids.extend(tokens);
                    loss_mask.extend(mask);
                    continue;
                }
                if let Some(text) = record.get("text").and_then(Value::as_str) {
                    append_document(&mut ids, &mut loss_mask, tokenizer, text)?;
                    continue;
                }
            }
            append_document(&mut ids, &mut loss_mask, tokenizer, &line)?;
        }
        ensure!(ids.len() >= 2, "dataset contains fewer than two tokens");
        ensure!(
            ids.len() == loss_mask.len(),
            "corpus IDs and loss mask differ in length"
        );
        Ok(Self { ids, loss_mask })
    }
}

fn append_document(
    ids: &mut Vec<u32>,
    loss_mask: &mut Vec<u8>,
    tokenizer: &BpeTokenizer,
    text: &str,
) -> Result<()> {
    ids.push(tokenizer.bos_id());
    loss_mask.push(1);
    let encoded = tokenizer.encode(text)?;
    loss_mask.extend(std::iter::repeat_n(1, encoded.len()));
    ids.extend(encoded);
    Ok(())
}
