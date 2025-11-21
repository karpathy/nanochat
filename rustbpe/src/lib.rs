use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation
#[pyclass]
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting
    pub pattern: String,
    /// Compiled regex for efficiency
    compiled_pattern: Regex,
}

// ------------------------ internal helpers ------------------------

#[derive(Clone, Debug)]
struct Word {
    ids: Vec<u32>,
}

impl Word {
    #[inline]
    fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    #[inline]
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }

    /// Merge all non-overlapping occurrences of pair -> new_id.
    /// Returns a small Vec of local pair-count deltas for THIS word only:
    ///   -1 for removed pairs, +1 for newly created pairs.
    ///
    /// NOTE: this version deliberately avoids a HashMap in the hot loop.
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        if n < 2 {
            return Vec::new();
        }

        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                let right = if i + 2 < n { Some(self.ids[i + 2]) } else { None };

                // remove old pairs
                if let Some(x) = left {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right {
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }

                // write merged token
                out.push(new_id);
                i += 2; // skip 'a' and 'b'
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }

        self.ids = out;
        deltas
    }
}

#[derive(Debug, Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    /// set of word indices where this pair may occur and needs processing
    pos: AHashSet<usize>,
}

impl PartialEq for MergeJob {
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by count; tie-break to ascending pair order (deterministic)
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else {
            // ascending order on the pair when counts tie
            other.pair.cmp(&self.pair)
        }
    }
}

#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    words
        .par_iter()
        .enumerate()
        .map(|(i, w)| {
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 {
                for (a, b) in w.pairs() {
                    *local_pc.entry((a, b)).or_default() += counts[i];
                    local_wtu.entry((a, b)).or_default().insert(i);
                }
            }
            (local_pc, local_wtu)
        })
        .reduce(
            || (AHashMap::new(), AHashMap::new()),
            |(mut acc_pc, mut acc_wtu), (pc, wtu)| {
                for (k, v) in pc {
                    *acc_pc.entry(k).or_default() += v;
                }
                for (k, s) in wtu {
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}

// ------------------------ END helpers ------------------------

impl Tokenizer {

    /// Core incremental BPE training given unique words and their counts.
    /// `words`: one entry per unique chunk (Vec<u32> of token-ids/bytes).
    /// `counts`: same length as `words`, count per chunk.
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        let num_merges = vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);
        self.merges.clear();

        // ---- Initial pair_counts and where_to_update (parallel) ----
        log::info!("Computing initial pair counts from {} unique sequences", words.len());
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);

        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
        for (pair, pos) in where_to_update.drain() {
            let c = *pair_counts.get(&pair).unwrap_or(&0);
            if c > 0 {
                heap.push(MergeJob {
                    pair,
                    count: c as u64,
                    pos,
                });
            }
        }

        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            let Some(mut top) = heap.pop() else { break; };

            // Lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 {
                top.count = current as u64;
                if top.count > 0 {
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break;
            }

            // Record merge
            let new_id = 256 + merges_done;
            self.merges.insert(top.pair, new_id);

            // Merge this pair in all words where it occurs
            let mut local_pos_updates: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos {
                // Apply merge to this word and collect pair-count deltas
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                // Update global pair counts based on this word's count
                for (pair, delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            // Add the updated pair counts back to the heap
            for (pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob {
                        pair,
                        count: cnt as u64,
                        pos,
                    });
                }
            }

            merges_done += 1;

            // Log progress every 1%
            let current_percent = (merges_done * 100) / num_merges;
            if current_percent > last_log_percent {
                log::info!(
                    "Progress: {}% ({}/{} merges) - Last merge: {:?} -> {} (frequency: {})",
                    current_percent, merges_done, num_merges, top.pair, new_id, top.count
                );
                last_log_percent = current_percent;
            }
        }

        log::info!("Finished training: {} merges completed", merges_done);
    }
}

/// Public methods for the Tokenizer class that will be exposed to Python.
#[pymethods]
impl Tokenizer {
    /// Create a new Tokenizer
    #[new]
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").expect("Empty regex should be valid"),
        }
    }

    /// Train from a streaming iterator (parallel ingestion).
    /// We refill a Rust Vec<String> buffer under the GIL, then release the GIL
    /// to do the heavy splitting and counting **in parallel** with rayon.
    #[pyo3(signature = (iterator, vocab_size, buffer_size=8192, pattern=None))]
    #[pyo3(text_signature = "(self, iterator, vocab_size, buffer_size=8192, pattern=None)")]
    pub fn train_from_iterator(
        &mut self,
        py: pyo3::Python<'_>,
        iterator: &pyo3::Bound<'_, pyo3::PyAny>,
        vocab_size: u32,
        buffer_size: usize,
        pattern: Option<String>,
    ) -> PyResult<()> {
        // Use provided pattern or default to GPT-4 pattern
        let pattern_str = pattern.unwrap_or_else(|| GPT4_PATTERN.to_string());

        // Update the stored pattern and compile it
        self.pattern = pattern_str.clone();
        self.compiled_pattern = Regex::new(&pattern_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid regex pattern: {}", e)))?;

        // Prepare a true Python iterator object
        let py_iter: pyo3::Py<pyo3::PyAny> = unsafe {
            pyo3::Py::from_owned_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
        };

        // Global chunk counts
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();

        // Temporary buffer we refill under the GIL
        let mut buf: Vec<String> = Vec::with_capacity(buffer_size);

        log::info!("Processing sequences from iterator (buffer_size: {})", buffer_size);
        let mut total_sequences = 0u64;

        // Helper: refill `buf` with up to `buffer_size` strings from the Python iterator.
        // Returns Ok(true) if the iterator is exhausted, Ok(false) otherwise.
        let refill = |buf: &mut Vec<String>| -> PyResult<bool> {
            pyo3::Python::with_gil(|py| {
                buf.clear();
                let it = py_iter.bind(py);
                loop {
                    if buf.len() >= buffer_size {
                        return Ok(false);
                    }
                    // next(it)
                    let next_obj = unsafe {
                        pyo3::Bound::from_owned_ptr_or_opt(py, pyo3::ffi::PyIter_Next(it.as_ptr()))
                    };
                    match next_obj {
                        Some(obj) => {
                            let s: String = obj.extract()?;
                            buf.push(s);
                        }
                        None => {
                            if pyo3::PyErr::occurred(py) {
                                return Err(pyo3::PyErr::fetch(py));
                            } else {
                                return Ok(true); // exhausted
                            }
                        }
                    }
                }
            })
        };

        // Stream ingestion loop: refill under GIL, process without GIL (parallel)
        loop {
            let exhausted = refill(&mut buf)?;
            if buf.is_empty() && exhausted {
                break;
            }

            total_sequences += buf.len() as u64;

            let pattern = self.compiled_pattern.clone();
            let local: AHashMap<CompactString, i32> = py.allow_threads(|| {
                buf.par_iter()
                    .map(|s| {
                        let mut m: AHashMap<CompactString, i32> = AHashMap::new();
                        for mat in pattern.find_iter(s) {
                            let piece = mat.expect("regex match failed").as_str();
                            *m.entry(CompactString::from(piece)).or_default() += 1;
                        }
                        m
                    })
                    .reduce(
                        || AHashMap::new(),
                        |mut a, b| {
                            for (k, v) in b {
                                *a.entry(k).or_default() += v;
                            }
                            a
                        },
                    )
            });

            // Merge local into global (single-threaded)
            for (k, v) in local {
                *counts.entry(k).or_default() += v;
            }

            if exhausted {
                break;
            }
        }
        log::info!("Processed {} sequences total, {} unique", total_sequences, counts.len());

        // Materialize words & counts
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }

        self.train_core_incremental(words, cvec, vocab_size);
        Ok(())
    }

    /// Return the regex pattern
    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    /// Return the mergeable ranks (token bytes -> token id / rank)
    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();

        // Build vocabulary incrementally from low to high token IDs
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|i| vec![i as u8]).collect();

        for (i, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), i as u32));
        }

        // Sort merges by token id (so we can reconstruct bytes progressively)
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);

        for (&pair, &merged_id) in sorted_merges {
            let (left, right) = pair;
            let mut merged_bytes = token_bytes[left as usize].clone();
            merged_bytes.extend(&token_bytes[right as usize]);

            if token_bytes.len() <= merged_id as usize {
                token_bytes.resize(merged_id as usize + 1, Vec::new());
            }
            token_bytes[merged_id as usize] = merged_bytes.clone();

            mergeable_ranks.push((merged_bytes, merged_id));
        }

        mergeable_ranks
    }

    /// Encode a string into token IDs
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();

        // Split text using the regex pattern
        for m in self.compiled_pattern.find_iter(text) {
            let chunk = m.expect("regex match failed").as_str();

            // Convert chunk to bytes then to u32 IDs
            let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();

            // Apply merges iteratively
            while ids.len() >= 2 {
                // Find the best pair to merge
                let mut best_pair: Option<(usize, Pair, u32)> = None;

                for i in 0..ids.len() - 1 {
                    let pair: Pair = (ids[i], ids[i + 1]);
                    if let Some(&new_id) = self.merges.get(&pair) {
                        if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                            best_pair = Some((i, pair, new_id));
                        }
                    }
                }

                // If we found a pair to merge, apply it
                if let Some((idx, _pair, new_id)) = best_pair {
                    ids[idx] = new_id;
                    ids.remove(idx + 1);
                } else {
                    // No more merges possible
                    break;
                }
            }

            all_ids.extend(ids);
        }

        all_ids
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}
