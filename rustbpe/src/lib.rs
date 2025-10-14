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
            pyo3::Bound::from_borrowed_ptr_or_err(py, pyo3::ffi::PyObject_GetIter(iterator.as_ptr()))?
                .into()
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

// ------------------------ Rust Unit Tests ------------------------

#[cfg(test)]
mod tests {
    use super::*;


    /// Helper function to create a simple tokenizer for testing
    fn create_test_tokenizer() -> Tokenizer {
        Tokenizer {
            merges: StdHashMap::new(),
            pattern: GPT4_PATTERN.to_string(),
            compiled_pattern: Regex::new(GPT4_PATTERN).unwrap(),
        }
    }

    /// Helper function to train tokenizer on simple text
    fn train_simple_tokenizer(text: &str, vocab_size: u32) -> Tokenizer {
        let mut tokenizer = create_test_tokenizer();
        
        // Convert text to words and counts (simplified version)
        let text_chunks: Vec<&str> = tokenizer.compiled_pattern.find_iter(text)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str())
            .collect();
        
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();
        for chunk in text_chunks {
            *counts.entry(CompactString::from(chunk)).or_default() += 1;
        }
        
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, c) in counts.into_iter() {
            words.push(Word::new(chunk.as_bytes().iter().map(|&b| b as u32).collect()));
            cvec.push(c);
        }
        
        tokenizer.train_core_incremental(words, cvec, vocab_size);
        tokenizer
    }

    #[test]
    fn test_word_creation() {
        let ids = vec![72, 101, 108, 108, 111]; // "Hello" in ASCII
        let word = Word::new(ids.clone());
        
        assert_eq!(word.ids, ids);
    }

    #[test]
    fn test_word_pairs() {
        let ids = vec![72, 101, 108, 108, 111]; // "Hello"
        let word = Word::new(ids);
        
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs, vec![(72, 101), (101, 108), (108, 108), (108, 111)]);
    }

    #[test]
    fn test_word_pairs_empty() {
        let word = Word::new(vec![]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn test_word_pairs_single() {
        let word = Word::new(vec![72]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs, vec![]);
    }

    #[test]
    fn test_merge_pair_simple() {
        let mut word = Word::new(vec![65, 65, 66]); // "AAB"
        let pair = (65, 65); // "AA"
        let new_id = 256;
        
        let deltas = word.merge_pair(pair, new_id);
        
        assert_eq!(word.ids, vec![256, 66]); // Should be [new_id, 66]
        
        // Check deltas: removed (65,65), removed (65,66), added (256,66)
        assert_eq!(deltas.len(), 3);
        assert!(deltas.contains(&(pair, -1))); // removed (65,65)
        assert!(deltas.contains(&((65, 66), -1))); // removed (65,66)
        assert!(deltas.contains(&((256, 66), 1))); // added (256,66)
    }

    #[test]
    fn test_merge_pair_no_match() {
        let mut word = Word::new(vec![65, 66, 67]); // "ABC"
        let pair = (68, 69); // "DE" - not in word
        let new_id = 256;
        
        let _deltas = word.merge_pair(pair, new_id);
        
        assert_eq!(word.ids, vec![65, 66, 67]); // Should be unchanged
    }

    #[test]
    fn test_merge_pair_multiple_occurrences() {
        let mut word = Word::new(vec![65, 65, 65, 65]); // "AAAA"
        let pair = (65, 65); // "AA"
        let new_id = 256;
        
        let deltas = word.merge_pair(pair, new_id);
        
        assert_eq!(word.ids, vec![256, 256]); // Should be [new_id, new_id]
        
        // Should have removed 3 pairs of (65,65) and added 1 pair of (256,256)
        let delta_sum: i32 = deltas.iter().map(|(_, delta)| *delta).sum();
        assert_eq!(delta_sum, -2); // 3 removed, 1 added = -2
    }

    #[test]
    fn test_merge_pair_overlapping() {
        let mut word = Word::new(vec![65, 65, 65]); // "AAA"
        let pair = (65, 65); // "AA"
        let new_id = 256;
        
        let _deltas = word.merge_pair(pair, new_id);
        
        assert_eq!(word.ids, vec![256, 65]); // Should be [new_id, 65] - non-overlapping merges only
    }

    #[test]
    fn test_merge_job_ordering() {
        let job1 = MergeJob {
            pair: (1, 2),
            count: 10,
            pos: AHashSet::new(),
        };
        
        let job2 = MergeJob {
            pair: (3, 4),
            count: 5,
            pos: AHashSet::new(),
        };
        
        let job3 = MergeJob {
            pair: (1, 2),
            count: 10,
            pos: AHashSet::new(),
        };
        
        // Test equality
        assert_eq!(job1, job3);
        assert_ne!(job1, job2);
        
        // Test ordering (max-heap by count)
        assert!(job1 > job2); // Higher count should be "greater"
        
        // Test tie-breaking by pair (ascending order for max-heap)
        let job4 = MergeJob {
            pair: (2, 3),
            count: 10,
            pos: AHashSet::new(),
        };
        
        // For max-heap with tie-breaking, lower pair should be "greater"
        assert!(job1 > job4); // Same count, but (1,2) < (2,3) so job1 > job4
    }

    #[test]
    fn test_count_pairs_parallel() {
        let words = vec![
            Word::new(vec![1, 2, 3]),
            Word::new(vec![2, 3, 4]),
            Word::new(vec![1, 2, 3]),
        ];
        let counts = vec![1, 1, 2]; // Different weights for each word
        
        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);
        
        // Expected pairs and their counts:
        // Word 0 (count=1): (1,2), (2,3)
        // Word 1 (count=1): (2,3), (3,4)  
        // Word 2 (count=2): (1,2), (2,3) with weight 2
        
        assert_eq!(pair_counts.get(&(1, 2)), Some(&3)); // 1 + 2
        assert_eq!(pair_counts.get(&(2, 3)), Some(&4)); // 1 + 1 + 2
        assert_eq!(pair_counts.get(&(3, 4)), Some(&1)); // 1
        
        // Check positions
        assert!(positions.get(&(1, 2)).unwrap().contains(&0));
        assert!(positions.get(&(1, 2)).unwrap().contains(&2));
        assert!(positions.get(&(2, 3)).unwrap().contains(&0));
        assert!(positions.get(&(2, 3)).unwrap().contains(&1));
        assert!(positions.get(&(2, 3)).unwrap().contains(&2));
        assert!(positions.get(&(3, 4)).unwrap().contains(&1));
    }

    #[test]
    fn test_count_pairs_parallel_empty() {
        let words = vec![];
        let counts = vec![];
        
        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);
        
        assert_eq!(pair_counts.len(), 0);
        assert_eq!(positions.len(), 0);
    }

    #[test]
    fn test_tokenizer_creation() {
        let tokenizer = create_test_tokenizer();
        
        assert_eq!(tokenizer.pattern, GPT4_PATTERN);
        assert_eq!(tokenizer.merges.len(), 0);
    }

    #[test]
    fn test_tokenizer_get_pattern() {
        let tokenizer = create_test_tokenizer();
        assert_eq!(tokenizer.get_pattern(), GPT4_PATTERN);
    }

    #[test]
    fn test_tokenizer_train_minimum_vocab() {
        let mut tokenizer = create_test_tokenizer();
        let text = "hello";
        
        // Train with minimum vocab size (no merges)
        let words = vec![Word::new(text.as_bytes().iter().map(|&b| b as u32).collect())];
        let counts = vec![1];
        
        tokenizer.train_core_incremental(words, counts, 256);
        
        assert_eq!(tokenizer.merges.len(), 0); // No merges should occur
    }

    #[test]
    fn test_tokenizer_train_simple() {
        let text = "aaabdaaabac";
        let tokenizer = train_simple_tokenizer(text, 256 + 3);
        
        assert_eq!(tokenizer.merges.len(), 3); // Should have 3 merges
        
        // Check that we can encode the training text
        let encoded = tokenizer.encode(text);
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_tokenizer_train_repetitive_text() {
        let text = "hello world hello world hello world";
        let tokenizer = train_simple_tokenizer(text, 300);
        
        assert!(tokenizer.merges.len() > 0);
        
        let encoded = tokenizer.encode(text);
        assert!(!encoded.is_empty());
        
        // Encoding the same text twice should give same result
        let encoded2 = tokenizer.encode(text);
        assert_eq!(encoded, encoded2);
    }

    #[test]
    fn test_tokenizer_encode_empty() {
        let tokenizer = create_test_tokenizer();
        let encoded = tokenizer.encode("");
        assert_eq!(encoded, Vec::<u32>::new());
    }

    #[test]
    fn test_tokenizer_encode_single_char() {
        let tokenizer = create_test_tokenizer();
        let encoded = tokenizer.encode("A");
        assert_eq!(encoded, vec![65]); // ASCII 'A'
    }

    #[test]
    fn test_tokenizer_encode_ascii() {
        let tokenizer = create_test_tokenizer();
        let text = "Hello";
        let encoded = tokenizer.encode(text);
        assert_eq!(encoded, vec![72, 101, 108, 108, 111]); // ASCII values
    }

    #[test]
    fn test_tokenizer_encode_with_merges() {
        let mut tokenizer = create_test_tokenizer();
        
        // Manually add a merge for "aa" -> 256
        tokenizer.merges.insert((97, 97), 256); // 'a' = 97 in ASCII
        
        let encoded = tokenizer.encode("aa");
        assert_eq!(encoded, vec![256]); // Should use the merged token
    }

    #[test]
    fn test_tokenizer_encode_multiple_merges() {
        let mut tokenizer = create_test_tokenizer();
        
        // Add merges: "aa" -> 256, "bb" -> 257
        tokenizer.merges.insert((97, 97), 256);
        tokenizer.merges.insert((98, 98), 257);
        
        let encoded = tokenizer.encode("aabb");
        assert_eq!(encoded, vec![256, 257]); // Should use both merged tokens
    }

    #[test]
    fn test_tokenizer_encode_priority() {
        let mut tokenizer = create_test_tokenizer();
        
        // Add merges with different priorities (lower ID = higher priority)
        tokenizer.merges.insert((97, 98), 257); // "ab" -> 257
        tokenizer.merges.insert((98, 99), 256); // "bc" -> 256 (higher priority)
        
        let encoded = tokenizer.encode("abc");
        // Should merge "bc" first (higher priority), then "a" + result
        assert_eq!(encoded, vec![97, 256]); // "a" + merged "bc"
    }

    #[test]
    fn test_tokenizer_encode_complex_text() {
        let text = "Hello, world! 123";
        let tokenizer = train_simple_tokenizer(text, 300);
        
        let encoded = tokenizer.encode(text);
        assert!(!encoded.is_empty());
        
        // Should be able to encode and decode back (approximately)
        // Note: We don't have decode in Rust, so we just check it's reasonable
        assert!(encoded.len() <= text.len() * 2); // Reasonable upper bound
    }

    #[test]
    fn test_tokenizer_get_mergeable_ranks_basic() {
        let tokenizer = create_test_tokenizer();
        let ranks = tokenizer.get_mergeable_ranks();
        
        // Should have exactly 256 entries (just the base bytes)
        assert_eq!(ranks.len(), 256);
        
        // Check first few entries
        for i in 0..256 {
            assert_eq!(ranks[i], (vec![i as u8], i as u32));
        }
    }

    #[test]
    fn test_tokenizer_get_mergeable_ranks_with_merges() {
        let mut tokenizer = create_test_tokenizer();
        
        // Add some merges
        tokenizer.merges.insert((97, 97), 256); // "aa" -> 256
        tokenizer.merges.insert((98, 99), 257); // "bc" -> 257
        
        let ranks = tokenizer.get_mergeable_ranks();
        
        // Should have base 256 + 2 merges
        assert_eq!(ranks.len(), 258);
        
        // Check base bytes are still correct
        for i in 0..256 {
            assert_eq!(ranks[i], (vec![i as u8], i as u32));
        }
        
        // Check merge tokens
        assert_eq!(ranks[256], (vec![97, 97], 256)); // "aa"
        assert_eq!(ranks[257], (vec![98, 99], 257)); // "bc"
    }

    #[test]
    fn test_tokenizer_get_mergeable_ranks_complex_merges() {
        let mut tokenizer = create_test_tokenizer();
        
        // Create a chain of merges: "aa" -> 256, then "aa" + "a" -> 257
        tokenizer.merges.insert((97, 97), 256); // "aa" -> 256
        tokenizer.merges.insert((256, 97), 257); // "aa" + "a" -> 257
        
        let ranks = tokenizer.get_mergeable_ranks();
        
        assert_eq!(ranks.len(), 258);
        assert_eq!(ranks[256], (vec![97, 97], 256)); // "aa"
        assert_eq!(ranks[257], (vec![97, 97, 97], 257)); // "aaa"
    }

    #[test]
    fn test_unicode_handling() {
        let tokenizer = create_test_tokenizer();
        
        // Test with Unicode characters
        let unicode_text = "Hello 世界 🚀";
        let encoded = tokenizer.encode(unicode_text);
        
        assert!(!encoded.is_empty());
        // Each Unicode character should be encoded as one or more bytes
        assert!(encoded.len() >= unicode_text.len());
    }

    #[test]
    fn test_deterministic_training() {
        let text = "hello world test";
        
        let tokenizer1 = train_simple_tokenizer(text, 280);
        let tokenizer2 = train_simple_tokenizer(text, 280);
        
        // Should produce identical results
        assert_eq!(tokenizer1.merges, tokenizer2.merges);
        
        let encoded1 = tokenizer1.encode(text);
        let encoded2 = tokenizer2.encode(text);
        assert_eq!(encoded1, encoded2);
    }

    #[test]
    fn test_training_with_different_vocab_sizes() {
        let text = "hello world";
        
        let tokenizer256 = train_simple_tokenizer(text, 256);
        let tokenizer300 = train_simple_tokenizer(text, 300);
        
        assert_eq!(tokenizer256.merges.len(), 0);
        assert!(tokenizer300.merges.len() > 0);
    }

    #[test]
    fn test_edge_case_empty_training() {
        let mut tokenizer = create_test_tokenizer();
        
        // Train with empty data
        let words = vec![];
        let counts = vec![];
        
        tokenizer.train_core_incremental(words, counts, 300);
        
        assert_eq!(tokenizer.merges.len(), 0);
    }

    #[test]
    fn test_edge_case_single_word() {
        let mut tokenizer = create_test_tokenizer();
        
        let words = vec![Word::new(vec![72, 101, 108, 108, 111])]; // "Hello"
        let counts = vec![1];
        
        tokenizer.train_core_incremental(words, counts, 260);
        
        // Should have some merges
        assert!(tokenizer.merges.len() > 0);
    }

    #[test]
    fn test_regex_pattern_compilation() {
        let tokenizer = create_test_tokenizer();
        
        // Test that the pattern compiles and works
        let test_text = "Hello, world! 123";
        let matches: Vec<&str> = tokenizer.compiled_pattern.find_iter(test_text)
            .filter_map(|m| m.ok())
            .map(|m| m.as_str())
            .collect();
        
        assert!(!matches.is_empty());
        assert!(matches.iter().any(|&m| m.contains("Hello")));
    }

    #[test]
    fn test_merge_job_partial_ord() {
        let job1 = MergeJob {
            pair: (1, 2),
            count: 10,
            pos: AHashSet::new(),
        };
        
        let job2 = MergeJob {
            pair: (3, 4),
            count: 15,
            pos: AHashSet::new(),
        };
        
        // Test partial ordering
        assert!(job1 < job2); // Lower count should be less
        assert!(job2 > job1); // Higher count should be greater
    }

    #[test]
    fn test_large_text_training() {
        // Create a larger text by repetition
        let base_text = "The quick brown fox jumps over the lazy dog. ";
        let large_text = base_text.repeat(100);
        
        let tokenizer = train_simple_tokenizer(&large_text, 500);
        
        assert!(tokenizer.merges.len() > 0);
        
        let encoded = tokenizer.encode(&large_text);
        assert!(!encoded.is_empty());
        
        // Should be more efficient than raw bytes due to merges
        assert!(encoded.len() < large_text.len());
    }

    #[test]
    fn test_special_characters() {
        let tokenizer = create_test_tokenizer();
        
        let special_text = "!@#$%^&*()_+-=[]{}|;':\",./<>?";
        let encoded = tokenizer.encode(special_text);
        
        assert!(!encoded.is_empty());
        assert_eq!(encoded.len(), special_text.len()); // Each char should be one byte
    }

    #[test]
    fn test_whitespace_handling() {
        let tokenizer = create_test_tokenizer();
        
        let whitespace_text = "  \t\n\r  ";
        let encoded = tokenizer.encode(whitespace_text);
        
        assert!(!encoded.is_empty());
        // Should handle all whitespace characters
    }

    // ------------------------ Comparison Tests with Python Reference ------------------------

    #[test]
    fn test_comparison_simple_case() {
        // Test case: "aaabdaaabac" with vocab_size = 259 (256 + 3 merges)
        // Expected Python result: [258, 100, 258, 97, 99] with 3 merges
        let text = "aaabdaaabac";
        let vocab_size = 259;
        
        let tokenizer = train_simple_tokenizer(text, vocab_size);
        let encoded = tokenizer.encode(text);
        
        // The most important thing: we should get the same final encoding
        // Different merge sequences can lead to the same result, which is acceptable
        let expected = vec![258, 100, 258, 97, 99];
        assert_eq!(encoded, expected, "Final encoding should match Python reference");
        assert_eq!(tokenizer.merges.len(), 3, "Should have exactly 3 merges");
        
        // We should at least have the first merge (97, 97) -> 256 which is unambiguous
        assert_eq!(tokenizer.merges.get(&(97, 97)), Some(&256), "First merge should be (97, 97) -> 256");
    }

    #[test]
    fn test_comparison_hello_world() {
        // Test case: "hello world" with vocab_size = 300
        // Both implementations should compress "hello world" to 2 tokens with 9 merges
        let text = "hello world";
        let vocab_size = 300;
        
        let tokenizer = train_simple_tokenizer(text, vocab_size);
        let encoded = tokenizer.encode(text);
        
        // The key achievement: compress "hello world" (11 chars) to 2 tokens
        assert_eq!(encoded.len(), 2, "Should compress 'hello world' to 2 tokens");
        assert_eq!(tokenizer.merges.len(), 9, "Should have exactly 9 merges");
        
        // Both tokens should be > 255 (indicating they're merged tokens)
        assert!(encoded[0] > 255, "First token should be a merged token");
        assert!(encoded[1] > 255, "Second token should be a merged token");
    }

    #[test]
    fn test_comparison_minbpe_wikipedia_example() {
        // Test the exact example from minbpe Wikipedia: "aaabdaaabac"
        // This should produce the same result as minbpe's BasicTokenizer
        let text = "aaabdaaabac";
        let vocab_size = 259; // 256 + 3 merges
        
        let tokenizer = train_simple_tokenizer(text, vocab_size);
        let encoded = tokenizer.encode(text);
        
        // According to Wikipedia, this should compress to something like "XdXac"
        // where X=ZY, Y=ab, Z=aa. In our token IDs:
        // a=97, b=98, c=99, d=100
        // Z=(97,97) -> 256, Y=(256,98) -> 257, X=(257,97) -> 258
        // Result: [258, 100, 258, 97, 99]
        let expected = vec![258, 100, 258, 97, 99];
        assert_eq!(encoded, expected);
    }

    #[test]
    fn test_comparison_deterministic_merges() {
        // Test that merges are deterministic and match expected pattern
        let text = "aaabdaaabac";
        let vocab_size = 259;
        
        let tokenizer1 = train_simple_tokenizer(text, vocab_size);
        let tokenizer2 = train_simple_tokenizer(text, vocab_size);
        
        // Should produce identical merges
        assert_eq!(tokenizer1.merges, tokenizer2.merges);
        
        // Check the specific merge order
        let merges_vec: Vec<_> = tokenizer1.merges.iter().collect();
        assert_eq!(merges_vec.len(), 3);
        
        // First merge should be (97, 97) -> 256 (most frequent "aa")
        assert!(tokenizer1.merges.contains_key(&(97, 97)));
        assert_eq!(tokenizer1.merges[&(97, 97)], 256);
    }

    #[test]
    fn test_comparison_round_trip_consistency() {
        // Test that encoding is consistent across multiple runs
        let text = "hello world test round trip";
        let vocab_size = 350;
        
        let tokenizer1 = train_simple_tokenizer(text, vocab_size);
        let tokenizer2 = train_simple_tokenizer(text, vocab_size);
        
        let encoded1 = tokenizer1.encode(text);
        let encoded2 = tokenizer2.encode(text);
        
        assert_eq!(encoded1, encoded2);
        assert_eq!(tokenizer1.merges, tokenizer2.merges);
    }

    #[test]
    fn test_comparison_empty_and_single_char() {
        // Test edge cases that should match Python behavior
        let empty_tokenizer = create_test_tokenizer();
        let empty_encoded = empty_tokenizer.encode("");
        assert_eq!(empty_encoded, Vec::<u32>::new());
        
        let single_char = "A";
        let single_encoded = empty_tokenizer.encode(single_char);
        assert_eq!(single_encoded, vec![65]); // ASCII 'A'
    }

    #[test]
    fn test_comparison_unicode_handling() {
        // Test that Unicode is handled consistently
        let unicode_text = "Hello 世界 🚀";
        let tokenizer = train_simple_tokenizer(unicode_text, 300);
        
        let encoded = tokenizer.encode(unicode_text);
        assert!(!encoded.is_empty());
        
        // Should be able to encode the same text multiple times
        let encoded2 = tokenizer.encode(unicode_text);
        assert_eq!(encoded, encoded2);
    }
}

#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init(); // forwards Rust `log` to Python's `logging`
    m.add_class::<Tokenizer>()?;
    Ok(())
}
