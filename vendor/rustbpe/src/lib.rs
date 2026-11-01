use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);

/// A Byte Pair Encoding tokenizer that matches the GPT-4 style implementation
pub struct Tokenizer {
    /// Maps pairs of token IDs to their merged token ID
    pub merges: StdHashMap<Pair, u32>,
    /// The regex pattern used for text splitting
    pub pattern: String,
    /// Compiled regex for efficiency
    compiled_pattern: Regex,
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
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
                let right = if i + 2 < n {
                    Some(self.ids[i + 2])
                } else {
                    None
                };

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
        log::info!(
            "Computing initial pair counts from {} unique sequences",
            words.len()
        );
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
            let Some(mut top) = heap.pop() else {
                break;
            };

            // Lazy refresh: if the count changed since we queued this job, update and requeue
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if current <= 0 {
                // Pair no longer exists or has non-positive count, skip it
                continue;
            }
            if top.count != current as u64 {
                top.count = current as u64;
                heap.push(top);
                continue;
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
                    current_percent,
                    merges_done,
                    num_merges,
                    top.pair,
                    new_id,
                    top.count
                );
                last_log_percent = current_percent;
            }
        }

        log::info!("Finished training: {} merges completed", merges_done);
    }
}

impl Tokenizer {
    /// Train from native Rust text input without initializing the Python runtime.
    /// Chunks are pre-tokenized in bounded batches and counted in parallel.
    pub fn train_from_texts<I>(&mut self, texts: I, vocab_size: u32, pattern: &str)
    where
        I: IntoIterator<Item = String>,
    {
        self.try_train_from_texts(
            texts.into_iter().map(Ok::<_, std::convert::Infallible>),
            vocab_size,
            pattern,
        )
        .expect("infallible tokenizer input");
    }

    /// Train from a fallible native text stream without materializing its input.
    pub fn try_train_from_texts<I, E>(
        &mut self,
        texts: I,
        vocab_size: u32,
        pattern: &str,
    ) -> Result<(), E>
    where
        I: IntoIterator<Item = Result<String, E>>,
    {
        assert!(vocab_size >= 256, "vocab_size must be at least 256");
        self.pattern = pattern.to_owned();
        self.compiled_pattern = Regex::new(pattern).expect("invalid BPE split pattern");
        let mut counts: AHashMap<CompactString, i32> = AHashMap::new();
        let mut batch = Vec::with_capacity(8192);
        for text in texts {
            let text = text?;
            batch.push(text);
            if batch.len() == 8192 {
                Self::count_text_batch(&self.compiled_pattern, &batch, &mut counts);
                batch.clear();
            }
        }
        if !batch.is_empty() {
            Self::count_text_batch(&self.compiled_pattern, &batch, &mut counts);
        }
        let mut words = Vec::with_capacity(counts.len());
        let mut cvec = Vec::with_capacity(counts.len());
        for (chunk, count) in counts {
            words.push(Word::new(
                chunk.as_bytes().iter().map(|&byte| byte as u32).collect(),
            ));
            cvec.push(count);
        }
        self.train_core_incremental(words, cvec, vocab_size);
        Ok(())
    }

    /// Return all mergeable byte sequences and their token IDs.
    pub fn get_mergeable_ranks_native(&self) -> Vec<(Vec<u8>, u32)> {
        let mut mergeable_ranks = Vec::new();
        let mut token_bytes: Vec<Vec<u8>> = (0..256_u32).map(|id| vec![id as u8]).collect();
        for (id, bytes) in token_bytes.iter().enumerate() {
            mergeable_ranks.push((bytes.clone(), id as u32));
        }
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

    fn count_text_batch(
        pattern: &Regex,
        batch: &[String],
        counts: &mut AHashMap<CompactString, i32>,
    ) {
        let local = batch
            .par_iter()
            .map(|text| {
                let mut local: AHashMap<CompactString, i32> = AHashMap::new();
                for matched in pattern.find_iter(text) {
                    if let Ok(piece) = matched {
                        *local
                            .entry(CompactString::from(piece.as_str()))
                            .or_default() += 1;
                    }
                }
                local
            })
            .reduce(AHashMap::new, |mut left, right| {
                for (piece, count) in right {
                    *left.entry(piece).or_default() += count;
                }
                left
            });
        for (piece, count) in local {
            *counts.entry(piece).or_default() += count;
        }
    }
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            merges: StdHashMap::new(),
            pattern: GPT4_PATTERN.to_owned(),
            compiled_pattern: Regex::new(GPT4_PATTERN).expect("default BPE pattern should compile"),
        }
    }

    pub fn get_pattern(&self) -> String {
        self.pattern.clone()
    }

    pub fn vocab_size(&self) -> u32 {
        256 + self.merges.len() as u32
    }

    pub fn get_mergeable_ranks(&self) -> Vec<(Vec<u8>, u32)> {
        self.get_mergeable_ranks_native()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();
        for matched in self.compiled_pattern.find_iter(text) {
            let chunk = match matched {
                Ok(matched) => matched.as_str(),
                Err(error) => {
                    log::warn!("Regex match error, skipping chunk: {}", error);
                    continue;
                }
            };
            let mut ids: Vec<u32> = chunk.bytes().map(u32::from).collect();
            while ids.len() >= 2 {
                let mut best_pair: Option<(usize, u32)> = None;
                for index in 0..ids.len() - 1 {
                    if let Some(&new_id) = self.merges.get(&(ids[index], ids[index + 1])) {
                        if best_pair.is_none_or(|(_, best_id)| new_id < best_id) {
                            best_pair = Some((index, new_id));
                        }
                    }
                }
                if let Some((index, new_id)) = best_pair {
                    ids[index] = new_id;
                    ids.remove(index + 1);
                } else {
                    break;
                }
            }
            all_ids.extend(ids);
        }
        all_ids
    }

    pub fn decode(&self, ids: Vec<u32>) -> Result<String, String> {
        let mut vocabulary: Vec<Vec<u8>> = (0..256_u32).map(|id| vec![id as u8]).collect();
        let mut sorted_merges: Vec<_> = self.merges.iter().collect();
        sorted_merges.sort_by_key(|&(_, &token_id)| token_id);
        for (&(left, right), &merged_id) in &sorted_merges {
            let mut merged_bytes = vocabulary
                .get(left as usize)
                .ok_or_else(|| format!("Invalid token id {left} in merge"))?
                .clone();
            merged_bytes.extend(
                vocabulary
                    .get(right as usize)
                    .ok_or_else(|| format!("Invalid token id {right} in merge"))?,
            );
            if vocabulary.len() <= merged_id as usize {
                vocabulary.resize(merged_id as usize + 1, Vec::new());
            }
            vocabulary[merged_id as usize] = merged_bytes;
        }
        let mut bytes = Vec::new();
        for id in ids {
            bytes.extend(
                vocabulary
                    .get(id as usize)
                    .ok_or_else(|| format!("Unknown token id: {id}"))?,
            );
        }
        String::from_utf8(bytes).map_err(|error| error.to_string())
    }

    pub fn batch_encode(&self, texts: &[String]) -> Vec<Vec<u32>> {
        texts.par_iter().map(|text| self.encode(text)).collect()
    }
}
// ============================================================================
// RUST TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_word_pairs() {
        let word = Word::new(vec![1, 2, 3, 4]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert_eq!(pairs, vec![(1, 2), (2, 3), (3, 4)]);
    }

    #[test]
    fn test_word_pairs_empty() {
        let word = Word::new(vec![]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_word_pairs_single() {
        let word = Word::new(vec![42]);
        let pairs: Vec<Pair> = word.pairs().collect();
        assert!(pairs.is_empty());
    }

    #[test]
    fn test_word_merge_pair() {
        // [1, 2, 3, 1, 2] with merge (1,2) -> 99 should become [99, 3, 99]
        let mut word = Word::new(vec![1, 2, 3, 1, 2]);
        let _deltas = word.merge_pair((1, 2), 99);
        assert_eq!(word.ids, vec![99, 3, 99]);
    }

    #[test]
    fn test_word_merge_pair_adjacent() {
        // [1, 2, 1, 2, 1, 2] -> [99, 99, 99] (non-overlapping)
        let mut word = Word::new(vec![1, 2, 1, 2, 1, 2]);
        let _deltas = word.merge_pair((1, 2), 99);
        assert_eq!(word.ids, vec![99, 99, 99]);
    }

    #[test]
    fn test_word_merge_no_match() {
        let mut word = Word::new(vec![1, 2, 3]);
        let deltas = word.merge_pair((4, 5), 99);
        assert_eq!(word.ids, vec![1, 2, 3]); // unchanged
                                             // Only delta should be for pairs that don't exist, so effectively empty useful deltas
        assert!(deltas.is_empty() || deltas.iter().all(|(_, d)| *d == 0));
    }

    #[test]
    fn test_tokenizer_new() {
        let tok = Tokenizer::new();
        assert!(tok.merges.is_empty());
        assert!(tok.pattern.is_empty());
    }

    #[test]
    fn test_encode_untrained_simple() {
        // With no merges and empty pattern, encode returns nothing (no regex matches)
        let tok = Tokenizer::new();
        let ids = tok.encode("hello");
        assert!(ids.is_empty()); // empty pattern matches nothing
    }

    #[test]
    fn test_encode_with_pattern_no_merges() {
        // With a simple pattern but no merges, should return raw byte values
        let tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };
        let ids = tok.encode("hi");
        // 'h' = 104, 'i' = 105
        assert_eq!(ids, vec![104, 105]);
    }

    #[test]
    fn test_encode_with_merges() {
        // Set up a tokenizer with one merge: (104, 105) -> 256  ('h','i' -> 256)
        let mut merges = StdHashMap::new();
        merges.insert((104, 105), 256); // 'hi' -> 256

        let tok = Tokenizer {
            merges,
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };

        let ids = tok.encode("hi");
        assert_eq!(ids, vec![256]); // merged into single token

        let ids2 = tok.encode("hip");
        // 'hi' merges to 256, 'p' stays as 112
        assert_eq!(ids2, vec![256, 112]);
    }

    #[test]
    fn test_get_mergeable_ranks_empty() {
        let tok = Tokenizer::new();
        let ranks = tok.get_mergeable_ranks();
        // Should have 256 byte-level tokens
        assert_eq!(ranks.len(), 256);
        // First should be [0] -> 0
        assert_eq!(ranks[0], (vec![0u8], 0));
        // Last should be [255] -> 255
        assert_eq!(ranks[255], (vec![255u8], 255));
    }

    #[test]
    fn test_get_mergeable_ranks_with_merge() {
        let mut merges = StdHashMap::new();
        // Merge bytes 65 ('A') and 66 ('B') into token 256
        merges.insert((65, 66), 256);

        let tok = Tokenizer {
            merges,
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        let ranks = tok.get_mergeable_ranks();
        assert_eq!(ranks.len(), 257); // 256 bytes + 1 merge

        // The merge should produce bytes [65, 66] -> 256
        let last = &ranks[256];
        assert_eq!(last.0, vec![65u8, 66u8]);
        assert_eq!(last.1, 256);
    }

    #[test]
    fn test_count_pairs_parallel() {
        let words = vec![Word::new(vec![1, 2, 3]), Word::new(vec![1, 2, 4])];
        let counts = vec![1, 2]; // first word appears 1x, second 2x

        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);

        // (1,2) appears in both: 1*1 + 1*2 = 3
        assert_eq!(pair_counts.get(&(1, 2)), Some(&3));
        // (2,3) appears only in first: 1*1 = 1
        assert_eq!(pair_counts.get(&(2, 3)), Some(&1));
        // (2,4) appears only in second: 1*2 = 2
        assert_eq!(pair_counts.get(&(2, 4)), Some(&2));

        // Check positions
        assert!(positions.get(&(1, 2)).unwrap().contains(&0));
        assert!(positions.get(&(1, 2)).unwrap().contains(&1));
    }

    #[test]
    fn test_train_core_incremental() {
        // Simple training test with repeated patterns
        let mut tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        // "ab" repeated 10 times, "cd" repeated 5 times
        let words = vec![
            Word::new(vec![97, 98]),  // "ab"
            Word::new(vec![99, 100]), // "cd"
        ];
        let counts = vec![10, 5];

        // Train with vocab_size = 257 (one merge)
        tok.train_core_incremental(words, counts, 257);

        // Should have merged (97, 98) since it has higher count
        assert_eq!(tok.merges.len(), 1);
        assert!(tok.merges.contains_key(&(97, 98)));
        assert_eq!(tok.merges.get(&(97, 98)), Some(&256));
    }

    // ==================== Additional comprehensive tests ====================

    #[test]
    fn test_default_trait() {
        let tok = Tokenizer::default();
        assert!(tok.merges.is_empty());
        assert!(tok.pattern.is_empty());
    }

    #[test]
    fn test_vocab_size() {
        let mut tok = Tokenizer::new();
        assert_eq!(tok.vocab_size(), 256);

        // Add some merges manually
        tok.merges.insert((97, 98), 256);
        assert_eq!(tok.vocab_size(), 257);

        tok.merges.insert((256, 99), 257);
        assert_eq!(tok.vocab_size(), 258);
    }

    #[test]
    fn test_word_merge_overlapping_pairs() {
        // "aaa" = [97, 97, 97] with merge (97, 97) -> 256
        // Should become [256, 97] (non-overlapping, left-to-right)
        let mut word = Word::new(vec![97, 97, 97]);
        let _deltas = word.merge_pair((97, 97), 256);
        assert_eq!(word.ids, vec![256, 97]);
    }

    #[test]
    fn test_word_merge_overlapping_pairs_even() {
        // "aaaa" = [97, 97, 97, 97] with merge (97, 97) -> 256
        // Should become [256, 256]
        let mut word = Word::new(vec![97, 97, 97, 97]);
        let _deltas = word.merge_pair((97, 97), 256);
        assert_eq!(word.ids, vec![256, 256]);
    }

    #[test]
    fn test_word_merge_multiple_occurrences() {
        // "abXab" where X doesn't match
        let mut word = Word::new(vec![1, 2, 99, 1, 2]);
        let deltas = word.merge_pair((1, 2), 256);
        assert_eq!(word.ids, vec![256, 99, 256]);

        // Count (1, 2) removals in deltas
        let ab_removals: i32 = deltas
            .iter()
            .filter(|(p, _)| *p == (1, 2))
            .map(|(_, d)| d)
            .sum();
        assert_eq!(ab_removals, -2); // two occurrences removed
    }

    #[test]
    fn test_encode_chained_merges() {
        // Set up a tokenizer with chained merges:
        // (97, 97) -> 256  ('aa' -> 256)
        // (256, 97) -> 257 ('aaa' effectively -> 257)
        let mut merges = StdHashMap::new();
        merges.insert((97, 97), 256); // 'aa' -> 256 (learned first)
        merges.insert((256, 97), 257); // 'aa' + 'a' -> 257 (learned second)

        let tok = Tokenizer {
            merges,
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };

        // "aaa" should encode as [257]
        // Step 1: [97, 97, 97]
        // Step 2: merge (97, 97) at pos 0 -> [256, 97]
        // Step 3: merge (256, 97) -> [257]
        let ids = tok.encode("aaa");
        assert_eq!(ids, vec![257]);

        // "aaaa" should encode as [256, 256]
        // Because (97, 97) has lower id than (256, 97), so we merge all 'aa' pairs first
        let ids = tok.encode("aaaa");
        assert_eq!(ids, vec![256, 256]);

        // "aaaaa" should be [257, 256]
        // [97, 97, 97, 97, 97]
        // -> [256, 97, 97, 97] (merge first aa)
        // -> [256, 256, 97] (merge second aa)
        // -> [257, 256] (merge (256, 97))
        // Wait, let me recalculate...
        // Actually the algorithm picks the pair with LOWEST new_id.
        // (97, 97) -> 256, (256, 97) -> 257
        // So 256 < 257, meaning (97, 97) is always preferred.
        // [97, 97, 97, 97, 97]
        // Pairs: (97,97) at 0,1,2,3. All map to 256.
        // Pick leftmost (position 0): [256, 97, 97, 97]
        // Pairs: (256,97)->257, (97,97)->256 at pos 1,2
        // 256 < 257, pick (97,97) at pos 1: [256, 256, 97]
        // Pairs: (256,256) not in merges, (256,97)->257
        // Only option is 257: [256, 257]
        let ids = tok.encode("aaaaa");
        assert_eq!(ids, vec![256, 257]);
    }

    #[test]
    fn test_encode_decode_roundtrip_simple() {
        // Set up tokenizer with some merges
        let mut merges = StdHashMap::new();
        merges.insert((104, 105), 256); // 'hi' -> 256

        let tok = Tokenizer {
            merges,
            pattern: r"\w+|\s+".to_string(),
            compiled_pattern: Regex::new(r"\w+|\s+").unwrap(),
        };

        let text = "hi";
        let ids = tok.encode(text);
        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip_with_spaces() {
        let mut merges = StdHashMap::new();
        merges.insert((104, 101), 256); // 'he' -> 256
        merges.insert((108, 108), 257); // 'll' -> 257
        merges.insert((256, 257), 258); // 'hell' -> 258

        let tok = Tokenizer {
            merges,
            pattern: r"\w+|\s+".to_string(),
            compiled_pattern: Regex::new(r"\w+|\s+").unwrap(),
        };

        let text = "hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(ids).unwrap();
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_decode_byte_level() {
        // Decode raw byte tokens (no merges)
        let tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        // [104, 105] = "hi"
        let decoded = tok.decode(vec![104, 105]).unwrap();
        assert_eq!(decoded, "hi");
    }

    #[test]
    fn test_decode_invalid_token() {
        let tok = Tokenizer::new();

        // Token 300 doesn't exist (only 0-255 in base vocab)
        let result = tok.decode(vec![300]);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_multiple_merges() {
        let mut tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        // "ab" appears 100 times, "bc" appears 50 times
        // After merging "ab", the corpus becomes "X c" where X=256
        // Then "Xc" (256, 99) should be merged next? No wait...
        // Let's use a simpler example:
        // "ab" appears 10 times
        let words = vec![
            Word::new(vec![97, 98]), // "ab"
        ];
        let counts = vec![10];

        // Train with vocab_size = 258 (2 merges)
        // But we only have one unique pair, so only one merge will happen
        tok.train_core_incremental(words, counts, 258);

        assert_eq!(tok.merges.len(), 1);
    }

    #[test]
    fn test_train_creates_chained_merges() {
        let mut tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        // "aaa" = [97, 97, 97]
        // First merge: (97, 97) -> 256, word becomes [256, 97]
        // Second merge: (256, 97) -> 257, word becomes [257]
        let words = vec![Word::new(vec![97, 97, 97])];
        let counts = vec![10];

        tok.train_core_incremental(words, counts, 258);

        assert_eq!(tok.merges.len(), 2);
        assert_eq!(tok.merges.get(&(97, 97)), Some(&256));
        assert_eq!(tok.merges.get(&(256, 97)), Some(&257));
    }

    #[test]
    fn test_get_mergeable_ranks_chained() {
        // Test that chained merges produce correct byte sequences
        let mut merges = StdHashMap::new();
        merges.insert((65, 66), 256); // 'AB' -> 256
        merges.insert((256, 67), 257); // 'ABC' -> 257

        let tok = Tokenizer {
            merges,
            pattern: String::new(),
            compiled_pattern: Regex::new("").unwrap(),
        };

        let ranks = tok.get_mergeable_ranks();
        assert_eq!(ranks.len(), 258);

        // Token 256 should be [65, 66] = "AB"
        assert_eq!(ranks[256], (vec![65u8, 66u8], 256));

        // Token 257 should be [65, 66, 67] = "ABC"
        assert_eq!(ranks[257], (vec![65u8, 66u8, 67u8], 257));
    }

    #[test]
    fn test_encode_empty_string() {
        let tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };

        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encode_no_matches() {
        // Pattern only matches words, input has no words
        let tok = Tokenizer {
            merges: StdHashMap::new(),
            pattern: r"\w+".to_string(),
            compiled_pattern: Regex::new(r"\w+").unwrap(),
        };

        let ids = tok.encode("   "); // only spaces
        assert!(ids.is_empty());
    }

    #[test]
    fn test_decode_empty() {
        let tok = Tokenizer::new();
        let decoded = tok.decode(vec![]).unwrap();
        assert_eq!(decoded, "");
    }

    #[test]
    fn test_word_merge_deltas_correctness() {
        // Verify that deltas are exactly correct for pair count updates
        // Word: [1, 2, 3, 1, 2] with merge (1, 2) -> 99
        // Before: pairs are (1,2), (2,3), (3,1), (1,2)
        // After:  [99, 3, 99], pairs are (99,3), (3,99)
        let mut word = Word::new(vec![1, 2, 3, 1, 2]);
        let deltas = word.merge_pair((1, 2), 99);

        // Aggregate deltas by pair
        let mut delta_map: StdHashMap<Pair, i32> = StdHashMap::new();
        for (pair, delta) in deltas {
            *delta_map.entry(pair).or_default() += delta;
        }

        // (1, 2) should have -2 (removed twice)
        assert_eq!(delta_map.get(&(1, 2)), Some(&-2));
        // (2, 3) should have -1 (removed once)
        assert_eq!(delta_map.get(&(2, 3)), Some(&-1));
        // (3, 1) should have -1 (removed once)
        assert_eq!(delta_map.get(&(3, 1)), Some(&-1));
        // (99, 3) should have +1 (created once)
        assert_eq!(delta_map.get(&(99, 3)), Some(&1));
        // (3, 99) should have +1 (created once)
        assert_eq!(delta_map.get(&(3, 99)), Some(&1));
    }

    #[test]
    fn test_count_pairs_parallel_empty() {
        let words: Vec<Word> = vec![];
        let counts: Vec<i32> = vec![];

        let (pair_counts, positions) = count_pairs_parallel(&words, &counts);
        assert!(pair_counts.is_empty());
        assert!(positions.is_empty());
    }

    #[test]
    fn test_count_pairs_parallel_zero_count() {
        // Words with zero count should not contribute
        let words = vec![Word::new(vec![1, 2, 3])];
        let counts = vec![0];

        let (pair_counts, _positions) = count_pairs_parallel(&words, &counts);
        assert!(pair_counts.is_empty());
    }
}
