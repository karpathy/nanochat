//! NumPy-compatible PCG64 seeding and one-dimensional permutation.
//!
//! The SeedSequence mixing algorithm follows NumPy's MIT-licensed
//! implementation (Copyright 2015 Melissa E. O'Neill; Copyright 2019 NumPy
//! developers). This module implements the scalar integer permutation path
//! used by `default_rng(seed).permutation(n)` in the dataset adapters.
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the "Software"),
//! to deal in the Software without restriction, including without limitation
//! the rights to use, copy, modify, merge, publish, distribute, sublicense,
//! and/or sell copies of the Software, and to permit persons to whom the
//! Software is furnished to do so, subject to the following conditions: The
//! above copyright notice and this permission notice shall be included in all
//! copies or substantial portions of the Software. THE SOFTWARE IS PROVIDED
//! "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED.

const INIT_A: u32 = 0x43b0_d7e5;
const MULT_A: u32 = 0x931e_8875;
const INIT_B: u32 = 0x8b51_f9dd;
const MULT_B: u32 = 0x58f3_8ded;
const MIX_MULT_L: u32 = 0xca01_f9dd;
const MIX_MULT_R: u32 = 0x4973_f715;
const PCG_MULTIPLIER: u128 = 0x2360_ed05_1fc6_5da4_4385_df64_9fcc_f645;

pub fn shuffle<T>(items: &mut Vec<T>, seed: u64) {
    let permutation = permutation(items.len(), seed);
    let mut values = std::mem::take(items)
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    items.reserve(values.len());
    for source in permutation {
        items.push(
            values[source]
                .take()
                .expect("permutation uses each index once"),
        );
    }
}

pub fn permutation(length: usize, seed: u64) -> Vec<usize> {
    assert!(
        length <= u32::MAX as usize,
        "NumPy permutation length exceeds supported range"
    );
    let mut values = (0..length).collect::<Vec<_>>();
    let mut rng = NumpyPcg64::seed(seed);
    for index in (1..length).rev() {
        let selected = rng.bounded_u32(index as u32) as usize;
        values.swap(index, selected);
    }
    values
}

struct NumpyPcg64 {
    state: u128,
    increment: u128,
    buffered_u32: Option<u32>,
}

impl NumpyPcg64 {
    fn seed(seed: u64) -> Self {
        let mut entropy = vec![seed as u32];
        if seed >> 32 != 0 {
            entropy.push((seed >> 32) as u32);
        }
        let pool = seed_sequence_pool(&entropy);
        let mut hash_const = INIT_B;
        let mut state_words = [0u32; 8];
        for (index, word) in state_words.iter_mut().enumerate() {
            let mut value = pool[index % pool.len()] ^ hash_const;
            hash_const = hash_const.wrapping_mul(MULT_B);
            value = value.wrapping_mul(hash_const);
            value ^= value >> 16;
            *word = value;
        }
        let state_words = state_words
            .chunks_exact(2)
            .map(|pair| u64::from(pair[0]) | (u64::from(pair[1]) << 32))
            .collect::<Vec<_>>();
        let state_init = (u128::from(state_words[0]) << 64) | u128::from(state_words[1]);
        let sequence = (u128::from(state_words[2]) << 64) | u128::from(state_words[3]);
        let increment = (sequence << 1) | 1;

        // PCG's reference seed procedure: step from zero, add the initial
        // state, then step once more before producing output.
        let state = increment
            .wrapping_add(state_init)
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(increment);
        Self {
            state,
            increment,
            buffered_u32: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(PCG_MULTIPLIER)
            .wrapping_add(self.increment);
        let rotation = (self.state >> 122) as u32;
        let xsl = (self.state >> 64) as u64 ^ self.state as u64;
        xsl.rotate_right(rotation)
    }

    fn next_u32(&mut self) -> u32 {
        if let Some(value) = self.buffered_u32.take() {
            value
        } else {
            let value = self.next_u64();
            self.buffered_u32 = Some((value >> 32) as u32);
            value as u32
        }
    }

    fn bounded_u32(&mut self, maximum: u32) -> u32 {
        if maximum == 0 {
            return 0;
        }
        let mut mask = maximum;
        mask |= mask >> 1;
        mask |= mask >> 2;
        mask |= mask >> 4;
        mask |= mask >> 8;
        mask |= mask >> 16;
        loop {
            let value = self.next_u32() & mask;
            if value <= maximum {
                return value;
            }
        }
    }
}

fn seed_sequence_pool(entropy: &[u32]) -> [u32; 4] {
    let mut pool = [0u32; 4];
    let mut hash_const = INIT_A;
    for index in 0..pool.len() {
        pool[index] = hash_mix(entropy.get(index).copied().unwrap_or(0), &mut hash_const);
    }
    for source in 0..pool.len() {
        for destination in 0..pool.len() {
            if source != destination {
                let mixed = hash_mix(pool[source], &mut hash_const);
                pool[destination] = mix(pool[destination], mixed);
            }
        }
    }
    for &extra in entropy.iter().skip(pool.len()) {
        for value in &mut pool {
            *value = mix(*value, hash_mix(extra, &mut hash_const));
        }
    }
    pool
}

fn hash_mix(mut value: u32, hash_const: &mut u32) -> u32 {
    value ^= *hash_const;
    *hash_const = hash_const.wrapping_mul(MULT_A);
    value = value.wrapping_mul(*hash_const);
    value ^ (value >> 16)
}

fn mix(left: u32, right: u32) -> u32 {
    let value = MIX_MULT_L
        .wrapping_mul(left)
        .wrapping_sub(MIX_MULT_R.wrapping_mul(right));
    value ^ (value >> 16)
}

#[cfg(test)]
mod tests {
    use super::{NumpyPcg64, permutation};

    #[test]
    fn pcg64_seed_42_matches_numpy_raw_output() {
        let mut rng = NumpyPcg64::seed(42);
        assert_eq!(
            (0..5).map(|_| rng.next_u64()).collect::<Vec<_>>(),
            [
                14_276_969_152_011_380_360,
                8_095_878_257_575_067_585,
                15_838_336_090_824_644_132,
                12_864_169_557_245_331_597,
                1_737_265_434_024_182_251,
            ]
        );
    }

    #[test]
    fn default_rng_permutation_matches_numpy_seed_42() {
        assert_eq!(
            permutation(20, 42),
            [
                15, 9, 14, 7, 12, 10, 6, 19, 3, 0, 16, 5, 11, 18, 2, 4, 17, 1, 13, 8
            ]
        );
    }

    #[test]
    fn larger_permutation_matches_numpy_sampled_positions() {
        let permutation = permutation(10_000, 42);
        assert_eq!(
            &permutation[..12],
            [
                8132, 8268, 719, 8237, 4555, 6651, 5650, 9034, 1260, 644, 9098, 3760
            ]
        );
        assert_eq!(
            &permutation[permutation.len() - 12..],
            [
                5952, 3375, 1137, 8941, 5004, 437, 491, 2189, 4802, 5668, 5729, 9864
            ]
        );
    }
}
