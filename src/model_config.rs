//! Typed GPT configuration shared by inference, training, and checkpoint IO.

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(default)]
pub struct GptConfig {
    pub sequence_len: usize,
    pub vocab_size: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    pub n_embd: usize,
    /// Layer pattern using `S` for quarter-context attention and `L` for full context.
    pub window_pattern: String,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            sequence_len: 2048,
            vocab_size: 32768,
            n_layer: 12,
            n_head: 6,
            n_kv_head: 6,
            n_embd: 768,
            window_pattern: "SSSL".into(),
        }
    }
}

impl GptConfig {
    /// Deserialize a checkpoint config and apply nanochat's legacy defaults.
    /// Before sliding-window attention was added, missing window patterns meant
    /// full-context attention on every layer.
    pub fn from_value(mut value: serde_json::Value) -> Result<Self> {
        if let Some(object) = value.as_object_mut() {
            if !object.contains_key("window_pattern") {
                object.insert(
                    "window_pattern".to_owned(),
                    serde_json::Value::String("L".to_owned()),
                );
            }
        }
        Ok(serde_json::from_value(value)?)
    }

    /// Check shape constraints before allocating model parameters or KV caches.
    pub fn validate(&self) -> Result<()> {
        if self.sequence_len == 0 || self.vocab_size == 0 || self.n_layer == 0 {
            bail!("sequence_len, vocab_size, and n_layer must be positive");
        }
        if self.n_head == 0 || self.n_kv_head == 0 || self.n_embd == 0 {
            bail!("n_embd, n_head, and n_kv_head must be positive");
        }
        if self.n_embd < 24 {
            bail!("n_embd must be at least 24 for nanochat's smear gate");
        }
        if self.n_embd % self.n_head != 0 {
            bail!(
                "n_embd ({}) must be divisible by n_head ({})",
                self.n_embd,
                self.n_head
            );
        }
        if self.n_kv_head > self.n_head || self.n_head % self.n_kv_head != 0 {
            bail!("n_kv_head must divide n_head and cannot exceed it");
        }
        let head_dim = self.n_embd / self.n_head;
        if head_dim % 2 != 0 {
            bail!("head dimension must be even for rotary embeddings");
        }
        if self.window_pattern.is_empty()
            || !self.window_pattern.bytes().all(|b| b == b'S' || b == b'L')
        {
            bail!("window_pattern must be a non-empty sequence of 'S' and 'L'");
        }
        Ok(())
    }

    /// The Python implementation tiles its pattern across layers and forces the
    /// final layer to full context regardless of the pattern's final character.
    pub fn is_long_window_layer(&self, layer_index: usize) -> bool {
        layer_index + 1 == self.n_layer
            || self.window_pattern.as_bytes()[layer_index % self.window_pattern.len()] == b'L'
    }

    pub fn attention_window(&self, layer_index: usize) -> (isize, isize) {
        if self.is_long_window_layer(layer_index) {
            (-1, 0)
        } else {
            ((self.sequence_len / 4) as isize, 0)
        }
    }
}
