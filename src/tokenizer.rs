//! Rust BPE tokenizer and nanoAgent conversation serialization.

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    fs,
    path::Path,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, bail, ensure};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const SPECIAL_TOKENS: [&str; 9] = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
];

pub const SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MergeableToken {
    bytes: String,
    id: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct SpecialToken {
    text: String,
    id: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TokenizerFile {
    pattern: String,
    mergeable_ranks: Vec<MergeableToken>,
    special_tokens: Vec<SpecialToken>,
}

#[derive(Clone)]
pub struct BpeTokenizer {
    pattern: Regex,
    mergeable: HashMap<Vec<u8>, u32>,
    special_to_id: HashMap<String, u32>,
    id_to_bytes: HashMap<u32, Vec<u8>>,
    id_to_special: HashMap<u32, String>,
    bos_id: u32,
    vocab_size: u32,
}

#[derive(Clone, Debug)]
enum PickleOp {
    String(String),
    Bytes(Vec<u8>),
    Integer(u32),
    CodecEncode,
    MemoRef(u32),
    EmptyDict,
    Mark,
    SetItems,
    Other,
}

struct BpePiece {
    bytes: Vec<u8>,
    token: u32,
    previous: Option<usize>,
    next: Option<usize>,
    generation: u32,
    active: bool,
}

impl BpeTokenizer {
    /// Convert nanochat's protocol-2/3/4/5 tiktoken `Encoding` pickle to the native
    /// JSON tokenizer format without importing or executing Python code.
    pub fn convert_tiktoken_pickle(
        input: impl AsRef<Path>,
        output: impl AsRef<Path>,
    ) -> Result<usize> {
        let input = input.as_ref();
        let bytes = fs::read(input)
            .with_context(|| format!("read tokenizer pickle {}", input.display()))?;
        let tokenizer = tokenizer_file_from_pickle(&bytes)?;
        let count = tokenizer.mergeable_ranks.len();
        let output = output.as_ref();
        if let Some(parent) = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(output, serde_json::to_vec(&tokenizer)?)
            .with_context(|| format!("write converted tokenizer {}", output.display()))?;
        Ok(count)
    }

    /// Train nanochat's byte-level BPE and write the native JSON tokenizer format.
    pub fn train_to_file<I>(texts: I, vocab_size: u32, path: impl AsRef<Path>) -> Result<Duration>
    where
        I: IntoIterator<Item = Result<String>>,
    {
        ensure!(
            vocab_size as usize >= SPECIAL_TOKENS.len() + 256,
            "vocabulary must include the 256 byte tokens and all special tokens"
        );
        let mut trainer = rustbpe::Tokenizer::new();
        let training_started = Instant::now();
        trainer.try_train_from_texts(
            texts,
            vocab_size - SPECIAL_TOKENS.len() as u32,
            SPLIT_PATTERN,
        )?;
        let training_time = training_started.elapsed();
        let mergeable_ranks = trainer
            .get_mergeable_ranks_native()
            .into_iter()
            .map(|(bytes, id)| MergeableToken {
                bytes: STANDARD.encode(bytes),
                id,
            })
            .collect::<Vec<_>>();
        let first_special_id = mergeable_ranks.len() as u32;
        let special_tokens = SPECIAL_TOKENS
            .iter()
            .enumerate()
            .map(|(index, text)| SpecialToken {
                text: (*text).to_owned(),
                id: first_special_id + index as u32,
            })
            .collect();
        let mut token_byte_lengths = vec![0i32; mergeable_ranks.len() + SPECIAL_TOKENS.len()];
        for token in &mergeable_ranks {
            let token_id = usize::try_from(token.id).context("token ID exceeds address space")?;
            ensure!(
                token_id < mergeable_ranks.len(),
                "trained tokenizer ranks are not contiguous"
            );
            let token_length = STANDARD
                .decode(&token.bytes)
                .context("decode trained BPE token bytes")?
                .len();
            token_byte_lengths[token_id] =
                i32::try_from(token_length).context("token byte length exceeds int32")?;
        }
        let output = TokenizerFile {
            pattern: SPLIT_PATTERN.to_owned(),
            mergeable_ranks,
            special_tokens,
        };
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec(&output)?)
            .with_context(|| format!("write tokenizer {}", path.display()))?;
        let token_bytes_path = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .join("token_bytes.pt");
        crate::torch_save::save_torch_value(
            crate::torch_save::TorchValue::IntTensor {
                shape: vec![token_byte_lengths.len()],
                values: token_byte_lengths,
            },
            &token_bytes_path,
        )
        .with_context(|| format!("write token byte lengths {}", token_bytes_path.display()))?;
        let trained = Self::load(path)?;
        let sanity_text = "Hello world! This is a test.\nNumbers: 123, 4567, 89\nContractions: I'm, you're, it's\nSpecial chars: @#$%^&*()\nUnicode: 你好世界 🌍";
        let sanity_ids = trained.encode(sanity_text)?;
        ensure!(
            trained.decode(&sanity_ids)? == sanity_text,
            "trained tokenizer failed its text round-trip sanity check"
        );
        Ok(training_time)
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes =
            fs::read(path).with_context(|| format!("read tokenizer file {}", path.display()))?;
        let is_pickle = matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("pkl" | "pickle")
        );
        let data = if is_pickle {
            tokenizer_file_from_pickle(&bytes)?
        } else {
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parse tokenizer file {}", path.display()))?
        };
        let pattern = Regex::new(&data.pattern).context("compile tokenizer split pattern")?;
        let mut mergeable = HashMap::with_capacity(data.mergeable_ranks.len());
        let mut id_to_bytes = HashMap::with_capacity(data.mergeable_ranks.len());
        let mut max_id = None;
        for token in data.mergeable_ranks {
            let bytes = STANDARD
                .decode(token.bytes)
                .context("decode BPE token bytes")?;
            ensure!(!bytes.is_empty(), "empty token in mergeable ranks");
            ensure!(
                mergeable.insert(bytes.clone(), token.id).is_none(),
                "duplicate BPE token bytes"
            );
            ensure!(
                id_to_bytes.insert(token.id, bytes).is_none(),
                "duplicate BPE token ID"
            );
            max_id = Some(max_id.map_or(token.id, |max: u32| max.max(token.id)));
        }
        let special_base = max_id.map_or(0, |id| id + 1);
        let mut special_to_id = HashMap::new();
        let mut id_to_special = HashMap::new();
        for token in data.special_tokens {
            ensure!(
                special_to_id.insert(token.text.clone(), token.id).is_none(),
                "duplicate special token"
            );
            ensure!(
                id_to_special.insert(token.id, token.text).is_none(),
                "duplicate special token ID"
            );
        }
        for text in SPECIAL_TOKENS {
            ensure!(
                special_to_id.contains_key(text),
                "tokenizer is missing special token {text}"
            );
        }
        let bos_id = special_to_id["<|bos|>"];
        let vocab_size = id_to_bytes
            .keys()
            .chain(id_to_special.keys())
            .copied()
            .max()
            .map_or(special_base, |id| id + 1);
        Ok(Self {
            pattern,
            mergeable,
            special_to_id,
            id_to_bytes,
            id_to_special,
            bos_id,
            vocab_size,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }
    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }

    pub fn special_id(&self, text: &str) -> Result<u32> {
        self.special_to_id
            .get(text)
            .copied()
            .with_context(|| format!("unknown special token {text}"))
    }

    /// Encode ordinary text. Special-token spellings are treated as regular text,
    /// matching tiktoken's `encode_ordinary` behavior.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        for matched in self.pattern.find_iter(text) {
            let piece = matched
                .context("tokenizer pattern matching failed")?
                .as_str();
            ids.extend(self.encode_piece(piece.as_bytes())?);
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
                .mergeable
                .get(&[*byte][..])
                .copied()
                .with_context(|| format!("tokenizer has no base-byte token for {byte}"))?;
            pieces.push(BpePiece {
                bytes: vec![*byte],
                token,
                previous: index.checked_sub(1),
                next: (index + 1 < bytes.len()).then_some(index + 1),
                generation: 0,
                active: true,
            });
        }

        let mut candidates = BinaryHeap::<Reverse<(u32, usize, usize, u32, u32)>>::new();
        for index in 0..pieces.len().saturating_sub(1) {
            push_bpe_candidate(index, &pieces, &self.mergeable, &mut candidates);
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
                push_bpe_candidate(previous, &pieces, &self.mergeable, &mut candidates);
            }
            push_bpe_candidate(left, &pieces, &self.mergeable, &mut candidates);
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

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut bytes = Vec::new();
        for id in ids {
            if let Some(token) = self.id_to_bytes.get(id) {
                bytes.extend_from_slice(token);
            } else if let Some(token) = self.id_to_special.get(id) {
                bytes.extend_from_slice(token.as_bytes());
            } else {
                bail!("unknown token ID {id}");
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    pub fn token_bytes(&self, id: u32) -> Result<Vec<u8>> {
        if let Some(bytes) = self.id_to_bytes.get(&id) {
            Ok(bytes.clone())
        } else if let Some(token) = self.id_to_special.get(&id) {
            Ok(token.as_bytes().to_vec())
        } else {
            bail!("unknown token ID {id}");
        }
    }

    pub fn token_byte_len(&self, id: u32) -> Result<usize> {
        if let Some(bytes) = self.id_to_bytes.get(&id) {
            Ok(bytes.len())
        } else if self.id_to_special.contains_key(&id) {
            Ok(0)
        } else {
            bail!("unknown token ID {id}");
        }
    }

    pub fn render_conversation(
        &self,
        conversation: &Value,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        let normalized = normalize_sft_messages(conversation)?;
        self.render_normalized_conversation(normalized, max_tokens)
    }

    fn render_normalized_conversation(
        &self,
        mut normalized: Vec<Value>,
        max_tokens: usize,
    ) -> Result<(Vec<u32>, Vec<u8>)> {
        if normalized
            .first()
            .and_then(|m| m.get("role"))
            .and_then(Value::as_str)
            == Some("system")
        {
            ensure!(
                normalized.len() >= 2,
                "system message must be followed by a user message"
            );
            ensure!(
                normalized[1].get("role").and_then(Value::as_str) == Some("user"),
                "system message must be followed by a user message"
            );
            let system_content = normalized[0]
                .get("content")
                .and_then(Value::as_str)
                .context("system content must be text")?
                .to_owned();
            let user_content = normalized[1]
                .get("content")
                .and_then(Value::as_str)
                .context("user content must be text")?
                .to_owned();
            normalized[1]["content"] = Value::String(format!("{system_content}\n\n{user_content}"));
            normalized.remove(0);
        }

        let mut ids = vec![self.bos_id];
        let mut mask = vec![0];
        for (index, message) in normalized.iter().enumerate() {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .context("message is missing role")?;
            let expected = if index % 2 == 0 { "user" } else { "assistant" };
            ensure!(
                role == expected,
                "message {index} must be from {expected}, got {role}"
            );
            match role {
                "user" => {
                    let content = message
                        .get("content")
                        .and_then(Value::as_str)
                        .context("user content must be a string")?;
                    push_special(&mut ids, &mut mask, self.special_id("<|user_start|>")?, 0);
                    push_text(&mut ids, &mut mask, self, content, 0)?;
                    push_special(&mut ids, &mut mask, self.special_id("<|user_end|>")?, 0);
                }
                "assistant" => {
                    push_special(
                        &mut ids,
                        &mut mask,
                        self.special_id("<|assistant_start|>")?,
                        0,
                    );
                    let content = message
                        .get("content")
                        .context("assistant message is missing content")?;
                    if let Some(content) = content.as_str() {
                        push_text(&mut ids, &mut mask, self, content, 1)?;
                    } else if let Some(parts) = content.as_array() {
                        for part in parts {
                            let kind = part
                                .get("type")
                                .and_then(Value::as_str)
                                .context("assistant part is missing type")?;
                            let text = part
                                .get("text")
                                .and_then(Value::as_str)
                                .context("assistant part is missing text")?;
                            match kind {
                                "text" => push_text(&mut ids, &mut mask, self, text, 1)?,
                                "python" => {
                                    push_special(
                                        &mut ids,
                                        &mut mask,
                                        self.special_id("<|python_start|>")?,
                                        1,
                                    );
                                    push_text(&mut ids, &mut mask, self, text, 1)?;
                                    push_special(
                                        &mut ids,
                                        &mut mask,
                                        self.special_id("<|python_end|>")?,
                                        1,
                                    );
                                }
                                "python_output" => {
                                    push_special(
                                        &mut ids,
                                        &mut mask,
                                        self.special_id("<|output_start|>")?,
                                        0,
                                    );
                                    push_text(&mut ids, &mut mask, self, text, 0)?;
                                    push_special(
                                        &mut ids,
                                        &mut mask,
                                        self.special_id("<|output_end|>")?,
                                        0,
                                    );
                                }
                                other => bail!("unknown assistant content part type {other}"),
                            }
                        }
                    } else {
                        bail!("assistant content must be text or a list of parts");
                    }
                    push_special(
                        &mut ids,
                        &mut mask,
                        self.special_id("<|assistant_end|>")?,
                        1,
                    );
                }
                other => bail!("unsupported role {other}"),
            }
        }
        ids.truncate(max_tokens);
        mask.truncate(max_tokens);
        Ok((ids, mask))
    }

    pub fn render_for_completion(&self, conversation: &Value) -> Result<Vec<u32>> {
        let mut messages = normalize_sft_messages(conversation)?;
        ensure!(
            messages
                .last()
                .and_then(|m| m.get("role"))
                .and_then(Value::as_str)
                == Some("assistant"),
            "last message must be from assistant"
        );
        messages.pop();
        let (mut ids, _) = self.render_normalized_conversation(messages, usize::MAX)?;
        ids.push(self.special_id("<|assistant_start|>")?);
        Ok(ids)
    }

    pub fn render_user_prompt(&self, user_text: &str) -> Result<Vec<u32>> {
        let mut ids = vec![self.bos_id, self.special_id("<|user_start|>")?];
        ids.extend(self.encode(user_text)?);
        ids.push(self.special_id("<|user_end|>")?);
        ids.push(self.special_id("<|assistant_start|>")?);
        Ok(ids)
    }
}

/// Normalize OpenAI-style SFT records to nanochat's alternating user/assistant
/// conversation format. This is also the conversion used by the Python
/// tools/post-training dataset adapters.
fn normalize_sft_messages(conversation: &Value) -> Result<Vec<Value>> {
    let raw_messages = conversation
        .get("messages")
        .context("conversation must have a messages array")?;
    let decoded_messages = decode_json_field(raw_messages, "messages")?;
    let messages = decoded_messages
        .as_array()
        .context("conversation must have a messages array")?;
    ensure!(
        !messages.is_empty(),
        "conversation must have at least one message"
    );

    let tools = conversation.get("tools").filter(|value| !value.is_null());
    let tools_text = if let Some(tools) = tools {
        let tools = decode_json_field(tools, "tools")?;
        Some(format!(
            "Available tools (JSON schema):\n{}",
            python_json(&tools)?
        ))
    } else {
        None
    };

    let mut converted = Vec::<Value>::new();
    for message in messages {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .context("SFT message is missing a string role")?;
        match role {
            "system" => {
                let text = sft_content_text(message.get("content"));
                let content = match tools_text.as_deref() {
                    Some(tools) if !text.is_empty() => format!("{text}\n\n{tools}"),
                    Some(tools) => tools.to_owned(),
                    None => text,
                };
                converted.push(serde_json::json!({"role":"system", "content":content}));
            }
            "user" => converted.push(serde_json::json!({
                "role":"user",
                "content":sft_content_text(message.get("content")),
            })),
            "assistant" => {
                let mut parts = Vec::new();
                let mut tool_calls = Vec::new();
                let content = message.get("content").unwrap_or(&Value::Null);
                if let Some(text) = content.as_str() {
                    if !text.is_empty() {
                        parts.push(serde_json::json!({"type":"text", "text":text}));
                    }
                } else if !content.is_null() && !content.is_array() {
                    parts.push(serde_json::json!({
                        "type":"text",
                        "text":sft_content_text(Some(content)),
                    }));
                } else if let Some(items) = content.as_array() {
                    let already_normalized = items.iter().any(|item| {
                        matches!(
                            item.get("type").and_then(Value::as_str),
                            Some("python" | "python_output")
                        )
                    });
                    if already_normalized {
                        for item in items {
                            match item.get("type").and_then(Value::as_str) {
                                Some("text") => {
                                    if let Some(text) = item.get("text").and_then(Value::as_str)
                                        && !text.is_empty()
                                    {
                                        parts.push(serde_json::json!({"type":"text", "text":text}));
                                    }
                                }
                                Some("python") | Some("python_output") => parts.push(item.clone()),
                                _ => {}
                            }
                        }
                    } else {
                        let text = sft_content_text(Some(content));
                        if !text.is_empty() {
                            parts.push(serde_json::json!({"type":"text", "text":text}));
                        }
                    }
                    for item in items {
                        if item.get("type").and_then(Value::as_str) == Some("toolCall") {
                            tool_calls.push(serde_json::json!({
                                "function": {
                                    "name": item.get("name").cloned().unwrap_or(Value::String(String::new())),
                                    "arguments": item.get("arguments").cloned().unwrap_or(serde_json::json!({})),
                                }
                            }));
                        }
                    }
                }
                if tool_calls.is_empty() {
                    if let Some(calls) = message.get("tool_calls").and_then(Value::as_array) {
                        tool_calls.extend(calls.iter().cloned());
                    }
                }
                if tool_calls.is_empty() {
                    if let Some(call) = message.get("function_call") {
                        tool_calls.push(serde_json::json!({"function":call}));
                    }
                }
                for call in tool_calls {
                    let function = call.get("function").unwrap_or(&call);
                    let name = function
                        .get("name")
                        .cloned()
                        .unwrap_or(Value::String(String::new()));
                    let arguments = match function.get("arguments") {
                        Some(value) if value.is_string() => {
                            decode_json_field(value, "tool arguments")
                                .unwrap_or_else(|_| value.clone())
                        }
                        Some(value) => value.clone(),
                        None => serde_json::json!({}),
                    };
                    let call_text = python_json(&serde_json::json!({
                        "name": name,
                        "arguments": arguments,
                    }))?;
                    parts.push(serde_json::json!({"type":"python", "text":call_text}));
                }
                if !parts.is_empty() {
                    converted.push(serde_json::json!({"role":"assistant", "content":parts}));
                }
            }
            "tool" => {
                let name = message
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or("tool");
                let call_id = message
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let result = sft_content_text(message.get("content"));
                converted.push(serde_json::json!({
                    "role":"user",
                    "content":format!("Tool result ({name}, {call_id}):\n{result}"),
                }));
            }
            other => bail!("unsupported role in SFT conversation: {other}"),
        }
    }

    if let Some(tools_text) = tools_text
        && converted
            .first()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            != Some("system")
    {
        converted.insert(
            0,
            serde_json::json!({"role":"system", "content":tools_text}),
        );
    }

    let mut normalized = Vec::<Value>::new();
    for message in converted {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if let Some(previous) = normalized.last_mut()
            && previous.get("role").and_then(Value::as_str) == Some(role)
        {
            if role == "assistant" {
                let previous_parts = previous
                    .get_mut("content")
                    .and_then(Value::as_array_mut)
                    .context("normalized assistant content is not a parts array")?;
                let mut current = message
                    .get("content")
                    .and_then(Value::as_array)
                    .context("normalized assistant content is not a parts array")?
                    .clone();
                previous_parts.append(&mut current);
            } else {
                let previous_content = previous
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                let current_content = message
                    .get("content")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                previous["content"] =
                    Value::String(format!("{previous_content}\n\n{current_content}"));
            }
        } else {
            normalized.push(message);
        }
    }

    if normalized
        .first()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        == Some("system")
    {
        ensure!(
            normalized
                .get(1)
                .and_then(|message| message.get("role"))
                .and_then(Value::as_str)
                == Some("user"),
            "SFT system message must be followed by a user message"
        );
    }
    ensure!(
        normalized
            .last()
            .and_then(|message| message.get("role"))
            .and_then(Value::as_str)
            == Some("assistant"),
        "SFT conversation must end with an assistant message"
    );
    Ok(normalized)
}

fn decode_json_field(value: &Value, field: &str) -> Result<Value> {
    if let Some(text) = value.as_str() {
        serde_json::from_str(text).with_context(|| format!("SFT {field} is not valid JSON"))
    } else {
        Ok(value.clone())
    }
}

fn sft_content_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(text)) => text.clone(),
        Some(Value::Array(parts)) => parts
            .iter()
            .filter_map(|part| {
                if let Some(text) = part.as_str() {
                    Some(text.to_owned())
                } else {
                    part.get("text")?.as_str().map(str::to_owned)
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        Some(other) => crate::api::python_str(other),
    }
}

fn python_json(value: &Value) -> Result<String> {
    match value {
        Value::Null | Value::Bool(_) => Ok(value.to_string()),
        Value::Number(number) => Ok(python_json_number(number)),
        Value::String(text) => Ok(python_json_string(text)),
        Value::Array(values) => Ok(format!(
            "[{}]",
            values
                .iter()
                .map(python_json)
                .collect::<Result<Vec<_>>>()?
                .join(", ")
        )),
        Value::Object(values) => {
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            let fields = entries
                .into_iter()
                .map(|(key, value)| {
                    Ok(format!(
                        "{}: {}",
                        python_json_string(key),
                        python_json(value)?
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(format!("{{{}}}", fields.join(", ")))
        }
    }
}

fn python_json_number(number: &serde_json::Number) -> String {
    if number.is_f64()
        && let Some(value) = number.as_f64()
    {
        return crate::json_compat::python_float_str(value);
    }
    let rendered = number.to_string();
    let Some((mantissa, exponent)) = rendered.split_once(['e', 'E']) else {
        return rendered;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return rendered;
    };
    format!(
        "{mantissa}e{}{abs_exponent:02}",
        if exponent < 0 { "-" } else { "+" },
        abs_exponent = exponent.unsigned_abs()
    )
}

/// Python's `json.dumps` escapes non-ASCII characters by default. Tool
/// schemas and decoded tool arguments are embedded directly in SFT prompts,
/// so preserve that spelling rather than emitting raw UTF-8.
fn python_json_string(text: &str) -> String {
    let mut output = String::with_capacity(text.len() + 2);
    output.push('"');
    for character in text.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{000c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character <= '\u{001f}' => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character if character.is_ascii() => output.push(character),
            character if (character as u32) <= 0xffff => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => {
                let value = character as u32 - 0x1_0000;
                let high = 0xd800 + (value >> 10);
                let low = 0xdc00 + (value & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    output.push('"');
    output
}

#[cfg(test)]
mod sft_normalization_tests {
    use super::{normalize_sft_messages, python_json};
    use serde_json::json;

    #[test]
    fn tool_arguments_and_tool_schemas_use_sorted_json_keys() {
        let conversation = json!({
            "tools": [{"z_schema": 2, "a_schema": 1}],
            "messages": [
                {"role":"user", "content":"look this up"},
                {
                    "role":"assistant",
                    "tool_calls":[{
                        "function":{
                            "name":"lookup",
                            "arguments":{"z":2, "a":1}
                        }
                    }]
                }
            ]
        });
        let normalized = normalize_sft_messages(&conversation).unwrap();
        assert_eq!(
            normalized[0]["content"],
            "Available tools (JSON schema):\n[{\"a_schema\": 1, \"z_schema\": 2}]"
        );
        assert_eq!(
            normalized[2]["content"][0]["text"],
            "{\"arguments\": {\"a\": 1, \"z\": 2}, \"name\": \"lookup\"}"
        );
    }

    #[test]
    fn sft_non_string_content_uses_python_str_formatting() {
        let conversation = json!({
            "messages": [
                {"role":"user", "content":{"first":true, "second":null}},
                {"role":"assistant", "content":"done"}
            ]
        });
        let normalized = normalize_sft_messages(&conversation).unwrap();
        assert_eq!(normalized[0]["content"], "{'first': True, 'second': None}");
    }

    #[test]
    fn sft_tool_json_pads_float_exponents_like_python_json_dumps() {
        let conversation = json!({
            "messages": [
                {"role":"user", "content":"use these values"},
                {"role":"assistant", "tool_calls":[{
                    "function":{"name":"f", "arguments":{"tiny":1e-7, "large":1e20}}
                }]}
            ]
        });
        let normalized = normalize_sft_messages(&conversation).unwrap();
        assert_eq!(
            normalized[1]["content"][0]["text"],
            "{\"arguments\": {\"large\": 1e+20, \"tiny\": 1e-07}, \"name\": \"f\"}"
        );
        assert_eq!(python_json(&json!(1e-5)).unwrap(), "1e-05");
        assert_eq!(python_json(&json!(1e-4)).unwrap(), "0.0001");
        assert_eq!(python_json(&json!(1e15)).unwrap(), "1000000000000000.0");
        assert_eq!(python_json(&json!(1e16)).unwrap(), "1e+16");
    }

    #[test]
    fn sft_tool_json_escapes_non_ascii_like_python_json_dumps() {
        let value = json!({"café": "snowman ☃ and emoji 😀"});
        assert_eq!(
            python_json(&value).unwrap(),
            r#"{"caf\u00e9": "snowman \u2603 and emoji \ud83d\ude00"}"#
        );
    }
}

fn tokenizer_file_from_pickle(bytes: &[u8]) -> Result<TokenizerFile> {
    let ops = parse_tiktoken_pickle(bytes)?;
    let pattern_position = ops
        .iter()
        .position(|op| matches!(op, PickleOp::String(key) if key == "pat_str"))
        .context("tiktoken pickle has no pat_str")?;
    let pattern = ops[pattern_position + 1..]
        .iter()
        .find_map(|op| match op {
            PickleOp::String(value) => Some(value.clone()),
            _ => None,
        })
        .context("tiktoken pickle pat_str is not a string")?;
    let mergeable_ranks = pickle_dict(&ops, "mergeable_ranks", false)?
        .into_iter()
        .map(|(key, id)| {
            let PickleOp::Bytes(bytes) = key else {
                anyhow::bail!("mergeable_ranks keys must be byte strings");
            };
            let PickleOp::Integer(id) = id else {
                anyhow::bail!("mergeable_ranks values must be non-negative integers");
            };
            Ok(MergeableToken {
                bytes: STANDARD.encode(bytes),
                id,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let special_tokens = pickle_dict(&ops, "special_tokens", true)?
        .into_iter()
        .map(|(key, id)| {
            let PickleOp::String(text) = key else {
                anyhow::bail!("special_tokens keys must be strings");
            };
            let PickleOp::Integer(id) = id else {
                anyhow::bail!("special_tokens values must be non-negative integers");
            };
            Ok(SpecialToken { text, id })
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        !mergeable_ranks.is_empty(),
        "tokenizer pickle has no mergeable ranks"
    );
    ensure!(
        !special_tokens.is_empty(),
        "tokenizer pickle has no special tokens"
    );
    Ok(TokenizerFile {
        pattern,
        mergeable_ranks,
        special_tokens,
    })
}

fn parse_tiktoken_pickle(bytes: &[u8]) -> Result<Vec<PickleOp>> {
    ensure!(
        bytes.len() >= 2 && bytes[0] == 0x80 && matches!(bytes[1], 2..=5),
        "unsupported tokenizer pickle; expected Python pickle protocol 2, 3, 4, or 5"
    );
    let mut ops = Vec::new();
    let mut memo = HashMap::<u32, PickleOp>::new();
    let mut memo_value = None::<PickleOp>;
    let mut offset = 2usize;
    while offset < bytes.len() {
        let opcode_offset = offset;
        let opcode = bytes[offset];
        offset += 1;
        let op = match opcode {
            0x95 => {
                let _ = read_pickle_u64(bytes, &mut offset)?;
                PickleOp::Other // FRAME boundaries do not affect pickle values.
            }
            0x63 => {
                // GLOBAL stores a module and qualified name as newline-delimited
                // strings. The converter only needs the scalar dictionaries in
                // the object's state, not the referenced Python class.
                let module = read_pickle_line(bytes, &mut offset)?;
                let name = read_pickle_line(bytes, &mut offset)?;
                if module == "_codecs" && name == "encode" {
                    PickleOp::CodecEncode
                } else {
                    PickleOp::Other
                }
            }
            0x71 => {
                let index = read_pickle_u8(bytes, &mut offset)? as u32;
                if let Some(value) = &memo_value {
                    memo.insert(index, value.clone());
                }
                continue;
            }
            0x72 => {
                let index = read_pickle_u32(bytes, &mut offset)?;
                if let Some(value) = &memo_value {
                    memo.insert(index, value.clone());
                }
                continue;
            }
            0x68 => {
                let index = read_pickle_u8(bytes, &mut offset)? as u32;
                memo.get(&index).cloned().unwrap_or(PickleOp::Other)
            }
            0x6a => {
                let index = read_pickle_u32(bytes, &mut offset)?;
                match memo.get(&index) {
                    Some(PickleOp::CodecEncode) => PickleOp::MemoRef(index),
                    Some(value) => value.clone(),
                    None => PickleOp::Other,
                }
            }
            0x52 => {
                let len = ops.len();
                let encoded = if len >= 4 {
                    let codec = match &ops[len - 4] {
                        PickleOp::CodecEncode => true,
                        PickleOp::MemoRef(index) => {
                            matches!(memo.get(index), Some(PickleOp::CodecEncode))
                        }
                        _ => false,
                    };
                    let text = match &ops[len - 3] {
                        PickleOp::String(text) => Some(text),
                        _ => None,
                    };
                    let latin1 =
                        matches!(&ops[len - 2], PickleOp::String(encoding) if encoding == "latin1");
                    let tuple = matches!(ops[len - 1], PickleOp::Other);
                    if codec && latin1 && tuple {
                        text.map(|text| {
                            text.chars()
                                .map(|character| {
                                    u8::try_from(character as u32).context(
                                        "non-Latin-1 character in tokenizer pickle byte key",
                                    )
                                })
                                .collect::<Result<Vec<_>>>()
                        })
                        .transpose()?
                    } else {
                        None
                    }
                } else {
                    None
                };
                if let Some(bytes) = encoded {
                    ops.truncate(len - 4);
                    PickleOp::Bytes(bytes)
                } else {
                    PickleOp::Other
                }
            }
            0x8c => PickleOp::String(read_pickle_string(bytes, &mut offset, 1)?),
            0x58 => PickleOp::String(read_pickle_string(bytes, &mut offset, 4)?),
            0x8d => PickleOp::String(read_pickle_string(bytes, &mut offset, 8)?),
            0x43 => PickleOp::Bytes(read_pickle_bytes(bytes, &mut offset, 1)?),
            0x42 => PickleOp::Bytes(read_pickle_bytes(bytes, &mut offset, 4)?),
            0x8e => PickleOp::Bytes(read_pickle_bytes(bytes, &mut offset, 8)?),
            0x4b => PickleOp::Integer(read_pickle_u8(bytes, &mut offset)? as u32),
            0x4d => PickleOp::Integer(read_pickle_u16(bytes, &mut offset)? as u32),
            0x4a => {
                let value = read_pickle_u32(bytes, &mut offset)? as i32;
                ensure!(value >= 0, "negative token ID in tokenizer pickle");
                PickleOp::Integer(value as u32)
            }
            0x7d => PickleOp::EmptyDict,
            0x28 => PickleOp::Mark,
            0x75 => PickleOp::SetItems,
            0x94 | 0x93 | 0x29 | 0x81 | 0x86 | 0x62 => PickleOp::Other,
            0x2e => {
                ops.push(PickleOp::Other);
                break;
            }
            _ => anyhow::bail!("unsupported pickle opcode 0x{opcode:02x} at byte {opcode_offset}"),
        };
        memo_value = match &op {
            PickleOp::String(_)
            | PickleOp::Bytes(_)
            | PickleOp::Integer(_)
            | PickleOp::CodecEncode => Some(op.clone()),
            PickleOp::Other | PickleOp::EmptyDict | PickleOp::Mark | PickleOp::SetItems => None,
            PickleOp::MemoRef(index) => memo.get(index).cloned(),
        };
        ops.push(op);
    }
    ensure!(offset <= bytes.len(), "truncated tokenizer pickle");
    Ok(ops)
}

fn pickle_dict(
    ops: &[PickleOp],
    name: &str,
    string_keys: bool,
) -> Result<Vec<(PickleOp, PickleOp)>> {
    let field_position = ops
        .iter()
        .position(|op| matches!(op, PickleOp::String(key) if key == name))
        .with_context(|| format!("tiktoken pickle has no {name}"))?;
    let dict_position = ops[field_position + 1..]
        .iter()
        .position(|op| matches!(op, PickleOp::EmptyDict))
        .map(|position| field_position + 1 + position)
        .with_context(|| format!("tiktoken pickle {name} is not a dictionary"))?;
    let mut entries = Vec::new();
    let mut cursor = dict_position + 1;
    let mut found_batch = false;
    loop {
        while matches!(ops.get(cursor), Some(PickleOp::Other)) {
            cursor += 1;
        }
        if !matches!(ops.get(cursor), Some(PickleOp::Mark)) {
            break;
        }
        found_batch = true;
        let mark_position = cursor;
        let end_position = ops[mark_position + 1..]
            .iter()
            .position(|op| matches!(op, PickleOp::SetItems))
            .map(|position| mark_position + 1 + position)
            .with_context(|| format!("tiktoken pickle {name} dictionary has no SETITEMS"))?;
        let values = ops[mark_position + 1..end_position]
            .iter()
            .filter(|op| !matches!(op, PickleOp::Other))
            .collect::<Vec<_>>();
        ensure!(
            values.len() % 2 == 0,
            "tiktoken pickle {name} has an unmatched key"
        );
        for pair in values.chunks_exact(2) {
            ensure!(
                if string_keys {
                    matches!(pair[0], PickleOp::String(_))
                } else {
                    matches!(pair[0], PickleOp::Bytes(_))
                },
                "tiktoken pickle {name} has a key of the wrong type: {:?}",
                pair[0]
            );
            ensure!(
                matches!(pair[1], PickleOp::Integer(_)),
                "tiktoken pickle {name} has a non-integer value"
            );
            entries.push((clone_pickle_scalar(pair[0])?, clone_pickle_scalar(pair[1])?));
        }
        cursor = end_position + 1;
    }
    ensure!(
        found_batch,
        "tiktoken pickle {name} dictionary has no MARK/SETITEMS"
    );
    Ok(entries)
}

fn clone_pickle_scalar(op: &PickleOp) -> Result<PickleOp> {
    match op {
        PickleOp::String(value) => Ok(PickleOp::String(value.clone())),
        PickleOp::Bytes(value) => Ok(PickleOp::Bytes(value.clone())),
        PickleOp::Integer(value) => Ok(PickleOp::Integer(*value)),
        _ => anyhow::bail!("unexpected non-scalar value in tiktoken pickle"),
    }
}

fn read_pickle_u8(bytes: &[u8], offset: &mut usize) -> Result<u8> {
    let value = *bytes.get(*offset).context("truncated tokenizer pickle")?;
    *offset += 1;
    Ok(value)
}

fn read_pickle_u16(bytes: &[u8], offset: &mut usize) -> Result<u16> {
    let slice = bytes
        .get(*offset..*offset + 2)
        .context("truncated tokenizer pickle")?;
    *offset += 2;
    Ok(u16::from_le_bytes([slice[0], slice[1]]))
}

fn read_pickle_u32(bytes: &[u8], offset: &mut usize) -> Result<u32> {
    let slice = bytes
        .get(*offset..*offset + 4)
        .context("truncated tokenizer pickle")?;
    *offset += 4;
    Ok(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
}

fn read_pickle_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let slice = bytes
        .get(*offset..*offset + 8)
        .context("truncated tokenizer pickle")?;
    *offset += 8;
    Ok(u64::from_le_bytes([
        slice[0], slice[1], slice[2], slice[3], slice[4], slice[5], slice[6], slice[7],
    ]))
}

fn read_pickle_string(bytes: &[u8], offset: &mut usize, length_width: usize) -> Result<String> {
    let length = match length_width {
        1 => read_pickle_u8(bytes, offset)? as usize,
        4 => read_pickle_u32(bytes, offset)? as usize,
        8 => usize::try_from(read_pickle_u64(bytes, offset)?)
            .context("tokenizer pickle string is too large")?,
        _ => anyhow::bail!("unsupported pickle string length width"),
    };
    String::from_utf8(read_pickle_payload(bytes, offset, length)?.to_vec())
        .context("tokenizer pickle contains invalid UTF-8")
}

fn read_pickle_bytes(bytes: &[u8], offset: &mut usize, length_width: usize) -> Result<Vec<u8>> {
    let length = match length_width {
        1 => read_pickle_u8(bytes, offset)? as usize,
        4 => read_pickle_u32(bytes, offset)? as usize,
        8 => usize::try_from(read_pickle_u64(bytes, offset)?)
            .context("tokenizer pickle byte string is too large")?,
        _ => anyhow::bail!("unsupported pickle bytes length width"),
    };
    Ok(read_pickle_payload(bytes, offset, length)?.to_vec())
}

fn read_pickle_payload<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .context("tokenizer pickle length overflows address space")?;
    let payload = bytes
        .get(*offset..end)
        .context("truncated tokenizer pickle")?;
    *offset = end;
    Ok(payload)
}

fn read_pickle_line<'a>(bytes: &'a [u8], offset: &mut usize) -> Result<&'a str> {
    let remaining = bytes.get(*offset..).context("truncated tokenizer pickle")?;
    let length = remaining
        .iter()
        .position(|byte| *byte == b'\n')
        .context("unterminated GLOBAL field in tokenizer pickle")?;
    let line = std::str::from_utf8(&remaining[..length])
        .context("GLOBAL field in tokenizer pickle is not UTF-8")?;
    *offset += length + 1;
    Ok(line)
}

fn push_bpe_candidate(
    left: usize,
    pieces: &[BpePiece],
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
    let mut merged = Vec::with_capacity(pieces[left].bytes.len() + pieces[right].bytes.len());
    merged.extend_from_slice(&pieces[left].bytes);
    merged.extend_from_slice(&pieces[right].bytes);
    if let Some(&rank) = ranks.get(&merged) {
        candidates.push(Reverse((
            rank,
            left,
            right,
            pieces[left].generation,
            pieces[right].generation,
        )));
    }
}

fn push_special(ids: &mut Vec<u32>, mask: &mut Vec<u8>, id: u32, supervised: u8) {
    ids.push(id);
    mask.push(supervised);
}

fn push_text(
    ids: &mut Vec<u32>,
    mask: &mut Vec<u8>,
    tokenizer: &BpeTokenizer,
    text: &str,
    supervised: u8,
) -> Result<()> {
    let encoded = tokenizer.encode(text)?;
    mask.extend(std::iter::repeat_n(supervised, encoded.len()));
    ids.extend(encoded);
    Ok(())
}
