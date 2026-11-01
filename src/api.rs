//! OpenAI-compatible chat completions HTTP API with SSE token streaming.

use std::{
    convert::Infallible,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::{Body, Bytes},
    extract::{DefaultBodyLimit, State},
    http::{
        HeaderMap, HeaderValue, StatusCode, Uri,
        header::{ALLOW, AUTHORIZATION},
    },
    response::{
        IntoResponse, Redirect, Response,
        sse::{Event, Sse},
    },
    routing::{get, post},
};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::task::JoinSet;
use tokio_stream::wrappers::UnboundedReceiverStream;

use crate::engine::{Engine, FinishReason};

const NONFINITE_NAN: &str = "\0rust-agent-api-float:nan";
const NONFINITE_INFINITY: &str = "\0rust-agent-api-float:inf";
const NONFINITE_NEG_INFINITY: &str = "\0rust-agent-api-float:-inf";

#[derive(Clone)]
struct ApiState {
    engine: Arc<Mutex<Engine>>,
    model_id: String,
    api_key: Option<String>,
    default_temperature: f64,
    default_top_k: Option<usize>,
}

#[derive(Debug, Serialize)]
struct ChatCompletionResponse {
    id: String,
    object: &'static str,
    created: u64,
    model: String,
    choices: Vec<ChatChoice>,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct ChatChoice {
    index: usize,
    message: ChatMessage,
    finish_reason: &'static str,
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: &'static str,
    content: String,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[derive(Debug, Serialize)]
struct ApiErrorBody {
    error: ApiErrorObject,
}

#[derive(Debug, Serialize)]
struct ApiErrorObject {
    message: String,
    r#type: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<&'static str>,
}

struct ApiError(StatusCode, String);

#[derive(Debug, PartialEq, Eq)]
enum MessageNormalizationError {
    InvalidRequest(String),
    UnhandledAttributeError,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response<Body> {
        let (status, message) = (self.0, self.1);
        (
            status,
            Json(ApiErrorBody {
                error: ApiErrorObject {
                    message,
                    r#type: "invalid_request_error",
                    code: None,
                },
            }),
        )
            .into_response()
    }
}

pub async fn serve(
    engine: Engine,
    model_id: String,
    address: String,
    api_key: Option<String>,
    default_temperature: f64,
    default_top_k: Option<usize>,
) -> anyhow::Result<()> {
    let addresses = tokio::net::lookup_host(address.as_str())
        .await?
        .collect::<Vec<_>>();
    anyhow::ensure!(
        !addresses.is_empty(),
        "bind address resolved to no socket addresses"
    );
    let state = ApiState {
        engine: Arc::new(Mutex::new(engine)),
        model_id,
        api_key,
        default_temperature,
        default_top_k,
    };
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .fallback(not_found)
        .method_not_allowed_fallback(method_not_allowed)
        .layer(DefaultBodyLimit::disable())
        .with_state(state);
    let mut servers = JoinSet::new();
    for address in addresses {
        let listener = TcpListener::bind(address).await?;
        println!("rustAgent API listening on http://{address}/v1");
        let app = app.clone();
        servers.spawn(async move { axum::serve(listener, app).await });
    }
    while let Some(result) = servers.join_next().await {
        result??;
    }
    Ok(())
}

async fn health(State(state): State<ApiState>) -> Json<Value> {
    Json(json!({"status": "ok", "model": state.model_id}))
}

async fn not_found(uri: Uri) -> Response {
    let path = uri.path();
    let route_path = if path.ends_with('/') {
        path.strip_suffix('/')
    } else {
        None
    };
    if route_path.is_some_and(is_api_path) {
        let mut location = path.trim_end_matches('/').to_owned();
        if let Some(query) = uri.query() {
            location.push('?');
            location.push_str(query);
        }
        return Redirect::temporary(&location).into_response();
    }
    (StatusCode::NOT_FOUND, Json(json!({"detail": "Not Found"}))).into_response()
}

async fn method_not_allowed(uri: Uri) -> Response {
    let mut response = (
        StatusCode::METHOD_NOT_ALLOWED,
        Json(json!({"detail": "Method Not Allowed"})),
    )
        .into_response();
    let allowed = match uri.path() {
        "/health" | "/v1/models" => Some("GET, HEAD"),
        "/v1/chat/completions" => Some("POST"),
        _ => None,
    };
    if let Some(allowed) = allowed {
        response
            .headers_mut()
            .insert(ALLOW, HeaderValue::from_static(allowed));
    }
    response
}

fn is_api_path(path: &str) -> bool {
    matches!(path, "/health" | "/v1/models" | "/v1/chat/completions")
}

async fn list_models(
    State(state): State<ApiState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authorize(&state, &headers)?;
    Ok(Json(json!({
        "object": "list",
        "data": [{"id": state.model_id, "object": "model", "created": now(), "owned_by": "nanochat"}]
    })))
}

async fn chat_completions(
    State(state): State<ApiState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, ApiError> {
    authorize(&state, &headers)?;
    let body = parse_python_json(&body).map_err(|_| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "Request body must be valid JSON".into(),
        )
    })?;
    if !body.is_object() {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Request body must be a JSON object".into(),
        ));
    }
    if let Some(requested_model) = body.get("model")
        && requested_model != &Value::String(state.model_id.clone())
    {
        return Err(ApiError(
            StatusCode::NOT_FOUND,
            format!("Model {} is not available", python_repr(requested_model)),
        ));
    }
    let messages = match body.get("messages") {
        Some(Value::Array(messages)) => messages.clone(),
        Some(Value::Bool(true)) => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "'bool' object is not iterable".into(),
            ));
        }
        Some(Value::Number(number)) if json_truthy(&Value::Number(number.clone())) => {
            let python_type = if number.is_i64() || number.is_u64() {
                "int"
            } else {
                "float"
            };
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                format!("'{python_type}' object is not iterable"),
            ));
        }
        Some(value) if nonfinite_float(value).is_some() => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "'float' object is not iterable".into(),
            ));
        }
        Some(value) if json_truthy(value) => {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "each message must be an object".into(),
            ));
        }
        _ => Vec::new(),
    };
    let normalized_messages = match normalize_messages(messages) {
        Ok(messages) => messages,
        Err(MessageNormalizationError::InvalidRequest(message)) => {
            return Err(ApiError(StatusCode::BAD_REQUEST, message));
        }
        Err(MessageNormalizationError::UnhandledAttributeError) => {
            return Ok((StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response());
        }
    };
    let (mut prompt, context_len) = {
        let engine = state.engine.lock().map_err(|_| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                "model lock was poisoned".into(),
            )
        })?;
        let prompt = render_openai_prompt(&engine.tokenizer, &normalized_messages)
            .map_err(|error| ApiError(StatusCode::BAD_REQUEST, error.to_string()))?;
        (prompt, engine.model_context())
    };
    if prompt.len() >= context_len {
        if context_len < 3 {
            return Err(ApiError(
                StatusCode::BAD_REQUEST,
                "Model context length is too small for chat".into(),
            ));
        }
        let end = prompt.len();
        prompt = std::iter::once(prompt[0])
            .chain(prompt[end - (context_len - 2)..end - 1].iter().copied())
            .chain(std::iter::once(prompt[end - 1]))
            .collect();
    }
    let available = context_len.saturating_sub(prompt.len());
    if available == 0 {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "Prompt exceeds the model context length".into(),
        ));
    }
    let requested_max = body
        .get("max_tokens")
        .or_else(|| body.get("max_completion_tokens"))
        .cloned()
        .unwrap_or_else(|| json!(256));
    if python_int_overflows(&requested_max) {
        return Ok((StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response());
    }
    let requested_max = python_int(&requested_max).map_err(|_| {
        ApiError(
            StatusCode::BAD_REQUEST,
            "max_tokens, temperature, and seed must be numeric".into(),
        )
    })?;
    if requested_max < 1 {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "max_tokens must be a positive integer".into(),
        ));
    }
    let max_tokens = requested_max.min(available as i128) as usize;
    let temperature = body
        .get("temperature")
        .map(python_float)
        .transpose()
        .map_err(|_| {
            ApiError(
                StatusCode::BAD_REQUEST,
                "max_tokens, temperature, and seed must be numeric".into(),
            )
        })?
        .unwrap_or(state.default_temperature);
    if temperature < 0.0 {
        return Err(ApiError(
            StatusCode::BAD_REQUEST,
            "temperature must be non-negative".into(),
        ));
    }
    // The Python API treats top-k as a server setting, not a per-request
    // OpenAI parameter, even if a client includes an unknown `top_k` field.
    let top_k = state.default_top_k;
    if body.get("seed").is_some_and(python_int_overflows) {
        return Ok((StatusCode::INTERNAL_SERVER_ERROR, "Internal Server Error").into_response());
    }
    let seed = body
        .get("seed")
        .map(python_int)
        .transpose()
        .map_err(|_| {
            ApiError(
                StatusCode::BAD_REQUEST,
                "max_tokens, temperature, and seed must be numeric".into(),
            )
        })?
        .unwrap_or(42);
    let seed = generator_seed(seed);
    let stream = body.get("stream").is_some_and(json_truthy);
    let id = completion_id();
    let created = now();

    if stream {
        return Ok(stream_completion(
            state,
            prompt,
            max_tokens,
            temperature,
            top_k,
            seed,
            id,
            created,
        ));
    }

    let seed = seed.ok_or_else(|| {
        ApiError(
            StatusCode::INTERNAL_SERVER_ERROR,
            "Generation failed: Overflow when unpacking long long".into(),
        )
    })?;

    let state_for_generation = state.clone();
    let model_id = state.model_id.clone();
    let prompt_tokens = prompt.len();
    let (tokens, finish_reason, content) =
        tokio::task::spawn_blocking(move || -> anyhow::Result<_> {
            let engine = state_for_generation
                .engine
                .lock()
                .map_err(|_| anyhow::anyhow!("model lock was poisoned"))?;
            let mut tokens = Vec::new();
            let finish_reason =
                engine.generate_with(&prompt, max_tokens, temperature, top_k, seed, |token| {
                    tokens.push(token);
                    true
                })?;
            let content = engine.tokenizer.decode(&tokens)?;
            Ok((tokens, finish_reason, content))
        })
        .await
        .map_err(|error| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation failed: {error}"),
            )
        })?
        .map_err(|error| {
            ApiError(
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Generation failed: {error}"),
            )
        })?;
    let completion_tokens = tokens.len();
    let response = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created,
        model: model_id,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant",
                content,
            },
            finish_reason: finish_reason_name(finish_reason),
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    };
    Ok(Json(response).into_response())
}

fn stream_completion(
    state: ApiState,
    prompt: Vec<u32>,
    max_tokens: usize,
    temperature: f64,
    top_k: Option<usize>,
    seed: Option<u64>,
    id: String,
    created: u64,
) -> Response {
    let (sender, receiver) = tokio::sync::mpsc::unbounded_channel();
    let model_id = state.model_id.clone();
    tokio::task::spawn_blocking(move || {
        let send_data = |data: Value| {
            sender
                .send(Ok::<Event, Infallible>(
                    Event::default().data(data.to_string()),
                ))
                .is_ok()
        };
        if !send_data(json!({
            "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
        })) {
            return;
        }
        let result = (|| -> anyhow::Result<FinishReason> {
            let seed = seed.ok_or_else(|| anyhow::anyhow!("Overflow when unpacking long long"))?;
            let engine = state
                .engine
                .lock()
                .map_err(|_| anyhow::anyhow!("model lock was poisoned"))?;
            let mut decoder = encoding_rs::UTF_8.new_decoder_without_bom_handling();
            let mut callback_error = None;
            let finish_reason = engine.generate_with(&prompt, max_tokens, temperature, top_k, seed, |token| {
                let bytes = match engine.tokenizer.token_bytes(token) {
                    Ok(bytes) => bytes,
                    Err(error) => {
                        callback_error = Some(error.to_string());
                        return false;
                    }
                };
                let mut delta = String::with_capacity(
                    bytes.len().saturating_add(3).saturating_mul(3).max(12),
                );
                let _ = decoder.decode_to_string(&bytes, &mut delta, false);
                if delta.is_empty() { return true; }
                send_data(json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": null}]
                }))
            })?;
            if let Some(error) = callback_error {
                anyhow::bail!("stream token rendering failed: {error}");
            }
            let mut tail = String::with_capacity(12);
            let _ = decoder.decode_to_string(&[], &mut tail, true);
            if !tail.is_empty() {
                send_data(json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": tail}, "finish_reason": null}]
                }));
            }
            Ok(finish_reason)
        })();
        match result {
            Ok(reason) => {
                send_data(json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason_name(reason)}]
                }));
            }
            Err(error) => {
                send_data(json!({"error": {"message": error.to_string(), "type": "server_error"}}));
            }
        }
        let _ = sender.send(Ok(Event::default().data("[DONE]")));
    });
    let stream = UnboundedReceiverStream::new(receiver);
    let mut response = Sse::new(stream).into_response();
    response
        .headers_mut()
        .insert("cache-control", "no-cache".parse().expect("valid header"));
    response
        .headers_mut()
        .insert("x-accel-buffering", "no".parse().expect("valid header"));
    response
}

fn normalize_messages(messages: Vec<Value>) -> Result<Vec<Value>, MessageNormalizationError> {
    if messages.is_empty() {
        return Err(MessageNormalizationError::InvalidRequest(
            "messages must contain at least one message".into(),
        ));
    }
    let mut system_parts = Vec::new();
    let mut turns = Vec::new();
    for message in messages {
        if !message.is_object() {
            return Err(MessageNormalizationError::InvalidRequest(
                "each message must be an object".into(),
            ));
        }
        let role_value = message.get("role").cloned().unwrap_or(Value::Null);
        let role = role_value.as_str();
        if matches!(role, Some("system" | "developer")) {
            let content = openai_content_text(message.get("content"))
                .map_err(MessageNormalizationError::InvalidRequest)?;
            system_parts.push(content);
            continue;
        }
        if role == Some("assistant") {
            let rendered_calls = message
                .get("tool_calls")
                .filter(|calls| json_truthy(calls))
                .map(render_tool_calls)
                .transpose()?;
            let mut content = openai_content_text(message.get("content"))
                .map_err(MessageNormalizationError::InvalidRequest)?;
            if let Some(rendered_calls) = rendered_calls {
                if !rendered_calls.is_empty() {
                    if !content.is_empty() {
                        content.push('\n');
                    }
                    content.push_str("Previous tool calls: ");
                    content.push_str(&rendered_calls.join("; "));
                }
            }
            turns.push(json!({"role": role, "content": content}));
        } else if role == Some("user") {
            let content = openai_content_text(message.get("content"))
                .map_err(MessageNormalizationError::InvalidRequest)?;
            turns.push(json!({"role": "user", "content": content}));
        } else if role == Some("tool") {
            let content = openai_content_text(message.get("content"))
                .map_err(MessageNormalizationError::InvalidRequest)?;
            let name = message
                .get("name")
                .map(python_str)
                .unwrap_or_else(|| "tool".to_owned());
            turns.push(
                json!({"role": "user", "content": format!("[Tool output from {name}]\n{content}")}),
            );
        } else {
            // Python extracts content before rejecting an unsupported role.
            openai_content_text(message.get("content"))
                .map_err(MessageNormalizationError::InvalidRequest)?;
            return Err(MessageNormalizationError::InvalidRequest(format!(
                "unsupported message role: {}",
                python_repr(&role_value)
            )));
        }
    }
    if turns
        .last()
        .and_then(|message| message.get("role"))
        .and_then(Value::as_str)
        != Some("user")
    {
        return Err(MessageNormalizationError::InvalidRequest(
            "the final message must be from the user".into(),
        ));
    }
    if !system_parts.is_empty() {
        let current = turns[0]["content"].as_str().unwrap_or_default();
        turns[0]["content"] = Value::String(format!("{}\n\n{current}", system_parts.join("\n\n")));
    }
    Ok(turns)
}

fn render_tool_calls(calls: &Value) -> Result<Vec<String>, MessageNormalizationError> {
    let Some(calls) = calls.as_array() else {
        if nonfinite_float(calls).is_some() {
            return Err(MessageNormalizationError::InvalidRequest(
                "'float' object is not iterable".into(),
            ));
        }
        if let Some(value) = calls.as_bool() {
            if value {
                return Err(MessageNormalizationError::InvalidRequest(
                    "'bool' object is not iterable".into(),
                ));
            }
        } else if let Some(number) = calls.as_number() {
            let python_type = if number.is_i64() || number.is_u64() {
                "int"
            } else {
                "float"
            };
            return Err(MessageNormalizationError::InvalidRequest(format!(
                "'{python_type}' object is not iterable"
            )));
        }
        // Python can iterate strings and objects, but calling `.get` on their
        // first yielded character/key raises an uncaught AttributeError.
        return Err(MessageNormalizationError::UnhandledAttributeError);
    };

    let mut rendered = Vec::with_capacity(calls.len());
    for call in calls {
        let Some(call) = call.as_object() else {
            return Err(MessageNormalizationError::UnhandledAttributeError);
        };
        let function = call.get("function");
        let function = match function {
            None => None,
            Some(value) => Some(
                value
                    .as_object()
                    .ok_or(MessageNormalizationError::UnhandledAttributeError)?,
            ),
        };
        let name = function
            .and_then(|function| function.get("name"))
            .map(python_str)
            .unwrap_or_else(|| "tool".to_owned());
        let arguments = function
            .and_then(|function| function.get("arguments"))
            .map(tool_arguments_text)
            .unwrap_or_else(|| "{}".to_owned());
        rendered.push(format!("{name}({arguments})"));
    }
    Ok(rendered)
}

fn render_openai_prompt(
    tokenizer: &crate::tokenizer::BpeTokenizer,
    turns: &[Value],
) -> anyhow::Result<Vec<u32>> {
    use anyhow::Context;

    let mut tokens = vec![tokenizer.bos_id()];
    for message in turns {
        let role = message
            .get("role")
            .and_then(Value::as_str)
            .context("normalized message is missing a role")?;
        let content = message
            .get("content")
            .and_then(Value::as_str)
            .context("normalized message content must be text")?;
        match role {
            "user" => {
                tokens.push(tokenizer.special_id("<|user_start|>")?);
                tokens.extend(tokenizer.encode(content)?);
                tokens.push(tokenizer.special_id("<|user_end|>")?);
            }
            "assistant" => {
                tokens.push(tokenizer.special_id("<|assistant_start|>")?);
                tokens.extend(tokenizer.encode(content)?);
                tokens.push(tokenizer.special_id("<|assistant_end|>")?);
            }
            other => anyhow::bail!("unsupported normalized role {other}"),
        }
    }
    tokens.push(tokenizer.special_id("<|assistant_start|>")?);
    Ok(tokens)
}

fn tool_arguments_text(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| python_str(value))
}

/// Match Python's `str()` formatting for JSON values used as tool arguments.
/// OpenAI normally sends arguments as a JSON string, but nanochat also accepts
/// already-decoded objects and includes their string form in the prompt.
pub(crate) fn python_str(value: &Value) -> String {
    if let Some(value) = nonfinite_float(value) {
        return nonfinite_repr(value).to_owned();
    }
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::Number(value) if value.is_f64() => crate::json_compat::python_float_str(
            value
                .as_f64()
                .expect("finite JSON float has an f64 representation"),
        ),
        Value::Number(value) => value.to_string(),
        Value::String(value) => value.clone(),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}: {}",
                    python_repr(&Value::String(key.clone())),
                    python_repr(value)
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn python_repr(value: &Value) -> String {
    if let Some(value) = nonfinite_float(value) {
        return nonfinite_repr(value).to_owned();
    }
    match value {
        Value::String(value) => {
            let quote = if value.contains('\'') && !value.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut rendered = String::with_capacity(value.len() + 2);
            rendered.push(quote);
            for character in value.chars() {
                match character {
                    '\\' => rendered.push_str("\\\\"),
                    '\n' => rendered.push_str("\\n"),
                    '\r' => rendered.push_str("\\r"),
                    '\t' => rendered.push_str("\\t"),
                    c if c == quote => {
                        rendered.push('\\');
                        rendered.push(c);
                    }
                    c if c.is_control() => {
                        let code = c as u32;
                        if code <= 0xff {
                            rendered.push_str(&format!("\\x{code:02x}"));
                        } else if code <= 0xffff {
                            rendered.push_str(&format!("\\u{code:04x}"));
                        } else {
                            rendered.push_str(&format!("\\U{code:08x}"));
                        }
                    }
                    c => rendered.push(c),
                }
            }
            rendered.push(quote);
            rendered
        }
        _ => python_str(value),
    }
}

fn json_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

/// `int(float('inf'))` raises OverflowError, which the reference endpoint
/// does not catch. NaN and string input fail with a caught ValueError instead.
fn python_int_overflows(value: &Value) -> bool {
    nonfinite_float(value).is_some_and(f64::is_infinite)
}

fn generator_seed(seed: i128) -> Option<u64> {
    (seed >= i64::MIN as i128 && seed <= u64::MAX as i128).then_some(seed as u64)
}

/// Coerce JSON values like Python's `int()` for request parameters.
fn python_int(value: &Value) -> Result<i128, ()> {
    if nonfinite_float(value).is_some() {
        return Err(());
    }
    match value {
        Value::Bool(value) => Ok(i128::from(*value)),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(value as i128)
            } else if let Some(value) = value.as_u64() {
                Ok(value as i128)
            } else {
                let value = value.as_f64().ok_or(())?;
                if !value.is_finite() {
                    return Err(());
                }
                if value <= i128::MIN as f64 {
                    Ok(i128::MIN)
                } else if value >= i128::MAX as f64 {
                    Ok(i128::MAX)
                } else {
                    Ok(value.trunc() as i128)
                }
            }
        }
        Value::String(value) => {
            let value = value.trim();
            let digits = value
                .strip_prefix('+')
                .or_else(|| value.strip_prefix('-'))
                .unwrap_or(value);
            if digits.is_empty() {
                return Err(());
            }
            let bytes = digits.as_bytes();
            if !underscores_between_digits(bytes)
                || !bytes
                    .iter()
                    .all(|byte| byte.is_ascii_digit() || *byte == b'_')
            {
                return Err(());
            }
            match value.replace('_', "").parse::<i128>() {
                Ok(value) => Ok(value),
                // Python integers are unbounded. Saturating here preserves the
                // later max-token cap while seed validation still rejects an
                // integer outside the supported seed range.
                Err(_) if value.starts_with('-') => Ok(i128::MIN),
                Err(_) => Ok(i128::MAX),
            }
        }
        _ => Err(()),
    }
}

/// Coerce JSON values like Python's `float()` for request parameters.
fn python_float(value: &Value) -> Result<f64, ()> {
    if let Some(value) = nonfinite_float(value) {
        return Ok(value);
    }
    match value {
        Value::Bool(value) => Ok(if *value { 1.0 } else { 0.0 }),
        Value::Number(value) => value.as_f64().ok_or(()),
        Value::String(value) => {
            let value = value.trim();
            if !underscores_between_digits(value.as_bytes()) {
                return Err(());
            }
            let normalized = value.replace('_', "");
            match normalized.to_ascii_lowercase().as_str() {
                "nan" | "+nan" | "-nan" => Ok(f64::NAN),
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
                "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
                _ => normalized.parse().map_err(|_| ()),
            }
        }
        _ => Err(()),
    }
}

fn nonfinite_float(value: &Value) -> Option<f64> {
    match value.as_str()? {
        NONFINITE_NAN => Some(f64::NAN),
        NONFINITE_INFINITY => Some(f64::INFINITY),
        NONFINITE_NEG_INFINITY => Some(f64::NEG_INFINITY),
        _ => None,
    }
}

fn nonfinite_repr(value: f64) -> &'static str {
    if value.is_nan() {
        "nan"
    } else if value.is_sign_negative() {
        "-inf"
    } else {
        "inf"
    }
}

/// Python's standard JSON decoder accepts bare NaN and Infinity constants.
/// Preserve those as private string sentinels so request coercion can retain
/// their float behavior despite serde_json's finite-number-only Value type.
fn parse_python_json(bytes: &[u8]) -> serde_json::Result<Value> {
    let mut normalized = Vec::with_capacity(bytes.len());
    let mut index = 0;
    let mut in_string = false;
    let mut escaped = false;
    while index < bytes.len() {
        let byte = bytes[index];
        if in_string {
            normalized.push(byte);
            if escaped {
                escaped = false;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            }
            index += 1;
            continue;
        }
        if byte == b'"' {
            normalized.push(byte);
            in_string = true;
            index += 1;
            continue;
        }
        let candidate = if bytes[index..].starts_with(b"-Infinity") {
            Some((b"-Infinity".as_slice(), NONFINITE_NEG_INFINITY))
        } else if bytes[index..].starts_with(b"Infinity") {
            Some((b"Infinity".as_slice(), NONFINITE_INFINITY))
        } else if bytes[index..].starts_with(b"NaN") {
            Some((b"NaN".as_slice(), NONFINITE_NAN))
        } else {
            None
        };
        if let Some((literal, sentinel)) = candidate {
            let left_boundary = index == 0
                || matches!(
                    bytes[index - 1],
                    b' ' | b'\t' | b'\r' | b'\n' | b':' | b',' | b'[' | b'{'
                );
            let end = index + literal.len();
            let right_boundary = end == bytes.len()
                || matches!(
                    bytes[end],
                    b' ' | b'\t' | b'\r' | b'\n' | b',' | b']' | b'}'
                );
            if left_boundary && right_boundary {
                serde_json::to_writer(&mut normalized, &sentinel)?;
                index = end;
                continue;
            }
        }
        if byte == b'-' || byte.is_ascii_digit() {
            let end = bytes[index..]
                .iter()
                .position(|byte| matches!(*byte, b',' | b']' | b'}' | b' ' | b'\t' | b'\r' | b'\n'))
                .map_or(bytes.len(), |offset| index + offset);
            let number = &bytes[index..end];
            if number
                .iter()
                .any(|byte| matches!(*byte, b'.' | b'e' | b'E'))
                && let Ok(number) = std::str::from_utf8(number)
                    .unwrap_or_default()
                    .parse::<f64>()
                && !number.is_finite()
            {
                let sentinel = if number.is_sign_negative() {
                    NONFINITE_NEG_INFINITY
                } else {
                    NONFINITE_INFINITY
                };
                serde_json::to_writer(&mut normalized, &sentinel)?;
                index = end;
                continue;
            }
        }
        normalized.push(byte);
        index += 1;
    }
    serde_json::from_slice(&normalized)
}

fn underscores_between_digits(bytes: &[u8]) -> bool {
    bytes.iter().enumerate().all(|(index, byte)| {
        *byte != b'_'
            || (index > 0
                && index + 1 < bytes.len()
                && bytes[index - 1].is_ascii_digit()
                && bytes[index + 1].is_ascii_digit())
    })
}

fn openai_content_text(content: Option<&Value>) -> Result<String, String> {
    let Some(content) = content else {
        return Ok(String::new());
    };
    if content.is_null() {
        return Ok(String::new());
    }
    if let Some(text) = content.as_str() {
        return Ok(text.to_owned());
    }
    if let Some(parts) = content.as_array() {
        let mut text = String::new();
        for part in parts {
            let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
            if !matches!(kind, "text" | "input_text") {
                return Err("nanochat currently accepts text-only message content".into());
            }
            let part_text = part
                .get("text")
                .cloned()
                .unwrap_or_else(|| Value::String(String::new()));
            text.push_str(&python_str(&part_text));
        }
        return Ok(text);
    }
    Err("message content must be a string or a list of text parts".into())
}

fn finish_reason_name(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn completion_id() -> String {
    let mut uuid = rand::random::<u128>();
    // Match uuid.uuid4().hex: RFC 4122 version 4 and variant bits in the
    // canonical 32-character lowercase hexadecimal representation.
    uuid = (uuid & !(0xFu128 << 76)) | (0x4u128 << 76);
    uuid = (uuid & !(0xCu128 << 60)) | (0x8u128 << 60);
    format!("chatcmpl-{uuid:032x}")
}

fn authorize(state: &ApiState, headers: &HeaderMap) -> Result<(), ApiError> {
    if let Some(expected) = state.api_key.as_deref().filter(|key| !key.is_empty()) {
        let supplied = headers
            .get(AUTHORIZATION)
            .and_then(|value| value.to_str().ok())
            .unwrap_or_default();
        let supplied = supplied.strip_prefix("Bearer ").unwrap_or_default();
        if supplied != expected {
            return Err(ApiError(
                StatusCode::UNAUTHORIZED,
                "Invalid or missing API key".into(),
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_string_formats_floats_with_python_notation_thresholds() {
        let values = parse_python_json(br#"[1e-5,1e-6,1e-7,1e15,1e16,1.0,-0.0]"#).unwrap();
        let formatted = values
            .as_array()
            .unwrap()
            .iter()
            .map(python_str)
            .collect::<Vec<_>>();
        assert_eq!(
            formatted,
            [
                "1e-05",
                "1e-06",
                "1e-07",
                "1000000000000000.0",
                "1e+16",
                "1.0",
                "-0.0"
            ]
        );
    }

    #[test]
    fn message_normalization_matches_reference_system_tool_and_content_formatting() {
        let messages = json!([
            {"role":"system", "content":"follow the spec"},
            {"role":"developer", "content":[{"type":"input_text", "text":"carefully"}]},
            {"role":"user", "content":[
                {"type":"text", "text":"find "},
                {"type":"input_text", "text":42}
            ]},
            {"role":"assistant", "content":"looking it up", "tool_calls":[{
                "function":{"name":"lookup", "arguments":{"z":2, "a":true}}
            }]},
            {"role":"tool", "name":"search", "content":[
                {"type":"text", "text":"found "},
                {"type":"text", "text":null}
            ]},
            {"role":"user", "content":"continue"}
        ]);
        let normalized = normalize_messages(messages.as_array().unwrap().clone()).unwrap();
        assert_eq!(
            normalized,
            json!([
                {"role":"user", "content":"follow the spec\n\ncarefully\n\nfind 42"},
                {"role":"assistant", "content":"looking it up\nPrevious tool calls: lookup({'z': 2, 'a': True})"},
                {"role":"user", "content":"[Tool output from search]\nfound None"},
                {"role":"user", "content":"continue"}
            ])
            .as_array()
            .unwrap()
            .clone()
        );
    }

    #[test]
    fn message_normalization_rejects_non_text_content_and_requires_final_user() {
        let non_text = json!([{"role":"user", "content":[{"type":"image_url"}]}]);
        assert_eq!(
            normalize_messages(non_text.as_array().unwrap().clone()).unwrap_err(),
            MessageNormalizationError::InvalidRequest(
                "nanochat currently accepts text-only message content".into()
            )
        );

        let final_assistant = json!([
            {"role":"user", "content":"question"},
            {"role":"assistant", "content":"answer"}
        ]);
        assert_eq!(
            normalize_messages(final_assistant.as_array().unwrap().clone()).unwrap_err(),
            MessageNormalizationError::InvalidRequest(
                "the final message must be from the user".into()
            )
        );
    }

    #[test]
    fn malformed_tool_calls_follow_python_type_and_exception_classification() {
        assert_eq!(
            render_tool_calls(&json!(true)).unwrap_err(),
            MessageNormalizationError::InvalidRequest("'bool' object is not iterable".into())
        );
        assert_eq!(
            render_tool_calls(&json!(2)).unwrap_err(),
            MessageNormalizationError::InvalidRequest("'int' object is not iterable".into())
        );
        assert_eq!(
            render_tool_calls(&json!(2.0)).unwrap_err(),
            MessageNormalizationError::InvalidRequest("'float' object is not iterable".into())
        );
        assert_eq!(
            render_tool_calls(&json!(["not a call"])).unwrap_err(),
            MessageNormalizationError::UnhandledAttributeError
        );
        assert_eq!(
            render_tool_calls(&json!([{"function":null}])).unwrap_err(),
            MessageNormalizationError::UnhandledAttributeError
        );
    }

    #[test]
    fn request_json_accepts_python_nonfinite_constants_without_rewriting_strings() {
        let value = parse_python_json(
            br#"{"nan":NaN,"positive":Infinity,"negative":-Infinity,"text":"NaN Infinity"}"#,
        )
        .unwrap();
        assert!(python_float(&value["nan"]).unwrap().is_nan());
        assert_eq!(python_float(&value["positive"]).unwrap(), f64::INFINITY);
        assert_eq!(python_float(&value["negative"]).unwrap(), f64::NEG_INFINITY);
        assert_eq!(value["text"], "NaN Infinity");
        assert_eq!(python_repr(&value["negative"]), "-inf");
        assert!(python_int(&value["nan"]).is_err());
    }

    #[test]
    fn generator_seed_preserves_pytorch_signed_and_unsigned_boundaries() {
        assert_eq!(generator_seed(-1), Some(u64::MAX));
        assert_eq!(generator_seed(i64::MIN as i128), Some(1u64 << 63));
        assert_eq!(generator_seed(u64::MAX as i128), Some(u64::MAX));
        assert_eq!(generator_seed(i64::MIN as i128 - 1), None);
        assert_eq!(generator_seed(u64::MAX as i128 + 1), None);
    }

    #[test]
    fn integer_conversion_distinguishes_infinity_from_nan_and_strings() {
        let value = parse_python_json(
            br#"{"positive":Infinity,"negative":-1e999,"nan":NaN,"string":"Infinity"}"#,
        )
        .unwrap();
        assert!(python_int_overflows(&value["positive"]));
        assert!(python_int_overflows(&value["negative"]));
        assert!(!python_int_overflows(&value["nan"]));
        assert!(!python_int_overflows(&value["string"]));
        assert!(python_int(&value["nan"]).is_err());
        assert!(python_int(&value["string"]).is_err());
    }

    #[test]
    fn request_json_rejects_nonfinite_words_inside_strings_as_normal_json() {
        assert_eq!(
            parse_python_json(br#"{"temperature":"NaN"}"#).unwrap()["temperature"],
            "NaN"
        );
        assert!(parse_python_json(br#"{"temperature":nan}"#).is_err());
    }

    #[test]
    fn request_json_maps_overflowing_float_literals_to_infinity() {
        let value = parse_python_json(br#"{"positive":1e400,"negative":-1e400}"#).unwrap();
        assert_eq!(python_float(&value["positive"]).unwrap(), f64::INFINITY);
        assert_eq!(python_float(&value["negative"]).unwrap(), f64::NEG_INFINITY);
    }

    #[tokio::test]
    async fn route_fallbacks_use_fastapi_json_error_shapes() {
        let missing_response = not_found(Uri::from_static("/missing")).await;
        assert_eq!(missing_response.status(), StatusCode::NOT_FOUND);
        assert_eq!(
            missing_response.headers()["content-type"],
            "application/json"
        );
        let method_response = method_not_allowed(Uri::from_static("/health")).await;
        assert_eq!(method_response.status(), StatusCode::METHOD_NOT_ALLOWED);
        assert_eq!(
            method_response.headers()["content-type"],
            "application/json"
        );
        assert_eq!(method_response.headers()[ALLOW], "GET, HEAD");
        let redirect = not_found(Uri::from_static("/health/?source=test")).await;
        assert_eq!(redirect.status(), StatusCode::TEMPORARY_REDIRECT);
        assert_eq!(redirect.headers()["location"], "/health?source=test");
    }
}
