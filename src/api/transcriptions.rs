//! POST /v1/audio/transcriptions: OpenAI-compatible multipart transcription.
//!
//! This is the Open WebUI dictation contract: multipart fields `file`
//! (required), `model`, `language`, `response_format` (json|text, default
//! json). Unknown extra fields are ignored, like the OpenAI API does.

use std::sync::Arc;
use std::time::Duration;

use axum::extract::multipart::MultipartError;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::api::error::ApiError;
use crate::audio::{TARGET_SR, decode_to_pcm_16k};
use crate::chunk::chunk_ranges;
use crate::engine::EngineError;
use crate::server::AppState;

/// How long a request may wait for a free engine slot before 503.
const QUEUE_TIMEOUT: Duration = Duration::from_secs(60);

enum ResponseFormat {
    Json,
    Text,
}

pub async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let limit_mb = state.cfg.max_upload_mb;
    let mut file: Option<Vec<u8>> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| multipart_error(e, limit_mb))?
    {
        match field.name().unwrap_or("") {
            "file" => {
                let bytes = field
                    .bytes()
                    .await
                    .map_err(|e| multipart_error(e, limit_mb))?;
                file = Some(bytes.to_vec());
            }
            "model" => model = Some(text_field(field, limit_mb).await?),
            "language" => language = Some(text_field(field, limit_mb).await?),
            "response_format" => response_format = Some(text_field(field, limit_mb).await?),
            _ => {} // ignore unknown fields (temperature, prompt, ...)
        }
    }

    let format = match response_format.as_deref() {
        None | Some("json") => ResponseFormat::Json,
        Some("text") => ResponseFormat::Text,
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "unsupported response_format: {other} (expected json or text)"
            )));
        }
    };
    let file = file.ok_or_else(|| ApiError::bad_request("missing required field: file"))?;
    let pcm = decode_to_pcm_16k(&file).map_err(|e| ApiError::bad_request(e.to_string()))?;
    let language = language.or_else(|| state.cfg.language.clone());

    let ranges = chunk_ranges(
        &pcm,
        TARGET_SR,
        state.cfg.chunk_max_sec,
        state.cfg.vad_threshold,
    );
    // Shared so per-chunk blocking tasks borrow ranges without copying PCM.
    let pcm = Arc::new(pcm);
    let mut parts: Vec<String> = Vec::with_capacity(ranges.len());
    for range in ranges {
        let permit = tokio::time::timeout(QUEUE_TIMEOUT, state.sem.acquire())
            .await
            .map_err(|_| ApiError::busy())?
            .map_err(|e| ApiError::internal(format!("semaphore closed: {e}")))?;
        let engine = Arc::clone(&state.engine);
        let pcm = Arc::clone(&pcm);
        let model = model.clone();
        let language = language.clone();
        let text = tokio::task::spawn_blocking(move || {
            engine.transcribe(&pcm[range], model.as_deref(), language.as_deref())
        })
        .await
        .map_err(|e| ApiError::internal(format!("transcription task failed: {e}")))?
        .map_err(engine_error)?;
        drop(permit);
        parts.push(text);
    }
    let text = parts.join(" ").trim().to_string();

    Ok(match format {
        ResponseFormat::Json => axum::Json(json!({ "text": text })).into_response(),
        ResponseFormat::Text => text.into_response(),
    })
}

async fn text_field(
    field: axum::extract::multipart::Field<'_>,
    limit_mb: usize,
) -> Result<String, ApiError> {
    field.text().await.map_err(|e| multipart_error(e, limit_mb))
}

/// Body-over-limit surfaces while reading multipart; keep our error shape.
fn multipart_error(err: MultipartError, limit_mb: usize) -> ApiError {
    if err.status() == StatusCode::PAYLOAD_TOO_LARGE {
        ApiError::too_large(limit_mb)
    } else {
        ApiError::bad_request(format!("invalid multipart body: {}", err.body_text()))
    }
}

fn engine_error(err: EngineError) -> ApiError {
    match err {
        EngineError::UnknownModel(_) => ApiError::bad_request(err.to_string()),
        EngineError::Failed(_) => ApiError::internal(err.to_string()),
    }
}
