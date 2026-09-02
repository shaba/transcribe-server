//! POST /v1/audio/transcriptions: OpenAI-compatible multipart transcription.
//!
//! This is the Open WebUI dictation contract: multipart fields `file`
//! (required), `model`, `language`, `response_format` (json|verbose_json|text,
//! default json). Unknown extra fields are ignored, like the OpenAI API does.

use std::sync::Arc;

use axum::extract::multipart::MultipartError;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::api::error::ApiError;
use crate::api::exec::transcribe_range;
use crate::api::verbose::{self, Chunk};
use crate::audio::{TARGET_SR, decode_to_pcm_16k, samples_to_sec};
use crate::chunk::chunk_ranges;
use crate::config::toggle;
use crate::engine::TranscribeOptions;
use crate::server::AppState;

enum ResponseFormat {
    Json,
    VerboseJson,
    Text,
}

pub async fn transcribe(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Response, ApiError> {
    let limit_mb = state.cfg.max_upload_mb;
    // The upload is buffered whole (both here and, for anything but the WAV
    // fast path, again as a tempfile in the decoder), so --max-upload-mb
    // doubles as the memory guard. Bytes rather than Vec<u8> so the buffer
    // axum already assembled is moved on, not copied a second time.
    let mut file: Option<axum::body::Bytes> = None;
    let mut model: Option<String> = None;
    let mut language: Option<String> = None;
    let mut response_format: Option<String> = None;
    let mut pnc: Option<String> = None;
    let mut itn: Option<String> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| multipart_error(e, limit_mb))?
    {
        match field.name().unwrap_or("") {
            "file" => {
                file = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| multipart_error(e, limit_mb))?,
                );
            }
            "model" => model = Some(text_field(field, limit_mb).await?),
            "language" => language = Some(text_field(field, limit_mb).await?),
            "response_format" => response_format = Some(text_field(field, limit_mb).await?),
            "pnc" => pnc = Some(text_field(field, limit_mb).await?),
            "itn" => itn = Some(text_field(field, limit_mb).await?),
            _ => {} // ignore unknown fields (temperature, prompt, ...)
        }
    }

    // Every field is validated only after the loop has drained the body.
    // Returning early from inside it drops the Multipart while the client is
    // still streaming the file, and the peer sees a reset connection instead
    // of the error JSON -- and field order is the client's choice, so a bad
    // toggle can arrive before a 256 MB upload.
    let pnc = parse_toggle_field("pnc", pnc)?;
    let itn = parse_toggle_field("itn", itn)?;
    let format = match response_format.as_deref() {
        None | Some("json") => ResponseFormat::Json,
        Some("verbose_json") => ResponseFormat::VerboseJson,
        Some("text") => ResponseFormat::Text,
        Some(other) => {
            return Err(ApiError::bad_request(format!(
                "unsupported response_format: {other} (expected json, verbose_json or text)"
            )));
        }
    };
    let file = file.ok_or_else(|| ApiError::bad_request("missing required field: file"))?;
    // Decoding (tempfile + libav) is CPU/IO-bound: keep it off the async
    // runtime threads.
    let pcm = tokio::task::spawn_blocking(move || decode_to_pcm_16k(&file))
        .await
        .map_err(|e| ApiError::internal(format!("decode task failed: {e}")))?
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let language = language.or_else(|| state.cfg.language.clone());
    // Request wins over the server-wide default; neither set leaves the model
    // family's own behavior untouched.
    let options = Arc::new(TranscribeOptions {
        model,
        language: language.clone(),
        pnc: pnc.or(state.cfg.pnc),
        itn: itn.or(state.cfg.itn),
    });

    let ranges = chunk_ranges(
        &pcm,
        TARGET_SR,
        state.cfg.chunk_max_sec,
        state.cfg.vad_threshold,
    );
    // Shared so per-chunk blocking tasks borrow ranges without copying PCM.
    let pcm = Arc::new(pcm);
    let duration = samples_to_sec(pcm.len());
    let mut chunks: Vec<Chunk> = Vec::with_capacity(ranges.len());
    for range in ranges {
        let span = samples_to_sec(range.start)..samples_to_sec(range.end);
        let transcript =
            transcribe_range(&state, Arc::clone(&pcm), range, Arc::clone(&options)).await?;
        chunks.push(Chunk { span, transcript });
    }

    Ok(match format {
        ResponseFormat::Json => {
            axum::Json(json!({ "text": verbose::joined_text(&chunks) })).into_response()
        }
        ResponseFormat::VerboseJson => {
            axum::Json(verbose::body(&chunks, duration, language.as_deref())).into_response()
        }
        ResponseFormat::Text => verbose::joined_text(&chunks).into_response(),
    })
}

/// A boolean request field, spelled exactly like its `--pnc`/`--itn` flag.
fn parse_toggle_field(name: &str, value: Option<String>) -> Result<Option<bool>, ApiError> {
    value
        .map(|text| {
            toggle(&text).map_err(|_| {
                // The value is echoed truncated: a field is only bounded by
                // --max-upload-mb, and a 200 MB error body would defeat the
                // limit that exists to bound memory.
                let mut shown: String = text.chars().take(32).collect();
                if shown.len() < text.len() {
                    shown.push_str("...");
                }
                ApiError::bad_request(format!("{name}: expected on or off, got: {shown}"))
            })
        })
        .transpose()
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
