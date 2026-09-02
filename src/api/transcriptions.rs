//! POST /v1/audio/transcriptions and /v1/audio/translations: OpenAI-compatible
//! multipart speech-to-text.
//!
//! The transcriptions route is the dictation contract every OpenAI STT client
//! sends: multipart fields `file` (required), `model`, `language`,
//! `response_format` (json|verbose_json|text, default json). Unknown extra
//! fields are ignored, like the OpenAI API does. The translations route takes
//! the same fields and asks the model to translate instead; both share one
//! handler because only the task differs.

use std::sync::Arc;

use axum::extract::multipart::MultipartError;
use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

use crate::api::error::ApiError;
use crate::api::exec::transcribe_range;
use crate::api::verbose::{self, Chunk};
use crate::audio::{decode_to_pcm_16k, samples_to_sec};
use crate::chunk::chunk_ranges;
use crate::config::toggle;
use crate::engine::{ModelInfo, Task, TranscribeOptions};
use crate::server::AppState;

enum ResponseFormat {
    Json,
    VerboseJson,
    Text,
}

pub async fn transcribe(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    handle(state, multipart, Task::Transcribe).await
}

/// POST /v1/audio/translations: same request shape, translated output. The
/// target language is the model's own (English for whisper); the
/// `target_language` field picks one on families that offer a choice.
pub async fn translate(
    State(state): State<AppState>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    handle(state, multipart, Task::Translate).await
}

async fn handle(
    state: AppState,
    mut multipart: Multipart,
    task: Task,
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
    let mut target_language: Option<String> = None;

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
            "target_language" => target_language = Some(text_field(field, limit_mb).await?),
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
    // The model this request will run on, resolved once: it answers whether a
    // translation is possible at all and what language the answer will be in.
    let model_info = state.engine.resolve_model(model.as_deref());
    // Checked here rather than left to the engine, which would only see it
    // after libav decoded the whole upload and the request waited its turn for
    // an engine slot -- a long way to go for an answer known from the alias.
    // The engine keeps its own check: it is the authority, this is the shortcut.
    let target_language = match task {
        Task::Transcribe => None,
        Task::Translate => {
            reject_impossible_translation(
                model.as_deref(),
                model_info.as_ref(),
                target_language.as_deref(),
            )?;
            // Normalized once, here: the check above accepts " EN " for a
            // model listing "en", and everything downstream -- the engine's
            // own check, the library, the language the response reports --
            // has to see the same spelling that was accepted.
            target_language.map(|target| match &model_info {
                Some(info) => info.canonical_translate_target(&target).to_string(),
                None => target.trim().to_string(),
            })
        }
    };
    // Decoding (tempfile + libav) is CPU/IO-bound: keep it off the async
    // runtime threads.
    let pcm = tokio::task::spawn_blocking(move || decode_to_pcm_16k(&file))
        .await
        .map_err(|e| ApiError::internal(format!("decode task failed: {e}")))?
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let language = language
        .map(|l| l.trim().to_string())
        .filter(|l| !l.is_empty())
        .or_else(|| state.cfg.language.clone());
    // Request wins over the server-wide default; neither set leaves the model
    // family's own behavior untouched.
    let options = Arc::new(TranscribeOptions {
        model,
        task,
        language: language.clone(),
        target_language,
        pnc: pnc.or(state.cfg.pnc),
        itn: itn.or(state.cfg.itn),
    });

    let ranges = chunk_ranges(
        &pcm,
        state.chunk_max_sec_for(model_info.as_ref()),
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
            let reported = match task {
                Task::Transcribe => language.clone(),
                // The text is in the target language, so saying the source
                // language here would label English output as Russian.
                Task::Translate => options
                    .target_language
                    .clone()
                    .or_else(|| single_translate_target(model_info.as_ref())),
            };
            axum::Json(verbose::body(&chunks, task, duration, reported.as_deref())).into_response()
        }
        ResponseFormat::Text => verbose::joined_text(&chunks).into_response(),
    })
}

/// Refuse a translation the model cannot produce, before anything expensive
/// happens. The rule itself lives on [`ModelInfo`], which is also what the
/// engine asks right before the run.
fn reject_impossible_translation(
    requested: Option<&str>,
    model: Option<&ModelInfo>,
    target: Option<&str>,
) -> Result<(), ApiError> {
    let Some(model) = model else {
        return Ok(()); // No model loaded: the engine reports that, not this.
    };
    // An unknown alias runs on the default model, so the message names the one
    // that was actually consulted; saying only "model 'x' cannot translate"
    // about a model the client never asked for reads as a different bug.
    let named = match requested {
        Some(alias) if alias != model.id => format!("model '{alias}' resolves to '{}'", model.id),
        _ => format!("model '{}'", model.id),
    };
    match model.translation_refusal(&named, target) {
        Some(refusal) => Err(ApiError::bad_request(refusal)),
        None => Ok(()),
    }
}

/// The language a translation will come out in when the caller named none:
/// knowable only when the model advertises exactly one target.
fn single_translate_target(model: Option<&ModelInfo>) -> Option<String> {
    match model?.translate_target_languages.as_slice() {
        [only] => Some(only.clone()),
        _ => None,
    }
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
