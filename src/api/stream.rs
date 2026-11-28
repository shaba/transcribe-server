//! WS /v1/audio/stream: streaming transcription over WebSocket.
//!
//! Protocol (control frames are JSON text):
//!   client -> {"type":"start","model":"<alias>"?,"language":"ru"?,
//!              "pnc":true?,"itn":false?}
//!   client -> binary PCM16LE mono 16 kHz frames (any framing)
//!   client -> {"type":"stop"}
//!   server -> {"type":"partial","text":"..."}   per drained chunk
//!   server -> {"type":"final","text":"..."}     on stop, then server closes
//!   server -> {"type":"error","message":"..."}  on any error, then closes
//!
//! Binary frames are only valid after start and must contain a whole
//! number of PCM16LE samples: a dangling odd byte is a protocol error.
//! The final text is every partial text plus the remainder joined with
//! a single space; empty chunk transcripts are skipped.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use serde::Deserialize;
use serde_json::json;

use crate::chunk::{first_cut, window_samples};
use crate::engine::{StreamUpdate, SttStream, TranscribeOptions};
use crate::server::AppState;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Command {
    Start {
        model: Option<String>,
        language: Option<String>,
        pnc: Option<bool>,
        itn: Option<bool>,
    },
    Stop,
}

enum SessionEnd {
    /// Final frame sent; close normally.
    Completed,
    /// Protocol or engine failure: send an error frame, then close.
    Error(String),
    /// Peer went away (transport error or close without stop).
    Disconnected,
}

pub async fn stream(State(state): State<AppState>, ws: WebSocketUpgrade) -> Response {
    stream_with_guard(state, ws, ())
}

/// Keep the caller's activity guard alive through the upgrade and session.
pub fn stream_with_guard(
    state: AppState,
    ws: WebSocketUpgrade,
    guard: impl Send + 'static,
) -> Response {
    // A single frame is turned into an f32 buffer inline; cap it so one
    // client cannot pin the current-thread runtime with a huge allocation.
    let ws = ws
        .max_message_size(MAX_WS_FRAME)
        .max_frame_size(MAX_WS_FRAME);
    ws.on_upgrade(move |socket| async move {
        let _guard = guard;
        handle(socket, state).await;
    })
}

async fn handle(mut socket: WebSocket, state: AppState) {
    match run_session(&mut socket, &state).await {
        SessionEnd::Error(message) => {
            let frame = json!({"type": "error", "message": message}).to_string();
            let _ = socket.send(Message::Text(frame.into())).await;
            let _ = socket.send(Message::Close(None)).await;
        }
        SessionEnd::Completed => {
            let _ = socket.send(Message::Close(None)).await;
        }
        SessionEnd::Disconnected => {}
    }
}

/// PCM16LE bytes -> f32 samples, rejecting a dangling byte.
fn pcm16_le(bytes: &[u8]) -> Result<Vec<f32>, String> {
    if !bytes.len().is_multiple_of(2) {
        return Err("odd-length binary frame: expected whole PCM16LE samples".to_string());
    }
    // as_chunks rather than chunks_exact: the odd byte is already rejected,
    // so the remainder is empty and the array chunks go straight into
    // from_le_bytes without re-indexing.
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|sample| f32::from(i16::from_le_bytes(*sample)) / 32768.0)
        .collect())
}

async fn native_feed(
    stream: &Arc<Mutex<Box<dyn SttStream>>>,
    pcm: Vec<f32>,
) -> Result<StreamUpdate, SessionEnd> {
    let stream = Arc::clone(stream);
    match tokio::task::spawn_blocking(move || {
        stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .feed(&pcm)
    })
    .await
    {
        Ok(Ok(update)) => Ok(update),
        Ok(Err(e)) => Err(SessionEnd::Error(e.to_string())),
        Err(e) => Err(SessionEnd::Error(format!("stream task failed: {e}"))),
    }
}

async fn native_finalize(
    stream: &Arc<Mutex<Box<dyn SttStream>>>,
) -> Result<StreamUpdate, SessionEnd> {
    let stream = Arc::clone(stream);
    match tokio::task::spawn_blocking(move || {
        stream
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .finalize()
    })
    .await
    {
        Ok(Ok(update)) => Ok(update),
        Ok(Err(e)) => Err(SessionEnd::Error(e.to_string())),
        Err(e) => Err(SessionEnd::Error(format!("stream task failed: {e}"))),
    }
}

/// Native stream loop: the family's own committed/tentative split drives the
/// frames; no chunker or local buffer is involved.
async fn run_native_stream(socket: &mut WebSocket, stream: Box<dyn SttStream>) -> SessionEnd {
    let stream = Arc::new(Mutex::new(stream));
    let mut last_committed = String::new();
    loop {
        let msg = match tokio::time::timeout(READ_IDLE_TIMEOUT, socket.recv()).await {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_)) | None) => return SessionEnd::Disconnected,
            Err(_) => return SessionEnd::Error("stream idle timeout".to_string()),
        };
        match msg {
            Message::Binary(bytes) => {
                let pcm = match pcm16_le(&bytes) {
                    Ok(pcm) => pcm,
                    Err(message) => return SessionEnd::Error(message),
                };
                let update = match native_feed(&stream, pcm).await {
                    Ok(update) => update,
                    Err(end) => return end,
                };
                if update.committed_changed {
                    let delta = update
                        .committed
                        .strip_prefix(&last_committed)
                        .unwrap_or(&update.committed)
                        .to_string();
                    last_committed = update.committed.clone();
                    if !delta.is_empty() {
                        let frame = json!({
                            "type": "partial",
                            "text": delta,
                            "committed": update.committed,
                            "tentative": ""
                        })
                        .to_string();
                        if socket.send(Message::Text(frame.into())).await.is_err() {
                            return SessionEnd::Disconnected;
                        }
                    }
                }
                if update.tentative_changed && !update.tentative.trim().is_empty() {
                    let frame = json!({
                        "type": "tentative",
                        "text": update.tentative.clone(),
                        "committed": update.committed,
                        "tentative": update.tentative
                    })
                    .to_string();
                    if socket.send(Message::Text(frame.into())).await.is_err() {
                        return SessionEnd::Disconnected;
                    }
                }
            }
            Message::Text(text) => match parse_command(&text) {
                Ok(Command::Stop) => break,
                Ok(Command::Start { .. }) => {
                    return SessionEnd::Error("duplicate start".to_string());
                }
                Err(message) => return SessionEnd::Error(message),
            },
            Message::Close(_) => return SessionEnd::Disconnected,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
    let update = match native_finalize(&stream).await {
        Ok(update) => update,
        Err(end) => return end,
    };
    let final_text = if update.full.trim().is_empty() {
        update.committed
    } else {
        update.full
    };
    let frame = json!({
        "type": "final",
        "text": final_text.clone(),
        "committed": final_text,
        "tentative": ""
    })
    .to_string();
    if socket.send(Message::Text(frame.into())).await.is_err() {
        return SessionEnd::Disconnected;
    }
    SessionEnd::Completed
}

async fn run_session(socket: &mut WebSocket, state: &AppState) -> SessionEnd {
    // A caller that opens the socket and never sends start must not hold the
    // engine lease forever: the on-demand worker would never retire.
    let start = match tokio::time::timeout(START_TIMEOUT, wait_for_start(socket)).await {
        Ok(Ok(session)) => session,
        Ok(Err(end)) => return end,
        Err(_) => return SessionEnd::Error("no start frame".to_string()),
    };
    let options = Arc::new(TranscribeOptions {
        model: start.model,
        // Streaming dictation is transcription; a translation stream would
        // need a protocol for the target language and has no caller yet.
        task: crate::engine::Task::Transcribe,
        language: start.language.or_else(|| state.cfg.language.clone()),
        target_language: None,
        pnc: start.pnc.or(state.cfg.pnc),
        itn: start.itn.or(state.cfg.itn),
    });
    // A model with a native stream owns the whole session: feed it directly
    // and let the family decide when text commits. GigaAM and whisper return
    // None here and take the chunked fallback below.
    match state.engine.start_stream(&options) {
        Ok(Some(stream)) => return run_native_stream(socket, stream).await,
        Ok(None) => {}
        Err(e) => return SessionEnd::Error(e.to_string()),
    }

    // The commit window is the operator's chunk length capped by the
    // streaming window: long-form models still get bounded re-decodes, and a
    // short --chunk-max-sec keeps its old per-chunk behaviour.
    let window_sec = state
        .chunk_max_sec(options.model.as_deref())
        .min(state.cfg.stream_window_sec);
    let window_len = window_samples(window_sec);
    let hop_sec = state.cfg.stream_hop_sec.min(window_sec);
    let hop_len = window_samples(hop_sec);

    let mut buffer: Vec<f32> = Vec::new();
    let mut committed: Vec<String> = Vec::new();
    // Samples already covered by the last tentative decode; reset on commit.
    let mut last_preview = 0usize;
    loop {
        // A client that sends start and then vanishes keeps the engine lease
        // (and, on-demand, the loaded model) until TCP notices. Bound it.
        let msg = match tokio::time::timeout(READ_IDLE_TIMEOUT, socket.recv()).await {
            Ok(Some(Ok(msg))) => msg,
            Ok(Some(Err(_)) | None) => return SessionEnd::Disconnected,
            Err(_) => return SessionEnd::Error("stream idle timeout".to_string()),
        };
        match msg {
            Message::Binary(bytes) => {
                let pcm = match pcm16_le(&bytes) {
                    Ok(pcm) => pcm,
                    Err(message) => return SessionEnd::Error(message),
                };
                buffer.extend(pcm);
                // Commit full windows first; a silence near the window end is
                // preferred, otherwise the window edge cuts.
                while buffer.len() >= window_len {
                    let cut = first_cut(&buffer, window_sec, state.cfg.vad_threshold);
                    let chunk: Vec<f32> = buffer.drain(..cut).collect();
                    let text = match transcribe(state, chunk, &options).await {
                        Ok(text) => text,
                        Err(end) => return end,
                    };
                    committed.push(text.clone());
                    let display = committed_text(&committed);
                    let frame = json!({
                        "type": "partial",
                        "text": text,
                        "committed": display,
                        "tentative": ""
                    })
                    .to_string();
                    if socket.send(Message::Text(frame.into())).await.is_err() {
                        return SessionEnd::Disconnected;
                    }
                    last_preview = 0;
                }
                // Re-decode only the uncommitted tail, at most once per hop.
                // Re-encoding the whole buffer every hop would cost more than
                // real time once the window is large.
                if buffer.len() >= hop_len && buffer.len() - last_preview >= hop_len {
                    let text = match transcribe(state, buffer.clone(), &options).await {
                        Ok(text) => text,
                        Err(end) => return end,
                    };
                    last_preview = buffer.len();
                    if !text.trim().is_empty() {
                        let display = committed_text(&committed);
                        let frame = json!({
                            "type": "tentative",
                            "text": text.clone(),
                            "committed": display,
                            "tentative": text
                        })
                        .to_string();
                        if socket.send(Message::Text(frame.into())).await.is_err() {
                            return SessionEnd::Disconnected;
                        }
                    }
                }
            }
            Message::Text(text) => match parse_command(&text) {
                Ok(Command::Stop) => break,
                Ok(Command::Start { .. }) => {
                    return SessionEnd::Error("duplicate start".to_string());
                }
                Err(message) => return SessionEnd::Error(message),
            },
            Message::Close(_) => return SessionEnd::Disconnected,
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }

    if !buffer.is_empty() {
        let remainder = std::mem::take(&mut buffer);
        match transcribe(state, remainder, &options).await {
            Ok(text) => committed.push(text),
            Err(end) => return end,
        }
    }
    let final_text = committed_text(&committed);
    let frame = json!({
        "type": "final",
        "text": final_text.clone(),
        "committed": final_text,
        "tentative": ""
    })
    .to_string();
    if socket.send(Message::Text(frame.into())).await.is_err() {
        return SessionEnd::Disconnected;
    }
    SessionEnd::Completed
}

/// The flicker-free committed prefix that tentative frames extend.
fn committed_text(parts: &[String]) -> String {
    crate::api::join_parts(parts.iter().map(String::as_str))
}

/// What the start frame asked for, before the server defaults are folded in.
struct Start {
    model: Option<String>,
    language: Option<String>,
    pnc: Option<bool>,
    itn: Option<bool>,
}

/// How long a client may hold the socket open before sending `start`.
const START_TIMEOUT: Duration = Duration::from_secs(30);

/// How long an established stream may stay silent before it is closed.
const READ_IDLE_TIMEOUT: Duration = Duration::from_secs(300);

/// Largest WebSocket frame/message accepted, in bytes.
const MAX_WS_FRAME: usize = 4 * 1024 * 1024;

/// First phase: only a start command (or ping/pong) is acceptable.
async fn wait_for_start(socket: &mut WebSocket) -> Result<Start, SessionEnd> {
    loop {
        let msg = match socket.recv().await {
            Some(Ok(msg)) => msg,
            Some(Err(_)) | None => return Err(SessionEnd::Disconnected),
        };
        match msg {
            Message::Text(text) => match parse_command(&text) {
                Ok(Command::Start {
                    model,
                    language,
                    pnc,
                    itn,
                }) => {
                    return Ok(Start {
                        model,
                        language,
                        pnc,
                        itn,
                    });
                }
                Ok(Command::Stop) => {
                    return Err(SessionEnd::Error("stop before start".to_string()));
                }
                Err(message) => return Err(SessionEnd::Error(message)),
            },
            Message::Binary(_) => {
                return Err(SessionEnd::Error("binary frame before start".to_string()));
            }
            Message::Close(_) => return Err(SessionEnd::Disconnected),
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}

fn parse_command(text: &str) -> Result<Command, String> {
    serde_json::from_str(text).map_err(|e| format!("invalid command: {e}"))
}

async fn transcribe(
    state: &AppState,
    chunk: Vec<f32>,
    options: &Arc<TranscribeOptions>,
) -> Result<String, SessionEnd> {
    let len = chunk.len();
    // The WS protocol carries text only; the transcript's timestamp rows,
    // which are relative to this drained chunk, are dropped here.
    crate::api::exec::transcribe_range(state, Arc::new(chunk), 0..len, Arc::clone(options))
        .await
        .map(|transcript| transcript.text)
        .map_err(|e| SessionEnd::Error(e.message))
}
