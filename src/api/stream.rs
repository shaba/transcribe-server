//! WS /v1/audio/stream: streaming transcription over WebSocket.
//!
//! Protocol (control frames are JSON text):
//!   client -> {"type":"start","model":"<alias>"?,"language":"ru"?}
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

use std::sync::Arc;

use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::response::Response;
use serde::Deserialize;
use serde_json::json;

use crate::audio::TARGET_SR;
use crate::chunk::chunk_ranges;
use crate::server::AppState;

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
enum Command {
    Start {
        model: Option<String>,
        language: Option<String>,
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
    ws.on_upgrade(move |socket| handle(socket, state))
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

async fn run_session(socket: &mut WebSocket, state: &AppState) -> SessionEnd {
    let (model, language) = match wait_for_start(socket).await {
        Ok(session) => session,
        Err(end) => return end,
    };
    let language = language.or_else(|| state.cfg.language.clone());
    let max_samples = ((state.cfg.chunk_max_sec * TARGET_SR as f32) as usize).max(1);

    let mut buffer: Vec<f32> = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    loop {
        let msg = match socket.recv().await {
            Some(Ok(msg)) => msg,
            Some(Err(_)) | None => return SessionEnd::Disconnected,
        };
        match msg {
            Message::Binary(bytes) => {
                if bytes.len() % 2 != 0 {
                    return SessionEnd::Error(
                        "odd-length binary frame: expected whole PCM16LE samples".to_string(),
                    );
                }
                buffer.extend(
                    bytes
                        .chunks_exact(2)
                        .map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0),
                );
                // Full window buffered: drain at the chunker's first cut
                // (a silence near the window end when there is one).
                while buffer.len() >= max_samples {
                    let cut = chunk_ranges(
                        &buffer,
                        TARGET_SR,
                        state.cfg.chunk_max_sec,
                        state.cfg.vad_threshold,
                    )[0]
                    .end;
                    let chunk: Vec<f32> = buffer.drain(..cut).collect();
                    let text = match transcribe(state, chunk, &model, &language).await {
                        Ok(text) => text,
                        Err(end) => return end,
                    };
                    let frame = json!({"type": "partial", "text": text}).to_string();
                    if socket.send(Message::Text(frame.into())).await.is_err() {
                        return SessionEnd::Disconnected;
                    }
                    parts.push(text);
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
        match transcribe(state, remainder, &model, &language).await {
            Ok(text) => parts.push(text),
            Err(end) => return end,
        }
    }
    let frame = json!({"type": "final", "text": crate::api::join_parts(&parts)}).to_string();
    if socket.send(Message::Text(frame.into())).await.is_err() {
        return SessionEnd::Disconnected;
    }
    SessionEnd::Completed
}

/// First phase: only a start command (or ping/pong) is acceptable.
async fn wait_for_start(
    socket: &mut WebSocket,
) -> Result<(Option<String>, Option<String>), SessionEnd> {
    loop {
        let msg = match socket.recv().await {
            Some(Ok(msg)) => msg,
            Some(Err(_)) | None => return Err(SessionEnd::Disconnected),
        };
        match msg {
            Message::Text(text) => match parse_command(&text) {
                Ok(Command::Start { model, language }) => return Ok((model, language)),
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
    model: &Option<String>,
    language: &Option<String>,
) -> Result<String, SessionEnd> {
    let len = chunk.len();
    crate::api::exec::transcribe_range(
        state,
        Arc::new(chunk),
        0..len,
        model.clone(),
        language.clone(),
    )
    .await
    .map_err(|e| SessionEnd::Error(e.message))
}
