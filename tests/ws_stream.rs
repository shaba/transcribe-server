//! Integration tests for the WebSocket streaming endpoint
//! /v1/audio/stream, running against a real server with FakeEngine.

mod common;

use std::sync::Arc;

use clap::Parser;
use common::{auth_keys, spawn_app_with_cfg, spawn_fake_app};
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::tungstenite::client::IntoClientRequest;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, connect_async};
use transcribe_server::config::Config;
use transcribe_server::engine::fake::FakeEngine;

type WsStream = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

/// Connect to ws://.../v1/audio/stream, optionally with a Bearer key.
async fn connect_ws(base: &str, bearer: Option<&str>) -> WsStream {
    let url = format!("{}/v1/audio/stream", base.replace("http://", "ws://"));
    let mut request = url.into_client_request().expect("ws request");
    if let Some(key) = bearer {
        request.headers_mut().insert(
            "authorization",
            format!("Bearer {key}").parse().expect("header value"),
        );
    }
    let (ws, _) = connect_async(request).await.expect("ws connect");
    ws
}

/// PCM16LE frame: `samples` i16 samples of a 440 Hz sine at half amplitude.
fn sine_frame(samples: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples * 2);
    for k in 0..samples {
        let v = (2.0 * std::f64::consts::PI * 440.0 * k as f64 / 16_000.0).sin() * 0.5;
        bytes.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
    }
    bytes
}

async fn send_json(ws: &mut WsStream, value: serde_json::Value) {
    ws.send(Message::Text(value.to_string().into()))
        .await
        .expect("send text");
}

/// Next JSON text frame from the server (skipping ping/pong).
async fn recv_json(ws: &mut WsStream) -> serde_json::Value {
    loop {
        let msg = ws
            .next()
            .await
            .expect("connection open")
            .expect("ws message");
        match msg {
            Message::Text(text) => return serde_json::from_str(&text).expect("JSON frame"),
            Message::Ping(_) | Message::Pong(_) => continue,
            other => panic!("expected text frame, got {other:?}"),
        }
    }
}

/// After the server closes, the client sees a Close frame and then None.
async fn assert_server_closed(ws: &mut WsStream) {
    loop {
        match ws.next().await {
            None | Some(Err(_)) | Some(Ok(Message::Close(_))) => return,
            Some(Ok(Message::Ping(_))) | Some(Ok(Message::Pong(_))) => continue,
            Some(Ok(other)) => panic!("expected close, got {other:?}"),
        }
    }
}

#[tokio::test]
async fn start_frames_stop_yields_final_text() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let mut ws = connect_ws(&base, Some("secret")).await;

    send_json(&mut ws, json!({"type": "start"})).await;
    for _ in 0..2 {
        ws.send(Message::Binary(sine_frame(8000).into()))
            .await
            .expect("send binary");
    }
    send_json(&mut ws, json!({"type": "stop"})).await;

    let final_msg = recv_json(&mut ws).await;
    assert_eq!(final_msg["type"], "final");
    assert_eq!(final_msg["text"], "fake:default:16000");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn start_frame_carries_the_toggles_to_the_engine() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let mut ws = connect_ws(&base, None).await;

    send_json(
        &mut ws,
        json!({"type": "start", "model": "ru", "pnc": false, "itn": true}),
    )
    .await;
    ws.send(Message::Binary(sine_frame(8000).into()))
        .await
        .expect("send binary");
    send_json(&mut ws, json!({"type": "stop"})).await;

    let final_msg = recv_json(&mut ws).await;
    assert_eq!(final_msg["type"], "final");
    assert_eq!(final_msg["text"], "fake:ru:8000:pnc=off:itn=on");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn binary_before_start_is_error() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let mut ws = connect_ws(&base, None).await;

    ws.send(Message::Binary(sine_frame(100).into()))
        .await
        .expect("send binary");

    let err = recv_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert!(err["message"].is_string());
    assert_server_closed(&mut ws).await;
}

/// With chunk_max_sec 0.5 (8000 samples) each frame drains as a partial;
/// the final is all partial texts joined with a space.
#[tokio::test]
async fn partials_drain_per_chunk_and_final_joins_all() {
    let cfg = Arc::new(
        Config::try_parse_from(["ts", "--parallel", "2", "--chunk-max-sec", "0.5"])
            .expect("parse config"),
    );
    let base = spawn_app_with_cfg(Arc::new(FakeEngine), auth_keys(&[]), cfg).await;
    let mut ws = connect_ws(&base, None).await;

    send_json(&mut ws, json!({"type": "start", "model": "m1"})).await;
    for _ in 0..2 {
        ws.send(Message::Binary(sine_frame(8000).into()))
            .await
            .expect("send binary");
    }
    send_json(&mut ws, json!({"type": "stop"})).await;

    for _ in 0..2 {
        let partial = recv_json(&mut ws).await;
        assert_eq!(partial["type"], "partial");
        assert_eq!(partial["text"], "fake:m1:8000");
    }
    let final_msg = recv_json(&mut ws).await;
    assert_eq!(final_msg["type"], "final");
    assert_eq!(final_msg["text"], "fake:m1:8000 fake:m1:8000");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn stop_without_audio_yields_empty_final() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let mut ws = connect_ws(&base, None).await;

    send_json(&mut ws, json!({"type": "start"})).await;
    send_json(&mut ws, json!({"type": "stop"})).await;

    let final_msg = recv_json(&mut ws).await;
    assert_eq!(final_msg["type"], "final");
    assert_eq!(final_msg["text"], "");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn odd_length_binary_frame_is_error() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let mut ws = connect_ws(&base, None).await;

    send_json(&mut ws, json!({"type": "start"})).await;
    ws.send(Message::Binary(vec![0u8, 1, 2].into()))
        .await
        .expect("send binary");

    let err = recv_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn unknown_command_type_is_error() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let mut ws = connect_ws(&base, None).await;

    send_json(&mut ws, json!({"type": "start"})).await;
    send_json(&mut ws, json!({"type": "bogus"})).await;

    let err = recv_json(&mut ws).await;
    assert_eq!(err["type"], "error");
    assert_server_closed(&mut ws).await;
}

#[tokio::test]
async fn ws_without_key_is_rejected_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let url = format!("{}/v1/audio/stream", base.replace("http://", "ws://"));
    let request = url.into_client_request().expect("ws request");
    let err = connect_async(request).await.expect_err("handshake fails");
    match err {
        tokio_tungstenite::tungstenite::Error::Http(resp) => {
            assert_eq!(resp.status(), 401);
        }
        other => panic!("expected HTTP 401 handshake error, got {other:?}"),
    }
}
