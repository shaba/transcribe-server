//! Integration tests for the HTTP API, running against a real server
//! bound to an ephemeral port with FakeEngine.

use std::sync::Arc;

use clap::Parser;
use transcribe_server::auth::AuthKeys;
use transcribe_server::config::Config;
use transcribe_server::engine::SttEngine;
use transcribe_server::engine::fake::FakeEngine;
use transcribe_server::server::{AppState, build_router};

/// Build AuthKeys from a list of accepted keys (empty = auth disabled).
pub fn auth_keys(keys: &[&str]) -> AuthKeys {
    AuthKeys(Arc::new(keys.iter().map(|k| k.to_string()).collect()))
}

/// Spawn the app on an ephemeral port; returns the base URL.
pub async fn spawn_app(engine: Arc<dyn SttEngine>, keys: AuthKeys) -> String {
    let cfg = Arc::new(Config::try_parse_from(["ts", "--parallel", "2"]).expect("parse config"));
    let router = build_router(AppState::new(engine, cfg), keys);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind ephemeral port");
    let addr = listener.local_addr().expect("local addr");
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    format!("http://{addr}")
}

pub async fn spawn_fake_app(keys: AuthKeys) -> String {
    spawn_app(Arc::new(FakeEngine), keys).await
}

#[tokio::test]
async fn health_reports_ok_backend_and_models() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let resp = reqwest::get(format!("{base}/health")).await.expect("GET");
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["status"], "ok");
    assert_eq!(json["backend"], "fake");
    assert_eq!(json["models"], serde_json::json!(["fake-model"]));
}

#[tokio::test]
async fn models_returns_openai_style_list() {
    let base = spawn_fake_app(auth_keys(&[])).await;
    let resp = reqwest::get(format!("{base}/v1/models"))
        .await
        .expect("GET");
    assert_eq!(resp.status(), 200);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["object"], "list");
    assert_eq!(json["data"][0]["id"], "fake-model");
    assert_eq!(json["data"][0]["object"], "model");
    assert_eq!(json["data"][0]["owned_by"], "transcribe-server");
    assert_eq!(json["data"].as_array().expect("data array").len(), 1);
}

#[tokio::test]
async fn models_without_key_is_401_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::get(format!("{base}/v1/models"))
        .await
        .expect("GET");
    assert_eq!(resp.status(), 401);
    let json: serde_json::Value = resp.json().await.expect("JSON body");
    assert_eq!(json["error"]["type"], "authentication_error");
}

#[tokio::test]
async fn models_with_valid_key_is_200() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::Client::new()
        .get(format!("{base}/v1/models"))
        .header("authorization", "Bearer secret")
        .send()
        .await
        .expect("GET");
    assert_eq!(resp.status(), 200);
}

#[tokio::test]
async fn health_stays_open_when_keys_configured() {
    let base = spawn_fake_app(auth_keys(&["secret"])).await;
    let resp = reqwest::get(format!("{base}/health")).await.expect("GET");
    assert_eq!(resp.status(), 200);
}
