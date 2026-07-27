//! Shared integration-test helpers: spawn the app on an ephemeral port
//! with FakeEngine and configurable auth keys.

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
    spawn_app_with_cfg(engine, keys, cfg).await
}

/// Spawn the app with an explicit Config (for non-default limits etc.).
pub async fn spawn_app_with_cfg(
    engine: Arc<dyn SttEngine>,
    keys: AuthKeys,
    cfg: Arc<Config>,
) -> String {
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
