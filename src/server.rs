//! AppState and router assembly.

use std::sync::Arc;

use axum::Router;
use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::get;

use crate::auth::{AuthKeys, require_api_key};
use crate::config::Config;
use crate::engine::SttEngine;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<dyn SttEngine>,
    pub cfg: Arc<Config>,
    pub sem: Arc<tokio::sync::Semaphore>, // permits = cfg.parallel
}

impl AppState {
    pub fn new(engine: Arc<dyn SttEngine>, cfg: Arc<Config>) -> Self {
        let sem = Arc::new(tokio::sync::Semaphore::new(cfg.parallel));
        Self { engine, cfg, sem }
    }
}

/// Auth applies to /v1/* only; /health is open (llama.cpp style).
pub fn build_router(state: AppState, keys: AuthKeys) -> Router {
    let v1 = Router::new()
        .route("/v1/models", get(crate::api::models::list_models))
        .layer(middleware::from_fn_with_state(keys, require_api_key));
    Router::new()
        .route("/health", get(crate::api::health::health))
        .merge(v1)
        .layer(DefaultBodyLimit::max(state.cfg.max_upload_mb * 1024 * 1024))
        .with_state(state)
}
